# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Validate the actual OpenAI request and typed caption recovery."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.image.recaption.vlm import Ego4DRecaptionStage, RecaptionWriter, parse_caption
from nemo_curator.tasks import ImageBatch, ImageObject

CAPTION = {
    "comprehensive_description": "A hand holds a red cup.",
    "prominent_elements": [
        {"name": "cup", "appearance": "red", "location": "center", "state": "", "relationships": ["held by a hand"]}
    ],
}


class CaptionHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    fail = True

    def do_GET(self) -> None:
        self._send({"object": "list", "data": [{"id": "test-vlm", "object": "model"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(body)
        context = json.loads(body["messages"][1]["content"][1]["text"].split("\n", 1)[1])
        limited = self.fail and context.get("anchor_narration", {}).get("text") == "limited"
        self._send(
            {
                "id": "caption",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if limited else "stop",
                        "message": {"role": "assistant", "content": "{" if limited else json.dumps(CAPTION)},
                    }
                ],
            }
        )

    def _send(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args: object) -> None:
        pass


def test_request_parquet_and_failed_only_retry(tmp_path: Path) -> None:
    CaptionHandler.requests = []
    CaptionHandler.fail = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stage = Ego4DRecaptionStage(
            source_kind="narration",
            source_name="ego4d-narration",
            output_dir=str(tmp_path),
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            requests_per_worker=2,
        )
        stage.setup()
        images = [
            ImageObject(
                image_id=str(index),
                image_path="sample.tar:512:a.jpg",
                image_data=np.zeros((24, 32, 3), dtype=np.uint8),
                metadata={
                    "context_json": json.dumps(
                        {
                            "target_frame": {"timestamp_sec": 1},
                            "anchor_narration": {"text": text, "timestamp_sec": 1},
                            "summary": None,
                        }
                    )
                },
            )
            for index, text in enumerate(("normal", "limited"))
        ]
        task = ImageBatch(dataset_name="test", data=images)
        result = stage.process(task)
        path = RecaptionWriter().process(result).data[0]
        saved = pq.ParquetFile(path).read()
        assert saved["status"].to_pylist() == ["ok", "failed"]
        assert saved["error_kind"].to_pylist() == [None, "length"]
        assert saved["caption_version"].to_pylist() == ["v1", "v1"]
        assert saved["prominent_elements"][0].as_py() == CAPTION["prominent_elements"]
        assert stage.process(task) is None
        request = CaptionHandler.requests[0]
        assert request["messages"][0]["role"] == "system"
        assert "#C refers to the camera wearer" in request["messages"][0]["content"]
        assert request["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert request["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        assert request["model"] == "test-vlm"
        CaptionHandler.fail = False
        stage.retry_statuses = ("failed",)
        retried = stage.process(task)
        RecaptionWriter().process(retried)
        assert len(CaptionHandler.requests) == 3
        assert pq.ParquetFile(path).read()["status"].to_pylist() == ["ok", "ok"]
        assert stage.process(task) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        '{"comprehensive_description":1,"prominent_elements":[]}',
        '{"comprehensive_description":"a","prominent_elements":[{}]}',
    ],
)
def test_rejects_wrong_caption_schema(response: str) -> None:
    with pytest.raises((ValueError, TypeError), match="Caption"):
        parse_caption(response)
