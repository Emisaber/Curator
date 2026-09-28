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

"""Tests for image annotation through an OpenAI-compatible VLM endpoint."""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import numpy as np
import pandas as pd
import pytest
from openai import APIStatusError

from nemo_curator.stages.image.annotation.vlm import ImageVLMAnnotationStage, VLMAnnotationWriter, parse_annotation
from nemo_curator.tasks import ImageBatch, ImageObject


class _VLMHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    models: ClassVar[list[str]] = ["test-vlm"]

    def do_GET(self) -> None:
        assert self.path == "/v1/models"
        self._send({"object": "list", "data": [{"id": model, "object": "model"} for model in self.models]})

    def do_POST(self) -> None:
        assert self.path == "/v1/chat/completions"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(body)
        self._send(
            {
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"clarity":"sharp","watermark":false}'},
                    }
                ],
            }
        )

    def _send(self, payload: dict) -> None:
        response = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args: object) -> None:
        pass


def test_selective_fields_and_image_request() -> None:
    _VLMHandler.requests = []
    _VLMHandler.models = ["test-vlm"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stage = ImageVLMAnnotationStage(
            fields=["clarity", "watermark"], base_url=f"http://127.0.0.1:{server.server_port}/v1"
        )
        stage.setup()
        image = np.full((24, 32, 3), 100, dtype=np.uint8)
        task = ImageBatch(
            dataset_name="images",
            data=[ImageObject(image_id="sample", image_path="a.tar:512:a.jpg", image_data=image)],
        )
        result = stage.process(task)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result is not None
    row = result.to_pandas().iloc[0]
    assert row["image_id"] == "sample"
    assert row["image_path"] == "a.tar:512:a.jpg"
    assert row["vlm_model"] == "test-vlm"
    assert row["clarity"] == "sharp"
    assert bool(row["watermark"]) is False
    request = _VLMHandler.requests[0]
    assert request["model"] == "test-vlm"
    assert request["response_format"]["json_schema"]["schema"]["required"] == ["clarity", "watermark"]
    prompt = request["messages"][0]["content"][0]["text"]
    assert '"clarity"' in prompt
    assert '"watermark"' in prompt
    assert '"scene_type"' not in prompt
    image_url = request["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\xff\xd8")


def test_multiple_served_models_require_explicit_selection() -> None:
    _VLMHandler.models = ["model-a", "model-b"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stage = ImageVLMAnnotationStage(fields=["clarity"], base_url=f"http://127.0.0.1:{server.server_port}/v1")
        with pytest.raises(ValueError, match="Expected one served model"):
            stage.setup()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_response_must_match_requested_fields_and_values() -> None:
    assert parse_annotation('{"clarity":"sharp","watermark":false}', ["clarity", "watermark"]) == {
        "clarity": "sharp",
        "watermark": False,
    }
    with pytest.raises(ValueError, match="fields differ"):
        parse_annotation('{"clarity":"sharp","watermark":false}', ["clarity"])
    with pytest.raises(ValueError, match="Invalid VLM annotation"):
        parse_annotation('{"clarity":"unclear"}', ["clarity"])


def test_batch_records_failures_and_retries_only_selected_rows(tmp_path: Path) -> None:
    stage = ImageVLMAnnotationStage(
        fields=["clarity", "rendered_text"], model="test-vlm", output_dir=str(tmp_path), source_name="blip"
    )
    images = [
        ImageObject(
            image_id=f"blip|a.tar|{i}",
            image_path=f"a.tar:{i}:image.jpg",
            image_data=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        for i in range(3)
    ]
    batch = ImageBatch(dataset_name="images", data=images)
    calls = []
    response = '{"clarity":"sharp","rendered_text":[]}'

    def mixed(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        index = len(calls)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response if index != 2 else '{"clarity":'),
                    finish_reason="stop" if index != 2 else "length",
                )
            ]
        )

    stage.client = SimpleNamespace(query_model_response=mixed)
    result = stage.process(batch)
    assert result is not None
    output = VLMAnnotationWriter().process(result)
    saved = pd.read_parquet(output.data[0])
    assert saved["status"].tolist() == ["ok", "length", "ok"]
    assert pd.isna(saved.loc[1, "clarity"])
    assert saved.loc[1, "rendered_text"] is None
    assert saved.loc[1, "raw_response"] == '{"clarity":'
    assert stage.process(batch) is None

    stage.retry_statuses = ("length",)
    stage.client = SimpleNamespace(
        query_model_response=lambda **kwargs: calls.append(kwargs) or mixed_result(response)
    )
    retried = stage.process(batch)
    assert retried is not None
    VLMAnnotationWriter().process(retried)
    saved = pd.read_parquet(output.data[0])
    assert saved["status"].tolist() == ["ok", "ok", "ok"]
    assert len(calls) == 4
    assert stage.process(batch) is None
    assert list(tmp_path.rglob("*.tmp")) == []


def mixed_result(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])


def test_request_error_writes_typed_failed_row(tmp_path: Path) -> None:
    stage = ImageVLMAnnotationStage(
        fields=["visual_quality", "watermark", "rendered_text"],
        model="test-vlm",
        output_dir=str(tmp_path),
        source_name="blip",
    )
    response = httpx.Response(503, request=httpx.Request("POST", "http://localhost/v1/chat/completions"))

    def unavailable(**_kwargs: object) -> None:
        msg = "service unavailable"
        raise APIStatusError(msg, response=response, body=None)

    stage.client = SimpleNamespace(query_model_response=unavailable)
    task = ImageBatch(
        dataset_name="images",
        data=[
            ImageObject(
                image_id="blip|a.tar|0",
                image_path="a.tar:512:image.jpg",
                image_data=np.zeros((8, 8, 3), dtype=np.uint8),
            )
        ],
    )
    result = stage.process(task)
    assert result is not None
    output = VLMAnnotationWriter().process(result)
    saved = pd.read_parquet(output.data[0])
    assert saved.loc[0, "status"] == "request_error"
    assert pd.isna(saved.loc[0, "visual_quality"])
    assert pd.isna(saved.loc[0, "watermark"])
    assert saved.loc[0, "rendered_text"] is None
    assert pd.isna(saved.loc[0, "raw_response"])


def test_requests_per_worker_are_concurrent_and_keep_image_order() -> None:
    class SlowHandler(_VLMHandler):
        active = 0
        peak = 0
        lock = threading.Lock()

        def do_POST(self) -> None:
            with self.lock:
                type(self).active += 1
                type(self).peak = max(type(self).peak, type(self).active)
            try:
                time.sleep(0.05)
                super().do_POST()
            finally:
                with self.lock:
                    type(self).active -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stage = ImageVLMAnnotationStage(
            fields=["clarity", "watermark"],
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="test-vlm",
            requests_per_worker=4,
        )
        stage.setup()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        task = ImageBatch(
            dataset_name="images",
            data=[
                ImageObject(image_id=f"image-{index}", image_path=f"a.tar:{index}:image.jpg", image_data=image)
                for index in range(8)
            ],
        )
        result = stage.process(task)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result is not None
    rows = result.to_pandas()
    assert rows["image_id"].tolist() == [f"image-{index}" for index in range(8)]
    assert rows["status"].tolist() == ["ok"] * 8
    assert SlowHandler.peak > 1


def test_multiple_endpoints_receive_requests() -> None:
    from nemo_curator.utils.hash_utils import get_deterministic_hash

    class FirstHandler(_VLMHandler):
        requests: ClassVar[list[dict]] = []

    class SecondHandler(_VLMHandler):
        requests: ClassVar[list[dict]] = []

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), handler) for handler in (FirstHandler, SecondHandler)]
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        stage = ImageVLMAnnotationStage(
            fields=["clarity", "watermark"],
            model="test-vlm",
            base_urls=tuple(f"http://127.0.0.1:{server.server_port}/v1" for server in servers),
        )
        stage.setup()
        images = []
        for endpoint in range(2):
            image_id = next(
                str(index) for index in range(10) if int(get_deterministic_hash([str(index)]), 16) % 2 == endpoint
            )
            images.append(
                ImageObject(
                    image_id=image_id,
                    image_path=f"a.tar:{endpoint}:image.jpg",
                    image_data=np.zeros((8, 8, 3), dtype=np.uint8),
                )
            )
        result = stage.process(ImageBatch(dataset_name="images", data=images))
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()

    assert result is not None
    assert result.to_pandas()["status"].tolist() == ["ok", "ok"]
    assert len(FirstHandler.requests) == 1
    assert len(SecondHandler.requests) == 1
