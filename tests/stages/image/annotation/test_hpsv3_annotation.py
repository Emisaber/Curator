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

import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from nemo_curator.stages.image.annotation.hpsv3_annotation import HPSv3AnnotationWriter, HPSv3ScoreStage
from nemo_curator.stages.image.annotation.hpsv3_server import HPSv3HTTPServer, HPSv3Scorer
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.hash_utils import get_deterministic_hash


class _ScoreHandler(BaseHTTPRequestHandler):
    captions: ClassVar[list[str]] = []
    requests: ClassVar[list[list[dict]]] = []
    fail_requests: ClassVar[set[int]] = set()

    def do_POST(self) -> None:
        assert self.path == "/score"
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        items = payload["items"]
        self.requests.append(items)
        if len(self.requests) in self.fail_requests:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.captions.extend(item["caption"] for item in items)
        rows = [
            {
                "image_id": item["image_id"],
                "status": "ok",
                "score_mu": float(len(item["caption"])),
                "score_sigma": 0.5,
                "error": None,
            }
            for item in items
        ]
        body = json.dumps({"items": rows}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def test_scores_and_persists_each_sample_caption(tmp_path: Path) -> None:
    _ScoreHandler.captions = []
    _ScoreHandler.requests = []
    _ScoreHandler.fail_requests = set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScoreHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stage = HPSv3ScoreStage(
            source_name="source",
            source_root=str(tmp_path),
            output_dir=str(tmp_path / "annotations"),
            endpoints=[f"http://127.0.0.1:{server.server_port}"],
        )
        samples = DocumentBatch(
            dataset_name="samples",
            data=pd.DataFrame(
                [
                    {"image_id": "a", "shard": "data.tar", "offset": 512, "size": 10, "caption_raw": "cat"},
                    {"image_id": "b", "shard": "data.tar", "offset": 512, "size": 10, "caption_raw": "red cat"},
                ]
            ),
            _metadata={"source_files": [str(tmp_path / "samples.parquet")]},
        )
        result = stage.process(samples)
        assert result is not None
        written = HPSv3AnnotationWriter().process(result)
        rows = pd.read_parquet(written.data[0])
        assert _ScoreHandler.captions == ["cat", "red cat"]
        assert rows["image_id"].tolist() == ["a", "b"]
        assert rows["score_mu"].tolist() == [3.0, 7.0]
        assert rows["caption_sha256"].nunique() == 2
        assert stage.process(samples) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_decode_input_preserves_captions_and_retries_only_failed_rows(tmp_path: Path) -> None:
    path = tmp_path / "vlv.parquet"
    captions = ["first caption", "  second\ncaption  ", "excluded", "last caption"]
    pq.write_table(pa.table({"caption": captions}), path, row_group_size=2)
    ids = [f"vlv|vlv.parquet|{index}" for index in range(4)]
    task = DocumentBatch(
        dataset_name="decode",
        data=pd.DataFrame(
            [
                {
                    "image_id": image_id,
                    "shard": "vlv.parquet",
                    "row_index": index,
                    "status": "failed" if index == 2 else "ok",
                }
                for index, image_id in enumerate(ids)
            ]
        ),
        _metadata={"source_files": [str(tmp_path / "decode.parquet")]},
    )
    _ScoreHandler.requests = []
    _ScoreHandler.captions = []
    _ScoreHandler.fail_requests = {2}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScoreHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stage = HPSv3ScoreStage(
        source_name="vlv",
        source_root=str(tmp_path),
        source=CaptionSource(root=str(tmp_path), format="vlv_parquet"),
        output_dir=str(tmp_path / "annotations"),
        endpoints=[f"http://127.0.0.1:{server.server_port}"],
        inference_batch_size=1,
    )
    try:
        result = stage.process(task)
        written = HPSv3AnnotationWriter().process(result)
        rows = pd.read_parquet(written.data[0])
        assert rows["image_id"].tolist() == [ids[0], ids[1], ids[3]]
        assert rows["status"].tolist() == ["ok", "request_error", "ok"]
        assert all("parquet_path" in items[0] and "tar_path" not in items[0] for items in _ScoreHandler.requests)
        assert stage.process(task) is None
        _ScoreHandler.fail_requests = set()
        _ScoreHandler.requests = []
        stage.retry_statuses = ("request_error",)
        result = stage.process(task)
        HPSv3AnnotationWriter().process(result)
        updated = pd.read_parquet(written.data[0])
        assert len(_ScoreHandler.requests) == 1
        assert _ScoreHandler.requests[0][0]["caption"] == captions[1]
        assert updated["status"].tolist() == ["ok", "ok", "ok"]
        assert updated.iloc[0].to_dict() == rows.iloc[0].to_dict()
        assert updated.iloc[2].to_dict() == rows.iloc[2].to_dict()
        assert not list(Path(written.data[0]).parent.glob("*.tmp"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.parametrize("source_name", ["long", "short", "vlv", "ego4d-fho"])
def test_decode_to_http_to_parquet_reads_each_native_source(tmp_path: Path, source_name: str) -> None:  # noqa: PLR0915
    caption = "  original\ncaption: café  "
    source = CaptionSource(root=str(tmp_path))
    pixels = np.full((3, 4, 4), 127, dtype=np.uint8)
    if source_name == "vlv":
        source.format = "vlv_parquet"
        source.image_shape = (3, 4, 4)
        pq.write_table(
            pa.table({"image": [None, pixels.tobytes()], "caption": ["unused", caption]}),
            tmp_path / "data.parquet",
            row_group_size=1,
        )
        record = {"image_id": "vlv|data.parquet|1", "shard": "data.parquet", "row_index": 1, "status": "ok"}
    else:
        buffer = io.BytesIO()
        Image.fromarray(pixels.transpose(1, 2, 0)).save(buffer, format="PNG")
        archive_path = tmp_path / "data.tar"
        with tarfile.open(archive_path, "w") as archive:
            for extension, content in (("png", buffer.getvalue()), ("txt", caption.encode("utf-8"))):
                member = tarfile.TarInfo(f"sample.{extension}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
        (tmp_path / "data.tar.idx").write_text(
            "v1.2 1\n"
            + " ".join(
                f"{member.name.rsplit('.', 1)[1]} {member.offset_data} {member.size} {member.name}"
                for member in members
            )
            + "\n",
            encoding="utf-8",
        )
        image_id = f"{source_name}|data.tar|sample"
        record = {
            "image_id": image_id,
            "shard": "data.tar",
            "offset": members[0].offset_data,
            "size": None,
            "status": "ok",
        }
        if source_name == "ego4d-fho":
            record["image_id"] = "ego4d-fho|video|10"
            source.format = "ego4d_recaption"
            source.sample_index_root = str(tmp_path / "indices")
            source.annotations_root = str(tmp_path / "captions")
            Path(source.sample_index_root).mkdir()
            Path(source.annotations_root).mkdir()
            pq.write_table(
                pa.Table.from_pylist([{**record, "decode_status": "ok", "caption_raw": None}]),
                Path(source.sample_index_root) / "video-000000.parquet",
            )
            name = get_deterministic_hash([record["image_id"]])
            pq.write_table(
                pa.table({"image_id": [record["image_id"]], "status": ["ok"], "comprehensive_description": [caption]}),
                Path(source.annotations_root) / f"part-{name}.parquet",
            )

    class Inferencer:
        def reward(self, prompts: list[str], images: list[Image.Image]) -> torch.Tensor:
            assert prompts == [caption]
            np.testing.assert_array_equal(np.asarray(images[0]), pixels.transpose(1, 2, 0))
            return torch.tensor([[1.5, 0.25]])

    scorer = HPSv3Scorer(Inferencer(), batch_wait_ms=0)
    server = HPSv3HTTPServer(("127.0.0.1", 0), scorer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stage = HPSv3ScoreStage(
            source_name=source_name,
            source_root=str(tmp_path),
            source=source,
            output_dir=str(tmp_path / "annotations"),
            endpoints=[f"http://127.0.0.1:{server.server_port}"],
            adjust_orientation=False,
        )
        task = DocumentBatch(
            dataset_name="decode",
            data=pd.DataFrame([record]),
            _metadata={"source_files": [str(tmp_path / "decode.parquet")]},
        )
        result = stage.process(task)
        written = HPSv3AnnotationWriter().process(result)
        rows = pq.ParquetFile(written.data[0]).read().to_pylist()
        assert rows[0]["image_id"] == record["image_id"]
        assert rows[0]["status"] == "ok"
        assert rows[0]["score_mu"] == 1.5
        assert rows[0]["score_sigma"] == 0.25
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        scorer.close()
