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
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from nemo_curator.stages.image.annotation import hpsv3_server
from nemo_curator.stages.image.annotation.hpsv3_server import HPSv3HTTPServer, HPSv3Scorer


class _Inferencer:
    def __init__(self) -> None:
        self.prompts = []
        self.batch_sizes = []

    def reward(self, prompts: list[str], images: list[Image.Image]) -> torch.Tensor:
        self.prompts.extend(prompts)
        self.batch_sizes.append(len(prompts))
        assert all(image.mode == "RGB" for image in images)
        return torch.tensor([[float(len(prompt)), 0.5] for prompt in prompts])


def _tar_image(tmp_path: Path) -> tuple[Path, int, int]:
    image = Image.new("RGB", (8, 8), color="red")
    data = io.BytesIO()
    image.save(data, format="PNG")
    content = data.getvalue()
    path = tmp_path / "images.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("sample.png")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with tarfile.open(path) as archive:
        offset = archive.getmember("sample.png").offset_data
    return path, offset, len(content)


def test_file_launch_imports_external_hpsv3(tmp_path: Path) -> None:
    package = tmp_path / "hpsv3"
    package.mkdir()
    (package / "__init__.py").write_text(
        "class HPSv3RewardInferencer:\n    def __init__(self, **kwargs):\n        pass\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), environment.get("PYTHONPATH", "")])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(  # noqa: S603 -- launch this repository's worker with a test-only model package
        [
            sys.executable,
            str(Path(hpsv3_server.__file__)),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--checkpoint",
            "unused",
            "--instance-id",
            "test-instance",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        announced = process.stdout.readline()
        assert announced.startswith("HPSV3_ENDPOINT=")
        address = json.loads(announced.removeprefix("HPSV3_ENDPOINT="))
        assert address["instance_id"] == "test-instance"
        port = address["port"]
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"HPSv3 server exited during startup: {process.stdout.read()}")
            try:
                with urlopen(f"{base}/health", timeout=0.5) as response:  # noqa: S310
                    health = json.load(response)
                    assert health["ready"]
                    assert health["instance_id"] == "test-instance"
                break
            except URLError:
                time.sleep(0.1)
        else:
            pytest.fail("HPSv3 server did not become ready")

        with urlopen(Request(f"{base}/drain", data=b"{}"), timeout=5) as response:  # noqa: S310
            assert json.load(response) == {"drained": True}
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        process.stdout.close()


def test_scores_same_image_with_each_caption_and_marks_missing_caption(tmp_path: Path) -> None:
    path, offset, size = _tar_image(tmp_path)
    inferencer = _Inferencer()
    items = [
        {"image_id": image_id, "tar_path": str(path), "offset": offset, "size": size, "caption": caption}
        for image_id, caption in [("a", "cat"), ("b", "red cat"), ("c", None)]
    ]
    scorer = HPSv3Scorer(inferencer, max_batch_size=1)
    try:
        rows = scorer.score(items)
    finally:
        scorer.close()
    assert inferencer.prompts == ["cat", "red cat"]
    assert inferencer.batch_sizes == [1, 1]
    assert [(row["image_id"], row["status"], row["score_mu"]) for row in rows] == [
        ("a", "ok", 3.0),
        ("b", "ok", 7.0),
        ("c", "skipped", None),
    ]


def test_concurrent_requests_share_one_model_batch(tmp_path: Path) -> None:
    path, offset, size = _tar_image(tmp_path)
    inferencer = _Inferencer()
    scorer = HPSv3Scorer(inferencer, max_batch_size=4, batch_wait_ms=1000)
    barrier = threading.Barrier(4)
    results = {}

    def score(index: int) -> None:
        barrier.wait()
        item = {
            "image_id": str(index),
            "tar_path": str(path),
            "offset": offset,
            "size": size,
            "caption": "cat" * (index + 1),
        }
        results[index] = scorer.score([item])[0]

    threads = [threading.Thread(target=score, args=(index,)) for index in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        assert all(not thread.is_alive() for thread in threads)
        assert inferencer.batch_sizes == [4]
        assert [results[index]["score_mu"] for index in range(4)] == [3.0, 6.0, 9.0, 12.0]
    finally:
        scorer.close()


def test_drain_waits_for_in_flight_score() -> None:
    entered = threading.Event()
    release = threading.Event()

    class _BlockingScorer:
        def score(self, items: list[dict]) -> list[dict]:
            entered.set()
            assert release.wait(5)
            return items

    server = HPSv3HTTPServer(("127.0.0.1", 0), _BlockingScorer())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    results = []

    def post(path: str) -> None:
        request = Request(f"{base}{path}", data=b'{"items": []}')  # noqa: S310 -- local HTTP test server
        with urlopen(request, timeout=5) as response:  # noqa: S310
            results.append((path, json.load(response)))

    score_thread = threading.Thread(target=post, args=("/score",))
    drain_thread = threading.Thread(target=post, args=("/drain",))
    try:
        score_thread.start()
        assert entered.wait(5)
        drain_thread.start()
        deadline = time.monotonic() + 5
        while not server.draining and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.draining
        assert drain_thread.is_alive()
        with pytest.raises(HTTPError) as error:
            post("/score")
        assert error.value.code == 503
        release.set()
        score_thread.join(5)
        drain_thread.join(5)
        assert not score_thread.is_alive()
        assert not drain_thread.is_alive()
        assert {path for path, _ in results} == {"/score", "/drain"}
        assert dict(results)["/drain"] == {"drained": True}
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_invalid_request_does_not_block_drain() -> None:
    scorer = HPSv3Scorer(_Inferencer())
    server = HPSv3HTTPServer(("127.0.0.1", 0), scorer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(f"{base}/score", data=b"invalid json"), timeout=5)  # noqa: S310
        assert error.value.code == 400
        with urlopen(Request(f"{base}/drain", data=b"{}"), timeout=5) as response:  # noqa: S310
            assert json.load(response) == {"drained": True}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        scorer.close()


@pytest.mark.parametrize("encoding", ["raw_chw_uint8", "encoded"])
def test_vlv_reads_original_rows_and_keeps_other_images_when_one_is_bad(tmp_path: Path, encoding: str) -> None:
    pixels = np.zeros((3, 8, 8), dtype=np.uint8)
    pixels[0] = 127
    if encoding == "encoded":
        buffer = io.BytesIO()
        Image.fromarray(pixels.transpose(1, 2, 0)).save(buffer, format="PNG")
        content = buffer.getvalue()
    else:
        content = pixels.tobytes()
    path = tmp_path / "vlv.parquet"
    pq.write_table(pa.table({"image": [None, content, b"broken", content]}), path, row_group_size=2)
    items = [
        {
            "image_id": str(index),
            "parquet_path": str(path),
            "row_index": index,
            "image_column": "image",
            "image_encoding": encoding,
            "image_shape": [3, 8, 8],
            "caption": "  original caption\n" if index else None,
        }
        for index in (3, 2, 1, 0)
    ]

    class Inferencer(_Inferencer):
        def reward(self, prompts: list[str], images: list[Image.Image]) -> torch.Tensor:
            assert all(image.getpixel((0, 0)) == (127, 0, 0) for image in images)
            return super().reward(prompts, images)

    inferencer = Inferencer()
    scorer = HPSv3Scorer(inferencer, batch_wait_ms=0)
    try:
        rows = scorer.score(items)
        assert [row["image_id"] for row in rows] == ["3", "2", "1", "0"]
        assert [row["status"] for row in rows] == ["ok", "failed", "ok", "skipped"]
        assert inferencer.prompts == ["  original caption\n", "  original caption\n"]
    finally:
        scorer.close()


def test_tar_size_can_be_read_from_header(tmp_path: Path) -> None:
    path, offset, _size = _tar_image(tmp_path)
    scorer = HPSv3Scorer(_Inferencer(), batch_wait_ms=0)
    try:
        rows = scorer.score(
            [{"image_id": "a", "tar_path": str(path), "offset": offset, "size": None, "caption": "cat"}]
        )
        assert rows[0]["status"] == "ok"
    finally:
        scorer.close()


def test_missing_source_is_a_read_failure_not_a_bad_image(tmp_path: Path) -> None:
    scorer = HPSv3Scorer(_Inferencer())
    try:
        with pytest.raises(FileNotFoundError):
            scorer.score(
                [
                    {
                        "image_id": "a",
                        "tar_path": str(tmp_path / "missing.tar"),
                        "offset": 512,
                        "size": 10,
                        "caption": "cat",
                    }
                ]
            )
    finally:
        scorer.close()
