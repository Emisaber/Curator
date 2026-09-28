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
import tarfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from nemo_curator.stages.image.annotation import ImageVLMAnnotationStage, VLMAnnotationWriter
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.decodable_reader import DecodableImageReaderStage, write_source_decode_records
from nemo_curator.stages.image.io.decoded_image_reader import DecodeRecordImageReaderStage
from nemo_curator.tasks import FileGroupTask
from tests.stages.image.annotation.test_vlm import _VLMHandler


def _tar(path: Path, bad: int | None = None) -> list[dict]:
    with tarfile.open(path, "w") as archive:
        for index in range(3):
            buffer = io.BytesIO()
            Image.new("RGB", (12, 8), "red").save(buffer, format="JPEG")
            content = b"broken image" if index == bad else buffer.getvalue()
            member = tarfile.TarInfo(f"{index}.jpg")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    with tarfile.open(path) as archive:
        return [
            {"shard": path.name, "member": member.name, "offset": member.offset_data, "size": member.size}
            for member in archive
        ]


def _source_decode(tmp_path: Path, name: str) -> tuple[CaptionSource, Path]:
    root = tmp_path / "media"
    root.mkdir()
    records_dir = tmp_path / "decode"
    if name in ("long", "short"):
        path = root / "part.tar"
        _tar(path, bad=1 if name == "short" else None)
        source = CaptionSource(str(root), index_suffix=None)
        DecodableImageReaderStage(
            source_name=name,
            source_root=str(root),
            index_suffix=None,
            records_dir=str(records_dir),
            dali_batch_size=2,
            num_threads=2,
        ).process(FileGroupTask(dataset_name="test", data=[str(path)]))
    elif name == "vlv":
        path = root / "part.parquet"
        pq.write_table(pa.table({"image": [bytes(range(12)), None, bytes(range(12))]}), path, row_group_size=1)
        source = CaptionSource(str(root), format="vlv_parquet", image_shape=(3, 2, 2))
        write_source_decode_records(
            [
                {
                    "image_id": f"vlv|part.parquet|{index}",
                    "shard": path.name,
                    "row_index": index,
                    "status": "failed" if index == 1 else "ok",
                }
                for index in range(3)
            ],
            str(records_dir),
            path.name,
        )
    else:
        path = root / "part.tar"
        rows = _tar(path)
        source = CaptionSource(str(root), format="ego4d_recaption")
        write_source_decode_records(
            [{**row, "image_id": f"fho|video|{index}", "status": "ok"} for index, row in enumerate(rows)],
            str(records_dir),
            path.name,
        )
    record_path = next(records_dir.glob("*.parquet"))
    return source, record_path


@pytest.mark.parametrize("name", ["long", "short", "vlv", "ego"])
def test_upstream_samples_reach_vlm_without_rewriting_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    if name != "vlv":
        pytest.importorskip("nvidia.dali")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source, record_path = _source_decode(tmp_path, name)
    original = record_path.read_bytes()
    expected = [row["image_id"] for row in pq.ParquetFile(record_path).read().to_pylist() if row["status"] == "ok"]
    reader = DecodeRecordImageReaderStage(name, source, image_batch_size=2, num_threads=2)
    task = FileGroupTask(dataset_name="test", data=[str(record_path)])
    batches = reader.process(task)
    assert [image.image_id for batch in batches for image in batch.data] == expected
    assert all(image.image_data.dtype == np.uint8 for batch in batches for image in batch.data)
    assert [image.image_id for batch in reader.process(task) for image in batch.data] == expected
    assert record_path.read_bytes() == original

    _VLMHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        annotation = ImageVLMAnnotationStage(
            fields=["clarity", "watermark"],
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="test-vlm",
            source_name=name,
            output_dir=str(tmp_path / "vlm"),
            requests_per_worker=2,
        )
        annotation.setup()
        writer = VLMAnnotationWriter()
        for batch in batches:
            writer.process(annotation.process(batch))
        outputs = sorted((tmp_path / "vlm" / f"source={name}").glob("*.parquet"))
        annotated = [row for path in outputs for row in pq.ParquetFile(path).read().to_pylist()]
        assert {row["image_id"] for row in annotated} == set(expected)
        assert all(row["status"] == "ok" and row["clarity"] == "sharp" for row in annotated)
        assert len(_VLMHandler.requests) == len(expected)
        for batch in reader.process(task):
            assert annotation.process(batch) is None
        assert len(_VLMHandler.requests) == len(expected)
        assert record_path.read_bytes() == original
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_all_failed_partition_produces_no_images(tmp_path: Path) -> None:
    write_source_decode_records(
        [{"image_id": "long|missing.tar|a", "status": "failed", "shard": "missing.tar"}],
        str(tmp_path),
        "missing.tar",
    )
    stage = DecodeRecordImageReaderStage("long", CaptionSource(str(tmp_path)))
    assert stage.process(FileGroupTask(dataset_name="test", data=[str(next(tmp_path.glob("*.parquet")))])) == []


def test_previously_decoded_tar_sample_failure_is_not_silently_skipped(tmp_path: Path) -> None:
    rows = _tar(tmp_path / "part.tar", bad=0)
    write_source_decode_records(
        [
            {**row, "image_id": f"long|part.tar|{index}", "status": "ok" if index == 0 else "failed"}
            for index, row in enumerate(rows)
        ],
        str(tmp_path / "decode"),
        "part.tar",
    )
    record = next((tmp_path / "decode").glob("*.parquet"))
    stage = DecodeRecordImageReaderStage("long", CaptionSource(str(tmp_path)))
    with pytest.raises(RuntimeError, match="Previously decoded sample"):
        stage.process(FileGroupTask(dataset_name="test", data=[str(record)]))
