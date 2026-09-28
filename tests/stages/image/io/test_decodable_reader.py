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
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from nemo_curator.stages.image.io.decodable_reader import (
    DecodableImageReaderStage,
    read_indexed_tar_with_dali,
    run_dali_decode,
)
from nemo_curator.stages.image.io.tar_image_reader import decode_tar_images
from nemo_curator.tasks import FileGroupTask


def _write_tar(path: Path, corrupt: set[int], *, exif: bool = False) -> None:
    with tarfile.open(path, "w") as archive:
        for index in range(4):
            buffer = io.BytesIO()
            image = Image.new("RGB", (12, 8), "red")
            orientation = image.getexif()
            orientation[274] = 6 if exif else 1
            image.save(buffer, format="JPEG", exif=orientation)
            content = b"broken image" if index in corrupt else buffer.getvalue()
            member = tarfile.TarInfo(f"sample_{index}.jpg")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    with tarfile.open(path) as archive:
        members = list(archive)
    with Path(f"{path}.idx").open("w", encoding="utf-8") as index_file:
        index_file.write(f"v1.2 {len(members)}\n")
        index_file.writelines(f"jpg {member.offset_data} {member.size} {member.name}\n" for member in members)


def _reader(root: Path, *, indexed: bool = True) -> DecodableImageReaderStage:
    return DecodableImageReaderStage(
        source_name="long",
        source_root=str(root),
        records_dir=str(root / "decode"),
        dali_batch_size=2,
        num_threads=2,
        index_suffix=".idx" if indexed else None,
    )


@pytest.mark.parametrize("corrupt", [set(), {2}, {0, 1, 2, 3}])
def test_real_dali_tar_keeps_valid_members_and_writes_one_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: set[int]
) -> None:
    pytest.importorskip("nvidia.dali")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "part.tar"
    _write_tar(path, corrupt)
    batches = _reader(tmp_path).process(FileGroupTask(dataset_name="test", data=[str(path)]))
    images = [image for batch in batches for image in batch.data]
    assert [image.image_id for image in images] == [f"long|part.tar|sample_{i}" for i in range(4) if i not in corrupt]
    assert all(image.image_data.shape == (8, 12, 3) and image.image_data.dtype == np.uint8 for image in images)
    files = list((tmp_path / "decode").glob("*.parquet"))
    assert len(files) == 1
    records = pq.ParquetFile(files[0]).read().to_pylist()
    assert len(records) == 4
    assert {row["image_id"] for row in records if row["status"] == "failed"} == {
        f"long|part.tar|sample_{i}" for i in corrupt
    }
    assert all(row["error"] and row["width"] is None for row in records if row["status"] == "failed")
    assert all(row["width"] == 12 and row["short_side_below_256"] for row in records if row["status"] == "ok")
    assert "source" not in records[0]


def test_tar_headers_support_cpu_fallback_without_an_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("nvidia.dali")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "part.tar"
    _write_tar(path, {2})
    batches = _reader(tmp_path, indexed=False).process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert [image.image_id for batch in batches for image in batch.data] == [
        "long|part.tar|sample_0",
        "long|part.tar|sample_1",
        "long|part.tar|sample_3",
    ]


def test_mixed_formats_fallback_uses_the_same_configured_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("nvidia.dali")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "mixed.tar"
    formats = {"a.JPG": "JPEG", "b.jpeg": "JPEG", "c.png": "PNG", "d.webp": "WEBP", "bad.jpg": None}
    with tarfile.open(path, "w") as archive:
        for name, image_format in formats.items():
            payload = io.BytesIO()
            if image_format:
                Image.new("RGB", (12, 8), "red").save(payload, format=image_format)
            else:
                payload.write(b"broken")
            member = tarfile.TarInfo(name)
            member.size = len(payload.getvalue())
            archive.addfile(member, io.BytesIO(payload.getvalue()))
    reader = _reader(tmp_path, indexed=False)
    reader.image_extensions = ("jpg", "jpeg", "png", "webp")
    reader.case_sensitive_extensions = False
    batches = reader.process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert {image.image_id for batch in batches for image in batch.data} == {f"long|mixed.tar|{key}" for key in "abcd"}
    rows = pq.ParquetFile(next((tmp_path / "decode").glob("*.parquet"))).read().to_pylist()
    assert len(rows) == 5
    assert {row["image_id"] for row in rows if row["status"] == "failed"} == {"long|mixed.tar|bad"}


@pytest.mark.parametrize("message", ["CUDA out of memory", "CUDA initialization error", "Could not open index file"])
def test_dali_runtime_failures_are_not_image_errors(message: str) -> None:
    class FailingPipeline:
        def run(self) -> None:
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message) as caught:
        run_dali_decode(FailingPipeline())
    assert type(caught.value) is RuntimeError


def test_dali_failure_outside_decode_does_not_write_failed_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reader = _reader(tmp_path)

    def fail_build(_paths: list[str]) -> None:
        message = "Unrecognized image format in reader setup"
        raise RuntimeError(message)

    monkeypatch.setattr(reader, "_create_dali_pipeline", fail_build)
    with pytest.raises(RuntimeError, match="reader setup"):
        reader.process(FileGroupTask(dataset_name="test", data=[str(tmp_path / "part.tar")]))
    assert not (tmp_path / "decode").exists()


def test_indexed_dali_and_cpu_fallback_keep_stored_exif_orientation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("nvidia.dali")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "part.tar"
    _write_tar(path, set(), exif=True)
    rows = _reader(tmp_path)._tar_records(path)
    dali_images = [image for batch in read_indexed_tar_with_dali(rows, str(tmp_path), 2, 2) for image in batch]
    cpu_images, _ = decode_tar_images(rows, {"long": str(tmp_path)}, adjust_orientation=False)
    assert [image.image_id for image in dali_images] == [image.image_id for image in cpu_images]
    assert all(image.image_data.shape == (8, 12, 3) for image in dali_images + cpu_images)


@pytest.mark.gpu
def test_mixed_dali_tar_fallback_keeps_valid_members(tmp_path: Path) -> None:
    pytest.importorskip("nvidia.dali")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mixed decode")
    path = tmp_path / "part.tar"
    _write_tar(path, {2}, exif=True)
    batches = _reader(tmp_path).process(FileGroupTask(dataset_name="test", data=[str(path)]))
    images = [image for batch in batches for image in batch.data]
    assert [image.image_id for image in images] == [f"long|part.tar|sample_{i}" for i in (0, 1, 3)]
    assert all(image.image_data.shape == (8, 12, 3) for image in images)
