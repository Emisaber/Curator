# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import io
import pathlib
import sys
import tarfile
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable
from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.tasks.image import ImageBatch, ImageObject


class _FakeTensorList:
    """Minimal stand-in for a DALI TensorList returned by Pipeline.run()."""

    def __init__(self, batch_size: int, height: int = 8, width: int = 8) -> None:
        self._arrays: list[np.ndarray] = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(batch_size)]

    def as_cpu(self) -> _FakeTensorList:
        return self

    def __len__(self) -> int:
        return len(self._arrays)

    def at(self, index: int) -> np.ndarray:
        return self._arrays[index]


@dataclass
class _FakePipeline:
    """A fake DALI pipeline that yields a fixed batch size until a total is reached."""

    total_samples: int
    batch_size: int

    def build(self) -> None:
        return None

    def epoch_size(self) -> dict[int, int]:
        return {0: self.total_samples}

    def run(self) -> _FakeTensorList:
        return _FakeTensorList(self.batch_size)


def _fake_create_pipeline_factory(per_tar_total: int, batch: int) -> Callable[[list[str]], _FakePipeline]:
    def _factory(tar_paths: list[str] | tuple[str, ...]) -> _FakePipeline:
        num_paths = len(tar_paths) if isinstance(tar_paths, (list, tuple)) else 1
        return _FakePipeline(total_samples=per_tar_total * num_paths, batch_size=batch)

    return _factory


@pytest.fixture(autouse=True)
def _stub_dali_modules() -> None:
    """Stub nvidia.dali only on CPU-only environments without real DALI.

    We avoid stubbing when CUDA is available so the GPU test can either use
    the real DALI (if installed) or skip cleanly if it's not.
    """
    import importlib.util

    if torch.cuda.is_available():
        return
    # Some environments may have a broken/partial installation where
    # nvidia.dali is present in sys.modules with __spec__ = None.
    # importlib.util.find_spec raises ValueError in that case. Treat this as
    # "not installed" so we provide our stub.
    try:
        dali_spec = importlib.util.find_spec("nvidia.dali")
    except (ValueError, ModuleNotFoundError, ImportError):
        dali_spec = None
    if dali_spec is not None:
        return

    nvidia = types.ModuleType("nvidia")
    dali = types.ModuleType("nvidia.dali")
    pipeline = types.ModuleType("nvidia.dali.pipeline")

    def pipeline_def(*_args: object, **_kwargs: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def _decorator(func: Callable[..., object]) -> Callable[..., object]:
            return func

        return _decorator

    class _Types:
        RGB = None

    dali.pipeline_def = pipeline_def
    dali.types = _Types
    dali.fn = types.SimpleNamespace(
        readers=types.SimpleNamespace(webdataset=lambda **_kwargs: None),
        decoders=types.SimpleNamespace(image=lambda *_a, **_k: None),
    )
    pipeline.Pipeline = type("Pipeline", (), {})

    sys.modules["nvidia"] = nvidia
    sys.modules["nvidia.dali"] = dali
    sys.modules["nvidia.dali.pipeline"] = pipeline


def test_inputs_outputs_and_name() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(dali_batch_size=3, verbose=False)
    assert stage.inputs() == ([], [])
    assert stage.outputs() == (["data"], ["image_data", "image_path", "image_id"])
    assert stage.name == "image_reader"
    assert stage.ray_stage_spec()["is_fanout_stage"] is True


def test_init_allows_cpu_when_no_cuda() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    # When CUDA is unavailable, the stage should initialize and use CPU DALI
    with patch("torch.cuda.is_available", return_value=False):
        stage = ImageReaderStage(dali_batch_size=2, verbose=False)
    assert stage is not None


def test_cpu_dali_pipeline_has_no_gpu_device_id() -> None:
    import nvidia.dali

    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    options = {}

    def pipeline_def(**kwargs) -> object:
        options.update(kwargs)
        return lambda _func: lambda: types.SimpleNamespace(build=lambda: None)

    with patch("torch.cuda.is_available", return_value=False), patch.object(nvidia.dali, "pipeline_def", pipeline_def):
        ImageReaderStage()._create_dali_pipeline(["a.tar"])
    assert options["device_id"] is None


def test_process_streams_batches_from_dali() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    # Two tar files; each has 5 total samples, emitted in batches of 2 (2,2,1)
    task = FileGroupTask(
        dataset_name="ds",
        data=["/data/a.tar", "/data/b.tar"],
        _metadata={"source_files": ["/data/a.tar", "/data/b.tar"]},
    )

    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(dali_batch_size=2, verbose=False)

    with patch.object(
        ImageReaderStage,
        "_create_dali_pipeline",
        side_effect=_fake_create_pipeline_factory(per_tar_total=5, batch=2),
    ):
        batches = stage.process(task)

    assert isinstance(batches, list)
    assert all(isinstance(b, ImageBatch) for b in batches)
    assert all(b.dataset_name == task.dataset_name and b._metadata == task._metadata for b in batches)
    assert len({id(b._metadata) for b in batches}) == len(batches)
    assert len({id(b._stage_perf) for b in batches}) == len(batches)

    total_images = sum(len(b.data) for b in batches)
    assert total_images == 10  # 2 tars * 5 images each
    # Spot-check a couple of ImageObject fields
    assert all(isinstance(img, ImageObject) for b in batches for img in b.data)


def test_source_info_preserves_webdataset_member_identity() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    task = FileGroupTask(dataset_name="ds", data=["/data/a.tar"])
    source = _FakeTensorList(2)
    source._arrays = [
        np.frombuffer(b"/data/a.tar:512:one.jpg", dtype=np.uint8),
        np.frombuffer(b"/data/a.tar:1536:two.jpg", dtype=np.uint8),
    ]
    pipe = _FakePipeline(total_samples=2, batch_size=2)
    pipe.run = lambda: (_FakeTensorList(2), source)

    with patch("torch.cuda.is_available", return_value=False):
        stage = ImageReaderStage(source_name="blip", source_root="/data")
    with patch.object(ImageReaderStage, "_create_dali_pipeline", return_value=pipe):
        batches = stage.process(task)

    assert [image.image_id for image in batches[0].data] == ["blip|a.tar|one", "blip|a.tar|two"]
    assert batches[0].data[0].image_path == "/data/a.tar:512:one.jpg"


def test_process_raises_on_empty_task() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    empty = FileGroupTask(dataset_name="ds", data=[])

    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(dali_batch_size=2, verbose=False)

    with pytest.raises(ValueError, match="No tar file paths"):
        stage.process(empty)


def test_resources_with_cuda_available() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    # Instantiate with CUDA available so __post_init__ passes
    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(dali_batch_size=2, verbose=False)
        res = stage.resources

    assert res.gpus == stage.num_gpus_per_worker
    assert res.requires_gpu is True


def test_resources_without_cuda() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    # Create the stage without CUDA available
    with patch("torch.cuda.is_available", return_value=False):
        stage = ImageReaderStage(dali_batch_size=2, verbose=False)
        res = stage.resources

    assert res.gpus == 0
    assert res.requires_gpu is False


@pytest.mark.gpu
def test_dali_image_reader_on_gpu() -> None:
    """Test DALI image reader on GPU."""

    # Reuse sample webdataset tar from repository-level tests assets
    # Project root is parents[4] from this file (tests/stages/image/io)
    tar_path = pathlib.Path(__file__).resolve().parents[4] / "tests" / "image_data" / "00000.tar"
    if not tar_path.exists():
        msg = f"Sample dataset not found at {tar_path}"
        raise FileNotFoundError(msg)

    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    from nemo_curator.tasks import FileGroupTask

    stage = ImageReaderStage(dali_batch_size=2, num_threads=2, verbose=False)
    task = FileGroupTask(dataset_name="ds", data=[str(tar_path)])

    batches = stage.process(task)

    # Should yield at least one batch with decoded images
    assert isinstance(batches, list)
    assert len(batches) >= 1
    total_images = 0
    for batch in batches:
        assert len(batch.data) >= 1
        for img in batch.data:
            # Validate decoded image
            assert img.image_data is not None
            assert img.image_data.ndim == 3  # H, W, C
            assert img.image_data.shape[2] == 3
            assert img.image_id != ""
            assert img.image_path.endswith(".jpg")
            total_images += 1

    assert total_images >= 1


@pytest.mark.gpu
def test_dali_source_info_matches_tar_members() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    tar_path = pathlib.Path(__file__).resolve().parents[4] / "tests" / "image_data" / "00000.tar"
    with tarfile.open(tar_path) as archive:
        expected = {
            f"test|00000.tar|{member.name.split('.', 1)[0]}"
            for member in archive
            if member.isfile() and member.name.endswith(".jpg")
        }
    stage = ImageReaderStage(dali_batch_size=2, num_threads=2, source_name="test", source_root=str(tar_path.parent))
    batches = stage.process(FileGroupTask(dataset_name="test", data=[str(tar_path)]))
    assert {image.image_id for batch in batches for image in batch.data} == expected


@pytest.mark.gpu
def test_dali_mixed_image_formats_match_sample_index(tmp_path: pathlib.Path) -> None:
    from nemo_curator.stages.image.io.caption_source import CaptionSource
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    from nemo_curator.stages.image.io.sample_index import WebDatasetImageSampleIndexStage

    tar_path = tmp_path / "mixed.tar"
    formats = {"a.jpg": "JPEG", "b.jpeg": "JPEG", "c.png": "PNG", "d.webp": "WEBP", "e.JPG": "JPEG"}
    with tarfile.open(tar_path, "w") as archive:
        for member_name, image_format in formats.items():
            buffer = io.BytesIO()
            Image.new("RGB", (16, 12), "red").save(buffer, format=image_format)
            payload = buffer.getvalue()
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(tar_path) as archive:
        members = list(archive)
    with open(f"{tar_path}.idx", "w", encoding="utf-8") as index_file:
        index_file.write(f"v1.2 {len(members)}\n")
        index_file.writelines(
            f"{member.name.rsplit('.', 1)[1]} {member.offset_data} {member.size} {member.name}\n" for member in members
        )

    task = FileGroupTask(dataset_name="mixed", data=[str(tar_path)])
    indexed = WebDatasetImageSampleIndexStage("mixed", CaptionSource(str(tmp_path))).process(task)
    expected_ids = set(indexed.data.column("image_id").to_pylist())
    stage = ImageReaderStage(
        dali_batch_size=2,
        num_threads=2,
        source_name="mixed",
        source_root=str(tmp_path),
        index_suffix=".idx",
        image_extensions=("jpg", "jpeg", "png", "webp"),
        case_sensitive_extensions=False,
    )
    batches = stage.process(task)
    images = [image for batch in batches for image in batch.data]

    assert {image.image_id for image in images} == expected_ids
    assert len(images) == len(formats)
    assert all(image.image_data.shape == (12, 16, 3) for image in images)


def test_max_images_per_partition_limits_sample_run() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage

    task = FileGroupTask(dataset_name="ds", data=["/data/a.tar"])
    with patch("torch.cuda.is_available", return_value=False):
        stage = ImageReaderStage(dali_batch_size=2, max_images_per_partition=3)
    with patch.object(ImageReaderStage, "_create_dali_pipeline", side_effect=_fake_create_pipeline_factory(10, 2)):
        batches = stage.process(task)

    assert [len(batch.data) for batch in batches] == [2, 1]
