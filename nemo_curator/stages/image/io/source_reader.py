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

"""Read native VLV shards and existing Ego4D recaption outputs."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from loguru import logger
from PIL import Image, ImageOps, UnidentifiedImageError

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.decodable_reader import (
    ImageDecodeError,
    image_decode_record,
    make_image_batches,
    read_indexed_tar_with_dali,
    write_source_decode_records,
)
from nemo_curator.stages.image.io.tar_image_reader import decode_tar_images
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, ImageBatch, ImageObject
from nemo_curator.utils.hash_utils import get_deterministic_hash


def decode_vlv_image(value: bytes | dict, source: CaptionSource) -> np.ndarray:
    if isinstance(value, dict):
        value = value["bytes"]
    if value is None:
        msg = "Missing image bytes"
        raise ValueError(msg)
    if source.image_encoding == "raw_chw_uint8":
        return np.frombuffer(value, dtype=np.uint8).reshape(source.image_shape).transpose(1, 2, 0)
    if source.image_encoding == "encoded":
        with Image.open(BytesIO(value)) as image:
            return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    msg = f"Unsupported VLV image encoding: {source.image_encoding}"
    raise ValueError(msg)


def read_vlv_rows(path: Path, row_indices: set[int], columns: list[str]) -> dict[int, dict]:
    """Read only row groups containing the requested original row positions."""
    parquet = pq.ParquetFile(path)
    rows = {}
    start = 0
    for group in range(parquet.num_row_groups):
        end = start + parquet.metadata.row_group(group).num_rows
        selected = sorted(index for index in row_indices if start <= index < end)
        if selected:
            table = parquet.read_row_group(group, columns=columns)
            values = table.take([index - start for index in selected]).to_pylist()
            rows.update(zip(selected, values, strict=True))
        start = end
    return rows


def read_ego4d_records(index_path: Path, source: CaptionSource) -> list[dict]:
    """Join one existing frame manifest to its corresponding recaption result."""
    records = pq.ParquetFile(index_path).read().to_pylist()
    filename = get_deterministic_hash([row["image_id"] for row in records])
    caption_path = Path(source.annotations_root) / f"part-{filename}.parquet"
    captions = {
        row["image_id"]: row["comprehensive_description"]
        for row in pq.ParquetFile(caption_path)
        .read(columns=["image_id", "status", "comprehensive_description"])
        .to_pylist()
        if row["status"] == "ok"
    }
    selected = [
        {**row, "caption_raw": captions[row["image_id"]]}
        for row in records
        if row["decode_status"] == "ok" and row["image_id"] in captions
    ]
    logger.info("Ego4D shard {}: {} usable / {} indexed frames", index_path.name, len(selected), len(records))
    return selected


def find_ego4d_records(image_ids: set[str], source: CaptionSource) -> dict[str, dict]:
    """Locate candidate frames through the existing per-video manifests."""
    video_uids = {image_id.split("|", 2)[1] for image_id in image_ids}
    paths = [
        path for path in Path(source.sample_index_root).glob("*.parquet") if path.stem.rsplit("-", 1)[0] in video_uids
    ]
    found = {}
    for path in sorted(paths):
        ids = set(pq.ParquetFile(path).read(columns=["image_id"])["image_id"].to_pylist())
        if ids & image_ids:
            found.update(
                (row["image_id"], row) for row in read_ego4d_records(path, source) if row["image_id"] in image_ids
            )
    return found


@dataclass
class SourceShardReaderStage(ProcessingStage[FileGroupTask, ImageBatch]):
    source_name: str
    source: CaptionSource
    image_batch_size: int = 32
    num_gpus_per_worker: float = 0.25
    num_threads: int = 8
    records_dir: str | None = None
    name: str = "source_shard_reader"

    def __post_init__(self) -> None:
        use_gpu = self.source.format == "ego4d_recaption" and torch.cuda.is_available()
        self.resources = Resources(gpus=self.num_gpus_per_worker) if use_gpu else Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "image_data", "image_path"]

    def process(self, task: FileGroupTask) -> list[ImageBatch]:
        batches = []
        for file_path in task.data:
            path = Path(file_path)
            if self.source.format == "vlv_parquet":
                shard_batches, records, shard = self._read_vlv(path, task)
            elif self.source.format == "ego4d_recaption":
                shard_batches, records, shard = self._read_ego4d(path, task)
            else:
                msg = f"Unsupported native source format: {self.source.format}"
                raise ValueError(msg)
            if self.records_dir is not None:
                write_source_decode_records(records, self.records_dir, shard)
            batches.extend(shard_batches)
        return batches

    def _read_vlv(self, path: Path, task: FileGroupTask) -> tuple[list[ImageBatch], list[dict], str]:
        if self.source.image_encoding not in ("raw_chw_uint8", "encoded"):
            msg = f"Unsupported VLV image encoding: {self.source.image_encoding}"
            raise ValueError(msg)
        shard = path.relative_to(self.source.root).as_posix()
        batches = []
        records = []
        start = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=self.image_batch_size, columns=[self.source.image_column]
        ):
            images = []
            for index, row in enumerate(batch.to_pylist()):
                location = {
                    "image_id": f"{self.source_name}|{shard}|{start + index}",
                    "shard": shard,
                    "row_index": start + index,
                }
                try:
                    pixels = decode_vlv_image(row[self.source.image_column], self.source)
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    UnidentifiedImageError,
                    OSError,
                    Image.DecompressionBombError,
                ) as exc:
                    record = image_decode_record(location, error=str(exc))
                else:
                    record = image_decode_record(location, pixels)
                    images.append(
                        ImageObject(
                            image_id=location["image_id"],
                            image_path=f"{path}:{start + index}",
                            image_data=pixels,
                            metadata={"source": self.source_name, **record},
                        )
                    )
                records.append(record)
            batches.extend(make_image_batches(images, task, self.image_batch_size))
            start += batch.num_rows
        return batches, records, shard

    def _read_ego4d(self, path: Path, task: FileGroupTask) -> tuple[list[ImageBatch], list[dict], str]:
        rows = [{**row, "source": self.source_name} for row in read_ego4d_records(path, self.source)]
        shard = rows[0]["shard"] if rows else f"{path.stem}.tar"
        if not rows:
            return [], [], shard
        try:
            images = [
                image
                for batch in read_indexed_tar_with_dali(
                    rows, self.source.root, self.image_batch_size, self.num_threads
                )
                for image in batch
            ]
        except ImageDecodeError as exc:
            logger.warning("DALI image decode failed for {}; retrying selected TAR members on CPU: {}", shard, exc)
            images = []
            records = []
            for start in range(0, len(rows), self.image_batch_size):
                decoded, checked = decode_tar_images(
                    rows[start : start + self.image_batch_size],
                    {self.source_name: self.source.root},
                    adjust_orientation=False,
                )
                images.extend(decoded)
                records.extend(checked)
        else:
            records = [image_decode_record(row, image.image_data) for row, image in zip(rows, images, strict=True)]
        return make_image_batches(images, task, self.image_batch_size), records, shard
