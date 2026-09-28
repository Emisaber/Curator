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

"""Decode source TARs with DALI and record per-image usability before CLIP."""

import tarfile
from collections.abc import Generator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from loguru import logger

from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.sample_index import _read_tar_locations
from nemo_curator.stages.image.io.tar_image_reader import SHORT_SIDE_THRESHOLD, decode_tar_images
from nemo_curator.tasks import FileGroupTask, ImageBatch, ImageObject
from nemo_curator.utils.hash_utils import get_deterministic_hash
from nemo_curator.utils.image_id import make_image_id

_DECODE_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("status", pa.string()),
        ("width", pa.int64()),
        ("height", pa.int64()),
        ("short_side_below_256", pa.bool_()),
        ("error", pa.string()),
        ("shard", pa.string()),
        ("member", pa.string()),
        ("offset", pa.int64()),
        ("size", pa.int64()),
        ("row_index", pa.int64()),
    ]
)


class ImageDecodeError(RuntimeError):
    """DALI reported a failure attributable to encoded image content."""


def run_dali_decode(pipe: object) -> object:
    try:
        return pipe.run()
    except RuntimeError as exc:
        message = str(exc).lower()
        # DALI reports image failures and runtime failures through the same exception type.
        decoder = "nvidia.dali.fn.decoders.image" in message
        image_error = any(
            text in message
            for text in (
                "unrecognized image format",
                "unsupported image type",
                "jpeg::getimageinfo",
            )
        )
        if decoder and image_error:
            raise ImageDecodeError(str(exc)) from exc
        raise


def image_decode_record(row: dict, pixels: np.ndarray | None = None, error: str | None = None) -> dict:
    height, width = pixels.shape[:2] if pixels is not None else (None, None)
    return {
        **row,
        "status": "ok" if pixels is not None else "failed",
        "width": width,
        "height": height,
        "short_side_below_256": min(height, width) < SHORT_SIDE_THRESHOLD if pixels is not None else None,
        "error": error,
    }


def write_source_decode_records(records: list[dict], records_dir: str, shard: str) -> None:
    output = Path(records_dir)
    output.mkdir(parents=True, exist_ok=True)
    filename = get_deterministic_hash([shard])
    pq.write_table(pa.Table.from_pylist(records, schema=_DECODE_SCHEMA), output / f"part-{filename}.parquet")


def make_image_batches(images: list[ImageObject], task: FileGroupTask, batch_size: int) -> list[ImageBatch]:
    return [
        ImageBatch(
            dataset_name=task.dataset_name,
            data=images[start : start + batch_size],
            _metadata=task._metadata.copy(),
            _stage_perf=task._stage_perf.copy(),
        )
        for start in range(0, len(images), batch_size)
    ]


def read_indexed_tar_with_dali(
    rows: list[dict], source_root: str, batch_size: int, num_threads: int
) -> Generator[list[ImageObject], None, None]:
    """Feed indexed JPEG bytes to DALI without changing frame identities."""
    from nvidia.dali import fn, pipeline_def, types

    use_gpu = torch.cuda.is_available()

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=0 if use_gpu else None,
        exec_async=False,
        exec_pipelined=False,
    )
    def indexed_pipeline() -> object:
        encoded = fn.external_source(name="encoded", dtype=types.UINT8, ndim=1)
        return fn.decoders.image(encoded, device="mixed" if use_gpu else "cpu", output_type=types.RGB)

    pipe = indexed_pipeline()
    pipe.build()
    with ExitStack() as stack:
        archives = {}
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            encoded = []
            for row in selected:
                path = Path(source_root) / row["shard"]
                if path not in archives:
                    archives[path] = stack.enter_context(path.open("rb"))
                archive = archives[path]
                archive.seek(row["offset"])
                encoded.append(np.frombuffer(archive.read(row["size"]), dtype=np.uint8))
            pipe.feed_input("encoded", encoded)
            pixels = run_dali_decode(pipe)[0].as_cpu()
            yield [
                ImageObject(
                    image_id=row["image_id"],
                    image_path=f"{Path(source_root) / row['shard']}:{row['offset']}:{row['member']}",
                    image_data=np.array(pixels.at(index)),
                    metadata={"source": row["source"], "shard": row["shard"], "member": row["member"]},
                )
                for index, row in enumerate(selected)
            ]


@dataclass(kw_only=True)
class DecodableImageReaderStage(ImageReaderStage):
    """Keep DALI's normal path and retry a corrupt TAR through the CPU decodable reader."""

    records_dir: str
    name: str = "decodable_image_reader"

    def _run_dali_pipeline(self, pipe: object) -> object:
        return run_dali_decode(pipe)

    def _accepts_member(self, member: str) -> bool:
        extension = member.rsplit(".", 1)[-1]
        if not self.case_sensitive_extensions:
            extension = extension.lower()
        return extension in self.image_extensions

    def _tar_records(self, path: Path) -> list[dict]:
        if self.index_suffix:
            images, _ = _read_tar_locations(path, self.index_suffix)
            locations = [location for location in images.values() if self._accepts_member(location[0])]
        else:
            with tarfile.open(path, "r:") as archive:
                locations = [
                    (member.name, member.offset_data, member.size)
                    for member in archive
                    if member.isfile() and self._accepts_member(member.name)
                ]
        relative = path.relative_to(self.source_root).as_posix()
        rows = [
            {
                "image_id": make_image_id(self.source_name, relative, member),
                "source": self.source_name,
                "shard": relative,
                "member": member,
                "offset": offset,
                "size": size,
            }
            for member, offset, size in locations
        ]
        return rows[: self.max_images_per_partition] if self.max_images_per_partition is not None else rows

    def process(self, task: FileGroupTask) -> list[ImageBatch]:
        output = []
        for file_path in task.data:
            path = Path(file_path)
            shard = path.relative_to(self.source_root).as_posix()
            shard_task = FileGroupTask(
                dataset_name=task.dataset_name,
                data=[file_path],
                _metadata=task._metadata.copy(),
                _stage_perf=task._stage_perf.copy(),
            )
            try:
                batches = super().process(shard_task)
            except ImageDecodeError as exc:
                logger.warning("DALI image decode failed for {}; retrying the entire TAR on CPU: {}", path, exc)
                rows = self._tar_records(path)
                images = []
                records = []
                for start in range(0, len(rows), self.dali_batch_size):
                    decoded, checked = decode_tar_images(
                        rows[start : start + self.dali_batch_size],
                        {self.source_name: self.source_root},
                        adjust_orientation=False,
                    )
                    images.extend(decoded)
                    records.extend(checked)
                batches = make_image_batches(images, task, self.dali_batch_size)
            else:
                records = []
                for batch in batches:
                    for image in batch.data:
                        _, offset, member = image.image_path.rsplit(":", 2)
                        record = image_decode_record(
                            {"image_id": image.image_id, "shard": shard, "member": member, "offset": int(offset)},
                            image.image_data,
                        )
                        image.metadata.update(
                            source=self.source_name,
                            shard=shard,
                            member=member,
                            width=record["width"],
                            height=record["height"],
                            short_side_below_256=record["short_side_below_256"],
                        )
                        records.append(record)
            write_source_decode_records(records, self.records_dir, shard)
            output.extend(batches)
        return output
