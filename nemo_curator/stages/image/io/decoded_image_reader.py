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

"""Read the successful samples of an existing source decode partition."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.decodable_reader import make_image_batches, read_indexed_tar_with_dali
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.source_reader import decode_vlv_image, read_vlv_rows
from nemo_curator.stages.image.io.tar_image_reader import decode_tar_images
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, ImageBatch, ImageObject


@dataclass
class DecodeRecordImageReaderStage(ProcessingStage[FileGroupTask, ImageBatch]):
    source_name: str
    source: CaptionSource
    image_batch_size: int = 32
    num_gpus_per_worker: float = 0.25
    num_threads: int = 8
    name: str = "decode_record_image_reader"

    def __post_init__(self) -> None:
        use_gpu = self.source.format != "vlv_parquet" and torch.cuda.is_available()
        self.resources = Resources(gpus=self.num_gpus_per_worker) if use_gpu else Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "image_data", "image_path"]

    def process(self, task: FileGroupTask) -> list[ImageBatch]:
        batches = []
        for file_path in task.data:
            records = pq.ParquetFile(file_path).read().to_pylist()
            rows = [{**row, "source": self.source_name} for row in records if row["status"] == "ok"]
            if not rows:
                continue
            path = Path(self.source.root) / rows[0]["shard"]
            if self.source.format == "vlv_parquet":
                values = read_vlv_rows(path, {row["row_index"] for row in rows}, [self.source.image_column])
                images = [
                    ImageObject(
                        image_id=row["image_id"],
                        image_path=f"{path}:{row['row_index']}",
                        image_data=decode_vlv_image(values[row["row_index"]][self.source.image_column], self.source),
                        metadata=row,
                    )
                    for row in rows
                ]
            elif self.source.format == "ego4d_recaption":
                images = [
                    image
                    for batch in read_indexed_tar_with_dali(
                        rows, self.source.root, self.image_batch_size, self.num_threads
                    )
                    for image in batch
                ]
            elif any(row["status"] == "failed" for row in records):
                images = []
                for start in range(0, len(rows), self.image_batch_size):
                    decoded, checked = decode_tar_images(
                        rows[start : start + self.image_batch_size],
                        {self.source_name: self.source.root},
                        adjust_orientation=False,
                    )
                    failures = [row for row in checked if row["status"] != "ok"]
                    if failures:
                        msg = f"Previously decoded sample cannot be read: {failures[0]['image_id']}: {failures[0]['error']}"
                        raise RuntimeError(msg)
                    images.extend(decoded)
            else:
                reader = ImageReaderStage(
                    dali_batch_size=self.image_batch_size,
                    num_threads=self.num_threads,
                    num_gpus_per_worker=self.num_gpus_per_worker,
                    source_name=self.source_name,
                    source_root=self.source.root,
                    index_suffix=self.source.index_suffix,
                    image_extensions=("jpg", "jpeg", "png", "webp"),
                    case_sensitive_extensions=False,
                )
                selected = {row["image_id"] for row in rows}
                decoded = {
                    image.image_id: image
                    for batch in reader._read_tars_with_dali([path])
                    for image in batch
                    if image.image_id in selected
                }
                images = [decoded[row["image_id"]] for row in rows]
            batches.extend(make_image_batches(images, task, self.image_batch_size))
        return batches
