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

"""Decode bounded batches of indexed members of local, uncompressed TAR files."""

from contextlib import ExitStack
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, ImageBatch, ImageObject
from nemo_curator.utils.hash_utils import get_deterministic_hash

SHORT_SIDE_THRESHOLD = 256


def decode_tar_images(
    rows: list[dict], source_roots: dict[str, str], *, adjust_orientation: bool = True
) -> tuple[list[ImageObject], list[dict]]:
    """Decode indexed members individually, keeping file I/O errors outside the image handler."""
    images = []
    records = []
    with ExitStack() as stack:
        archives = {}
        for row in rows:
            tar_path = str(Path(source_roots[row["source"]]) / row["shard"])
            if tar_path not in archives:
                archives[tar_path] = stack.enter_context(Path(tar_path).open("rb"))
            archive = archives[tar_path]
            archive.seek(row["offset"])
            content = archive.read(row["size"])
            record = {
                **row,
                "status": "failed",
                "width": None,
                "height": None,
                "short_side_below_256": None,
                "error": None,
            }
            try:
                with Image.open(BytesIO(content)) as image:
                    image.load()
                    oriented = ImageOps.exif_transpose(image) if adjust_orientation else image
                    pixels = np.array(oriented.convert("RGB"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                record["error"] = str(exc)
            else:
                height, width = pixels.shape[:2]
                short_side_below_256 = min(height, width) < SHORT_SIDE_THRESHOLD
                record.update(status="ok", width=width, height=height, short_side_below_256=short_side_below_256)
                images.append(
                    ImageObject(
                        image_path=f"{tar_path}:{row['offset']}:{row['member']}",
                        image_id=row["image_id"],
                        image_data=pixels,
                        metadata={
                            "source": row["source"],
                            "shard": row["shard"],
                            "member": row["member"],
                            "caption_raw": row.get("caption_raw"),
                            "width": width,
                            "height": height,
                            "short_side_below_256": short_side_below_256,
                        },
                    )
                )
            records.append(record)

    return images, records


def write_decode_records(records: list[dict], records_dir: str, filename: str) -> None:
    output_dir = Path(records_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_table = pd.DataFrame(records).astype(
        {
            "image_id": "string",
            "status": "string",
            "width": "Int64",
            "height": "Int64",
            "short_side_below_256": "boolean",
            "error": "string",
        }
    )
    record_table.to_parquet(output_dir / f"{filename}.parquet", index=False)


@dataclass
class TarImageDecodeStage(ProcessingStage[DocumentBatch, ImageBatch]):
    """Read bounded batches of members described by source, shard, offset and size."""

    source_roots: dict[str, str]
    records_dir: str
    name: str = "tar_image_decode"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "source", "shard", "member", "offset", "size"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> ImageBatch | None:
        images, records = decode_tar_images(task.to_pandas().to_dict("records"), self.source_roots)
        filename = get_deterministic_hash([row["image_id"] for row in records])
        write_decode_records(records, self.records_dir, filename)
        if not images:
            return None
        return ImageBatch(
            dataset_name=task.dataset_name,
            data=images,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
