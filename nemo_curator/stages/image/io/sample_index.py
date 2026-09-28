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

"""Index every WebDataset image and its original caption without decoding pixels."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.io.caption_source import CaptionSource, read_parquet_captions
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.image_id import make_image_id, sample_key_from_member

_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
_SAMPLE_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("source", pa.string()),
        ("shard", pa.string()),
        ("member", pa.string()),
        ("offset", pa.int64()),
        ("size", pa.int64()),
        ("caption_raw", pa.string()),
    ]
)


def _read_tar_locations(tar_path: Path, index_suffix: str) -> tuple[dict, dict]:
    images = {}
    text_locations = {}
    with open(f"{tar_path}{index_suffix}", encoding="utf-8") as index_file:
        header = index_file.readline().split()
        if not header or header[0] != "v1.2":
            msg = f"Expected a DALI v1.2 index for {tar_path}"
            raise ValueError(msg)
        for line in index_file:
            fields = line.split()
            if len(fields) % 4:
                msg = f"Malformed DALI index for {tar_path}"
                raise ValueError(msg)
            for start in range(0, len(fields), 4):
                extension, offset, size, member = fields[start : start + 4]
                key = sample_key_from_member(member)
                if extension.lower() in _IMAGE_EXTENSIONS:
                    if key in images:
                        msg = f"Duplicate image sample key {key!r} in {tar_path}"
                        raise ValueError(msg)
                    images[key] = (member, int(offset), int(size))
                elif extension.lower() == "txt":
                    text_locations[key] = (int(offset), int(size))
    return images, text_locations


def _read_captions(
    tar_path: Path, root: Path, source: CaptionSource, images: dict, text_locations: dict
) -> dict[str, str | None]:
    if source.format == "parquet":
        return read_parquet_captions(tar_path, root, set(images), source.caption_column)
    if source.format != "webdataset_txt":
        msg = f"Unsupported caption source format: {source.format}"
        raise ValueError(msg)

    captions: dict[str, str | None] = dict.fromkeys(images)
    with tar_path.open("rb") as archive:
        for key in images:
            if key in text_locations:
                offset, size = text_locations[key]
                archive.seek(offset)
                captions[key] = archive.read(size).decode("utf-8")
    return captions


@dataclass
class WebDatasetImageSampleIndexStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Build the sample index for one source from DALI v1.2 TAR indexes."""

    source_name: str
    source: CaptionSource
    name: str = "webdataset_image_sample_index"

    def process(self, task: FileGroupTask) -> DocumentBatch:
        root = Path(self.source.root).resolve()
        rows = []
        for tar_file in task.data:
            tar_path = Path(tar_file).resolve()
            relative_tar = tar_path.relative_to(root).as_posix()
            images, text_locations = _read_tar_locations(tar_path, self.source.index_suffix)
            captions = _read_captions(tar_path, root, self.source, images, text_locations)

            for key, (member, offset, size) in images.items():
                rows.append(
                    {
                        "image_id": make_image_id(self.source_name, relative_tar, member),
                        "source": self.source_name,
                        "shard": relative_tar,
                        "member": member,
                        "offset": offset,
                        "size": size,
                        "caption_raw": captions[key],
                    }
                )

        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pylist(rows, schema=_SAMPLE_SCHEMA),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
