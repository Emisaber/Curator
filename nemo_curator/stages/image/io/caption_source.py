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

"""Source locations for WebDataset, VLV, and Ego4D image-caption records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq


@dataclass
class CaptionSource:
    """Image storage and caption lookup settings for one source partition."""

    root: str
    format: Literal["webdataset_txt", "parquet", "vlv_parquet", "ego4d_recaption"] = "webdataset_txt"
    caption_column: str = "caption"
    index_suffix: str = ".idx"
    image_column: str = "image"
    image_encoding: Literal["raw_chw_uint8", "encoded"] = "raw_chw_uint8"
    image_shape: tuple[int, int, int] = (3, 384, 384)
    sample_index_root: str | None = None
    annotations_root: str | None = None


def read_parquet_captions(
    tar_path: Path, source_root: Path, sample_keys: set[str], caption_column: str
) -> dict[str, str | None]:
    """Resolve selected sample keys through `<archive>.caption_refs.parquet`."""
    refs = pq.read_table(f"{tar_path}.caption_refs.parquet").to_pandas()
    refs = refs[refs["sample_key"].isin(sample_keys)]
    captions: dict[str, str | None] = dict.fromkeys(sample_keys)
    for (parquet_path, row_group), group in refs.groupby(["parquet_path", "row_group"]):
        source_file = pq.ParquetFile(source_root / parquet_path)
        values = source_file.read_row_group(int(row_group), columns=[caption_column])[caption_column]
        for row in group.itertuples(index=False):
            captions[row.sample_key] = values[int(row.row_index)].as_py()
    return captions
