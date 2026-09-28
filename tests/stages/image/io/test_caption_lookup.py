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

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.image.io.caption_lookup import resolve_captions
from nemo_curator.stages.image.io.caption_source import CaptionSource


def test_external_parquet_caption_references_keep_original_text(tmp_path: Path) -> None:
    caption = "  café\nleft  "
    pq.write_table(pa.table({"description": ["unused", caption]}), tmp_path / "captions.parquet", row_group_size=1)
    pq.write_table(
        pa.table({"sample_key": ["a"], "parquet_path": ["captions.parquet"], "row_group": [1], "row_index": [0]}),
        tmp_path / "data.tar.caption_refs.parquet",
    )
    source = CaptionSource(str(tmp_path), format="parquet", caption_column="description")
    assert resolve_captions({"source|data.tar|a", "source|data.tar|missing"}, {"source": source}) == {
        "source|data.tar|a": caption,
        "source|data.tar|missing": None,
    }
