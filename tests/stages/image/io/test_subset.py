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

import shutil
from pathlib import Path

from nemo_curator.stages.image.io.subset import load_subset, prepare_subset


def test_whole_shard_subset_is_reproducible_and_excludes_journeydb(tmp_path: Path) -> None:
    root = tmp_path / "long"
    root.mkdir()
    for name in ("sa_0.tar", "sa_1.tar", "sa_2.tar", "sa_3.tar", "x_jdb_0.tar", "JourneyDB_1.tar"):
        (root / name).write_bytes(b"not a readable tar")
    config = {
        "sources": {"long": {"root": str(root), "exclude_patterns": ["*journeydb*", "*_jdb_*"]}},
        "subset": {"source_sample_rates": {"long": 0.5}},
        "seed": 12,
    }
    prepare_subset(config, tmp_path / "a")
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    config["sources"]["long"]["root"] = str(moved)
    prepare_subset(config, tmp_path / "b")
    info, shards = load_subset(tmp_path / "a")
    assert shards == load_subset(tmp_path / "b")[1]
    assert len(shards["long"]) == 2
    assert all(row["shard"].startswith("sa_") for row in shards["long"])
    assert info["statistics"]["long"]["excluded"] == 2
    assert sorted(path.name for path in (tmp_path / "a").iterdir()) == ["config.json", "shards.jsonl"]


def test_subset_accepts_all_source_formats_without_reading_contents(tmp_path: Path) -> None:
    sources = {}
    formats = {
        "long": "webdataset_txt",
        "short": "webdataset_txt",
        "vlv": "vlv_parquet",
        "ego4d-fho": "ego4d_recaption",
    }
    for name, source_format in formats.items():
        root = tmp_path / name
        root.mkdir()
        suffix = ".parquet" if source_format == "vlv_parquet" else ".tar"
        (root / f"part{suffix}").touch()
        sources[name] = {"root": str(root), "format": source_format}
    prepare_subset({"sources": sources}, tmp_path / "subset")
    _, shards = load_subset(tmp_path / "subset")
    assert shards["vlv"] == [{"source": "vlv", "shard": "part.parquet"}]
    assert shards["ego4d-fho"] == [{"source": "ego4d-fho", "shard": "part.tar", "sample_index": "part.parquet"}]
    assert all(len(records) == 1 for records in shards.values())
