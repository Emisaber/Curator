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

import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def test_cpu_pipeline_and_portable_report(
    tutorial_modules: dict[str, ModuleType],
    image_tars: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = tutorial_modules["run"]
    output = tmp_path / "experiment"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--source",
            f"long={image_tars}",
            "--output",
            str(output),
            "--mode",
            "phash",
            "--source-sample-rate",
            "long=1.0",
            "--batch-size",
            "3",
            "--report-pairs",
            "20",
        ],
    )
    module.run(module.parse_args())
    stats = json.loads((output / "report/stats.json").read_text())
    assert stats["decoded"] == 12
    assert stats["decode_failures"] == 2
    features = pd.concat(pd.read_parquet(path) for path in (output / "features/clip/schema-v1").rglob("*.parquet"))
    assert len(features) == 12
    assert all(
        "laplacian_variance" in metadata and "is_blurry" in metadata for metadata in features["metadata"]
    )
    assert stats["phash_hit_pairs"] > 0
    assert 0 < stats["review_pairs"] <= 20
    assert (output / "report/index.html").is_file()
    data = json.loads((output / "report/data.js").read_text().removeprefix("window.REVIEW = ").removesuffix(";\n"))
    decode = tutorial_modules["report"].read_parquets(output / "annotations/decode/schema-v1").set_index("image_id")
    blur = tutorial_modules["report"].read_parquets(output / "annotations/blur/schema-v1").set_index("image_id")
    assert len(decode) == 14
    assert len(blur) == 12
    assert set(blur.index) == set(decode.index[decode["status"] == "ok"])
    records = pd.read_parquet(output / "samples/schema-v1/manifest.parquet").set_index("image_id")
    for image_id, image in data["images"].items():
        row = records.loc[image_id]
        with (image_tars / row["shard"]).open("rb") as archive:
            archive.seek(int(row["offset"]))
            original = archive.read(int(row["size"]))
        assert (output / "report" / image["original"]).read_bytes() == original
    assert len(list(image_tars.glob("*.tar"))) == 2
