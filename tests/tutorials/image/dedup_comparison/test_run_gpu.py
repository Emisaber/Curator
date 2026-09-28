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

"""Opt-in native GPU smoke test; use a Ray cluster with the intended GPU allocation."""

import os
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@pytest.mark.gpu
def test_native_clip_workflow(
    tutorial_modules: dict[str, ModuleType],
    image_tars: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_dir = os.environ.get("CURATOR_IMAGE_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set CURATOR_IMAGE_MODEL_DIR to the existing CLIP model cache root")
    module = tutorial_modules["run"]
    output = tmp_path / "clip-experiment"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--source",
            f"fixture={image_tars}",
            "--output",
            str(output),
            "--mode",
            "both",
            "--model-dir",
            model_dir,
            "--source-sample-rate",
            "fixture=1.0",
            "--batch-size",
            "4",
            "--n-clusters",
            "2",
            "--report-pairs",
            "20",
        ],
    )
    module.run(module.parse_args())
    read = tutorial_modules["report"].read_parquets
    candidates = read(output / "dedup/clip/schema-v1/cache/pairwise_results")
    duplicates = read(output / "dedup/clip/schema-v1/duplicates")
    expected = set(candidates.loc[candidates["cosine_sim_score"] >= 0.99, "id"])
    assert expected
    assert set(duplicates["id"]) == expected
    assert len(read(output / "features/clip/schema-v1")) == 12
    report_pairs = pd.read_parquet(output / "report/candidates.parquet")
    assert report_pairs["clip_hit"].any()
