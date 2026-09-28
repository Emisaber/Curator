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
from types import ModuleType

import pandas as pd


def test_sampling_stable_across_mounts_and_batches(
    tutorial_modules: dict[str, ModuleType],
    image_tars: Path,
    tmp_path: Path,
):
    prepare = tutorial_modules["prepare_samples"].prepare_samples
    moved = tmp_path / "moved"
    shutil.copytree(image_tars, moved)
    prepare({"long": str(image_tars)}, str(tmp_path / "a"), {"long": 0.75}, batch_size=2)
    prepare({"long": str(moved)}, str(tmp_path / "b"), {"long": 0.75}, batch_size=5)
    a = pd.read_parquet(tmp_path / "a/samples.parquet")
    b = pd.read_parquet(tmp_path / "b/samples.parquet")
    pd.testing.assert_frame_equal(a, b)
    assert a["image_id"].nunique() == 9
    assert all(len(pd.read_parquet(path)) <= 2 for path in (tmp_path / "a/manifests").rglob("*.parquet"))


def test_equal_source_sampling(tutorial_modules: dict[str, ModuleType], image_tars: Path, tmp_path: Path):
    prepare = tutorial_modules["prepare_samples"].prepare_samples
    prepare(
        {"short": str(image_tars), "long": str(image_tars)},
        str(tmp_path / "input"),
        {"short": 5 / 12, "long": 5 / 12},
    )
    samples = pd.read_parquet(tmp_path / "input/samples.parquet")
    assert samples.groupby("source").size().to_dict() == {"long": 5, "short": 5}
    assert samples["image_id"].is_unique
