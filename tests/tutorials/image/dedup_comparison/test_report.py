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


def test_native_clip_pairs_direction_and_precision(tutorial_modules: dict[str, ModuleType], tmp_path: Path):
    for name in ("features", "phash", "clip/cache/pairwise_results", "clip/duplicates"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "run.json").write_text(json.dumps({"phash_threshold": 0}))
    pd.DataFrame(
        [
            {"image_id": "a", "metadata": {"phash": "0000000000000000"}, "embedding": [1.0, 0.0]},
            {"image_id": "b", "metadata": {"phash": "0000000000000000"}, "embedding": [1.0, 0.0]},
            {"image_id": "c", "metadata": {"phash": "ffffffffffffffff"}, "embedding": [0.0, 1.0]},
        ]
    ).to_parquet(tmp_path / "features/f.parquet")
    pd.DataFrame([{"id_a": "a", "id_b": "b", "hamming_distance": 0, "matched": True}]).to_parquet(
        tmp_path / "phash/p.parquet"
    )
    pd.DataFrame(
        [
            {"id": "a", "max_id": "a", "cosine_sim_score": 0.0},
            {"id": "b", "max_id": "a", "cosine_sim_score": 0.98974609375},
            {"id": "c", "max_id": "b", "cosine_sim_score": 0.990234375},
        ]
    ).to_parquet(tmp_path / "clip/cache/pairwise_results/p.parquet")
    pd.DataFrame({"id": ["c"]}).to_parquet(tmp_path / "clip/duplicates/d.parquet")
    pairs, stats = tutorial_modules["report"].collect_candidates(tmp_path)
    observed = {(p["id_a"], p["id_b"]): p for p in pairs}
    assert set(observed) == {("a", "b"), ("b", "c")}
    assert observed[("a", "b")]["clip_hit"] is False
    assert observed[("b", "c")]["clip_hit"] is True
    assert observed[("a", "b")]["clip_from"] == "b"
    assert stats["clip_duplicate_ids"] == 1


def test_review_budget_and_cohorts(tutorial_modules: dict[str, ModuleType]):
    sample = tutorial_modules["report"].sample_pairs
    pairs = [
        {
            "id_a": str(i),
            "id_b": str(i + 500),
            "phash_hit": i % 2 == 0,
            "clip_hit": i % 3 == 0,
            "hamming_distance": 4 if i % 2 == 0 else 7,
            "clip_score": 0.995 if i % 3 == 0 else 0.985,
        }
        for i in range(400)
    ]
    result = sample(pairs, 200, 42)
    assert result == sample(pairs, 200, 42)
    assert len(result) == 200
    assert len({(p["id_a"], p["id_b"]) for p in result}) == 200
    assert sum("phash_random" in p["cohorts"] for p in result) == 60
    assert sum("clip_random" in p["cohorts"] for p in result) == 60
    assert not sample([], 200, 42)


def test_clip_disagreement_reasons(tutorial_modules: dict[str, ModuleType], tmp_path: Path):
    for name in ("features", "phash", "clip/cache/pairwise_results", "clip/duplicates"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "run.json").write_text(json.dumps({"phash_threshold": 6}))
    pd.DataFrame(
        [{"image_id": key, "metadata": {"phash": "0000000000000000"}, "embedding": [1.0, 0.0]} for key in "abcd"]
    ).to_parquet(tmp_path / "features/f.parquet")
    pd.DataFrame([{"id_a": "a", "id_b": key, "hamming_distance": 0, "matched": True} for key in "bcd"]).to_parquet(
        tmp_path / "phash/p.parquet"
    )
    pd.DataFrame(
        [
            {"id": "a", "max_id": "a", "cosine_sim_score": 0.0},
            {"id": "b", "max_id": "a", "cosine_sim_score": 0.95},
            {"id": "c", "max_id": "b", "cosine_sim_score": 1.0},
            {"id": "d", "max_id": "d", "cosine_sim_score": 0.0},
        ]
    ).to_parquet(tmp_path / "clip/cache/pairwise_results/p.parquet")
    pd.DataFrame({"id": ["c"]}).to_parquet(tmp_path / "clip/duplicates/d.parquet")
    for cluster, ids in enumerate((["a", "b", "c"], ["d"])):
        directory = tmp_path / f"clip/cache/kmeans_results/centroid={cluster}"
        directory.mkdir(parents=True)
        pd.DataFrame({"image_id": ids}).to_parquet(directory / "f.parquet")
    pairs, _ = tutorial_modules["report"].collect_candidates(tmp_path)
    observed = {(p["id_a"], p["id_b"]): p for p in pairs}
    assert observed[("a", "b")]["clip_status"] == "below_threshold"
    assert observed[("a", "c")]["clip_status"] == "not_best_match"
    assert observed[("a", "d")]["clip_status"] == "cross_cluster"
    assert observed[("a", "d")]["clip_diagnostic"] == 1.0
    assert observed[("a", "d")]["clip_hit"] is False
    assert observed[("b", "c")]["clip_removed_b"] is True
    assert observed[("a", "d")]["group_a"] == observed[("a", "d")]["group_b"]
