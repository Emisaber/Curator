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

import pandas as pd
import pytest

from nemo_curator.stages.image.deduplication.phash_candidates import PhashCandidateStage
from nemo_curator.tasks import FileGroupTask


@pytest.mark.parametrize("block_size", [1, 3])
def test_cross_file_pairs_and_boundary(tmp_path: Path, block_size: int):
    paths = []
    for index, rows in enumerate(
        [
            [("a", "0000000000000000"), ("d", "ffffffffffffffff")],
            [("b", "0000000000000000"), ("c", "0000000000000001")],
        ]
    ):
        path = tmp_path / f"features_{index}.parquet"
        pd.DataFrame([{"image_id": key, "metadata": {"phash": value}} for key, value in rows]).to_parquet(path)
        paths.append(str(path))
    stage = PhashCandidateStage(str(tmp_path / "pairs"), distance_threshold=0, max_distance=1, block_size=block_size)
    result = stage.process(FileGroupTask(dataset_name="images", data=paths))
    pairs = pd.concat([pd.read_parquet(path) for path in result.data])
    observed = {(row.id_a, row.id_b): (row.hamming_distance, row.matched) for row in pairs.itertuples()}
    assert observed == {("a", "b"): (0, True), ("a", "c"): (1, False), ("b", "c"): (1, False)}
    assert len(pairs) == 3
