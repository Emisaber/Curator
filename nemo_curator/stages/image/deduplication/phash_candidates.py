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

"""Bounded-memory all-pairs hash comparison for small image experiments."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask

_POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
_HASH_BITS = 64


def find_phash_pairs(
    hashes: np.ndarray, max_distance: int, block_size: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield indices and distances once per unordered pair, without self-pairs."""
    for start in range(0, len(hashes), block_size):
        stop = min(start + block_size, len(hashes))
        distances = _POPCOUNT[np.bitwise_xor(hashes[start:stop, None, :], hashes[None, :, :])].sum(axis=2)
        mask = distances <= max_distance
        mask &= np.arange(len(hashes))[None, :] > np.arange(start, stop)[:, None]
        left, right = np.nonzero(mask)
        yield start, left + start, right, distances[left, right]


@dataclass
class PhashCandidateStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Compare all feature files in one task, including cross-file duplicates.

    Distances up to max_distance are saved for inspection; distance_threshold
    only controls the matched flag. This is an O(N²) small-experiment stage.
    """

    output_path: str
    distance_threshold: int = 6
    max_distance: int = 8
    block_size: int = 256
    name: str = "phash_candidates"
    is_resumable = False

    def __post_init__(self) -> None:
        if not 0 <= self.distance_threshold <= self.max_distance <= _HASH_BITS or self.block_size < 1:
            msg = "Require 0 <= distance_threshold <= max_distance <= 64 and block_size > 0"
            raise ValueError(msg)

    def num_workers(self) -> int:
        return 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: FileGroupTask) -> FileGroupTask:
        features = pd.concat(
            [pd.read_parquet(path, columns=["image_id", "metadata"]) for path in task.data],
            ignore_index=True,
        ).sort_values("image_id", ignore_index=True)
        hashes = np.array([list(bytes.fromhex(meta["phash"])) for meta in features["metadata"]], dtype=np.uint8)
        ids = features["image_id"].to_numpy()
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for start, left, right, distances in find_phash_pairs(hashes, self.max_distance, self.block_size):
            pairs = pd.DataFrame(
                {
                    "id_a": ids[left],
                    "id_b": ids[right],
                    "hamming_distance": distances.astype(np.uint8),
                    "matched": distances <= self.distance_threshold,
                }
            )
            path = output_dir / f"block_{start:08d}.parquet"
            pairs.to_parquet(path, index=False)
            outputs.append(str(path))
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=outputs,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
