# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Resolve captions only for image pairs found by semantic deduplication."""

import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.io.caption_lookup import (
    read_webdataset_components,  # noqa: F401 -- retain the report import
    resolve_captions,
)
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask


def normalize_caption(caption: str | None) -> str | None:
    """Normalize only for comparison; never replace the source caption."""
    if caption is None:
        return None
    return " ".join(unicodedata.normalize("NFC", caption).split()) or None


class CandidatePairPartitioningStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Group candidate-pair Parquet files by image cluster."""

    name = "candidate_pair_partitioning"
    resources = Resources(cpus=0.5)

    def __init__(self, input_path: str):
        self.input_path = input_path

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def num_workers(self) -> int:
        return 1

    def process(self, task: EmptyTask) -> list[FileGroupTask]:
        clusters: dict[int, list[str]] = defaultdict(list)
        for path in Path(self.input_path).glob("cluster_*_*.parquet"):
            cluster_id = int(path.stem.removeprefix("cluster_").rsplit("_", 1)[0])
            clusters[cluster_id].append(str(path))
        return [
            FileGroupTask(
                dataset_name=task.dataset_name,
                data=sorted(paths),
                _metadata={"centroid_id": cluster_id, "filetype": "parquet"},
            )
            for cluster_id, paths in sorted(clusters.items())
        ]


class CaptionAwareDeduplicationStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Compare source captions on every image-similar edge without removing images."""

    name = "caption_aware_deduplication"
    resources = Resources(cpus=1.0)

    def __init__(self, sources: dict[str, CaptionSource], output_path: str):
        self.sources = sources
        self.output_path = output_path

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: FileGroupTask) -> FileGroupTask:
        cluster_id = task._metadata["centroid_id"]
        image_ids = set()
        for path in task.data:
            pairs = pq.read_table(path, columns=["id_a", "id_b"])
            image_ids.update(pairs["id_a"].to_pylist())
            image_ids.update(pairs["id_b"].to_pylist())

        captions = resolve_captions(image_ids, self.sources)
        normalized = {image_id: normalize_caption(caption) for image_id, caption in captions.items()}
        output_dir = Path(self.output_path)
        captions_dir = output_dir / "captions"
        pairs_dir = output_dir / "pairs"
        captions_dir.mkdir(parents=True, exist_ok=True)
        pairs_dir.mkdir(parents=True, exist_ok=True)
        caption_path = captions_dir / f"cluster_{cluster_id}_captions.parquet"
        sorted_ids = sorted(image_ids)
        pq.write_table(
            pa.table(
                {
                    "image_id": sorted_ids,
                    "caption_raw": [captions[image_id] for image_id in sorted_ids],
                    "caption_normalized": [normalized[image_id] for image_id in sorted_ids],
                }
            ),
            caption_path,
        )

        decisions = []
        for path in task.data:
            pairs = pq.read_table(path).to_pandas()
            caption_a = pairs["id_a"].map(normalized)
            caption_b = pairs["id_b"].map(normalized)
            pairs["caption_relation"] = "different_caption"
            pairs.loc[caption_a.isna() | caption_b.isna(), "caption_relation"] = "missing_caption"
            pairs.loc[caption_a.notna() & (caption_a == caption_b), "caption_relation"] = "same_caption"
            decision_path = pairs_dir / Path(path).name
            pq.write_table(pa.Table.from_pandas(pairs, preserve_index=False), decision_path)
            decisions.append(str(decision_path))

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=decisions,
            _metadata={**task._metadata, "captions_path": str(caption_path)},
            _stage_perf=task._stage_perf,
        )
