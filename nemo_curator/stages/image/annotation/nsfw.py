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

"""Annotate persisted CLIP image embeddings with Curator's NSFW scorer."""

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.nsfw import NSFWScorer
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.annotation.file_utils import annotation_path
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, FileGroupTask

_ANNOTATION_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("status", pa.string()),
        ("nsfw_score", pa.float64()),
        ("error", pa.string()),
    ]
)


@dataclass
class ImageNSFWAnnotationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Score image embeddings without applying a filtering threshold."""

    model_dir: str
    output_dir: str
    source_name: str
    model_inference_batch_size: int = 32
    num_gpus_per_worker: float = 0.25
    name: str = "image_nsfw_annotation"

    def __post_init__(self) -> None:
        if self.model_inference_batch_size < 1:
            msg = "NSFW inference batch size must be positive"
            raise ValueError(msg)
        self.resources = Resources(gpus=self.num_gpus_per_worker) if torch.cuda.is_available() else Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "embedding"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "status", "nsfw_score"]

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        NSFWScorer.download_weights_on_node(self.model_dir)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self.model = NSFWScorer(model_dir=self.model_dir)
        self.model.setup()

    def process(self, task: DocumentBatch) -> DocumentBatch | None:
        source_files = task._metadata["source_files"]
        if len(source_files) != 1:
            msg = "NSFW annotation requires one embedding file per task"
            raise ValueError(msg)
        path = annotation_path(Path(self.output_dir), self.source_name, source_files[0])
        if path.exists():
            return None

        frame = task.to_pandas()
        if frame.empty:
            return None
        scores = []
        for start in range(0, len(frame), self.model_inference_batch_size):
            embeddings = np.stack(frame["embedding"].iloc[start : start + self.model_inference_batch_size]).astype(
                np.float32
            )
            scores.extend(self.model(embeddings).detach().cpu().tolist())
        annotations = pa.Table.from_pydict(
            {
                "image_id": frame["image_id"].tolist(),
                "status": ["ok"] * len(frame),
                "nsfw_score": scores,
                "error": [None] * len(frame),
            },
            schema=_ANNOTATION_SCHEMA,
        )
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=annotations,
            _metadata={**task._metadata, "annotation_path": str(path)},
            _stage_perf=task._stage_perf,
        )


@dataclass
class NSFWAnnotationWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    name: str = "nsfw_annotation_writer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "status", "nsfw_score"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> FileGroupTask:
        path = Path(task._metadata["annotation_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            pq.write_table(task.data, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[str(path)],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
