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

"""Score indexed image-caption samples through job-managed HPSv3 services."""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.annotation.file_utils import annotation_path
from nemo_curator.stages.image.io.caption_lookup import resolve_captions
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, FileGroupTask

_SCORE_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("caption_version", pa.string()),
        ("caption_sha256", pa.string()),
        ("status", pa.string()),
        ("score_mu", pa.float64()),
        ("score_sigma", pa.float64()),
        ("error", pa.string()),
    ]
)


@dataclass
class HPSv3ScoreStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    source_name: str
    source_root: str
    output_dir: str
    endpoints: list[str]
    inference_batch_size: int = 16
    request_timeout: float = 300.0
    source: CaptionSource | None = None
    adjust_orientation: bool = True
    retry_statuses: tuple[str, ...] = ()
    name: str = "hpsv3_score"

    def __post_init__(self) -> None:
        if not self.endpoints or self.inference_batch_size < 1:
            msg = "HPSv3 requires endpoints and a positive inference batch size"
            raise ValueError(msg)
        self.resources = Resources(cpus=1)

    def inputs(self) -> tuple[list[str], list[str]]:
        if self.source is None:
            columns = ["image_id", "shard", "offset", "size", "caption_raw"]
        else:
            location = "row_index" if self.source.format == "vlv_parquet" else "offset"
            columns = ["image_id", "status", "shard", location]
        return ["data"], columns

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "status", "score_mu"]

    def _output_path(self, task: DocumentBatch) -> Path:
        source_files = task._metadata["source_files"]
        if len(source_files) != 1:
            msg = "HPSv3 scoring requires one input Parquet per task"
            raise ValueError(msg)
        return annotation_path(Path(self.output_dir), self.source_name, source_files[0])

    def _request_items(self, chunk: list[dict]) -> list[dict]:
        items = []
        for row in chunk:
            item = {
                "image_id": row["image_id"],
                "caption": row["caption_raw"] if isinstance(row["caption_raw"], str) else None,
            }
            if self.source is not None and self.source.format == "vlv_parquet":
                item.update(
                    parquet_path=str(Path(self.source_root) / row["shard"]),
                    row_index=int(row["row_index"]),
                    image_column=self.source.image_column,
                    image_encoding=self.source.image_encoding,
                    image_shape=self.source.image_shape,
                )
            else:
                item.update(
                    tar_path=str(Path(self.source_root) / row["shard"]),
                    offset=int(row["offset"]),
                    size=int(row["size"]) if pd.notna(row.get("size")) else None,
                    adjust_orientation=self.adjust_orientation,
                )
            items.append(item)
        return items

    def _score(self, chunk: list[dict]) -> list[dict]:
        endpoint_index = int(hashlib.sha256(chunk[0]["image_id"].encode()).hexdigest(), 16) % len(self.endpoints)
        request = Request(  # noqa: S310 -- endpoints are job-managed HTTP services
            f"{self.endpoints[endpoint_index]}/score",
            data=json.dumps({"items": self._request_items(chunk)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:  # noqa: S310
                scores = json.load(response)["items"]
        except HTTPError as exc:
            if exc.code not in (408, 409, 429) and exc.code < HTTPStatus.INTERNAL_SERVER_ERROR:
                raise
            error = f"{exc}: {exc.read().decode('utf-8', errors='replace')}"
        except (URLError, TimeoutError, ConnectionError) as exc:
            error = str(exc)
        else:
            if [score["image_id"] for score in scores] != [row["image_id"] for row in chunk]:
                msg = "HPSv3 response does not match the requested image IDs"
                raise ValueError(msg)
            return scores
        return [
            {
                "image_id": row["image_id"],
                "status": "request_error",
                "score_mu": None,
                "score_sigma": None,
                "error": error,
            }
            for row in chunk
        ]

    def process(self, task: DocumentBatch) -> DocumentBatch | None:  # noqa: C901
        path = self._output_path(task)
        previous = None
        if path.exists():
            if not self.retry_statuses:
                return None
            previous = pd.read_parquet(path).set_index("image_id").to_dict("index")
        elif self.retry_statuses:
            return None
        frame = task.to_pandas()
        if self.source is not None:
            frame = frame[frame["status"] == "ok"]
        records = frame.to_dict("records")
        if not records:
            return None
        pending = [
            row for row in records if previous is None or previous[row["image_id"]]["status"] in self.retry_statuses
        ]
        if not pending:
            return None
        if self.source is not None:
            captions = resolve_captions({row["image_id"] for row in pending}, {self.source_name: self.source})
            for row in pending:
                row["caption_raw"] = captions[row["image_id"]]
        rows = {image_id: {"image_id": image_id, **row} for image_id, row in (previous or {}).items()}
        for start in range(0, len(pending), self.inference_batch_size):
            chunk = pending[start : start + self.inference_batch_size]
            scores = self._score(chunk)
            for sample, score in zip(chunk, scores, strict=True):
                caption = sample["caption_raw"]
                rows[sample["image_id"]] = {
                    "image_id": sample["image_id"],
                    "caption_version": "raw",
                    "caption_sha256": (
                        hashlib.sha256(caption.encode("utf-8")).hexdigest() if isinstance(caption, str) else None
                    ),
                    "status": score["status"],
                    "score_mu": score["score_mu"],
                    "score_sigma": score["score_sigma"],
                    "error": score["error"],
                }
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pd.DataFrame([rows[row["image_id"]] for row in records]),
            _metadata={**task._metadata, "annotation_path": str(path)},
            _stage_perf=task._stage_perf,
        )


@dataclass
class HPSv3AnnotationWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    name: str = "hpsv3_annotation_writer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "status", "score_mu"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> FileGroupTask:
        path = Path(task._metadata["annotation_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            table = pa.Table.from_pandas(task.to_pandas(), schema=_SCORE_SCHEMA, preserve_index=False)
            pq.write_table(table, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[str(path)],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
