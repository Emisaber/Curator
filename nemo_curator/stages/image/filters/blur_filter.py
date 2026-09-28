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

"""Score decoded images using the interleaved blur filter's Laplacian metric."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import ImageBatch
from nemo_curator.utils.hash_utils import get_deterministic_hash

_ANNOTATION_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("status", pa.string()),
        ("laplacian_variance", pa.float64()),
        ("is_blurry", pa.bool_()),
        ("error", pa.string()),
    ]
)

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class ImageBlurFilterStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Annotate RGB images with Laplacian variance; optionally remove blurry images."""

    score_threshold: float = 100.0
    drop_blurry: bool = False
    annotations_dir: str | None = None
    name: str = "image_blur_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: ImageBatch) -> ImageBatch:
        if cv2 is None:
            msg = (
                "opencv-python-headless is required for the image blur filter. "
                "Install with: pip install nemo_curator[cv2]"
            )
            raise ImportError(msg)

        kept = []
        records = []
        for image in task.data:
            if image.image_data is None:
                msg = f"Image {image.image_id} has no decoded image_data"
                raise ValueError(msg)
            status = "ok"
            error = None
            try:
                score = float(cv2.Laplacian(image.image_data, cv2.CV_64F).var())
            except cv2.error as exc:
                logger.debug(
                    "cv2.Laplacian failed (image_id={} image_shape={}): {}",
                    image.image_id,
                    image.image_data.shape,
                    exc,
                )
                score = None
                status = "failed"
                error = str(exc)
            image.metadata["laplacian_variance"] = score
            image.metadata["is_blurry"] = score < self.score_threshold if score is not None else None
            if self.annotations_dir is not None:
                if not image.image_id:
                    msg = "Cannot persist blur annotation without image_id"
                    raise ValueError(msg)
                records.append(
                    {
                        "image_id": image.image_id,
                        "status": status,
                        "laplacian_variance": score,
                        "is_blurry": image.metadata["is_blurry"],
                        "error": error,
                    }
                )
            if not (self.drop_blurry and image.metadata["is_blurry"]):
                kept.append(image)

        if records:
            image_ids = [record["image_id"] for record in records]
            output_dir = Path(self.annotations_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"part-{get_deterministic_hash(image_ids)}.parquet"
            pq.write_table(pa.Table.from_pylist(records, schema=_ANNOTATION_SCHEMA), path)

        if not self.drop_blurry:
            return task
        return ImageBatch(
            data=kept,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
