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


"""Extract selected video frames into bounded TAR and manifest shards."""

import io
import os
import tarfile
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.recaption.ego4d import PLAN_SCHEMA
from nemo_curator.tasks import DocumentBatch, FileGroupTask, ImageBatch, ImageObject

MANIFEST_SCHEMA = pa.schema(
    [
        *PLAN_SCHEMA,
        ("shard", pa.string()),
        ("member", pa.string()),
        ("offset", pa.int64()),
        ("size", pa.int64()),
        ("decode_status", pa.string()),
        ("width", pa.int64()),
        ("height", pa.int64()),
        ("decode_error", pa.string()),
    ]
)


def write_parquet(rows: list[dict], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        table = pa.Table.from_pylist(rows, schema=schema)
        if "source" in table.column_names and path.parent.name.startswith("source="):
            table = table.drop(["source"])
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Ego4DFrameExtractStage(ProcessingStage[DocumentBatch, FileGroupTask]):
    source_root: str
    source_name: str
    output: str
    batch_size: int = 64
    jpeg_quality: int = 90
    retry_failed: bool = False
    name: str = "ego4d_frame_extract"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "video", "video_uid", "frame_number"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> FileGroupTask:
        import cv2

        rows = sorted(task.to_pyarrow().to_pylist(), key=lambda row: row["frame_number"])
        capture = None
        manifests = []
        try:
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start : start + self.batch_size]
                stem = f"{chunk[0]['video_uid']}-{start // self.batch_size:06d}"
                root = Path(self.output)
                manifest = root / "samples/schema-v1/manifests" / f"source={self.source_name}" / f"{stem}.parquet"
                media = root / "media" / f"source={self.source_name}" / f"{stem}.tar"
                manifests.append(str(manifest))
                if manifest.exists() and media.exists():
                    statuses = pq.ParquetFile(manifest).read(columns=["decode_status"])["decode_status"].to_pylist()
                    if not self.retry_failed or "failed" not in statuses:
                        continue
                if capture is None:
                    capture = cv2.VideoCapture(str(Path(self.source_root) / chunk[0]["video"]))
                media.parent.mkdir(parents=True, exist_ok=True)
                temporary = media.with_name(f".{media.name}.{uuid.uuid4().hex}.tmp")
                records = []
                try:
                    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
                        for row in chunk:
                            record = {
                                **row,
                                "source": self.source_name,
                                "shard": media.name,
                                "member": f"{row['video_uid']}/{row['frame_number']:09d}.jpg",
                                "offset": None,
                                "size": None,
                                "decode_status": "failed",
                                "width": None,
                                "height": None,
                                "decode_error": None,
                            }
                            capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_number"])
                            ok, pixels = capture.read()
                            if not ok:
                                record["decode_error"] = (
                                    f"Cannot decode frame {row['frame_number']} from {row['video']}"
                                )
                            else:
                                encoded, payload = cv2.imencode(
                                    ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                                )
                                if not encoded:
                                    msg = f"Cannot encode frame {row['frame_number']}"
                                    raise OSError(msg)
                                info = tarfile.TarInfo(record["member"])
                                info.size = len(payload)
                                record.update(
                                    offset=archive.offset + tarfile.BLOCKSIZE,
                                    size=info.size,
                                    decode_status="ok",
                                    width=pixels.shape[1],
                                    height=pixels.shape[0],
                                )
                                archive.addfile(info, io.BytesIO(payload.tobytes()))
                            records.append(record)
                    os.replace(temporary, media)
                    write_parquet(records, MANIFEST_SCHEMA, manifest)
                finally:
                    temporary.unlink(missing_ok=True)
        finally:
            if capture is not None:
                capture.release()
        return FileGroupTask(
            dataset_name=task.dataset_name, data=manifests, _metadata=task._metadata, _stage_perf=task._stage_perf
        )


@dataclass
class RecaptionImageReader(ProcessingStage[DocumentBatch, ImageBatch]):
    media_root: str
    source_name: str
    name: str = "recaption_image_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "shard", "member", "offset", "size", "context_json"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> ImageBatch:
        images = []
        with ExitStack() as stack:
            archives = {}
            for row in task.to_pyarrow().to_pylist():
                path = Path(self.media_root) / row["shard"]
                pixels = None
                error = row["decode_error"]
                if row["decode_status"] == "ok":
                    if path not in archives:
                        archives[path] = stack.enter_context(path.open("rb"))
                    archive = archives[path]
                    archive.seek(row["offset"])
                    try:
                        with Image.open(io.BytesIO(archive.read(row["size"]))) as image:
                            pixels = np.array(image.convert("RGB"))
                    except (UnidentifiedImageError, OSError) as exc:
                        error = str(exc)
                images.append(
                    ImageObject(
                        image_id=row["image_id"],
                        image_path=f"{path}:{row['offset']}:{row['member']}",
                        image_data=pixels,
                        metadata={
                            "source": self.source_name,
                            "context_json": row["context_json"],
                            "decode_error": error,
                        },
                    )
                )
        return ImageBatch(
            dataset_name=task.dataset_name, data=images, _metadata=task._metadata, _stage_perf=task._stage_perf
        )
