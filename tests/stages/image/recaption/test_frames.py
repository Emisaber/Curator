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


"""Exercise real video extraction, TAR offsets and failed-frame persistence."""

from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.image.recaption.ego4d import PLAN_SCHEMA
from nemo_curator.stages.image.recaption.frames import Ego4DFrameExtractStage, RecaptionImageReader
from nemo_curator.tasks import DocumentBatch


def test_extract_video_tar_and_resume(tmp_path: Path) -> None:
    video_path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(12):
        pixels = np.zeros((48, 64, 3), dtype=np.uint8)
        pixels[:, :, 2] = index * 15
        writer.write(pixels)
    writer.release()
    rows = [
        {
            "image_id": f"ego4d-fho|v|{number}",
            "source": "ego4d-fho",
            "video_uid": "v",
            "video": "v.mp4",
            "frame_number": number,
            "timestamp_sec": number / 10,
            "caption_raw": None,
            "context_json": "{}",
            "annotation_json": "[]",
        }
        for number in (1, 8, 50)
    ]
    batch = DocumentBatch(dataset_name="test", data=pa.Table.from_pylist(rows, schema=PLAN_SCHEMA))
    stage = Ego4DFrameExtractStage(
        source_root=str(tmp_path), source_name="ego4d-fho", output=str(tmp_path / "out"), batch_size=3
    )
    result = stage.process(batch)
    manifest = Path(result.data[0])
    table = pq.ParquetFile(manifest).read()
    assert table["decode_status"].to_pylist() == ["ok", "ok", "failed"]
    assert "source" not in table.column_names
    reader = RecaptionImageReader(media_root=str(tmp_path / "out/media/source=ego4d-fho"), source_name="ego4d-fho")
    images = reader.process(DocumentBatch(dataset_name="test", data=table))
    assert [image.image_id for image in images.data] == [row["image_id"] for row in rows]
    assert images.data[0].image_data.shape == (48, 64, 3)
    assert abs(int(images.data[1].image_data[0, 0, 0]) - 120) < 10
    assert images.data[2].image_data is None
    mtime = manifest.stat().st_mtime_ns
    assert stage.process(batch).data == result.data
    assert manifest.stat().st_mtime_ns == mtime
    assert not list((tmp_path / "out").rglob("*.tmp"))
