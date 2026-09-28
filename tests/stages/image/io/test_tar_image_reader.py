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

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nemo_curator.stages.image.io.tar_image_reader import TarImageDecodeStage
from nemo_curator.tasks import DocumentBatch


def test_decode_orientation_and_corrupt_member(tmp_path: Path):
    image = Image.new("RGB", (12, 8), "red")
    exif = image.getexif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    content = buffer.getvalue()
    (tmp_path / "input.tar").write_bytes(content + b"broken")
    rows = [
        {
            "image_id": "valid",
            "source": "s",
            "shard": "input.tar",
            "member": "a.jpg",
            "offset": 0,
            "size": len(content),
        },
        {"image_id": "bad", "source": "s", "shard": "input.tar", "member": "b.jpg", "offset": len(content), "size": 6},
    ]
    stage = TarImageDecodeStage({"s": str(tmp_path)}, str(tmp_path / "records"))
    result = stage.process(DocumentBatch(dataset_name="images", data=pd.DataFrame(rows)))
    assert result is not None
    assert len(result.data) == 1
    assert result.data[0].image_id == "valid"
    assert result.data[0].image_data.shape == (12, 8, 3)
    assert result.data[0].image_data.dtype == np.uint8
    assert result.data[0].metadata["short_side_below_256"] is True
    records = pd.read_parquet(tmp_path / "records").set_index("image_id")
    assert records.loc["valid", "status"] == "ok"
    assert records.loc["bad", "status"] == "failed"
    assert pd.isna(records.loc["valid", "error"])
    assert bool(records.loc["valid", "short_side_below_256"])
    assert records.loc["bad", "error"]
    assert pd.isna(records.loc["bad", "short_side_below_256"])
    assert (tmp_path / "input.tar").read_bytes() == content + b"broken"


def test_short_side_tag_keeps_decodable_images(tmp_path: Path):
    rows = []
    content = bytearray()
    for image_id, size in (("at_boundary", (256, 256)), ("narrow", (255, 256))):
        buffer = BytesIO()
        Image.new("RGB", size, "red").save(buffer, format="JPEG")
        payload = buffer.getvalue()
        rows.append(
            {
                "image_id": image_id,
                "source": "s",
                "shard": "input.tar",
                "member": f"{image_id}.jpg",
                "offset": len(content),
                "size": len(payload),
            }
        )
        content.extend(payload)
    (tmp_path / "input.tar").write_bytes(content)

    stage = TarImageDecodeStage({"s": str(tmp_path)}, str(tmp_path / "records"))
    result = stage.process(DocumentBatch(dataset_name="images", data=pd.DataFrame(rows)))
    assert result is not None
    assert {image.image_id for image in result.data} == {"at_boundary", "narrow"}
    assert {image.image_id: image.metadata["short_side_below_256"] for image in result.data} == {
        "at_boundary": False,
        "narrow": True,
    }
    records = pd.read_parquet(tmp_path / "records").set_index("image_id")
    assert not bool(records.loc["at_boundary", "short_side_below_256"])
    assert bool(records.loc["narrow", "short_side_below_256"])


def test_all_corrupt_still_writes_record(tmp_path: Path):
    (tmp_path / "input.tar").write_bytes(b"broken")
    task = DocumentBatch(
        dataset_name="images",
        data=pd.DataFrame(
            [
                {
                    "image_id": "bad",
                    "source": "s",
                    "shard": "input.tar",
                    "member": "a.jpg",
                    "offset": 0,
                    "size": 6,
                }
            ]
        ),
    )
    assert TarImageDecodeStage({"s": str(tmp_path)}, str(tmp_path / "records")).process(task) is None
    records = pd.read_parquet(tmp_path / "records")
    assert len(records) == 1
    assert records.iloc[0]["status"] == "failed"
