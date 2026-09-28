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

import importlib.util
import io
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemo_curator.stages.image.deduplication.caption import CaptionSource


@pytest.mark.parametrize(
    ("other_extension", "image_format"),
    [("jpg", "JPEG"), ("jpeg", "JPEG"), ("png", "PNG"), ("webp", "WEBP"), ("JPG", "JPEG")],
)
def test_report_exports_sampled_pairs_and_original_captions(
    tmp_path: Path, other_extension: str, image_format: str
) -> None:
    source_root = tmp_path / "images"
    source_root.mkdir()
    tar_path = source_root / "part.tar"
    with tarfile.open(tar_path, "w") as archive:
        for key, extension, format_name in (("a", "jpg", "JPEG"), ("b", other_extension, image_format)):
            image = io.BytesIO()
            Image.new("RGB", (16, 16), "red").save(image, format=format_name)
            info = tarfile.TarInfo(f"{key}.{extension}")
            info.size = len(image.getvalue())
            archive.addfile(info, io.BytesIO(image.getvalue()))
    with tarfile.open(tar_path) as archive:
        members = list(archive)
    with open(f"{tar_path}.idx", "w", encoding="utf-8") as index_file:
        index_file.write("v1.2 2\n")
        index_file.writelines(
            f"{member.name.rsplit('.', 1)[1]} {member.offset_data} {member.size} {member.name}\n" for member in members
        )

    decisions = tmp_path / "decisions"
    decisions.mkdir()
    (decisions / "pairs").mkdir()
    (decisions / "captions").mkdir()
    ids = ["source|part.tar|a", "source|part.tar|b"]
    pq.write_table(
        pa.table(
            {"id_a": [ids[0]], "id_b": [ids[1]], "cosine_sim_score": [0.999], "caption_relation": ["same_caption"]}
        ),
        decisions / "pairs/cluster_0_00000000.parquet",
    )
    pq.write_table(
        pa.table({"image_id": ids, "caption_raw": ["red square", "red  square"]}),
        decisions / "captions/cluster_0_captions.parquet",
    )

    report_path = Path(__file__).resolve().parents[4] / "tutorials/image/caption-dedup/report.py"
    spec = importlib.util.spec_from_file_location("caption_dedup_report", report_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = module.create_report(decisions, {"source": CaptionSource(str(source_root))}, tmp_path / "report", 3)
    page = index.read_text(encoding="utf-8")
    assert "same_caption" in page
    assert "red  square" in page
    assert len(list((tmp_path / "report/assets").glob("*.jpg"))) == 2
