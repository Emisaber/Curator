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

"""Tests for the static VLM annotation review."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def test_report_separates_failures_from_field_distribution(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/report.py"
    spec = importlib.util.spec_from_file_location("vlm_annotation_report", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    image_bytes = io.BytesIO()
    Image.new("RGB", (16, 16)).save(image_bytes, format="JPEG")
    archive = tmp_path / "images.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("a.jpg")
        info.size = image_bytes.tell()
        tar.addfile(info, io.BytesIO(image_bytes.getvalue()))
    image_path = f"{archive}:512:a.jpg"
    annotations = tmp_path / "annotations"
    (annotations / "source=test").mkdir(parents=True)
    (annotations / "run.json").write_text(json.dumps({"fields": ["clarity"]}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "image_id": "ok",
                "status": "ok",
                "image_path": image_path,
                "clarity": "sharp",
                "raw_response": '{"clarity":"sharp"}',
                "error": None,
            },
            {
                "image_id": "failed",
                "status": "length",
                "image_path": image_path,
                "clarity": None,
                "raw_response": '{"clarity":',
                "error": "Output reached max_output_tokens",
            },
        ]
    ).to_parquet(annotations / "source=test/part-1.parquet", index=False)

    index = module.create_report(annotations, tmp_path / "report")
    page = index.read_text(encoding="utf-8")
    assert "2 written rows" in page
    assert 'id="failures"' in page
    assert "Output reached max_output_tokens" in page
    assert 'id="clarity"' in page
    assert "<h3>sharp <small>1</small></h3>" in page
    assert "<h3>None" not in page
    assert len(list((tmp_path / "report/assets").glob("*.jpg"))) == 2


def test_report_reads_vlv_success_and_failure_previews(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/report.py"
    spec = importlib.util.spec_from_file_location("vlm_native_report", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shard = tmp_path / "images.parquet"
    pixels = np.full((3, 16, 16), 100, dtype=np.uint8)
    pd.DataFrame({"image": [pixels.tobytes(), pixels.tobytes()]}).to_parquet(shard, index=False)
    annotations = tmp_path / "annotations"
    (annotations / "source=vlv").mkdir(parents=True)
    (annotations / "run.json").write_text(
        json.dumps(
            {
                "fields": ["clarity"],
                "sources": {
                    "vlv": {"root": str(tmp_path), "format": "vlv_parquet", "image_shape": [3, 16, 16]}
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"image_id": "vlv|images.parquet|0", "image_path": f"{shard}:0", "status": "ok", "clarity": "sharp"},
            {"image_id": "vlv|images.parquet|1", "image_path": f"{shard}:1", "status": "length", "clarity": None},
        ]
    ).to_parquet(annotations / "source=vlv/part-1.parquet", index=False)
    report = module.create_report(annotations, tmp_path / "report")
    assert 'id="failures"' in report.read_text(encoding="utf-8")
    assert len(list((tmp_path / "report/assets").glob("*.jpg"))) == 2
