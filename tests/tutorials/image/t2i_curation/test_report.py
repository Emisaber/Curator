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

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.parametrize("changed_caption", [False, True])
def test_report_reviews_original_captions_and_checks_scored_caption_identity(
    tmp_path: Path, changed_caption: bool
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/t2i-curation/report.py"
    spec = importlib.util.spec_from_file_location("t2i_score_report", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    media = tmp_path / "media"
    media.mkdir()
    caption = "  caption <tag>\nsecond line  "
    pixels = np.zeros((3, 2, 2), dtype=np.uint8).tobytes()
    pq.write_table(pa.table({"image": [pixels], "caption": [caption]}), media / "data.parquet")
    (tmp_path / "run.json").write_text(
        json.dumps({"sources": {"vlv": {"root": str(media), "format": "vlv_parquet", "image_shape": [3, 2, 2]}}})
    )
    image_id = "vlv|data.parquet|0"
    for stage, row in (
        ("decode", {"image_id": image_id, "status": "ok"}),
        ("nsfw", {"image_id": image_id, "status": "ok", "nsfw_score": 0.4, "error": None}),
        (
            "hpsv3",
            {
                "image_id": image_id,
                "status": "ok",
                "score_mu": 1.5,
                "score_sigma": 0.25,
                "caption_version": "raw",
                "caption_sha256": hashlib.sha256(("old" if changed_caption else caption).encode()).hexdigest(),
                "error": None,
            },
        ),
    ):
        directory = tmp_path / f"annotations/{stage}/schema-v1/source=vlv"
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([row]), directory / "part-test.parquet")
    if changed_caption:
        with pytest.raises(ValueError, match="differs from the scored caption"):
            module.create_report(tmp_path, tmp_path / "report", per_group=1)
        return
    page = module.create_report(tmp_path, tmp_path / "report", per_group=1)
    text = page.read_text(encoding="utf-8")
    assert "caption &lt;tag&gt;" in text
    assert "score_sigma" in text
    assert "nsfw_score" in text
    assert len(list((page.parent / "assets").glob("*.jpg"))) == 1
    summary = json.loads((page.parent / "summary.json").read_text())
    assert summary["decode_success"] == {"vlv": 1}
    assert summary["scores"]["hpsv3/vlv"]["statuses"] == {"ok": 1}
