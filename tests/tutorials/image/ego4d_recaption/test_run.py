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

"""Exercise both source pipelines, persistent results and a static review report."""

import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from tests.stages.image.recaption.test_vlm import CaptionHandler


def load_tutorial(name: str):
    path = Path(__file__).resolve().parents[4] / "tutorials/image/ego4d-recaption" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"ego4d_recaption_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_sources_pipeline_report_and_resume(tmp_path: Path) -> None:
    root = tmp_path / "data"
    annotations = root / "v2/annotations"
    annotations.mkdir(parents=True)
    videos = []
    for uid in ("fho", "narration"):
        writer = cv2.VideoWriter(str(root / f"{uid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        assert writer.isOpened()
        for index in range(30):
            writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
        writer.release()
        videos.append(
            {
                "video_uid": uid,
                "duration_sec": 3.0,
                "video_metadata": {"fps": 10, "num_frames": 30, "width": 64, "height": 48},
                "redacted_intervals": [],
                "video_components": [],
            }
        )
    (root / "v2/ego4d.json").write_text(json.dumps({"videos": videos}))
    action = {
        "uid": "action",
        "is_valid_action": True,
        "start_frame": 0,
        "end_frame": 19,
        "critical_frames": {"pre_frame": 5, "contact_frame": 12},
        "frames": [],
        "narration_text": "#C C picks up a cup",
    }
    (annotations / "fho_main.json").write_text(
        json.dumps(
            {"videos": [{**videos[0], "annotated_intervals": [{"redacted": False, "narrated_actions": [action]}]}]}
        )
    )
    narration = {
        "narration_pass_1": {
            "narrations": [
                {
                    "timestamp_sec": 2.4,
                    "timestamp_frame": 24,
                    "narration_text": "#C C holds a cup",
                    "annotation_uid": "narration",
                }
            ],
            "summaries": [],
        }
    }
    (annotations / "narration.json").write_text(json.dumps({"fho": narration, "narration": narration}))
    CaptionHandler.requests = []
    CaptionHandler.fail = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "output"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "data_root": str(root),
                "output": str(output),
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "batch_size": 64,
                "extraction_workers": 1,
                "caption_workers": 1,
                "requests_per_worker": 2,
                "num_cpus": 4,
                "num_gpus": 0,
                "sample_counts": {"fho": 3, "narration": 1},
            }
        )
    )
    try:
        runtime = load_tutorial("run")
        report = load_tutorial("experiment").run_experiment(config)
        paths = list((output / "annotations/recaption/schema-v1/prompt-v1").glob("source=*/*.parquet"))
        assert len(paths) == 2
        rows = [row for path in paths for row in pq.ParquetFile(path).read().to_pylist()]
        assert len(rows) == 4
        assert all(row["status"] == "ok" for row in rows)
        assert len(CaptionHandler.requests) == 4
        assert (output / "annotations/recaption/schema-v1/prompt-v1/run.json").exists()
        page = report.read_text()
        assert "Original annotations and associations" in page
        assert "#C C picks up a cup" in page
        assert "A hand holds a red cup." in page
        assert len(list((report.parent / "assets").glob("*.jpg"))) == 4
        assert page.count("<article>") == 4
        assert "all results shown" in page
        assert "0 pending" in page
        assert "fho system prompt" in page
        assert "<script" not in page
        runtime.run(config)
        assert len(CaptionHandler.requests) == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_fixed_counts_preserve_selection_and_reject_changed_settings(tmp_path: Path) -> None:
    root = tmp_path / "data"
    annotations = root / "v2/annotations"
    annotations.mkdir(parents=True)
    videos = []
    fho = []
    narration = {}
    for index in range(8):
        uid = f"video-{index}"
        (root / f"{uid}.mp4").touch()
        info = {
            "video_uid": uid,
            "duration_sec": 20,
            "video_metadata": {"fps": 10, "num_frames": 200, "width": 64, "height": 48},
            "redacted_intervals": [],
            "video_components": [],
        }
        videos.append(info)
        if index < 4:
            fho.append(
                {
                    **info,
                    "annotated_intervals": [
                        {
                            "redacted": False,
                            "narrated_actions": [
                                {
                                    "uid": f"action-{index}",
                                    "is_valid_action": True,
                                    "start_frame": 0,
                                    "end_frame": 99,
                                    "critical_frames": {"contact_frame": 50},
                                    "frames": [],
                                    "narration_text": "#C C holds a cup",
                                }
                            ],
                        }
                    ],
                }
            )
        narration[uid] = {
            "narration_pass_1": {
                "narrations": [
                    {"timestamp_sec": t, "timestamp_frame": int(t * 10), "narration_text": "#C C holds a cup"}
                    for t in (2.5, 7.5, 12.5, 17.5)
                ]
            }
        }
    (root / "v2/ego4d.json").write_text(json.dumps({"videos": videos}))
    (annotations / "fho_main.json").write_text(json.dumps({"videos": fho}))
    (annotations / "narration.json").write_text(json.dumps(narration))
    config = {
        "data_root": str(root),
        "output": str(tmp_path / "output"),
        "sample_counts": {"fho": 5, "narration": 6},
        "sampling_seed": 42,
    }
    runtime = load_tutorial("run")
    plans = runtime.prepare(config)
    rows = {
        kind: [row for path in paths for row in pq.ParquetFile(path).read().to_pylist()]
        for kind, paths in plans.items()
    }
    assert len(rows["fho"]) == 5
    assert len(rows["narration"]) == 6
    assert all(row["video_uid"] in {f"video-{i}" for i in range(4, 8)} for row in rows["narration"])
    assert len({row["image_id"] for values in rows.values() for row in values}) == 11
    before = {path: Path(path).stat().st_mtime_ns for paths in plans.values() for path in paths}
    assert runtime.prepare(config) == plans
    assert before == {path: Path(path).stat().st_mtime_ns for path in before}
    other = runtime.prepare({**config, "output": str(tmp_path / "same-seed")})
    other_ids = {
        row["image_id"]
        for paths in other.values()
        for path in paths
        for row in pq.ParquetFile(path).read().to_pylist()
    }
    assert other_ids == {row["image_id"] for values in rows.values() for row in values}
    with pytest.raises(ValueError, match="different settings"):
        runtime.prepare({**config, "sampling_seed": 43})
    with pytest.raises(ValueError, match="fewer than"):
        runtime.prepare(
            {**config, "output": str(tmp_path / "insufficient"), "sample_counts": {"fho": 21, "narration": 6}}
        )
