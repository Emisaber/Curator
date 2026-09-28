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


"""Validate source annotations, exact frame matching and fixed-bin selection."""

import json

import pytest

from nemo_curator.stages.image.recaption.ego4d import sample_fho, sample_narration


def metadata() -> dict:
    return {
        "duration_sec": 30,
        "video_metadata": {"fps": 10, "num_frames": 300},
        "redacted_intervals": [],
        "video_components": [],
    }


def test_fho_merges_frames_and_keeps_exact_boxes_and_partial_actions() -> None:
    frame = {
        "frame_number": 50,
        "boxes": [
            {
                "object_type": "object_of_change",
                "structured_noun": "cup",
                "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
            }
        ],
    }
    action = {
        "uid": "a",
        "is_valid_action": True,
        "is_partial": True,
        "start_frame": 20,
        "end_frame": 99,
        "critical_frames": {"pre_45": 10, "pre_frame": 50, "contact_frame": 60, "pnr_frame": 70, "post_frame": 80},
        "narration_text": "#C C holds a cup",
        "frames": [frame],
    }
    video = {
        "video_uid": "video",
        "video_metadata": {"width": 100, "height": 80},
        "annotated_intervals": [{"redacted": False, "narrated_actions": [action, {**action, "uid": "b"}]}],
    }
    rows = sample_fho(video, metadata(), "ego4d-fho")
    assert [r["frame_number"] for r in rows] == [10, 30, 50, 60, 70, 80, 90]
    selected = next(row for row in rows if row["frame_number"] == 50)
    context = json.loads(selected["context_json"])
    assert context["action_context"]["is_partial"] is True
    assert context["frame_context"]["sampling_roles"] == ["pre_frame", "uniform_1"]
    assert context["object_hints"][0]["bbox"] == [0.1, 0.25, 0.3, 0.5]
    assert len(json.loads(selected["annotation_json"])) == 2
    assert json.loads(rows[-1]["context_json"])["object_hints"] == []


@pytest.mark.parametrize("critical_frames", [None, {"pre_frame": 50, "contact_frame": 60}])
def test_fho_samples_actions_with_null_frame_annotations(critical_frames: dict | None) -> None:
    action = {
        "is_valid_action": True,
        "is_partial": False,
        "start_frame": 20,
        "end_frame": 99,
        "critical_frames": critical_frames,
        "narration_text": "#C C holds a cup",
        "frames": None,
    }
    video = {
        "video_uid": "video",
        "video_metadata": {"width": 100, "height": 80},
        "annotated_intervals": [{"redacted": False, "narrated_actions": [action]}],
    }
    rows = sample_fho(video, metadata(), "ego4d-fho")
    assert [row["frame_number"] for row in rows] == ([30, 50, 60, 70, 90] if critical_frames else [30, 50, 70, 90])
    for row in rows:
        context = json.loads(row["context_json"])
        assert context["action_context"]["narration_text"] == action["narration_text"]
        assert context["object_hints"] == []
        assert json.loads(row["annotation_json"])[0]["frame_annotation"] is None
    if critical_frames:
        selected = next(row for row in rows if row["frame_number"] == 50)
        assert json.loads(selected["context_json"])["frame_context"]["sampling_roles"] == ["pre_frame", "uniform_1"]


def test_narration_selects_bin_center_and_summary_from_selected_pass() -> None:
    def narration(t: float, text: str) -> dict:
        return {"timestamp_sec": t, "timestamp_frame": round(t * 10), "narration_text": text, "annotation_uid": text}

    summary = {"start_sec": 0, "end_sec": 10, "summary_text": "kitchen", "annotation_uid": "summary"}
    record = {
        "narration_pass_1": {
            "narrations": [
                narration(1, "early"),
                narration(2.4, "selected"),
                narration(4.9, "late"),
                narration(7.5, "#unsure"),
            ],
            "summaries": [summary],
        },
        "narration_pass_2": {
            "narrations": [
                narration(2.4, "same frame other caption"),
                narration(5.1, "next bin"),
                narration(20.1, "no summary"),
            ],
            "summaries": [],
        },
    }
    rows = sample_narration("v", record, metadata(), "ego4d-narration")
    assert [r["frame_number"] for r in rows] == [24, 51, 201]
    first = json.loads(rows[0]["context_json"])
    assert first["anchor_narration"]["text"] == "selected"
    assert first["summary"]["text"] == "kitchen"
    assert len(json.loads(rows[0]["annotation_json"])[0]["anchors"]) == 2
    assert json.loads(rows[1]["context_json"])["summary"] is None


def test_narration_filters_redacted_and_out_of_bounds_frames() -> None:
    info = metadata()
    info["redacted_intervals"] = [{"start_frame": 20, "end_frame": 30}]
    rows = [
        {"timestamp_sec": t, "timestamp_frame": frame, "narration_text": "visible"}
        for t, frame in [(2.5, 25), (7.5, 75), (35, 350), (-1, -10)]
    ]
    selected = sample_narration("v", {"narration_pass_1": {"narrations": rows}}, info, "ego4d-narration")
    assert [r["frame_number"] for r in selected] == [75]
