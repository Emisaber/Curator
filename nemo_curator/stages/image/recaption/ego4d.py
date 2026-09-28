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


"""Select Ego4D FHO and narration frames without decoding video."""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa

CRITICAL_ROLES = ("pre_45", "pre_30", "pre_15", "pre_frame", "contact_frame", "pnr_frame", "post_frame")
PLAN_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("source", pa.string()),
        ("video_uid", pa.string()),
        ("video", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp_sec", pa.float64()),
        ("caption_raw", pa.string()),
        ("context_json", pa.string()),
        ("annotation_json", pa.string()),
    ]
)


def valid_frame(frame: int, timestamp: float, metadata: dict) -> bool:
    info = metadata["video_metadata"]
    if not 0 <= frame < info["num_frames"] or not 0 <= timestamp <= metadata["duration_sec"]:
        return False
    for interval in metadata.get("redacted_intervals", []):
        if interval["start_frame"] <= frame <= interval["end_frame"]:
            return False
    return not any(
        component["redacted"]
        and component["canonical_video_start_frame"] <= frame <= component["canonical_video_end_frame"]
        for component in metadata.get("video_components", [])
    )


def _sample_row(source: str, uid: str, frame: int, timestamp: float, context: dict, annotations: list) -> dict:  # noqa: PLR0913
    return {
        "image_id": f"{source}|{uid}|{frame}",
        "source": source,
        "video_uid": uid,
        "video": f"{uid}.mp4",
        "frame_number": frame,
        "timestamp_sec": timestamp,
        "caption_raw": None,
        "context_json": json.dumps(context, ensure_ascii=False),
        "annotation_json": json.dumps(annotations, ensure_ascii=False),
    }


def _object_hints(frame: dict | None, width: int, height: int) -> list[dict]:
    if frame is None:
        return []
    hints = []
    for box in frame["boxes"]:
        bbox = box["bbox"]
        hints.append(
            {
                **{
                    name: box.get(name)
                    for name in ("object_type", "structured_noun", "freeform_noun", "unsure_noun", "instance_number")
                },
                "bbox": [bbox["x"] / width, bbox["y"] / height, bbox["width"] / width, bbox["height"] / height],
            }
        )
    return hints


def sample_fho(video: dict, metadata: dict, source: str) -> list[dict]:
    fps = metadata["video_metadata"]["fps"]
    candidates = defaultdict(list)
    for interval_index, interval in enumerate(video["annotated_intervals"]):
        if interval["redacted"]:
            continue
        for action_index, action in enumerate(interval["narrated_actions"]):
            if not action["is_valid_action"] or action.get("is_rejected") or action.get("is_invalid_annotation"):
                continue
            critical = {
                role: value
                for role, value in (action.get("critical_frames") or {}).items()
                if role in CRITICAL_ROLES and isinstance(value, int) and not isinstance(value, bool)
            }
            roles = defaultdict(list)
            for role, frame in critical.items():
                roles[frame].append(role)
            start, end = action["start_frame"], action["end_frame"]
            for index in range(4):
                roles[start + (2 * index + 1) * (end - start + 1) // 8].append(f"uniform_{index}")
            frame_annotations = {frame["frame_number"]: frame for frame in (action.get("frames") or [])}
            for number, labels in roles.items():
                timestamp = number / fps
                if not valid_frame(number, timestamp, metadata):
                    continue
                exact_frame = frame_annotations.get(number)
                context = {
                    "action_context": {
                        **{
                            key: action.get(key)
                            for key in (
                                "narration_text",
                                "structured_verb",
                                "freeform_verb",
                                "state_transition",
                                "is_partial",
                            )
                        }
                    },
                    "frame_context": {
                        "timestamp_sec": timestamp,
                        "seconds_from_window_start": (number - start) / fps,
                        "sampling_roles": labels,
                        "seconds_from_landmarks": {
                            role: (number - value) / fps
                            for role, value in critical.items()
                            if role in ("pre_frame", "contact_frame", "pnr_frame", "post_frame")
                        },
                    },
                    "object_hints": _object_hints(
                        exact_frame, video["video_metadata"]["width"], video["video_metadata"]["height"]
                    ),
                }
                annotation = {
                    "interval_index": interval_index,
                    "action_index": action_index,
                    "action_uid": action.get("uid"),
                    "narration_annotation_uid": action.get("narration_annotation_uid"),
                    "sampling_roles": labels,
                    "action": {key: value for key, value in action.items() if key != "frames"},
                    "frame_annotation": exact_frame,
                }
                priority = (
                    exact_frame is None,
                    not any(label in CRITICAL_ROLES for label in labels),
                    interval_index,
                    action_index,
                )
                candidates[number].append((priority, context, annotation))
    rows = []
    for number, associations in sorted(candidates.items()):
        selected = min(associations, key=lambda item: item[0])
        rows.append(
            _sample_row(source, video["video_uid"], number, number / fps, selected[1], [a[2] for a in associations])
        )
    return rows


def sample_narration(uid: str, record: dict, metadata: dict, source: str) -> list[dict]:
    by_frame = defaultdict(list)
    for pass_name in ("narration_pass_1", "narration_pass_2"):
        for index, narration in enumerate(record.get(pass_name, {}).get("narrations", [])):
            text = narration.get("narration_text") or ""
            number, timestamp = narration.get("timestamp_frame"), narration.get("timestamp_sec")
            if not text.strip() or "#unsure" in text.lower():
                continue
            if (
                not isinstance(number, int)
                or isinstance(number, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or not valid_frame(number, timestamp, metadata)
            ):
                continue
            by_frame[number].append({"pass": pass_name, "narration_index": index, "narration": narration})
    bins = defaultdict(list)
    for number, associations in by_frame.items():
        anchor = associations[0]
        timestamp = anchor["narration"]["timestamp_sec"]
        bins[int(timestamp // 5)].append((number, timestamp, anchor, associations))
    rows = []
    for interval, candidates in sorted(bins.items()):
        center = interval * 5 + 2.5
        number, timestamp, anchor, associations = min(candidates, key=lambda item: (abs(item[1] - center), item[0]))
        summaries = record.get(anchor["pass"], {}).get("summaries", [])
        covering = [s for s in summaries if s["start_sec"] <= timestamp <= s["end_sec"]]
        summary = (
            min(covering, key=lambda s: abs((s["start_sec"] + s["end_sec"]) / 2 - timestamp)) if covering else None
        )
        context = {
            "target_frame": {"timestamp_sec": number / metadata["video_metadata"]["fps"]},
            "anchor_narration": {"text": anchor["narration"]["narration_text"], "timestamp_sec": timestamp},
            "summary": {
                "text": summary["summary_text"],
                "start_sec": summary["start_sec"],
                "end_sec": summary["end_sec"],
            }
            if summary
            else None,
        }
        rows.append(
            _sample_row(source, uid, number, timestamp, context, [{"anchors": associations, "summary": summary}])
        )
    return rows


def read_video_metadata(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        videos = json.load(stream)["videos"]
    return {video["video_uid"]: video for video in videos}
