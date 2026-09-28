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

"""Structured image annotations from an OpenAI-compatible vision-language service."""

import base64
import hashlib
import io
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from openai import APIConnectionError, APIStatusError
from PIL import Image

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.models.client import OpenAIClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask, ImageBatch, ImageObject
from nemo_curator.utils.hash_utils import get_deterministic_hash

_FIELD_SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "visual_quality": (
        '"visual_quality": An integer from 1 to 5 for technical visual usability, not beauty or '
        "subject preference. 5: clear and coherent, with no material defect. 4: minor defects, main "
        "content easy to understand. 3: noticeable defects, but main content identifiable. 2: severe "
        "defects substantially hide the main content. 1: unusable, corrupt, black, or nearly "
        "obscured. Do not lower the score solely because the image is synthetic or depicts an "
        "unusual subject.",
        {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    ),
    "clarity": (
        '"clarity": One of [sharp, mild_blur, severe_blur]. Choose sharp when the main content is '
        "clear; mild_blur when it is visibly softened but identifiable; severe_blur when blur "
        "substantially prevents understanding it. Deliberate background depth of field alone is not "
        "a blur defect. Do not label compression blocks as blur.",
        {"type": "string", "enum": ["sharp", "mild_blur", "severe_blur"]},
    ),
    "exposure": (
        '"exposure": One of [normal, low_light, overexposed]. Choose low_light when lost shadow '
        "detail hides important content; overexposed when clipped bright regions hide important "
        "content; otherwise normal. A night scene or bright lighting alone is insufficient. If both "
        "problems occur, choose the one that obscures more of the main content.",
        {"type": "string", "enum": ["normal", "low_light", "overexposed"]},
    ),
    "color_cast": (
        '"color_cast": True only when a conspicuous overall tint makes colors across much of the '
        "image visibly distorted. Do not flag a locally colored light, a naturally colored scene, or "
        "a coherent stylized palette solely for its color.",
        {"type": "boolean"},
    ),
    "rendered_text": (
        '"rendered_text": List at most 12 distinct, clearly legible text strings visible in the '
        "scene or overlaid on it, in approximate reading order. Include repeated text, including "
        "watermarks, only once; do not repeat or slightly rephrase a string to fill the list. For "
        "text-dense images, choose the most prominent headings, labels, and short phrases instead "
        "of transcribing every line or paragraph. Copy selected characters as seen; do not complete "
        "obscured words or guess uncertain characters. Use [] when no text is confidently legible.",
        {"type": "array", "items": {"type": "string"}},
    ),
    "text_coverage": (
        '"text_coverage": One of [none, small, moderate, large]. Count visible text whether or not '
        "it is legible: none = no visible text; small = isolated, visually minor text; moderate = "
        "text occupies a noticeable part but the visual scene remains primary; large = text "
        "dominates the image. This is a visual estimate, not OCR box-area measurement.",
        {"type": "string", "enum": ["none", "small", "moderate", "large"]},
    ),
    "watermark": (
        '"watermark": True for a creator, publisher, or ownership mark composited over the image, '
        "including a translucent mark. A brand physically printed on a depicted object or sign is "
        "not a watermark.",
        {"type": "boolean"},
    ),
    "logo": (
        '"logo": One of [none, in_scene, overlay, both]. In_scene = a logo physically present on a '
        "depicted object or sign; overlay = a logo composited over the image; both = both kinds are "
        "visible. An overlaid ownership logo may also make watermark true.",
        {"type": "string", "enum": ["none", "in_scene", "overlay", "both"]},
    ),
    "visible_artifacts": (
        '"visible_artifacts": List only clearly visible defects from [noise, compression, '
        "corruption]: noise = disruptive random speckling; compression = blocking or ringing; "
        "corruption = broken or missing image regions. Do not count ordinary scene texture, blur, or "
        "exposure here. Use [] if none is clearly visible.",
        {"type": "array", "items": {"type": "string", "enum": ["noise", "compression", "corruption"]}},
    ),
    "scene_type": (
        '"scene_type": One of [real_world, rendered, game, animation, mixed]. Real_world = '
        "camera-captured scene; rendered = visibly computer-generated 3D scene; game = clear "
        "game-interface or gameplay evidence; animation = drawn or animated visual style; mixed = "
        "clearly combined kinds. Do not infer game solely from a 3D-rendered appearance.",
        {"type": "string", "enum": ["real_world", "rendered", "game", "animation", "mixed"]},
    ),
}


def annotation_prompt(fields: list[str]) -> str:
    sections = [
        "You are annotating one image for T2I data curation. Judge only what is visible in the "
        "image. Do not infer its source, creator intent, or unseen content. Evaluate each requested "
        "field independently. Return one JSON object containing exactly the requested keys and "
        "allowed values:"
    ]
    sections.extend(_FIELD_SPECS[name][0] for name in fields)
    sections.append("Use only visible evidence. Return only the JSON object, without markdown or extra fields.")
    return "\n\n".join(sections)


def annotation_schema(fields: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": fields,
        "properties": {name: _FIELD_SPECS[name][1] for name in fields},
    }


def parse_annotation(response: str, fields: list[str]) -> dict[str, Any]:
    result = json.loads(response)
    if not isinstance(result, dict) or set(result) != set(fields):
        msg = f"VLM response fields differ from requested fields: {fields}"
        raise ValueError(msg)
    for name, value in result.items():
        schema = _FIELD_SPECS[name][1]
        kind = schema["type"]
        if kind == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "boolean":
            valid = isinstance(value, bool)
        elif kind == "string":
            valid = isinstance(value, str)
        else:
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
            if valid and "enum" in schema["items"]:
                valid = all(item in schema["items"]["enum"] for item in value)
        if valid and "enum" in schema:
            valid = value in schema["enum"]
        if not valid:
            msg = f"Invalid VLM annotation {name}={value!r}"
            raise ValueError(msg)
    return result


@dataclass
class ImageVLMAnnotationStage(ProcessingStage[ImageBatch, DocumentBatch]):
    """Annotate decoded images through an external OpenAI-compatible VLM."""

    fields: list[str]
    base_url: str = "http://127.0.0.1:8000/v1"
    base_urls: tuple[str, ...] | None = None
    model: str | None = None
    api_key: str = "EMPTY"
    timeout: float = 120.0
    max_output_tokens: int = 512
    requests_per_worker: int = 1
    max_image_edge: int = 1024
    jpeg_quality: int = 90
    output_dir: str | None = None
    source_name: str | None = None
    retry_statuses: tuple[str, ...] = ()
    name: str = "image_vlm_annotation"
    prompt: str = field(init=False)
    prompt_sha256: str = field(init=False)
    response_format: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        if not self.fields or len(set(self.fields)) != len(self.fields):
            msg = "fields must be a nonempty list without duplicates"
            raise ValueError(msg)
        unknown = set(self.fields) - _FIELD_SPECS.keys()
        if unknown:
            msg = f"Unknown VLM annotation fields: {sorted(unknown)}"
            raise ValueError(msg)
        self.prompt = annotation_prompt(self.fields)
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        self.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "image_annotation", "strict": True, "schema": annotation_schema(self.fields)},
        }

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "status", *self.fields]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        urls = self.base_urls or (self.base_url,)
        self.clients = [OpenAIClient(base_url=url, api_key=self.api_key, timeout=self.timeout) for url in urls]
        for client in self.clients:
            client.setup()
        self.client = self.clients[0]
        if self.model is None:
            models = self.client.client.models.list().data
            if len(models) != 1:
                msg = f"Expected one served model, found {len(models)}; set model explicitly"
                raise ValueError(msg)
            self.model = models[0].id

    def _image_url(self, image_data: np.ndarray) -> str:
        image = Image.fromarray(image_data)
        image.thumbnail((self.max_image_edge, self.max_image_edge))
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=self.jpeg_quality)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _client_for(self, image_id: str) -> OpenAIClient:
        if self.base_urls:
            return self.clients[int(get_deterministic_hash([image_id]), 16) % len(self.clients)]
        return self.client

    def _annotate(self, image: ImageObject) -> dict[str, Any]:
        row = {
            "image_id": image.image_id,
            "image_path": image.image_path,
            "vlm_model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "raw_response": None,
            "error": None,
            **dict.fromkeys(self.fields),
        }
        content = [
            {"type": "text", "text": self.prompt},
            {"type": "image_url", "image_url": {"url": self._image_url(image.image_data)}},
        ]
        try:
            completion = self._client_for(image.image_id).query_model_response(
                messages=[{"role": "user", "content": content}],
                model=self.model,
                generation_config=GenerationConfig(
                    max_tokens=self.max_output_tokens,
                    temperature=0,
                    seed=0,
                    extra_kwargs={"response_format": self.response_format},
                ),
            )
        except APIConnectionError as exc:
            row.update(status="request_error", error=str(exc))
        except APIStatusError as exc:
            if exc.status_code not in (408, 409, 429) and exc.status_code < HTTPStatus.INTERNAL_SERVER_ERROR:
                raise
            row.update(status="request_error", error=str(exc))
        else:
            choice = completion.choices[0]
            response = choice.message.content
            row["raw_response"] = response
            if choice.finish_reason == "length":
                row.update(status="length", error="Output reached max_output_tokens")
            else:
                try:
                    row.update(parse_annotation(response, self.fields))
                    row["status"] = "ok"
                except (ValueError, TypeError) as exc:
                    row.update(status="invalid_response", error=str(exc))
        if row["status"] != "ok":
            logger.warning("VLM annotation failed for {}: {}", image.image_id, row["status"])
        return row

    def process(self, task: ImageBatch) -> DocumentBatch | None:  # noqa: C901
        if not task.data:
            return None
        annotation_path = None
        previous = None
        if self.output_dir is not None:
            filename = get_deterministic_hash([image.image_id for image in task.data])
            annotation_path = Path(self.output_dir) / f"source={self.source_name}" / f"part-{filename}.parquet"
            if annotation_path.exists():
                if not self.retry_statuses:
                    return None
                previous = pd.read_parquet(annotation_path).set_index("image_id").to_dict("index")
            elif self.retry_statuses:
                return None
        rows = [None] * len(task.data)
        pending = []
        for index, image in enumerate(task.data):
            old = previous.get(image.image_id) if previous is not None else None
            if previous is not None and (old is None or old["status"] not in self.retry_statuses):
                rows[index] = {"image_id": image.image_id, **old}
                continue
            pending.append((index, image))
        if not pending:
            return None
        if self.requests_per_worker == 1:
            annotated = (self._annotate(image) for _, image in pending)
        else:
            with ThreadPoolExecutor(max_workers=self.requests_per_worker) as pool:
                annotated = list(pool.map(self._annotate, (image for _, image in pending)))
        for (index, _), row in zip(pending, annotated, strict=True):
            rows[index] = row
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pd.DataFrame(rows),
            _metadata=(
                {**task._metadata, "annotation_path": str(annotation_path)} if annotation_path else task._metadata
            ),
            _stage_perf=task._stage_perf,
        )


@dataclass
class VLMAnnotationWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Commit a complete annotation batch as one Parquet file."""

    name: str = "vlm_annotation_writer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "status"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> FileGroupTask:
        path = Path(task._metadata["annotation_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            frame = task.to_pandas()
            types = {
                "image_id": pa.string(),
                "status": pa.string(),
                "image_path": pa.string(),
                "vlm_model": pa.string(),
                "prompt_sha256": pa.string(),
                "raw_response": pa.string(),
                "error": pa.string(),
            }
            for name in frame.columns:
                if name in _FIELD_SPECS:
                    kind = _FIELD_SPECS[name][1]["type"]
                    if kind == "integer":
                        frame[name] = frame[name].astype("Int64")
                    elif kind == "boolean":
                        frame[name] = frame[name].astype("boolean")
                    types[name] = {
                        "integer": pa.int64(),
                        "boolean": pa.bool_(),
                        "string": pa.string(),
                        "array": pa.list_(pa.string()),
                    }[kind]
            schema = pa.schema([(name, types[name]) for name in frame.columns])
            pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[str(path)],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
