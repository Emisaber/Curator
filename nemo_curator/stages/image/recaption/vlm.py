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


"""Caption decoded Ego4D frames using the existing VLM execution machinery."""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from openai import APIConnectionError, APIStatusError

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.annotation.vlm import ImageVLMAnnotationStage
from nemo_curator.stages.image.recaption.prompts import FHO_PROMPT, NARRATION_PROMPT
from nemo_curator.tasks import DocumentBatch, FileGroupTask, ImageObject

_ELEMENT_FIELDS = ("name", "appearance", "location", "state", "relationships")
_ELEMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_ELEMENT_FIELDS),
    "properties": {
        **{name: {"type": "string"} for name in _ELEMENT_FIELDS[:-1]},
        "relationships": {"type": "array", "items": {"type": "string"}},
    },
}
CAPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["comprehensive_description", "prominent_elements"],
    "properties": {
        "comprehensive_description": {"type": "string"},
        "prominent_elements": {"type": "array", "items": _ELEMENT_SCHEMA},
    },
}
CAPTION_PARQUET_SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("caption_version", pa.string()),
        ("status", pa.string()),
        ("image_path", pa.string()),
        ("vlm_model", pa.string()),
        ("prompt_sha256", pa.string()),
        ("comprehensive_description", pa.string()),
        (
            "prominent_elements",
            pa.list_(
                pa.struct(
                    [(name, pa.string()) for name in _ELEMENT_FIELDS[:-1]] + [("relationships", pa.list_(pa.string()))]
                )
            ),
        ),
        ("raw_response", pa.string()),
        ("error_kind", pa.string()),
        ("error", pa.string()),
    ]
)


def parse_caption(response: str) -> dict[str, Any]:
    result = json.loads(response)
    if not isinstance(result, dict) or set(result) != set(CAPTION_SCHEMA["required"]):
        msg = "Caption response must contain comprehensive_description and prominent_elements"
        raise ValueError(msg)
    if not isinstance(result["comprehensive_description"], str) or not isinstance(result["prominent_elements"], list):
        msg = "Caption description must be a string and elements must be a list"
        raise TypeError(msg)
    for element in result["prominent_elements"]:
        if not isinstance(element, dict) or set(element) != set(_ELEMENT_FIELDS):
            msg = "Caption element fields do not match the requested schema"
            raise ValueError(msg)
        if not all(isinstance(element[name], str) for name in _ELEMENT_FIELDS[:-1]):
            msg = "Caption element attributes must be strings"
            raise TypeError(msg)
        relationships = element["relationships"]
        if not isinstance(relationships, list) or not all(isinstance(item, str) for item in relationships):
            msg = "Caption relationships must be a list of strings"
            raise TypeError(msg)
    return result


@dataclass
class Ego4DRecaptionStage(ImageVLMAnnotationStage):
    """Reuse client setup, image encoding, concurrency and batch recovery."""

    source_kind: str = "fho"
    caption_version: str = "v1"
    fields: list[str] = field(init=False, default_factory=lambda: list(CAPTION_SCHEMA["required"]))
    max_output_tokens: int = 2048
    name: str = "ego4d_recaption"

    def __post_init__(self) -> None:
        self.prompt = {"fho": FHO_PROMPT, "narration": NARRATION_PROMPT}[self.source_kind]
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        self.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "image_caption", "strict": True, "schema": CAPTION_SCHEMA},
        }

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "caption_version", "status", *self.fields]

    def _annotate(self, image: ImageObject) -> dict[str, Any]:
        row = {
            "image_id": image.image_id,
            "caption_version": self.caption_version,
            "status": "failed",
            "image_path": image.image_path,
            "vlm_model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "comprehensive_description": None,
            "prominent_elements": None,
            "raw_response": None,
            "error_kind": None,
            "error": None,
        }
        if image.image_data is None:
            row.update(error_kind="decode_error", error=image.metadata["decode_error"])
            return row
        content = [
            {"type": "image_url", "image_url": {"url": self._image_url(image.image_data)}},
            {"type": "text", "text": "Ego4D annotation context:\n" + image.metadata["context_json"]},
        ]
        try:
            completion = self._client_for(image.image_id).query_model_response(
                messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": content}],
                model=self.model,
                generation_config=GenerationConfig(
                    max_tokens=self.max_output_tokens,
                    temperature=0,
                    seed=0,
                    extra_kwargs={"response_format": self.response_format},
                ),
            )
        except APIConnectionError as exc:
            row.update(error_kind="request_error", error=str(exc))
        except APIStatusError as exc:
            if exc.status_code not in (408, 409, 429) and exc.status_code < HTTPStatus.INTERNAL_SERVER_ERROR:
                raise
            row.update(error_kind="request_error", error=str(exc))
        else:
            choice = completion.choices[0]
            row["raw_response"] = choice.message.content
            if choice.finish_reason == "length":
                row.update(error_kind="length", error="Output reached max_output_tokens")
            else:
                try:
                    row.update(parse_caption(choice.message.content))
                    row["status"] = "ok"
                except (ValueError, TypeError) as exc:
                    row.update(error_kind="invalid_response", error=str(exc))
        if row["status"] != "ok":
            logger.warning("Recaption failed for {}: {}", image.image_id, row["error_kind"])
        return row


@dataclass
class RecaptionWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    name: str = "recaption_writer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_id", "status", "caption_version"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> FileGroupTask:
        path = Path(task._metadata["annotation_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            table = pa.Table.from_pandas(task.to_pandas(), schema=CAPTION_PARQUET_SCHEMA, preserve_index=False)
            pq.write_table(table, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[str(path)],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
