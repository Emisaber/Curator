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

"""Input identities and persisted partitions for image scoring stages."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def annotation_path(output: Path, source_name: str, input_file: str) -> Path:
    digest = hashlib.sha256(input_file.encode("utf-8")).hexdigest()[:16]
    return output / f"source={source_name}" / f"part-{digest}.parquet"


def pending_files(
    files: list[Path], output: Path, source_name: str, retry_statuses: tuple[str, ...] = ()
) -> list[str]:
    selected = []
    for path in files:
        result = annotation_path(output, source_name, str(path))
        if retry_statuses:
            if result.exists() and set(
                pq.ParquetFile(result).read(columns=["status"])["status"].to_pylist()
            ).intersection(retry_statuses):
                selected.append(str(path))
        elif not result.exists():
            selected.append(str(path))
    return selected


def load_input_run(input_run: Path) -> dict:
    settings = json.loads((input_run / "run.json").read_text(encoding="utf-8"))
    version = "schema-v1" if settings.get("subset") else "v1"
    completed = input_run / "dedup/clip" / version / "completed.json"
    if json.loads(completed.read_text(encoding="utf-8")) != settings:
        msg = f"Deduplication completion does not match its input settings: {completed}"
        raise ValueError(msg)
    return settings


def write_run_info(directory: Path, settings: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != settings:
            msg = f"Existing annotations use different settings: {path}"
            raise ValueError(msg)
    else:
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
