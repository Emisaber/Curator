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

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.image.annotation.file_utils import (
    annotation_path,
    load_input_run,
    pending_files,
    write_run_info,
)


def test_recovery_skips_written_files_and_retry_selects_only_failed_partitions(tmp_path: Path) -> None:
    files = [tmp_path / f"input-{index}.parquet" for index in range(3)]
    output = tmp_path / "annotations"
    for index, status in enumerate(("ok", "request_error")):
        path = annotation_path(output, "vlv", str(files[index]))
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"status": [status]}), path)
    assert pending_files(files, output, "vlv") == [str(files[2])]
    assert pending_files(files, output, "vlv", ("request_error",)) == [str(files[1])]
    assert pending_files(files, output, "short") == [str(path) for path in files]


def test_settings_must_match_completed_input_and_existing_annotations(tmp_path: Path) -> None:
    settings = {"sources": {"long": {"root": "/media"}}, "subset": "/subset"}
    (tmp_path / "run.json").write_text(json.dumps(settings))
    completed = tmp_path / "dedup/clip/schema-v1/completed.json"
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps(settings))
    assert load_input_run(tmp_path) == settings
    write_run_info(tmp_path / "annotations", settings)
    write_run_info(tmp_path / "annotations", settings)
    with pytest.raises(ValueError, match="different settings"):
        write_run_info(tmp_path / "annotations", {**settings, "subset": "/other"})
    completed.write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        load_input_run(tmp_path)
