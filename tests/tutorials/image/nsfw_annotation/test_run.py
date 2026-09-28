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

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.annotation.file_utils import annotation_path


@pytest.mark.parametrize("subset", [True, False])
def test_nsfw_reads_current_feature_layout_and_skips_completed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subset: bool
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/nsfw-annotation/run.py"
    spec = importlib.util.spec_from_file_location("nsfw_annotation_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = {
        "sources": {"long": {"root": "/media/long"}, "vlv": {"root": "/media/vlv", "format": "vlv_parquet"}},
        "subset": "/subset" if subset else None,
    }
    (tmp_path / "run.json").write_text(json.dumps(settings))
    version = "schema-v1" if subset else "v1"
    completed = tmp_path / f"dedup/clip/{version}/completed.json"
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps(settings))
    inputs = {}
    for name in settings["sources"]:
        directory = (
            tmp_path / "features/clip/schema-v1" / f"source={name}"
            if subset
            else tmp_path / "dedup/clip/v1/cache/features" / name
        )
        directory.mkdir(parents=True)
        path = directory / "part-test.parquet"
        pq.write_table(pa.table({"image_id": [name], "embedding": [[0.0] * 768]}), path)
        inputs[name] = path
    result = annotation_path(tmp_path / "annotations/nsfw/schema-v1", "long", str(inputs["long"]))
    result.parent.mkdir(parents=True)
    pq.write_table(pa.table({"status": ["ok"]}), result)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {"input_run": str(tmp_path), "output": str(tmp_path), "model_dir": "/models", "annotation_workers": 4}
        )
    )
    pipelines = []

    class Pipeline:
        def __init__(self, name: str) -> None:
            self.stages = []
            pipelines.append(self)

        def add_stage(self, stage: ProcessingStage) -> None:
            self.stages.append(stage)

        def run(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module, "RayClient", Mock())
    module.run(config_file)
    assert len(pipelines) == 1
    assert pipelines[0].stages[0].file_paths == [str(inputs["vlv"])]
    assert pipelines[0].stages[1].source_name == "vlv"
    assert pipelines[0].stages[1].num_workers() == 4
    assert pipelines[0].kwargs["checkpoint_path"] == tmp_path / "checkpoints/nsfw/schema-v1/source=vlv"
    metadata = json.loads((tmp_path / "annotations/nsfw/schema-v1/run.json").read_text())
    assert metadata["sources"] == settings["sources"]
