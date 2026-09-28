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


def test_all_sources_share_one_independent_service_and_separate_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/hpsv3-annotation/run.py"
    spec = importlib.util.spec_from_file_location("hpsv3_annotation_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = {
        "subset": "/subset",
        "sources": {
            "long": {"root": "/media/long", "format": "webdataset_txt"},
            "vlv": {"root": "/media/vlv", "format": "vlv_parquet"},
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(settings))
    completed = tmp_path / "dedup/clip/schema-v1/completed.json"
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps(settings))
    decode = tmp_path / "annotations/decode/schema-v1"
    decode.mkdir(parents=True)
    (decode / "run.json").write_text(json.dumps({"tar_exif_orientation": "stored_pixels"}))
    for name in settings["sources"]:
        directory = decode / f"source={name}"
        directory.mkdir()
        pq.write_table(pa.table({"image_id": [name], "status": ["ok"]}), directory / "part-test.parquet")
    config = {
        "input_run": str(tmp_path),
        "output": str(tmp_path),
        "hps_python": "/separate/env/bin/python",
        "checkpoint": "/models/hps",
        "replicas": 2,
        "annotation_workers": 8,
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    pipelines = []

    class Pipeline:
        def __init__(self, name: str) -> None:
            self.stages = []
            pipelines.append(self)

        def add_stage(self, stage: ProcessingStage) -> None:
            self.stages.append(stage)

        def run(self, **kwargs) -> None:
            self.kwargs = kwargs

    service = Mock()
    service.start.return_value = ["http://host:1", "http://host:2"]
    factory = Mock(return_value=service)
    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module, "HPSv3Service", factory)
    monkeypatch.setattr(module, "RayClient", Mock())
    monkeypatch.setattr(module.ray, "shutdown", Mock())
    module.run(config_file)
    assert factory.call_args.kwargs["python"] == config["hps_python"]
    assert factory.call_args.kwargs["replicas"] == 2
    assert factory.call_args.kwargs["read_parquet"] is True
    service.start.assert_called_once()
    service.stop.assert_called_once_with(raise_on_drain_failure=True)
    assert len(pipelines) == 2
    for pipeline, name in zip(pipelines, settings["sources"], strict=True):
        stage = pipeline.stages[1]
        assert stage.source_name == name
        assert stage.source.format == settings["sources"][name]["format"]
        assert stage.adjust_orientation is False
        assert stage.num_workers() == 8
        assert pipeline.kwargs["checkpoint_path"] == tmp_path / "checkpoints/hpsv3/schema-v1" / f"source={name}"
    metadata = json.loads((tmp_path / "annotations/hpsv3/schema-v1/run.json").read_text())
    assert metadata["sources"] == settings["sources"]
