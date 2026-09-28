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
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("subset", [True, False])
def test_workflow_runs_dedup_then_one_vlm_entry_and_resumes_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subset: bool
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/t2i-curation/run.py"
    spec = importlib.util.spec_from_file_location("t2i_curation_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "output"
    config = {"output": str(output), "dedup": {"model_dir": "/models/clip"}, "vlm": {"fields": ["clarity"]}}
    if subset:
        config["subset"] = "/frozen/subset"
    else:
        config["sources"] = {"vlv": {"root": "/media/vlv", "format": "vlv_parquet"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    commands = []
    fail_vlm = True

    def run(command: list[str]) -> None:
        commands.append(command)
        settings = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        if "caption-dedup" in command[1]:
            assert settings["skip_report"] is True
            if subset:
                assert settings["subset"] == config["subset"]
            else:
                assert json.loads(Path(settings["sources"]).read_text()) == config["sources"]
        else:
            assert settings["input_run"] == str(output)
            assert settings["batch_size"] == 32
            assert "manifest" not in settings
            assert "source" not in settings
            if fail_vlm:
                raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module, "_run", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.run(path)
    assert len(commands) == 2
    assert "caption-dedup" in commands[0][1]
    assert "vlm-annotation" in commands[1][1]
    fail_vlm = False
    module.run(path)
    assert len(commands) == 4


def test_dedup_failure_prevents_vlm_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/t2i-curation/run.py"
    spec = importlib.util.spec_from_file_location("t2i_failed_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"output": str(tmp_path / "output"), "subset": "/subset", "dedup": {}, "vlm": {}}),
        encoding="utf-8",
    )
    commands = []

    def run(command: list[str]) -> None:
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module, "_run", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.run(path)
    assert len(commands) == 1
    assert "caption-dedup" in commands[0][1]


def test_scores_run_once_after_vlm_without_legacy_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/t2i-curation/run.py"
    spec = importlib.util.spec_from_file_location("t2i_scores_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "output"
    config = {
        "output": str(output),
        "subset": "/subset",
        "dedup": {},
        "vlm": {"fields": ["clarity"]},
        "nsfw": {"annotation_workers": 4},
        "hpsv3": {"replicas": 2, "annotation_workers": 16},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    commands = []

    def run(command: list[str]) -> None:
        commands.append(command)
        if command[2] == "--config" and "caption-dedup" not in command[1]:
            settings = json.loads(Path(command[-1]).read_text())
            assert settings["input_run"] == str(output)
            assert "manifest" not in settings
            assert "source" not in settings

    monkeypatch.setattr(module, "_run", run)
    module.run(path)
    assert [Path(command[1]).parent.name for command in commands[:4]] == [
        "caption-dedup",
        "vlm-annotation",
        "nsfw-annotation",
        "hpsv3-annotation",
    ]
    assert Path(commands[4][1]).name == "report.py"
    assert not (output / "samples").exists()
