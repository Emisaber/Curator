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
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


def _mock_pipeline(module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_caption: bool) -> list[str]:
    events = []

    client = Mock()
    client.start.side_effect = lambda: events.append("start")
    client.stop.side_effect = lambda: events.append("stop")

    class Pipeline:
        def __init__(self, name: str = "embedding") -> None:
            self.name = name

        def add_stage(self, stage: object) -> None:
            pass

        def run(self, **_kwargs: object) -> None:
            events.append(self.name)
            if self.name == "embedding":
                (tmp_path / "output/dedup/clip/v1/cache/features/long").mkdir(parents=True)
            elif fail_caption:
                msg = "caption lookup failed"
                raise RuntimeError(msg)

    class SemanticWorkflow:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["eps"] is None

        def run(self) -> None:
            events.append("semantic")

    monkeypatch.setattr(module, "RayClient", Mock(return_value=client))
    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module, "SemanticDeduplicationWorkflow", SemanticWorkflow)
    monkeypatch.setattr(module, "create_embedding_pipeline", lambda *_args: Pipeline())
    return events


@pytest.mark.parametrize("fail_caption", [False, True])
def test_dedup_completion_is_committed_after_caption_checks_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_caption: bool
) -> None:
    folder = Path(__file__).resolve().parents[4] / "tutorials/image/caption-dedup"
    report_spec = importlib.util.spec_from_file_location("report", folder / "report.py")
    report_module = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report_module)
    monkeypatch.setitem(sys.modules, "report", report_module)
    spec = importlib.util.spec_from_file_location("caption_dedup_run", folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"long": {"root": str(tmp_path / "media")}}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"sources": str(sources), "output": str(tmp_path / "output"), "model_dir": "/model", "skip_report": True}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config)])
    args = module.parse_args()
    events = _mock_pipeline(module, tmp_path, monkeypatch, fail_caption)
    completed = tmp_path / "output/dedup/clip/v1/completed.json"
    if fail_caption:
        with pytest.raises(RuntimeError, match="caption lookup failed"):
            module.run(args)
        assert not completed.exists()
        before = (tmp_path / "output/run.json").read_bytes()
        with pytest.raises(RuntimeError, match="incomplete"):
            module.run(args)
        assert (tmp_path / "output/run.json").read_bytes() == before
    else:
        module.run(args)
        assert events == ["start", "embedding", "semantic", "caption_aware_image_deduplication", "stop"]
        assert json.loads(completed.read_text()) == json.loads((tmp_path / "output/run.json").read_text())
        events.clear()
        module.run(args)
        assert events == []
        args.batch_size = 64
        with pytest.raises(ValueError, match="different settings"):
            module.run(args)
