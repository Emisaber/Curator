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

"""Run a fixed-size Ego4D recaption experiment and review every result."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _tutorial(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"ego4d_recaption_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_experiment(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "sample_counts" not in config:
        msg = "Experiments require sample_counts to limit the input frames"
        raise ValueError(msg)
    _tutorial("run").run(config_path)
    report = _tutorial("report").create_report(
        Path(config["output"]).resolve(), config.get("caption_version", "v1"), per_group=None
    )
    print(f"Review report: {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run_experiment(parser.parse_args().config)
