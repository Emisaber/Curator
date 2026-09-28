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

"""Run CLIP-caption checks, then annotate their decoded source samples."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _write_config(output: Path, name: str, config: dict) -> Path:
    path = output / "configs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)  # noqa: S603 -- arguments target repository entry points, without a shell


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"]).resolve()
    schema_version = str(config.get("schema_version", 1))
    if schema_version != "1":
        msg = "The current image contract is schema-v1"
        raise ValueError(msg)
    dedup = {**config["dedup"], "output": str(output), "seed": config.get("seed", 42), "skip_report": True}
    if "subset" in config:
        dedup["subset"] = config["subset"]
    else:
        dedup["sources"] = str(_write_config(output, "sources", config["sources"]))
    dedup_config = _write_config(output, "dedup", dedup)
    _run([sys.executable, str(Path(__file__).parents[1] / "caption-dedup/run.py"), "--config", str(dedup_config)])

    if "vlm" in config:
        vlm = {
            **config["vlm"],
            "input_run": str(output),
            "output": str(output),
            "schema_version": schema_version,
            "batch_size": config["vlm"].get("batch_size", 32),
        }
        vlm_config = _write_config(output, "vlm", vlm)
        _run([sys.executable, str(Path(__file__).parents[1] / "vlm-annotation/run.py"), "--config", str(vlm_config)])

    for stage in ("nsfw", "hpsv3"):
        if stage not in config:
            continue
        settings = {
            **config[stage],
            "input_run": str(output),
            "output": str(output),
            "schema_version": schema_version,
        }
        stage_config = _write_config(output, stage, settings)
        _run(
            [
                sys.executable,
                str(Path(__file__).parents[1] / f"{stage}-annotation/run.py"),
                "--config",
                str(stage_config),
            ]
        )

    if ("nsfw" in config or "hpsv3" in config) and not config.get("skip_report", False):
        _run(
            [
                sys.executable,
                str(Path(__file__).with_name("report.py")),
                "--input-run",
                str(output),
                "--output",
                str(output / "report-scores"),
                "--per-group",
                str(config.get("review_per_group", 8)),
                "--seed",
                str(config.get("seed", 42)),
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run(parser.parse_args().config)
