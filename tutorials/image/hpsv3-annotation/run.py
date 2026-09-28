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

"""Score source images and original captions with independent HPSv3 model processes."""

import argparse
import json
import uuid
from pathlib import Path

import ray

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.image.annotation.file_utils import load_input_run, pending_files, write_run_info
from nemo_curator.stages.image.annotation.hpsv3_annotation import HPSv3AnnotationWriter, HPSv3ScoreStage
from nemo_curator.stages.image.annotation.hpsv3_service import HPSv3Service
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.text.io.reader.parquet import ParquetReader


def run(config_path: Path) -> None:  # noqa: C901, PLR0912, PLR0915
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"]).resolve()
    schema_version = str(config.get("schema_version", "1"))
    if schema_version != "1":
        msg = "The current image contract is schema-v1"
        raise ValueError(msg)
    annotations = output / "annotations/hpsv3/schema-v1"
    input_run = Path(config["input_run"]).resolve() if "input_run" in config else None
    input_settings = load_input_run(input_run) if input_run is not None else None
    if input_settings is not None:
        sources = input_settings["sources"]
        inputs = {name: input_run / "annotations/decode/schema-v1" / f"source={name}" for name in sources}
        decode_info = json.loads((input_run / "annotations/decode/schema-v1/run.json").read_text(encoding="utf-8"))
        adjust_orientation = decode_info["tar_exif_orientation"] != "stored_pixels"
    else:
        sources = {config["source"]["name"]: {"root": config["source"]["root"]}}
        inputs = {config["source"]["name"]: Path(config["manifest"]).resolve()}
        adjust_orientation = True
    run_info = {
        "sources": sources,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "schema_version": schema_version,
        "caption_version": "raw",
        "adjust_orientation": adjust_orientation,
        "python": config["hps_python"],
        "checkpoint": config["checkpoint"],
        "model_config": config.get("model_config"),
        "replicas": config.get("replicas", 1),
        "inference_batch_size": config.get("inference_batch_size", 16),
        "max_model_batch_size": config.get("max_model_batch_size", 256),
        "batch_wait_ms": config.get("batch_wait_ms", 50),
    }
    if input_run is not None:
        run_info.update(input_run=str(input_run), input_settings=input_settings)
    write_run_info(annotations, run_info)
    retry = config.get("retry")
    retry_statuses = tuple(retry["statuses"]) if retry else ()
    if retry is not None:
        if retry["pass"] < 1 or not retry_statuses or set(retry_statuses) - {"failed", "request_error"}:
            msg = "HPSv3 retry requires a positive pass and failure statuses"
            raise ValueError(msg)
        retry_file = annotations / f"retry-{retry['pass']}.json"
        if retry_file.exists() and json.loads(retry_file.read_text(encoding="utf-8")) != retry:
            msg = f"Existing retry pass uses different settings: {retry_file}"
            raise ValueError(msg)
        if not retry_file.exists():
            retry_file.write_text(json.dumps(retry, indent=2), encoding="utf-8")
    files = {
        name: pending_files(sorted(path.glob("*.parquet")), annotations, name, retry_statuses)
        for name, path in inputs.items()
    }
    if not any(files.values()):
        return

    client = RayClient(num_cpus=config.get("num_cpus", 4), num_gpus=config.get("num_gpus", 1))
    client.start()
    service = HPSv3Service(
        python=config["hps_python"],
        checkpoint=config["checkpoint"],
        model_config=config.get("model_config"),
        replicas=config.get("replicas", 1),
        namespace="nemo_curator_hpsv3",
        actor_prefix=f"hpsv3_{uuid.uuid4().hex[:12]}",
        log_dir=str(output / "logs/hpsv3/schema-v1"),
        start_timeout=config.get("start_timeout", 600.0),
        drain_timeout=config.get("drain_timeout", 120.0),
        max_batch_size=config.get("max_model_batch_size", 256),
        batch_wait_ms=config.get("batch_wait_ms", 50),
        read_parquet=any(settings.get("format") == "vlv_parquet" for settings in sources.values()),
    )
    completed = False
    try:
        endpoints = service.start()
        for name, paths in files.items():
            if not paths:
                continue
            pipeline = Pipeline(name=f"hpsv3_image_caption_annotation_{name}")
            fields = None if input_run is not None else ["image_id", "shard", "offset", "size", "caption_raw"]
            pipeline.add_stage(ParquetReader(file_paths=paths, files_per_partition=1, fields=fields))
            pipeline.add_stage(
                HPSv3ScoreStage(
                    source_name=name,
                    source_root=str(Path(sources[name]["root"]).resolve()),
                    source=CaptionSource(**sources[name]) if input_run is not None else None,
                    output_dir=str(annotations),
                    endpoints=endpoints,
                    inference_batch_size=config.get("inference_batch_size", 16),
                    request_timeout=config.get("request_timeout", 300.0),
                    adjust_orientation=adjust_orientation,
                    retry_statuses=retry_statuses,
                ).with_(num_workers=config.get("annotation_workers"))
            )
            pipeline.add_stage(HPSv3AnnotationWriter())
            checkpoint = output / "checkpoints/hpsv3/schema-v1" / f"source={name}"
            if retry is not None:
                checkpoint = checkpoint / f"retry-{retry['pass']}"
            pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint)
        completed = True
    finally:
        try:
            service.stop(raise_on_drain_failure=completed)
        finally:
            try:
                ray.shutdown()
            finally:
                client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run(parser.parse_args().config)
