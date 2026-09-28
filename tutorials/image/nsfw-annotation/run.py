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

"""Score persisted CLIP features for every source in a completed curation run."""

import argparse
import json
from pathlib import Path

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.image.annotation import ImageNSFWAnnotationStage, NSFWAnnotationWriter
from nemo_curator.stages.image.annotation.file_utils import load_input_run, pending_files, write_run_info
from nemo_curator.stages.text.io.reader.parquet import ParquetReader


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"]).resolve()
    schema_version = str(config.get("schema_version", "1"))
    if schema_version != "1":
        msg = "The current image contract is schema-v1"
        raise ValueError(msg)
    annotations = output / "annotations/nsfw/schema-v1"
    input_run = Path(config["input_run"]).resolve() if "input_run" in config else None
    input_settings = load_input_run(input_run) if input_run is not None else None
    if input_settings is not None:
        sources = input_settings["sources"]
        features = {
            name: (
                input_run / "features/clip/schema-v1" / f"source={name}"
                if input_settings.get("subset")
                else input_run / "dedup/clip/v1/cache/features" / name
            )
            for name in sources
        }
    else:
        sources = {config["source"]["name"]: config["source"]}
        features = {config["source"]["name"]: Path(config["features"]).resolve()}
    run_info = {
        "sources": sources,
        "features": {name: str(path) for name, path in features.items()},
        "schema_version": schema_version,
        "embedding_model": "openai/clip-vit-large-patch14",
        "embedding_model_revision": "32bd642",
        "model": "laion/clip-autokeras-binary-nsfw",
        "model_dir": config["model_dir"],
        "model_inference_batch_size": config.get("model_inference_batch_size", 32),
    }
    if input_run is not None:
        run_info.update(input_run=str(input_run), input_settings=input_settings)
    write_run_info(annotations, run_info)
    files = {name: pending_files(sorted(path.glob("*.parquet")), annotations, name) for name, path in features.items()}
    if not any(files.values()):
        return

    client = RayClient(num_cpus=config.get("num_cpus", 4), num_gpus=config.get("num_gpus", 1))
    client.start()
    try:
        for name, paths in files.items():
            if not paths:
                continue
            pipeline = Pipeline(name=f"image_nsfw_annotation_{name}")
            pipeline.add_stage(
                ParquetReader(file_paths=paths, files_per_partition=1, fields=["image_id", "embedding"])
            )
            pipeline.add_stage(
                ImageNSFWAnnotationStage(
                    model_dir=config["model_dir"],
                    output_dir=str(annotations),
                    source_name=name,
                    model_inference_batch_size=config.get("model_inference_batch_size", 32),
                    num_gpus_per_worker=config.get("num_gpus_per_worker", 0.25),
                ).with_(num_workers=config.get("annotation_workers"))
            )
            pipeline.add_stage(NSFWAnnotationWriter())
            pipeline.run(checkpoint_path=output / "checkpoints/nsfw/schema-v1" / f"source={name}")
    finally:
        client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run(parser.parse_args().config)
