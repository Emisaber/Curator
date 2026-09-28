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


"""Sample Ego4D frames and generate structured T2I captions."""

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from types import ModuleType

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.image.recaption.ego4d import PLAN_SCHEMA, read_video_metadata, sample_fho, sample_narration
from nemo_curator.stages.image.recaption.frames import (
    MANIFEST_SCHEMA,
    Ego4DFrameExtractStage,
    RecaptionImageReader,
    write_parquet,
)
from nemo_curator.stages.image.recaption.vlm import Ego4DRecaptionStage, RecaptionWriter
from nemo_curator.stages.text.io.reader.parquet import ParquetReader


def _save_run(path: Path, settings: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != settings:
            msg = f"Existing results use different settings: {path}"
            raise ValueError(msg)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def _serving_runtime() -> ModuleType:
    path = Path(__file__).parents[1] / "vlm-annotation/run.py"
    spec = importlib.util.spec_from_file_location("vlm_annotation_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_plans(  # noqa: PLR0913
    records: dict, candidates: set, metadata: dict, kind: str, directory: Path, config: dict
) -> list[str]:
    selection = directory / "selection.json"
    if selection.exists():
        return json.loads(selection.read_text(encoding="utf-8"))
    count = config["sample_counts"][kind]
    rng = random.Random(config.get("sampling_seed", 42))  # noqa: S311 - reproducible experiment inputs
    reservoir = []
    seen = 0
    source = f"ego4d-{kind}"
    for uid in sorted(candidates):
        rows = (
            sample_fho(records[uid], metadata[uid], source)
            if kind == "fho"
            else sample_narration(uid, records[uid], metadata[uid], source)
        )
        for row in rows:
            seen += 1
            if len(reservoir) < count:
                reservoir.append(row)
            else:
                slot = rng.randrange(seen)
                if slot < count:
                    reservoir[slot] = row
    if seen < count:
        msg = f"{kind} has {seen} eligible frames, fewer than the requested {count}"
        raise ValueError(msg)
    by_video = defaultdict(list)
    for row in reservoir:
        by_video[row["video_uid"]].append(row)
    paths = []
    for uid, rows in sorted(by_video.items()):
        path = directory / f"{uid}.parquet"
        write_parquet(sorted(rows, key=lambda row: row["frame_number"]), PLAN_SCHEMA, path)
        paths.append(str(path))
    selection.write_text(json.dumps(paths, indent=2), encoding="utf-8")
    return paths


def prepare(config: dict) -> dict[str, list[str]]:  # noqa: C901
    root = Path(config["data_root"]).resolve()
    output = Path(config["output"]).resolve()
    metadata_path = Path(config.get("metadata", root / "v2/ego4d.json"))
    fho_path = Path(config.get("fho_annotations", root / "v2/annotations/fho_main.json"))
    narration_path = Path(config.get("narration_annotations", root / "v2/annotations/narration.json"))
    sources = config.get("sources", ["fho", "narration"])
    experiment = {}
    if "sample_counts" in config:
        counts = config["sample_counts"]
        if set(counts) != set(sources) or any(not isinstance(count, int) or count <= 0 for count in counts.values()):
            msg = "sample_counts must specify a positive count for each selected source"
            raise ValueError(msg)
        experiment = {"sample_counts": counts, "sampling_seed": config.get("sampling_seed", 42)}
    _save_run(
        output / "samples/schema-v1/run.json",
        {
            "data_root": str(root),
            "metadata": str(metadata_path),
            "fho_annotations": str(fho_path),
            "narration_annotations": str(narration_path),
            "sources": sources,
            "video_uids": config.get("video_uids"),
            "batch_size": config.get("batch_size", 64),
            "jpeg_quality": config.get("jpeg_quality", 90),
            "sampling": {"fho_uniform_frames": 4, "narration_bin_seconds": 5},
            **experiment,
        },
    )
    metadata = read_video_metadata(metadata_path)
    with fho_path.open(encoding="utf-8") as stream:
        fho = {video["video_uid"]: video for video in json.load(stream)["videos"]}
    local = {path.stem for path in root.glob("*.mp4")}
    selected = local & metadata.keys()
    if "video_uids" in config:
        selected &= set(config["video_uids"])
    plans = {kind: [] for kind in sources}
    for kind in sources:
        source = f"ego4d-{kind}"
        directory = output / "samples/schema-v1/plans" / f"source={source}"
        if kind == "fho":
            candidates = selected & fho.keys()
            records = fho
        elif kind == "narration":
            with narration_path.open(encoding="utf-8") as stream:
                records = json.load(stream)
            candidates = (selected - fho.keys()) & records.keys()
        else:
            msg = f"Unknown Ego4D source: {kind}"
            raise ValueError(msg)
        if experiment:
            plans[kind] = _sample_plans(records, candidates, metadata, kind, directory, config)
            continue
        for uid in sorted(candidates):
            path = directory / f"{uid}.parquet"
            if path.exists():
                plans[kind].append(str(path))
                continue
            rows = (
                sample_fho(records[uid], metadata[uid], source)
                if kind == "fho"
                else sample_narration(uid, records[uid], metadata[uid], source)
            )
            if rows:
                write_parquet(rows, PLAN_SCHEMA, path)
                plans[kind].append(str(path))
    return plans


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output"]).resolve()
    plans = prepare(config)
    client = RayClient(num_cpus=config.get("num_cpus", 8), num_gpus=config.get("num_gpus", 0))
    client.start()
    server = None
    try:
        for kind, paths in plans.items():
            if not paths:
                continue
            source = f"ego4d-{kind}"
            pipeline = Pipeline(name=f"ego4d_extract_{kind}")
            pipeline.add_stage(
                ParquetReader(
                    file_paths=paths,
                    files_per_partition=1,
                    fields=[name for name in PLAN_SCHEMA.names if name != "source"],
                )
            )
            pipeline.add_stage(
                Ego4DFrameExtractStage(
                    source_root=config["data_root"],
                    source_name=source,
                    output=str(output),
                    batch_size=config.get("batch_size", 64),
                    jpeg_quality=config.get("jpeg_quality", 90),
                    retry_failed=config.get("retry_failed", False),
                ).with_(num_workers=config.get("extraction_workers", 4))
            )
            checkpoint = (
                None if config.get("retry_failed", False) else output / "checkpoints/recaption-extract" / source
            )
            pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint)

        if config.get("extract_only", False):
            return
        runtime = _serving_runtime()
        if "inference_server" in config:
            server = runtime._VLLMServerPool(config["inference_server"], output)
            server.start()
            model, urls = server.model_name, tuple(server.endpoints)
        else:
            urls = tuple(config.get("base_urls", [config.get("base_url", "http://127.0.0.1:8000/v1")]))
            model = runtime._served_model({**config, "base_url": urls[0]})
        version = config.get("caption_version", "v1")
        annotations = output / "annotations/recaption/schema-v1" / f"prompt-{version}"
        stages = {}
        for kind, paths in plans.items():
            if not paths:
                continue
            source = f"ego4d-{kind}"
            stage = Ego4DRecaptionStage(
                source_kind=kind,
                source_name=source,
                output_dir=str(annotations),
                caption_version=version,
                base_url=urls[0],
                base_urls=urls,
                model=model,
                max_image_edge=config.get("max_image_edge", 1024),
                max_output_tokens=config.get("max_output_tokens", 2048),
                timeout=config.get("timeout", 120.0),
                requests_per_worker=config.get("requests_per_worker", 8),
                retry_statuses=("failed",) if config.get("retry_failed", False) else (),
            )
            stages[kind] = stage
        _save_run(
            annotations / "run.json",
            {
                "method": "ego4d_recaption",
                "model": model,
                "caption_version": version,
                "input": str(output / "samples/schema-v1/run.json"),
                "sources": {
                    kind: {
                        "source": stage.source_name,
                        "prompt": stage.prompt,
                        "prompt_sha256": stage.prompt_sha256,
                        "max_image_edge": stage.max_image_edge,
                        "max_output_tokens": stage.max_output_tokens,
                        "jpeg_quality": stage.jpeg_quality,
                    }
                    for kind, stage in stages.items()
                },
                "batch_size": config.get("batch_size", 64),
            },
        )
        for kind, stage in stages.items():
            source = stage.source_name
            pipeline = Pipeline(name=f"ego4d_recaption_{kind}")
            pipeline.add_stage(
                ParquetReader(
                    file_paths=str(output / "samples/schema-v1/manifests" / f"source={source}"),
                    files_per_partition=1,
                    fields=[name for name in MANIFEST_SCHEMA.names if name != "source"],
                )
            )
            pipeline.add_stage(
                RecaptionImageReader(media_root=str(output / "media" / f"source={source}"), source_name=source)
            )
            pipeline.add_stage(stage.with_(num_workers=config.get("caption_workers", 8)))
            pipeline.add_stage(RecaptionWriter())
            checkpoint = (
                None if config.get("retry_failed", False) else output / "checkpoints/recaption" / version / source
            )
            pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint)
    finally:
        try:
            if server is not None:
                server.stop()
        finally:
            client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    run(parser.parse_args().config)
