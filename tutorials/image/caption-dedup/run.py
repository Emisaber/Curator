# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Run CLIP candidates and caption comparison on source shards."""

import argparse
import json
from pathlib import Path

from report import create_report

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.deduplication.caption import (
    CandidatePairPartitioningStage,
    CaptionAwareDeduplicationStage,
    CaptionSource,
)
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.decodable_reader import DecodableImageReaderStage
from nemo_curator.stages.image.io.sample_index import WebDatasetImageSampleIndexStage
from nemo_curator.stages.image.io.source_reader import SourceShardReaderStage
from nemo_curator.stages.image.io.subset import load_subset
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import EmptyTask


def create_sample_index_pipeline(name: str, source: CaptionSource, output: Path) -> Pipeline:
    pipeline = Pipeline(name=f"image_sample_index_{name}")
    pipeline.add_stage(FilePartitioningStage(file_paths=source.root, files_per_partition=1, file_extensions=[".tar"]))
    pipeline.add_stage(WebDatasetImageSampleIndexStage(source_name=name, source=source))
    pipeline.add_stage(
        ParquetWriter(
            path=str(output / "samples/v1" / f"source={name}"),
            fields=["image_id", "shard", "member", "offset", "size", "caption_raw"],
        )
    )
    return pipeline


def create_embedding_pipeline(
    name: str, source: CaptionSource, output: Path, args: argparse.Namespace, shards: list[dict] | None = None
) -> Pipeline:
    pipeline = Pipeline(name=f"image_embeddings_{name}")
    records_dir = str(Path(args.output) / "annotations/decode/schema-v1" / f"source={name}")
    if shards is None:
        file_paths = source.root
    elif source.format == "ego4d_recaption":
        file_paths = [str(Path(source.sample_index_root) / row["sample_index"]) for row in shards]
    else:
        file_paths = [str(Path(source.root) / row["shard"]) for row in shards]
    extension = ".parquet" if source.format in ("vlv_parquet", "ego4d_recaption") else ".tar"
    pipeline.add_stage(
        FilePartitioningStage(file_paths=file_paths, files_per_partition=1, file_extensions=[extension])
    )
    if source.format in ("webdataset_txt", "parquet"):
        reader = DecodableImageReaderStage(
            dali_batch_size=args.batch_size,
            num_gpus_per_worker=args.reader_gpus_per_worker,
            source_name=name,
            source_root=source.root,
            index_suffix=source.index_suffix,
            image_extensions=("jpg", "jpeg", "png", "webp"),
            case_sensitive_extensions=False,
            records_dir=records_dir,
        )
    else:
        reader = SourceShardReaderStage(
            name,
            source,
            image_batch_size=args.batch_size,
            num_gpus_per_worker=args.reader_gpus_per_worker,
            records_dir=records_dir,
        )
    pipeline.add_stage(reader.with_(num_workers=args.reader_workers))
    pipeline.add_stage(
        ImageEmbeddingStage(
            model_dir=args.model_dir,
            num_gpus_per_worker=args.clip_gpus_per_worker,
            model_inference_batch_size=args.batch_size,
            remove_image_data=True,
        ).with_(num_workers=args.clip_workers)
    )
    pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(fields=["image_id", "embedding"]))
    feature_path = output / f"source={name}" if args.subset else output / "features" / name
    pipeline.add_stage(ParquetWriter(path=str(feature_path)))
    return pipeline


def _check_completed_run(run_file: Path, completed: Path, run_info: dict) -> None:
    if json.loads(run_file.read_text(encoding="utf-8")) != run_info:
        msg = f"Existing deduplication uses different settings: {run_file}"
        raise ValueError(msg)
    if not completed.exists():
        msg = f"Deduplication is incomplete; partial deduplication resume is not supported: {run_file.parent}"
        raise RuntimeError(msg)
    if json.loads(completed.read_text(encoding="utf-8")) != run_info:
        msg = f"Deduplication completion does not match its settings: {completed}"
        raise ValueError(msg)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    dedup_output = output / ("dedup/clip/schema-v1" if args.subset else "dedup/clip/v1")
    selected_shards = None
    if args.subset:
        subset_config, selected_shards = load_subset(Path(args.subset))
        source_data = {
            name: {key: value for key, value in source.items() if key != "exclude_patterns"}
            for name, source in subset_config["sources"].items()
            if selected_shards[name]
        }
    else:
        source_data = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    sources = {name: CaptionSource(**config) for name, config in source_data.items()}
    run_info = {
        **{key: value for key, value in vars(args).items() if key not in ("config", "skip_report")},
        "sources": source_data,
        "shards": selected_shards,
        "candidate_eps": 0.01,
    }
    run_file = output / "run.json"
    completed = dedup_output / "completed.json"
    if run_file.exists():
        _check_completed_run(run_file, completed, run_info)
        if not args.skip_report:
            create_report(dedup_output, sources, output / "report", args.review_pairs, args.seed)
        return
    output.mkdir(parents=True, exist_ok=True)
    dedup_output.mkdir(parents=True)
    (output / "run.json").write_text(
        json.dumps(run_info, indent=2),
        encoding="utf-8",
    )
    (dedup_output / "run.json").write_text(
        json.dumps(
            {
                "method": "clip_image_similarity_with_source_caption_relation",
                "candidate_eps": 0.01,
                "n_clusters": args.n_clusters,
                "caption_source_format": {name: source.format for name, source in sources.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    decode_output = output / "annotations/decode/schema-v1"
    decode_output.mkdir(parents=True)
    (decode_output / "run.json").write_text(
        json.dumps(
            {
                "method": "dali_with_tar_cpu_fallback_and_native_pixel_restore",
                "batch_size": args.batch_size,
                "tar_fallback": "whole_shard_on_image_decode_error",
                "tar_exif_orientation": "stored_pixels",
                "sources": source_data,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    client = RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    client.start()
    try:
        feature_output = output / "features/clip/schema-v1" if args.subset else dedup_output / "cache"
        for name, source in sources.items():
            create_embedding_pipeline(
                name,
                source,
                feature_output,
                args,
                selected_shards[name] if selected_shards is not None else None,
            ).run()
        feature_paths = [
            feature_output / f"source={name}" if args.subset else feature_output / "features" / name
            for name in sources
        ]
        input_paths = [str(path) for path in feature_paths if path.is_dir()]
        if not input_paths:
            msg = "No usable images produced CLIP features"
            raise ValueError(msg)
        SemanticDeduplicationWorkflow(
            input_path=input_paths,
            output_path=str(dedup_output / "cache/workflow"),
            cache_path=str(dedup_output / "cache/workflow"),
            id_field="image_id",
            embedding_field="embedding",
            n_clusters=args.n_clusters,
            eps=None,
            candidate_eps=0.01,
            read_kwargs={"categorical_partitions": False},
        ).run()
        pipeline = Pipeline(name="caption_aware_image_deduplication")
        pipeline.add_stage(CandidatePairPartitioningStage(str(dedup_output / "cache/workflow/candidate_pairs")))
        pipeline.add_stage(CaptionAwareDeduplicationStage(sources, str(dedup_output)))
        pipeline.run(initial_tasks=[EmptyTask(dataset_name="image_dedup")])
    finally:
        client.stop()
    temporary = completed.with_suffix(".tmp")
    temporary.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    temporary.replace(completed)
    if not args.skip_report:
        create_report(dedup_output, sources, output / "report", args.review_pairs, args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON configuration; command-line options override its values")
    config_args, _ = parser.parse_known_args()
    config = json.loads(Path(config_args.config).read_text(encoding="utf-8")) if config_args.config else {}
    inputs = parser.add_mutually_exclusive_group(required=not ("subset" in config or "sources" in config))
    inputs.add_argument("--sources", help="JSON map of source names to CaptionSource fields")
    inputs.add_argument("--subset", help="Frozen shard subset directory containing config.json and shards.jsonl")
    parser.add_argument("--output", required="output" not in config, help="New experiment directory")
    parser.add_argument("--model-dir", required="model_dir" not in config, help="Existing CLIP model directory")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reader-workers", type=int, help="Reader workers per source; omit for NeMo autoscaling")
    parser.add_argument("--clip-workers", type=int, help="CLIP workers per source; omit for NeMo autoscaling")
    parser.add_argument("--reader-gpus-per-worker", type=float, default=0.25)
    parser.add_argument("--clip-gpus-per-worker", type=float, default=1.0)
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--review-pairs", type=int, default=240)
    parser.add_argument(
        "--skip-report", action=argparse.BooleanOptionalAction, default=False, help="Skip review generation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(**config)
    args = parser.parse_args()
    if bool(args.subset) == bool(args.sources):
        parser.error("Specify exactly one of subset or sources")
    return args


if __name__ == "__main__":
    run(parse_args())
