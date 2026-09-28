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

"""Compare pHash and the native image semantic-dedup workflow on local TARs."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from prepare_samples import parse_sources, prepare_samples
from report import create_report

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.image.deduplication.phash import PerceptualHashStage
from nemo_curator.stages.image.deduplication.phash_candidates import PhashCandidateStage
from nemo_curator.stages.image.filters.blur_filter import ImageBlurFilterStage
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.tar_image_reader import TarImageDecodeStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.utils import TaskPerfUtils


def create_feature_pipeline(args: argparse.Namespace, roots: dict[str, str], source_name: str) -> Pipeline:
    output = Path(args.output)
    pipeline = Pipeline(name="image_dedup_features")
    pipeline.add_stage(
        ParquetReader(
            file_paths=str(output / "samples/schema-v1/manifests" / f"source={source_name}"),
            files_per_partition=1,
        )
    )
    pipeline.add_stage(
        TarImageDecodeStage(
            {source_name: roots[source_name]}, str(output / "annotations/decode/schema-v1"),
        ).with_(num_workers=1)
    )
    pipeline.add_stage(
        ImageBlurFilterStage(
            score_threshold=args.blur_threshold,
            annotations_dir=str(output / "annotations/blur/schema-v1"),
        ).with_(num_workers=1)
    )
    fields = ["image_id", "metadata"]
    if args.mode in {"phash", "both"}:
        pipeline.add_stage(PerceptualHashStage().with_(num_workers=1))
    if args.mode in {"clip", "both"}:
        from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage

        pipeline.add_stage(
            ImageEmbeddingStage(
                model_dir=args.model_dir,
                num_gpus_per_worker=1,
                model_inference_batch_size=args.batch_size,
                remove_image_data=True,
            ).with_(num_workers=1)
        )
        fields.append("embedding")
    pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(fields=fields))
    pipeline.add_stage(ParquetWriter(path=str(output / "features/clip/schema-v1" / f"source={source_name}")))
    return pipeline


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    roots = parse_sources(args.source)
    caption_sources = {
        name: CaptionSource(
            root=root,
            format=args.caption_source_formats.get(name, "webdataset_txt"),
            caption_column=args.caption_columns.get(name, "caption"),
        )
        for name, root in roots.items()
    }
    prepare_samples(
        roots,
        str(output / "samples/schema-v1"),
        args.source_sample_rates,
        args.batch_size,
        args.seed,
        samples_path=str(output / "samples/schema-v1/manifest.parquet"),
        caption_sources=caption_sources,
    )
    for stage_name, stage_config in (
        ("decode", {"method": "pillow_exif_transpose_rgb", "short_side_tag": 256}),
        ("blur", {"method": "opencv_laplacian_variance", "score_threshold": args.blur_threshold}),
    ):
        stage_dir = output / "annotations" / stage_name / "schema-v1"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "run.json").write_text(json.dumps(stage_config, indent=2), encoding="utf-8")
    config = {
        **vars(args),
        "clip_eps": 0.01,
        "clip_which_to_keep": "hard",
        "clip_compute_dtype": "float16",
        "clip_storage_dtype": "float16",
        "versions": {name: importlib.metadata.version(name) for name in ("nemo-curator", "ray", "numpy", "pandas")},
        "timings_seconds": {},
    }
    (output / "run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    client = RayClient(
        num_cpus=args.num_cpus, num_gpus=0 if args.mode == "phash" else args.num_gpus, object_store_memory=2 * 1024**3
    )
    client.start()
    try:
        start = time.perf_counter()
        tasks = []
        for source_name in roots:
            manifest_dir = output / "samples/schema-v1/manifests" / f"source={source_name}"
            if not manifest_dir.is_dir():
                continue
            tasks.extend(create_feature_pipeline(args, roots, source_name).run())
        config["timings_seconds"]["features"] = time.perf_counter() - start
        config["feature_stage_metrics"] = TaskPerfUtils.aggregate_task_metrics(tasks)
        files = [path for task in tasks for path in task.data]
        if not files:
            msg = "No images decoded successfully; inspect annotations/decode/schema-v1/*.parquet"
            raise ValueError(msg)
        if args.mode in {"phash", "both"}:
            start = time.perf_counter()
            pipeline = Pipeline(name="image_phash_candidates")
            pipeline.add_stage(
                PhashCandidateStage(str(output / "phash"), args.phash_threshold, args.phash_max_distance)
            )
            pipeline.run(initial_tasks=[FileGroupTask(dataset_name="image_dedup", data=files)])
            config["timings_seconds"]["phash_comparison"] = time.perf_counter() - start
        if args.mode in {"clip", "both"}:
            from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow

            start = time.perf_counter()
            SemanticDeduplicationWorkflow(
                input_path=[
                    str(output / "features/clip/schema-v1" / f"source={source}")
                    for source in roots
                    if (output / "features/clip/schema-v1" / f"source={source}").is_dir()
                ],
                output_path=str(output / "dedup/clip/schema-v1"),
                cache_path=str(output / "dedup/clip/schema-v1/cache"),
                id_field="image_id",
                embedding_field="embedding",
                n_clusters=args.n_clusters,
                eps=0.01,
            ).run()
            config["timings_seconds"]["clip_dedup"] = time.perf_counter() - start
    finally:
        client.stop()
        (output / "run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    create_report(output, args.report_pairs, args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True, help="NAME=/path to uncompressed TARs")
    parser.add_argument("--output", required=True, help="New experiment directory")
    parser.add_argument("--mode", choices=["phash", "clip", "both"], default="both")
    parser.add_argument("--model-dir", default="./models", help="Native CLIP model cache root")
    parser.add_argument(
        "--source-sample-rate",
        nargs="*",
        default=[],
        metavar="SOURCE=RATE",
        help="Independent sampling fraction for each source; omit to use every record",
    )
    parser.add_argument(
        "--caption-source-format",
        nargs="*",
        default=[],
        metavar="SOURCE=FORMAT",
        help="Caption format for a source (webdataset_txt or parquet)",
    )
    parser.add_argument(
        "--caption-column",
        nargs="*",
        default=[],
        metavar="SOURCE=COLUMN",
        help="Caption column for a Parquet caption source",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Laplacian variance threshold used to mark blurry images; does not remove them",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Ray GPU count; select devices with CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument("--n-clusters", type=int, default=100)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--phash-max-distance", type=int, default=8)
    parser.add_argument("--report-pairs", type=int, default=200)
    args = parser.parse_args()
    args.source_sample_rates = {
        name: float(rate) for name, rate in (value.split("=", 1) for value in args.source_sample_rate)
    }
    args.caption_source_formats = dict(value.split("=", 1) for value in args.caption_source_format)
    args.caption_columns = dict(value.split("=", 1) for value in args.caption_column)
    return args


if __name__ == "__main__":
    run(parse_args())
