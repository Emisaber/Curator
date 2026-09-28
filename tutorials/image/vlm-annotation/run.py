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

"""Annotate source shards using a local OpenAI-compatible VLM."""

import argparse
import contextlib
import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

import pyarrow.parquet as pq
import ray

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.core.serve.dynamo.infra import engine_kwargs_to_cli_flags
from nemo_curator.core.serve.placement import build_pg, get_bundle_node_ip, get_free_port_in_bundle
from nemo_curator.core.serve.subprocess_mgr import ManagedSubprocess, reacquire_detached_actor_handles
from nemo_curator.models.client import OpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.annotation import ImageVLMAnnotationStage, VLMAnnotationWriter
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.decodable_reader import DecodableImageReaderStage
from nemo_curator.stages.image.io.decoded_image_reader import DecodeRecordImageReaderStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.source_reader import SourceShardReaderStage
from nemo_curator.stages.image.io.subset import load_subset


def _served_model(config: dict) -> str:
    model = config.get("model")
    if model is not None:
        return model
    client = OpenAIClient(base_url=config.get("base_url", "http://127.0.0.1:8000/v1"), api_key="EMPTY")
    client.setup()
    models = client.client.models.list().data
    if len(models) != 1:
        msg = f"Expected one served model, found {len(models)}; set model explicitly"
        raise ValueError(msg)
    return models[0].id


def _retry_archives(
    annotations: Path, retry: dict, source_name: str | None = None, source_format: str = "webdataset_txt"
) -> list[str]:
    if (
        retry["pass"] < 1
        or not retry["statuses"]
        or set(retry["statuses"]) - {"request_error", "length", "invalid_response"}
    ):
        msg = "retry requires a positive pass and nonempty failure statuses"
        raise ValueError(msg)
    retry_file = annotations / f"retry-{retry['pass']}.json"
    if retry_file.exists() and json.loads(retry_file.read_text(encoding="utf-8")) != retry:
        msg = f"Existing retry pass uses different settings: {retry_file}"
        raise ValueError(msg)
    if not retry_file.exists():
        retry_file.write_text(json.dumps(retry, indent=2), encoding="utf-8")
    archives = set()
    pattern = f"source={source_name}/*.parquet" if source_name is not None else "source=*/*.parquet"
    for path in annotations.glob(pattern):
        for batch in pq.ParquetFile(path).iter_batches(columns=["status", "image_path"]):
            for row in batch.to_pylist():
                if row["status"] in retry["statuses"]:
                    parts = 1 if source_format == "vlv_parquet" else 2
                    archives.add(row["image_path"].rsplit(":", parts)[0])
    return sorted(archives)


def _source_files(source: CaptionSource, shards: list[dict] | None, archives: list[str] | None) -> str | list[str]:
    if shards is not None:
        selected = shards
        if archives is not None:
            selected = [row for row in shards if str(Path(source.root) / row["shard"]) in archives]
        if source.format == "ego4d_recaption":
            return [str(Path(source.sample_index_root) / row["sample_index"]) for row in selected]
        return [str(Path(source.root) / row["shard"]) for row in selected]
    if source.format == "ego4d_recaption":
        if archives is not None:
            return [
                str(Path(source.sample_index_root) / Path(path).relative_to(source.root).with_suffix(".parquet"))
                for path in archives
            ]
        return sorted(str(path) for path in Path(source.sample_index_root).rglob("*.parquet"))
    return archives if archives is not None else source.root


def _source_reader(name: str, source: CaptionSource, config: dict, output: Path) -> ProcessingStage:
    records_dir = str(output / "annotations/decode/schema-v1" / f"source={name}")
    if source.format in ("webdataset_txt", "parquet"):
        reader = DecodableImageReaderStage(
            dali_batch_size=config.get("batch_size", 8),
            num_gpus_per_worker=config.get("reader_gpus_per_worker", 0.25),
            source_name=name,
            source_root=source.root,
            index_suffix=source.index_suffix,
            max_images_per_partition=config.get("max_images_per_partition"),
            image_extensions=("jpg", "jpeg", "png", "webp"),
            case_sensitive_extensions=False,
            records_dir=records_dir,
        )
    else:
        reader = SourceShardReaderStage(
            name,
            source,
            image_batch_size=config.get("batch_size", 8),
            num_gpus_per_worker=config.get("reader_gpus_per_worker", 0.25),
            records_dir=records_dir,
        )
    return reader.with_(num_workers=config.get("reader_workers"))


def _decode_files(input_run: Path, name: str, source: CaptionSource, archives: list[str] | None) -> list[str]:
    paths = sorted((input_run / "annotations/decode/schema-v1" / f"source={name}").glob("*.parquet"))
    if archives is None:
        return [str(path) for path in paths]
    selected = []
    for path in paths:
        shards = pq.ParquetFile(path).read(columns=["shard"])["shard"].to_pylist()
        if any(str(Path(source.root) / shard) in archives for shard in shards):
            selected.append(str(path))
    return selected


class _VLLMServerPool:
    """Run vLLM from a separate environment on Ray-reserved GPUs."""

    def __init__(self, config: dict, output: Path) -> None:
        self.config = config
        self.model_name = config.get("model_name", Path(config["model_path"]).name)
        self.log_dir = output / "logs" / "vllm"
        self.namespace = "image_vlm_annotation"
        self.name_prefix = f"image_vlm_{uuid.uuid4().hex[:8]}"
        self.placement_groups = []
        self.processes = []
        self.endpoints = []

    def start(self) -> None:
        serving = self.config
        tp_size = serving.get("engine_kwargs", {}).get("tensor_parallel_size", 1)
        ray_gpus = serving.get("ray_gpus_per_replica", tp_size)
        environment = Path(serving["environment"])
        command = [
            str(environment / "bin" / "vllm"),
            "serve",
            serving["model_path"],
            "--served-model-name",
            self.model_name,
            "--host",
            "0.0.0.0",  # noqa: S104
            *engine_kwargs_to_cli_flags(serving.get("engine_kwargs", {})),
        ]
        subprocess_env = {
            "VIRTUAL_ENV": str(environment),
            "PATH": f"{environment / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": "",
            **serving.get("env_vars", {}),
        }
        with ray.init(namespace=self.namespace, ignore_reinit_error=True):
            try:
                for index in range(serving.get("num_replicas", 1)):
                    pg = build_pg(
                        [{"CPU": 1, "GPU": ray_gpus}],
                        "STRICT_PACK",
                        name=f"{self.name_prefix}_{index}",
                        bundle_label_selector=None,
                        ready_timeout_s=120,
                    )
                    self.placement_groups.append(pg)
                    port = get_free_port_in_bundle(pg, 0, serving.get("port", 8000) + index)
                    host = get_bundle_node_ip(pg, 0)
                    process = ManagedSubprocess.spawn(
                        label=f"VLLM_{index}",
                        pg=pg,
                        bundle_index=0,
                        num_gpus=ray_gpus,
                        command=[*command, "--port", str(port)],
                        runtime_dir=str(self.log_dir),
                        actor_name_prefix=self.name_prefix,
                        subprocess_env=subprocess_env,
                    )
                    self.processes.append(process)
                    endpoint = f"http://{host}:{port}/v1"
                    self.endpoints.append(endpoint)
                for process, endpoint in zip(self.processes, self.endpoints, strict=True):
                    self._wait_for_model(process, endpoint)
            except Exception:
                self._stop_connected()
                raise

    def _wait_for_model(self, process: ManagedSubprocess, endpoint: str) -> None:
        deadline = time.monotonic() + self.config.get("health_check_timeout_s", 600)
        while time.monotonic() < deadline:
            if not process.is_alive():
                msg = f"vLLM exited before becoming ready: {process.read_log_tail()}"
                raise RuntimeError(msg)
            try:
                with urllib.request.urlopen(f"{endpoint}/models", timeout=2) as response:  # noqa: S310
                    models = json.load(response)["data"]
                if self.model_name in {model["id"] for model in models}:
                    return
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(1)
        msg = f"vLLM did not become ready at {endpoint}: {process.read_log_tail()}"
        raise TimeoutError(msg)

    def _stop_connected(self) -> None:
        processes = reacquire_detached_actor_handles(
            self.processes, actor_name_prefix=self.name_prefix, namespace=self.namespace
        )
        ManagedSubprocess.stop_many(processes)
        self.processes.clear()
        for pg in self.placement_groups:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)
        self.placement_groups.clear()

    def stop(self) -> None:
        if self.processes or self.placement_groups:
            with ray.init(namespace=self.namespace, ignore_reinit_error=True):
                self._stop_connected()


def _input_sources(config: dict) -> tuple[dict, dict | None, dict | None]:
    selected_shards = None
    input_settings = None
    input_run = Path(config["input_run"]).resolve() if "input_run" in config else None
    if input_run is not None:
        input_settings = json.loads((input_run / "run.json").read_text(encoding="utf-8"))
        dedup_version = "schema-v1" if input_settings.get("subset") else "v1"
        completed = input_run / "dedup/clip" / dedup_version / "completed.json"
        if json.loads(completed.read_text(encoding="utf-8")) != input_settings:
            msg = f"Deduplication completion does not match its input settings: {completed}"
            raise ValueError(msg)
        source_data = input_settings["sources"]
    elif "subset" in config:
        subset_config, selected_shards = load_subset(Path(config["subset"]))
        source_data = {
            name: {key: value for key, value in source.items() if key != "exclude_patterns"}
            for name, source in subset_config["sources"].items()
            if selected_shards[name]
        }
    elif "sources" in config:
        source_data = config["sources"]
    else:
        source = config["source"]
        source_data = {source["name"]: {key: value for key, value in source.items() if key != "name"}}
    return source_data, selected_shards, input_settings


def _pipeline_reader(config: dict, name: str, source: CaptionSource, batch_size: int) -> ProcessingStage:
    if "input_run" in config:
        return DecodeRecordImageReaderStage(
            source_name=name,
            source=source,
            image_batch_size=batch_size,
            num_gpus_per_worker=config.get("reader_gpus_per_worker", 0.25),
        ).with_(num_workers=config.get("reader_workers"))
    if "subset" in config or "sources" in config:
        return _source_reader(name, source, config, Path(config["output"]).resolve())
    return ImageReaderStage(
        dali_batch_size=config.get("batch_size", 8),
        num_gpus_per_worker=config.get("reader_gpus_per_worker", 0.25),
        source_name=name,
        source_root=source.root,
        index_suffix=config["source"].get("index_suffix"),
        max_images_per_partition=config.get("max_images_per_partition"),
        image_extensions=("jpg", "jpeg", "png", "webp"),
        case_sensitive_extensions=False,
    ).with_(num_workers=config.get("reader_workers"))


def _write_run_info(annotations: Path, run_info: dict) -> None:
    run_file = annotations / "run.json"
    if run_file.exists():
        previous = json.loads(run_file.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in run_info.items()):
            msg = f"Existing annotations use different settings: {run_file}"
            raise ValueError(msg)
    else:
        annotations.mkdir(parents=True, exist_ok=True)
        run_file.write_text(json.dumps(run_info, indent=2), encoding="utf-8")


def _checkpoint_path(output: Path, version_path: Path, name: str, multi_source: bool, retry: dict | None) -> Path:
    checkpoint = output / "checkpoints" / "vlm-basic" / version_path
    if multi_source:
        checkpoint = checkpoint / f"source={name}"
    if retry is not None:
        checkpoint = checkpoint / f"retry-{retry['pass']}"
    return checkpoint


def _run_pipeline(config: dict, model: str, base_url: str, base_urls: tuple[str, ...] | None = None) -> None:
    output = Path(config["output"]).resolve()
    multi_source = "input_run" in config or "subset" in config or "sources" in config
    source_data, selected_shards, input_settings = _input_sources(config)
    input_run = Path(config["input_run"]).resolve() if "input_run" in config else None
    batch_size = config.get("batch_size", 32 if input_run is not None else 8)
    version = config.get("version", "v1")
    schema_version = str(config.get("schema_version", "1"))
    if multi_source and schema_version != "1":
        msg = "The current image contract is schema-v1"
        raise ValueError(msg)
    prompt_version = config.get("prompt_version", "v2")
    version_path = Path(f"schema-v{schema_version}") / f"prompt-{prompt_version}" if multi_source else Path(version)
    annotations = output / "annotations" / "vlm-basic" / version_path
    retry = config.get("retry")
    annotation_stage = ImageVLMAnnotationStage(
        fields=config["fields"],
        base_url=base_url,
        base_urls=base_urls,
        model=model,
        max_image_edge=config.get("max_image_edge", 1024),
        max_output_tokens=config.get("max_output_tokens", 512),
        requests_per_worker=config.get("requests_per_worker", 1),
        timeout=config.get("timeout", 120.0),
        output_dir=str(annotations),
        retry_statuses=tuple(retry["statuses"]) if retry else (),
    )
    run_info = {
        "fields": config["fields"],
        "model": model,
        "base_url": None if "inference_server" in config else annotation_stage.base_url,
        "prompt_sha256": annotation_stage.prompt_sha256,
        "max_image_edge": annotation_stage.max_image_edge,
        "max_output_tokens": annotation_stage.max_output_tokens,
        "batch_size": batch_size,
        "max_shards": config.get("max_shards"),
        "max_images_per_partition": config.get("max_images_per_partition"),
    }
    if multi_source:
        run_info.update(sources=source_data, schema_version=schema_version, prompt_version=prompt_version)
        if input_run is not None:
            run_info.update(input_run=str(input_run), input_settings=input_settings)
        if selected_shards is not None:
            run_info.update(subset=str(Path(config["subset"]).resolve()), shards=selected_shards)
    else:
        run_info["source"] = config["source"]
    if "requests_per_worker" in config:
        run_info["requests_per_worker"] = annotation_stage.requests_per_worker
    if "inference_server" in config:
        run_info["inference_server"] = config["inference_server"]
    _write_run_info(annotations, run_info)
    for name, settings in source_data.items():
        source = CaptionSource(**settings)
        archives = _retry_archives(annotations, retry, name, source.format) if retry else None
        file_paths = (
            _decode_files(input_run, name, source, archives)
            if input_run is not None
            else _source_files(source, selected_shards[name] if selected_shards is not None else None, archives)
        )
        if not file_paths:
            continue
        checkpoint = _checkpoint_path(output, version_path, name, multi_source, retry)
        pipeline = Pipeline(name=f"image_vlm_annotation_{name}" if multi_source else "image_vlm_annotation")
        extension = (
            ".parquet" if input_run is not None or source.format in ("vlv_parquet", "ego4d_recaption") else ".tar"
        )
        pipeline.add_stage(
            FilePartitioningStage(
                file_paths=file_paths,
                file_extensions=[extension],
                files_per_partition=1,
                limit=config.get("max_shards"),
            )
        )
        pipeline.add_stage(_pipeline_reader(config, name, source, batch_size))
        annotation_stage.source_name = name
        pipeline.add_stage(annotation_stage.with_(num_workers=config.get("annotation_workers")))
        pipeline.add_stage(VLMAnnotationWriter())

        if "inference_server" in config:
            pipeline.run(executor=RayDataExecutor(), checkpoint_path=checkpoint)
        else:
            pipeline.run(checkpoint_path=checkpoint)


def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    client = RayClient(num_cpus=config.get("num_cpus", 4), num_gpus=config.get("num_gpus", 1))
    client.start()
    server = None
    try:
        if "inference_server" in config:
            server = _VLLMServerPool(config["inference_server"], Path(config["output"]).resolve())
            server.start()
            model = server.model_name
            base_url = server.endpoints[0]
        else:
            model = _served_model(config)
            base_url = config.get("base_url", "http://127.0.0.1:8000/v1")
        _run_pipeline(config, model, base_url, tuple(server.endpoints) if server else None)
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
