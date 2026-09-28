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

"""Tests for the VLM annotation tutorial launcher."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.annotation import ImageVLMAnnotationStage
from nemo_curator.stages.image.io.decoded_image_reader import DecodeRecordImageReaderStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.source_reader import SourceShardReaderStage
from nemo_curator.stages.image.io.subset import load_subset, prepare_subset


def test_launcher_versions_prompt_and_reads_all_indexed_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/run.py"
    spec = importlib.util.spec_from_file_location("vlm_annotation_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pipelines = []

    class Pipeline:
        def __init__(self, name: str) -> None:
            self.stages = []
            pipelines.append(self)

        def add_stage(self, stage: object) -> None:
            self.stages.append(stage)

        def run(self, checkpoint_path: Path) -> None:
            self.checkpoint_path = checkpoint_path

    class RayClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module, "RayClient", RayClient)

    config = {
        "source": {"name": "sample", "root": str(tmp_path / "images")},
        "output": str(tmp_path / "output"),
        "version": "v1",
        "fields": ["clarity"],
        "model": "test-vlm",
        "annotation_workers": 16,
        "requests_per_worker": 4,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module.run(config_path)

    reader = next(stage for stage in pipelines[-1].stages if isinstance(stage, ImageReaderStage))
    assert reader.image_extensions == ("jpg", "jpeg", "png", "webp")
    assert reader.case_sensitive_extensions is False
    annotation = next(stage for stage in pipelines[-1].stages if isinstance(stage, ImageVLMAnnotationStage))
    assert annotation.num_workers() == 16
    assert annotation.requests_per_worker == 4
    assert pipelines[-1].checkpoint_path == tmp_path / "output/checkpoints/vlm-basic/v1"
    run_file = tmp_path / "output/annotations/vlm-basic/v1/run.json"
    run_info = json.loads(run_file.read_text(encoding="utf-8"))
    assert "code_sha256" not in run_info
    assert run_info["requests_per_worker"] == 4

    module.run(config_path)
    run_info["prompt_sha256"] = "different-prompt"
    run_file.write_text(json.dumps(run_info), encoding="utf-8")
    with pytest.raises(ValueError, match="different settings"):
        module.run(config_path)

    config["version"] = "v2"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module.run(config_path)
    assert pipelines[-1].checkpoint_path == tmp_path / "output/checkpoints/vlm-basic/v2"
    assert (tmp_path / "output/annotations/vlm-basic/v2/run.json").is_file()


def test_launcher_retry_selects_failed_archives_and_uses_separate_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/run.py"
    spec = importlib.util.spec_from_file_location("vlm_annotation_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipelines = []

    class Pipeline:
        def __init__(self, name: str) -> None:
            self.stages = []
            pipelines.append(self)

        def add_stage(self, stage: object) -> None:
            self.stages.append(stage)

        def run(self, checkpoint_path: Path) -> None:
            self.checkpoint_path = checkpoint_path

    class RayClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module, "RayClient", RayClient)
    config = {
        "source": {"name": "sample", "root": str(tmp_path / "images")},
        "output": str(tmp_path / "output"),
        "version": "v1",
        "fields": ["clarity"],
        "model": "test-vlm",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module.run(config_path)
    annotations = tmp_path / "output/annotations/vlm-basic/v1/source=sample"
    annotations.mkdir(parents=True)
    failed_tar = str(tmp_path / "images/failed.tar")
    pd.DataFrame(
        [
            {"status": "ok", "image_path": str(tmp_path / "images/ok.tar") + ":512:a.jpg"},
            {"status": "length", "image_path": failed_tar + ":1024:b.jpg"},
        ]
    ).to_parquet(annotations / "part-1.parquet", index=False)
    config["retry"] = {"pass": 1, "statuses": ["length"]}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module.run(config_path)
    partition = next(stage for stage in pipelines[-1].stages if isinstance(stage, FilePartitioningStage))
    annotation = next(stage for stage in pipelines[-1].stages if isinstance(stage, ImageVLMAnnotationStage))
    assert partition.file_paths == [failed_tar]
    assert annotation.retry_statuses == ("length",)
    assert pipelines[-1].checkpoint_path == tmp_path / "output/checkpoints/vlm-basic/v1/retry-1"
    assert (annotations.parent / "retry-1.json").is_file()


def test_launcher_manages_replicated_vllm_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/run.py"
    spec = importlib.util.spec_from_file_location("vlm_annotation_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = []
    captured = {}

    class RayClient:
        def __init__(self, **kwargs: object) -> None:
            captured["cluster"] = kwargs

        def start(self) -> None:
            events.append("cluster_start")

        def stop(self) -> None:
            events.append("cluster_stop")

    class VLLMServerPool:
        def __init__(self, config: dict, output: Path) -> None:
            captured["serving"] = config
            self.model_name = config["model_name"]
            self.endpoints = ["http://localhost:8010/v1", "http://localhost:8011/v1"]

        def start(self) -> None:
            events.append("server_start")

        def stop(self) -> None:
            events.append("server_stop")

    class Pipeline:
        def __init__(self, name: str) -> None:
            self.stages = []

        def add_stage(self, stage: object) -> None:
            self.stages.append(stage)

        def run(self, executor: object, checkpoint_path: Path) -> None:
            assert isinstance(executor, module.RayDataExecutor)
            annotation = next(stage for stage in self.stages if isinstance(stage, ImageVLMAnnotationStage))
            assert annotation.base_urls == ("http://localhost:8010/v1", "http://localhost:8011/v1")
            assert annotation.model == "test-vlm"
            events.append("pipeline_run")

    monkeypatch.setattr(module, "RayClient", RayClient)
    monkeypatch.setattr(module, "_VLLMServerPool", VLLMServerPool)
    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module, "RayDataExecutor", type("RayDataExecutor", (), {}))
    config = {
        "source": {"name": "sample", "root": str(tmp_path / "images")},
        "output": str(tmp_path / "output"),
        "fields": ["clarity"],
        "num_gpus": 3,
        "inference_server": {
            "environment": "/envs/vllm",
            "model_path": "/models/vlm",
            "model_name": "test-vlm",
            "num_replicas": 2,
            "engine_kwargs": {
                "max_num_seqs": 512,
                "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            },
            "env_vars": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module.run(config_path)

    assert captured["serving"] == config["inference_server"]
    assert captured["cluster"]["num_gpus"] == 3
    assert events == ["cluster_start", "server_start", "pipeline_run", "server_stop", "cluster_stop"]
    run_info = json.loads((tmp_path / "output/annotations/vlm-basic/v1/run.json").read_text(encoding="utf-8"))
    assert run_info["inference_server"] == config["inference_server"]
    assert run_info["base_url"] is None


def test_server_pool_uses_independent_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/run.py"
    spec = importlib.util.spec_from_file_location("vlm_annotation_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launches = []

    class RayContext:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *_args: object) -> None:
            pass

    class ManagedSubprocess:
        @classmethod
        def spawn(cls, **kwargs: object) -> object:
            launches.append(kwargs)
            return object()

        @classmethod
        def stop_many(cls, procs: list) -> None:
            assert len(procs) == 2

    monkeypatch.setattr(module.ray, "init", lambda **_kwargs: RayContext())
    monkeypatch.setattr(module, "build_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(module, "get_free_port_in_bundle", lambda _pg, _bundle, port: port)
    monkeypatch.setattr(module, "get_bundle_node_ip", lambda _pg, _bundle: "127.0.0.1")
    monkeypatch.setattr(module, "ManagedSubprocess", ManagedSubprocess)
    monkeypatch.setattr(module, "reacquire_detached_actor_handles", lambda procs, **_kwargs: procs)
    monkeypatch.setattr(module.ray.util, "remove_placement_group", lambda _pg: None)
    config = {
        "environment": "/envs/qwen",
        "model_path": "/models/qwen",
        "model_name": "qwen",
        "num_replicas": 2,
        "port": 8010,
        "engine_kwargs": {
            "tensor_parallel_size": 1,
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
        },
        "env_vars": {"VLLM_USE_RUST_FRONTEND": "1"},
    }
    pool = module._VLLMServerPool(config, tmp_path)
    monkeypatch.setattr(pool, "_wait_for_model", lambda _process, _endpoint: None)
    pool.start()

    assert pool.endpoints == ["http://127.0.0.1:8010/v1", "http://127.0.0.1:8011/v1"]
    assert len(launches) == 2
    for launch in launches:
        assert launch["command"][:3] == ["/envs/qwen/bin/vllm", "serve", "/models/qwen"]
        assert launch["subprocess_env"]["VIRTUAL_ENV"] == "/envs/qwen"
        assert launch["subprocess_env"]["VLLM_USE_RUST_FRONTEND"] == "1"
        assert "--speculative-config" in launch["command"]
    pool.stop()


def _source_subset(tmp_path: Path) -> tuple[dict, Path]:
    sources = {}
    formats = {"long": "webdataset_txt", "short": "webdataset_txt", "vlv": "vlv_parquet", "ego": "ego4d_recaption"}
    for name, source_format in formats.items():
        root = tmp_path / name
        root.mkdir()
        extension = ".parquet" if source_format == "vlv_parquet" else ".tar"
        for stem in ("a", "b"):
            (root / f"{stem}{extension}").write_bytes(b"unread shard")
        sources[name] = {"root": str(root), "format": source_format, "index_suffix": None}
    sources["ego"].update(sample_index_root=str(tmp_path / "frame-index"), annotations_root=str(tmp_path / "captions"))
    subset = tmp_path / "subset"
    prepare_subset({"sources": sources}, subset)
    return sources, subset


def _assert_subset_pipeline(name: str, pipeline: object, source: dict, shards: list[dict], output: Path) -> None:
    partition, reader, annotation, _writer = pipeline.stages
    expected = (
        [str(Path(source["sample_index_root"]) / row["sample_index"]) for row in shards]
        if name == "ego"
        else [str(Path(source["root"]) / row["shard"]) for row in shards]
    )
    assert partition.file_paths == expected
    assert reader.num_workers() == 4
    assert annotation.source_name == name
    assert annotation.num_workers() == 64
    assert annotation.requests_per_worker == 8
    assert pipeline.checkpoint_path == output / f"checkpoints/vlm-basic/schema-v1/prompt-v2/source={name}"
    if name in ("long", "short"):
        assert isinstance(reader, ImageReaderStage)
        assert reader.dali_batch_size == 32
    else:
        assert isinstance(reader, SourceShardReaderStage)
        assert reader.image_batch_size == 32


def test_subset_sources_share_services_and_keep_source_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/run.py"
    spec = importlib.util.spec_from_file_location("vlm_subset_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources, subset = _source_subset(tmp_path)
    pipelines = []
    events = []

    class Pipeline:
        def __init__(self, name: str) -> None:
            self.name = name
            self.stages = []
            pipelines.append(self)

        def add_stage(self, stage: object) -> None:
            self.stages.append(stage)

        def run(self, executor: object, checkpoint_path: Path) -> None:
            self.checkpoint_path = checkpoint_path
            events.append(self.name)

    class VLLMServerPool:
        model_name = "test-vlm"
        endpoints = ("http://localhost:8010/v1", "http://localhost:8011/v1")

        def __init__(self, _config: dict, _output: Path) -> None:
            pass

        def start(self) -> None:
            events.append("server_start")

        def stop(self) -> None:
            events.append("server_stop")

    monkeypatch.setattr(module, "Pipeline", Pipeline)
    monkeypatch.setattr(module, "RayClient", Mock())
    monkeypatch.setattr(module, "_VLLMServerPool", VLLMServerPool)
    output = tmp_path / "output"
    config = {
        "subset": str(subset),
        "output": str(output),
        "fields": ["clarity"],
        "batch_size": 32,
        "reader_workers": 4,
        "annotation_workers": 64,
        "requests_per_worker": 8,
        "inference_server": {"environment": "/envs/vllm", "model_path": "/models/vlm"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module.run(config_path)
    _, selected = load_subset(subset)
    assert events == ["server_start", *(f"image_vlm_annotation_{name}" for name in sources), "server_stop"]
    annotations = output / "annotations/vlm-basic/schema-v1/prompt-v2"
    for name, pipeline in zip(sources, pipelines, strict=True):
        _assert_subset_pipeline(name, pipeline, sources[name], selected[name], output)
    run_info = json.loads((annotations / "run.json").read_text(encoding="utf-8"))
    assert run_info["sources"] == sources
    assert run_info["shards"] == selected
    assert not list(subset.rglob("*.parquet"))

    for name, location in {
        "vlv": f"{tmp_path / 'vlv/a.parquet'}:1",
        "ego": f"{tmp_path / 'ego/b.tar'}:512:frame.jpg",
    }.items():
        partition_dir = annotations / f"source={name}"
        partition_dir.mkdir()
        pd.DataFrame([{"status": "length", "image_path": location}]).to_parquet(partition_dir / "part-1.parquet")
    config["retry"] = {"pass": 1, "statuses": ["length"]}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    pipelines.clear()
    module.run(config_path)
    assert len(pipelines) == 2
    assert pipelines[0].stages[0].file_paths == [str(tmp_path / "vlv/a.parquet")]
    assert pipelines[1].stages[0].file_paths == [str(tmp_path / "frame-index/b.parquet")]
    for pipeline in pipelines:
        assert pipeline.checkpoint_path.name == "retry-1"
        assert pipeline.stages[2].retry_statuses == ("length",)


def _upstream_decode(tmp_path: Path) -> tuple[Path, dict, dict, Path, dict]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    formats = {"long": "webdataset_txt", "short": "webdataset_txt", "vlv": "vlv_parquet", "ego": "ego4d_recaption"}
    sources = {name: {"root": str(tmp_path / name), "format": kind} for name, kind in formats.items()}
    settings = {"sources": sources, "subset": "/frozen/subset", "shards": {name: [] for name in sources}}
    (upstream / "run.json").write_text(json.dumps(settings), encoding="utf-8")
    dedup = upstream / "dedup/clip/schema-v1"
    dedup.mkdir(parents=True)
    completed = dedup / "completed.json"
    completed.write_text(json.dumps(settings), encoding="utf-8")
    inputs = {}
    for name in sources:
        folder = upstream / "annotations/decode/schema-v1" / f"source={name}"
        folder.mkdir(parents=True)
        shard = "part.parquet" if name == "vlv" else "part.tar"
        path = folder / "part-check.parquet"
        pd.DataFrame([{"image_id": f"{name}|{shard}|a", "shard": shard, "status": "ok"}]).to_parquet(path)
        inputs[name] = str(path)
    return upstream, sources, settings, completed, inputs


def test_completed_upstream_decode_partitions_are_the_vlm_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Path(__file__).resolve().parents[4] / "tutorials/image/vlm-annotation/run.py"
    spec = importlib.util.spec_from_file_location("vlm_upstream_run", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream, sources, settings, completed, inputs = _upstream_decode(tmp_path)
    original = {path: Path(path).read_bytes() for path in inputs.values()}
    pipelines = []

    class Pipeline:
        def __init__(self, name: str) -> None:
            self.stages = []
            pipelines.append(self)

        def add_stage(self, stage: object) -> None:
            self.stages.append(stage)

        def run(self, checkpoint_path: Path) -> None:
            self.checkpoint_path = checkpoint_path

    monkeypatch.setattr(module, "Pipeline", Pipeline)
    config = {"input_run": str(upstream), "output": str(tmp_path / "output"), "fields": ["clarity"]}
    module._run_pipeline(config, "test-vlm", "http://localhost:8000/v1")
    for name, pipeline in zip(sources, pipelines, strict=True):
        partition, reader, annotation, _writer = pipeline.stages
        assert partition.file_paths == [inputs[name]]
        assert partition.file_extensions == [".parquet"]
        assert isinstance(reader, DecodeRecordImageReaderStage)
        assert reader.image_batch_size == 32
        assert annotation.source_name == name
        assert pipeline.checkpoint_path == tmp_path / f"output/checkpoints/vlm-basic/schema-v1/prompt-v2/source={name}"
    annotations = tmp_path / "output/annotations/vlm-basic/schema-v1/prompt-v2"
    run_info = json.loads((annotations / "run.json").read_text(encoding="utf-8"))
    assert run_info["batch_size"] == 32
    assert run_info["input_settings"] == settings
    assert not (tmp_path / "output/annotations/decode").exists()
    assert all(Path(path).read_bytes() == value for path, value in original.items())

    folder = annotations / "source=vlv"
    folder.mkdir()
    pd.DataFrame([{"status": "length", "image_path": f"{tmp_path / 'vlv/part.parquet'}:1"}]).to_parquet(
        folder / "part-failed.parquet"
    )
    config["retry"] = {"pass": 1, "statuses": ["length"]}
    pipelines.clear()
    module._run_pipeline(config, "test-vlm", "http://localhost:8000/v1")
    assert len(pipelines) == 1
    assert pipelines[0].stages[0].file_paths == [inputs["vlv"]]
    assert pipelines[0].checkpoint_path.name == "retry-1"
    assert pipelines[0].stages[2].retry_statuses == ("length",)

    config["batch_size"] = 64
    with pytest.raises(ValueError, match="different settings"):
        module._run_pipeline(config, "test-vlm", "http://localhost:8000/v1")
    completed.unlink()
    with pytest.raises(FileNotFoundError):
        module._run_pipeline(config, "test-vlm", "http://localhost:8000/v1")
