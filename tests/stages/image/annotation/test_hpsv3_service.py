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

import io
from pathlib import Path
from unittest.mock import Mock
from urllib.request import Request

import pytest

from nemo_curator.stages.image.annotation import hpsv3_service


def test_service_drains_before_releasing_ray_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = []
    service = hpsv3_service.HPSv3Service(
        python="/unused/python",
        checkpoint="/unused/checkpoint",
        model_config=None,
        replicas=1,
        namespace="test_hpsv3",
        actor_prefix="test_hpsv3_run",
        log_dir=str(tmp_path),
    )
    service.endpoints = ["http://127.0.0.1:18080"]
    service.processes = [object()]
    service.placement_group = object()

    def drain(request: Request, *, timeout: float) -> io.BytesIO:
        assert request.full_url.endswith("/drain")
        assert timeout == service.drain_timeout
        events.append("drain")
        return io.BytesIO(b'{"drained": true}')

    monkeypatch.setattr(hpsv3_service, "urlopen", drain)
    monkeypatch.setattr(hpsv3_service.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(hpsv3_service, "reacquire_detached_actor_handles", lambda *_args, **_kwargs: service.processes)
    monkeypatch.setattr(hpsv3_service.ManagedSubprocess, "stop_many", lambda _processes: events.append("stop"))
    monkeypatch.setattr(hpsv3_service, "sweep_orphan_actors_by_prefix", lambda **_kwargs: events.append("sweep"))
    monkeypatch.setattr(hpsv3_service, "remove_named_pgs_with_prefix", lambda _prefix: events.append("release"))

    service.stop()

    assert events == ["drain", "stop", "sweep", "release"]
    assert service.endpoints == []
    assert service.processes == []
    assert service.placement_group is None


def test_ready_uses_port_announced_by_matching_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = hpsv3_service.HPSv3Service(
        python="/unused/python",
        checkpoint="/unused/checkpoint",
        model_config=None,
        replicas=1,
        namespace="test_hpsv3",
        actor_prefix="test_hpsv3_run",
        log_dir=str(tmp_path),
    )

    class _Process:
        def is_alive(self) -> bool:
            return True

        def read_log_tail(self) -> str:
            return 'HPSV3_ENDPOINT={"port": 24567, "instance_id": "expected"}'

    def health(url: str, *, timeout: float) -> io.BytesIO:
        assert url == "http://127.0.0.1:24567/health"
        assert timeout == 2
        return io.BytesIO(b'{"ready": true, "instance_id": "expected"}')

    monkeypatch.setattr(hpsv3_service, "urlopen", health)
    assert service._wait_ready("127.0.0.1", "expected", _Process()) == "http://127.0.0.1:24567"


def test_replicas_spawn_before_waiting_and_use_independent_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    service = hpsv3_service.HPSv3Service(
        python="/separate/env/bin/python",
        checkpoint="/models/hps",
        model_config=None,
        replicas=2,
        namespace="test",
        actor_prefix="test_run",
        log_dir=str(tmp_path),
        read_parquet=True,
    )
    monkeypatch.setattr(hpsv3_service.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(hpsv3_service, "build_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(hpsv3_service, "get_bundle_node_ip", lambda *_args: "host")

    def spawn(label: str, _pg: object, index: int, **kwargs) -> Mock:
        events.append("spawn")
        assert label == f"worker{index}"
        assert kwargs["command"][0] == service.python
        assert "--read-parquet" in kwargs["command"]
        assert kwargs["subprocess_env"] == {"PYTHONPATH": ""}
        return Mock(index=index)

    def ready(_host: str, _instance_id: str, process: Mock) -> str:
        events.append("ready")
        return f"http://host:{8000 + process.index}"

    monkeypatch.setattr(hpsv3_service.ManagedSubprocess, "spawn", spawn)
    monkeypatch.setattr(service, "_wait_ready", ready)
    assert service.start() == ["http://host:8000", "http://host:8001"]
    assert events == ["spawn", "spawn", "ready", "ready"]
