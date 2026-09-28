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

"""Job-scoped HPSv3 model processes on Ray-assigned GPUs."""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import ray
from loguru import logger

from nemo_curator.core.serve.constants import PLACEMENT_GROUP_READY_TIMEOUT_S
from nemo_curator.core.serve.placement import (
    build_pg,
    get_bundle_node_ip,
    remove_named_pgs_with_prefix,
)
from nemo_curator.core.serve.subprocess_mgr import (
    ManagedSubprocess,
    reacquire_detached_actor_handles,
    sweep_orphan_actors_by_prefix,
)


@dataclass
class HPSv3Service:
    python: str
    checkpoint: str
    model_config: str | None
    replicas: int
    namespace: str
    actor_prefix: str
    log_dir: str
    start_timeout: float = 600.0
    drain_timeout: float = 120.0
    max_batch_size: int = 256
    batch_wait_ms: float = 50.0
    read_parquet: bool = False
    processes: list[ManagedSubprocess] = field(default_factory=list, init=False)
    endpoints: list[str] = field(default_factory=list, init=False)
    placement_group: object | None = field(default=None, init=False)

    def start(self) -> list[str]:
        if self.replicas < 1:
            msg = "HPSv3 replicas must be positive"
            raise ValueError(msg)
        if not ray.is_initialized():
            ray.init(address="auto", namespace=self.namespace)
        self.placement_group = build_pg(
            [{"CPU": 1, "GPU": 1} for _ in range(self.replicas)],
            "SPREAD",
            name=self.actor_prefix,
            bundle_label_selector=None,
            ready_timeout_s=PLACEMENT_GROUP_READY_TIMEOUT_S,
        )
        script = str(Path(__file__).with_name("hpsv3_server.py"))
        try:
            starting = []
            for index in range(self.replicas):
                host = get_bundle_node_ip(self.placement_group, index)
                instance_id = uuid.uuid4().hex
                command = [
                    self.python,
                    script,
                    "--host",
                    host,
                    "--port",
                    "0",
                    "--checkpoint",
                    self.checkpoint,
                    "--instance-id",
                    instance_id,
                    "--max-batch-size",
                    str(self.max_batch_size),
                    "--batch-wait-ms",
                    str(self.batch_wait_ms),
                ]
                if self.model_config:
                    command.extend(["--model-config", self.model_config])
                if self.read_parquet:
                    command.append("--read-parquet")
                process = ManagedSubprocess.spawn(
                    f"worker{index}",
                    self.placement_group,
                    index,
                    num_gpus=1,
                    command=command,
                    runtime_dir=self.log_dir,
                    actor_name_prefix=self.actor_prefix,
                    subprocess_env={"PYTHONPATH": ""},
                )
                self.processes.append(process)
                starting.append((host, instance_id, process))
            for host, instance_id, process in starting:
                endpoint = self._wait_ready(host, instance_id, process)
                if endpoint in self.endpoints:
                    msg = f"HPSv3 replicas share an endpoint: {endpoint}"
                    raise RuntimeError(msg)  # noqa: TRY301 -- clean up every spawned replica on startup failure
                self.endpoints.append(endpoint)
        except Exception:
            try:
                self.stop(raise_on_drain_failure=False)
            except Exception:  # noqa: BLE001
                logger.exception("HPSv3 startup cleanup failed")
            raise
        return self.endpoints

    def _wait_ready(self, host: str, instance_id: str, process: ManagedSubprocess) -> str:
        deadline = time.monotonic() + self.start_timeout
        endpoint = None
        while time.monotonic() < deadline:
            if not process.is_alive():
                msg = f"HPSv3 exited during startup: {process.read_log_tail()}"
                raise RuntimeError(msg)
            if endpoint is None:
                for line in process.read_log_tail().splitlines():
                    if line.startswith("HPSV3_ENDPOINT="):
                        announced = json.loads(line.removeprefix("HPSV3_ENDPOINT="))
                        if announced["instance_id"] == instance_id:
                            endpoint = f"http://{host}:{announced['port']}"
                            break
            try:
                if endpoint is not None:
                    with urlopen(f"{endpoint}/health", timeout=2) as response:  # noqa: S310
                        health = json.load(response)
                    if health.get("ready") and health.get("instance_id") == instance_id:
                        return endpoint
            except (OSError, URLError, ValueError):
                pass
            time.sleep(1)
        msg = f"HPSv3 did not become ready: {process.read_log_tail()}"
        raise TimeoutError(msg)

    def stop(self, *, raise_on_drain_failure: bool = True) -> None:
        drain_errors = []
        for endpoint in self.endpoints:
            try:
                request = Request(f"{endpoint}/drain", data=b"{}", headers={"Content-Type": "application/json"})  # noqa: S310
                with urlopen(request, timeout=self.drain_timeout) as response:  # noqa: S310
                    if not json.load(response).get("drained"):
                        msg = "HPSv3 service did not confirm drain"
                        raise RuntimeError(msg)  # noqa: TRY301 -- release resources even when drain fails
            except Exception as exc:  # noqa: BLE001
                drain_errors.append(f"{endpoint}: {exc}")
        if self.placement_group is not None:
            if not ray.is_initialized():
                ray.init(address="auto", namespace=self.namespace)
            processes = reacquire_detached_actor_handles(
                self.processes, actor_name_prefix=self.actor_prefix, namespace=self.namespace
            )
            try:
                ManagedSubprocess.stop_many(processes)
                sweep_orphan_actors_by_prefix(prefix=self.actor_prefix, namespace=self.namespace)
            finally:
                remove_named_pgs_with_prefix(self.actor_prefix)
                self.placement_group = None
                self.processes.clear()
                self.endpoints.clear()
        if drain_errors:
            message = f"HPSv3 service drain failed: {'; '.join(drain_errors)}"
            if raise_on_drain_failure:
                raise RuntimeError(message)
            logger.warning(message)
