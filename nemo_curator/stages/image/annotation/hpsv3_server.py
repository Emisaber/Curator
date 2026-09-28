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

"""HPSv3 HTTP worker, executed with the separate HPSv3 Python interpreter."""

import argparse
import json
import tarfile
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def _read_vlv_values(items: list[dict]) -> dict[str, object]:
    """Read selected original rows once per Parquet row group and request."""
    import pyarrow.parquet as pq

    groups = defaultdict(list)
    for item in items:
        if "parquet_path" in item and isinstance(item["caption"], str) and item["caption"].strip():
            groups[item["parquet_path"], item["image_column"]].append(item)
    values = {}
    for (path, column), selected in groups.items():
        with pq.ParquetFile(path) as parquet:
            start = 0
            for group in range(parquet.num_row_groups):
                end = start + parquet.metadata.row_group(group).num_rows
                rows = [item for item in selected if start <= item["row_index"] < end]
                if rows:
                    table = parquet.read_row_group(group, columns=[column])
                    for item in rows:
                        values[item["image_id"]] = table[column][item["row_index"] - start].as_py()
                start = end
    return values


def _vlv_image(value: bytes | dict, item: dict) -> Image.Image:
    if isinstance(value, dict):
        value = value["bytes"]
    if value is None:
        msg = "Missing image bytes"
        raise ValueError(msg)
    if item["image_encoding"] == "raw_chw_uint8":
        pixels = np.frombuffer(value, dtype=np.uint8).reshape(item["image_shape"]).transpose(1, 2, 0)
        return Image.fromarray(pixels)
    if item["image_encoding"] == "encoded":
        with Image.open(BytesIO(value)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    msg = f"Unsupported VLV image encoding: {item['image_encoding']}"
    raise ValueError(msg)


@dataclass
class _PendingScore:
    rows: list[dict]
    remaining: int
    done: threading.Event
    error: Exception | None = None


class HPSv3Scorer:
    def __init__(self, inferencer: object, max_batch_size: int = 256, batch_wait_ms: float = 50):
        if max_batch_size < 1 or batch_wait_ms < 0:
            msg = "HPSv3 batch size must be positive and wait must be nonnegative"
            raise ValueError(msg)
        self.inferencer = inferencer
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000
        self.queue: Queue[tuple[_PendingScore, int, str, Image.Image] | None] = Queue(maxsize=max_batch_size * 2)
        self.worker = threading.Thread(target=self._run_batches, daemon=True)
        self.worker.start()

    def _run_batches(self) -> None:  # noqa: C901
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get(timeout=max(0, deadline - time.monotonic())))
                except Empty:
                    break
            try:
                rewards = (
                    self.inferencer.reward([item[2] for item in batch], [item[3] for item in batch]).detach().cpu()
                )
                if len(rewards) != len(batch):
                    msg = "HPSv3 returned a different number of scores than images"
                    raise RuntimeError(msg)  # noqa: TRY301 -- record model failures on each pending request
                for (pending, position, _, _), reward in zip(batch, rewards, strict=True):
                    pending.rows[position].update(status="ok", score_mu=float(reward[0]), score_sigma=float(reward[1]))
            except Exception as exc:  # noqa: BLE001
                for pending, _, _, _ in batch:
                    pending.error = exc
            finally:
                for pending, _, _, _ in batch:
                    pending.remaining -= 1
                    if pending.remaining == 0:
                        pending.done.set()

    def close(self) -> None:
        self.queue.put(None)
        self.worker.join()

    def score(self, items: list[dict]) -> list[dict]:  # noqa: C901, PLR0915
        rows = []
        images = []
        prompts = []
        positions = []
        vlv_values = _read_vlv_values(items) if any("parquet_path" in item for item in items) else {}
        with ExitStack() as stack:
            archives = {}
            for item in items:
                row = {
                    "image_id": item["image_id"],
                    "status": "failed",
                    "score_mu": None,
                    "score_sigma": None,
                    "error": None,
                }
                rows.append(row)
                caption = item["caption"]
                if not isinstance(caption, str) or not caption.strip():
                    row["status"] = "skipped"
                    row["error"] = "empty_caption"
                    continue
                if "parquet_path" in item:
                    try:
                        image = _vlv_image(vlv_values[item["image_id"]], item)
                    except (ValueError, TypeError, OSError, Image.DecompressionBombError) as exc:
                        row["error"] = str(exc)
                        continue
                else:
                    path = item["tar_path"]
                    if path not in archives:
                        archives[path] = stack.enter_context(Path(path).open("rb"))
                    archive = archives[path]
                    size = item["size"]
                    if size is None:
                        archive.seek(item["offset"] - tarfile.BLOCKSIZE)
                        header = tarfile.TarInfo.frombuf(archive.read(tarfile.BLOCKSIZE), "utf-8", "surrogateescape")
                        size = header.size
                    archive.seek(item["offset"])
                    content = archive.read(size)
                    if len(content) != size:
                        msg = "TAR member is truncated"
                        raise OSError(msg)
                    try:
                        with Image.open(BytesIO(content)) as decoded:
                            decoded.load()
                            oriented = (
                                ImageOps.exif_transpose(decoded) if item.get("adjust_orientation", True) else decoded
                            )
                            image = oriented.convert("RGB")
                    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                        row["error"] = str(exc)
                        continue
                images.append(image)
                prompts.append(caption)
                positions.append(len(rows) - 1)

        if images:
            pending = _PendingScore(rows=rows, remaining=len(images), done=threading.Event())
            for position, prompt, image in zip(positions, prompts, images, strict=True):
                self.queue.put((pending, position, prompt, image))
            pending.done.wait()
            if pending.error is not None:
                raise pending.error
        return rows


class HPSv3HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], scorer: HPSv3Scorer, instance_id: str = ""):
        super().__init__(address, HPSv3RequestHandler)
        self.scorer = scorer
        self.instance_id = instance_id
        self.condition = threading.Condition()
        self.in_flight = 0
        self.draining = False


class HPSv3RequestHandler(BaseHTTPRequestHandler):
    def _send(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        with self.server.condition:
            draining = self.server.draining
            in_flight = self.server.in_flight
        self._send(
            HTTPStatus.OK,
            {"ready": not draining, "in_flight": in_flight, "instance_id": self.server.instance_id},
        )

    def do_POST(self) -> None:
        if self.path == "/score":
            self._score()
        elif self.path == "/drain":
            self._drain()
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _score(self) -> None:
        with self.server.condition:
            if self.server.draining:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "draining"})
                return
            self.server.in_flight += 1
        try:
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                items = payload["items"]
                if not isinstance(items, list):
                    msg = "items must be a list"
                    raise TypeError(msg)  # noqa: TRY301 -- malformed payloads return HTTP 400
            except (KeyError, TypeError, ValueError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            try:
                self._send(HTTPStatus.OK, {"items": self.server.scorer.score(items)})
            except Exception as exc:  # noqa: BLE001
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        finally:
            with self.server.condition:
                self.server.in_flight -= 1
                self.server.condition.notify_all()

    def _drain(self) -> None:
        with self.server.condition:
            self.server.draining = True
            self.server.condition.wait_for(lambda: self.server.in_flight == 0)
        self._send(HTTPStatus.OK, {"drained": True})
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        print(f"HPSv3 HTTP: {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--batch-wait-ms", type=float, default=50)
    parser.add_argument("--read-parquet", action="store_true")
    args = parser.parse_args()

    if args.read_parquet:
        import pyarrow.parquet  # noqa: F401 -- fail at startup when the configured source requires Parquet

    from hpsv3 import HPSv3RewardInferencer

    inferencer = HPSv3RewardInferencer(config_path=args.model_config, checkpoint_path=args.checkpoint, device="cuda:0")
    scorer = HPSv3Scorer(inferencer, args.max_batch_size, args.batch_wait_ms)
    server = HPSv3HTTPServer((args.host, args.port), scorer, args.instance_id)
    print(f"HPSV3_ENDPOINT={json.dumps({'port': server.server_port, 'instance_id': args.instance_id})}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        scorer.close()


if __name__ == "__main__":
    main()
