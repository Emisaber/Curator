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

"""Select complete source shards without reading images or captions."""

import fnmatch
import hashlib
import json
from pathlib import Path


def prepare_subset(config: dict, output: Path) -> None:
    seed = config.get("seed", 42)
    sources = config["sources"]
    rates = config.get("subset", {}).get("source_sample_rates", dict.fromkeys(sources, 1.0))
    if set(rates) != set(sources) or any(not 0 < rate <= 1 for rate in rates.values()):
        msg = "source_sample_rates must contain every source with a value in (0, 1]"
        raise ValueError(msg)
    records = []
    stats = {}
    for name, source in sources.items():
        source_format = source.get("format", "webdataset_txt")
        if source_format not in ("webdataset_txt", "parquet", "vlv_parquet", "ego4d_recaption"):
            msg = f"Unsupported source format: {source_format}"
            raise ValueError(msg)
        root = Path(source["root"])
        extension = "*.parquet" if source_format == "vlv_parquet" else "*.tar"
        candidates = sorted(path.relative_to(root).as_posix() for path in root.rglob(extension))
        patterns = source.get("exclude_patterns", [])
        eligible = [
            path
            for path in candidates
            if not any(fnmatch.fnmatch(Path(path).name.casefold(), pattern.casefold()) for pattern in patterns)
        ]
        ranked = sorted(
            eligible,
            key=lambda path: (hashlib.sha256(f"{seed}:{name}:{path}".encode()).digest(), path),
        )
        selected = sorted(ranked[: int(len(eligible) * rates[name])])
        for shard in selected:
            record = {"source": name, "shard": shard}
            if source_format == "ego4d_recaption":
                record["sample_index"] = Path(shard).with_suffix(".parquet").as_posix()
            records.append(record)
        stats[name] = {
            "available": len(candidates),
            "excluded": len(candidates) - len(eligible),
            "selected": len(selected),
            "sample_rate": rates[name],
        }
    if not records:
        msg = "No shards selected; increase the source sampling rates"
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps({"seed": seed, "sources": sources, "statistics": stats}, indent=2), encoding="utf-8"
    )
    with (output / "shards.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def load_subset(directory: Path) -> tuple[dict, dict[str, list[dict]]]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    shards = {name: [] for name in config["sources"]}
    with (directory / "shards.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            shards[record["source"]].append(record)
    return config, shards
