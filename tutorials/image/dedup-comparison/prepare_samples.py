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

"""Select reproducible records and write small manifests for the image reader."""

import argparse
import hashlib
import heapq
import json
import tarfile
from pathlib import Path

import pandas as pd

from nemo_curator.stages.image.io.caption_source import CaptionSource, read_parquet_captions
from nemo_curator.utils.image_id import make_image_id

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_sources(values: list[str]) -> dict[str, str]:
    """Parse NAME=/path entries; the name becomes part of each sample's ID."""
    roots = {}
    for value in values:
        name, path = value.split("=", 1)
        if not name or name in roots:
            msg = f"Source names must be nonempty and unique: {name!r}"
            raise ValueError(msg)
        roots[name] = str(Path(path).resolve(strict=True))
    return roots


def _resolve_captions(
    selected: list[dict], source_roots: dict[str, str], caption_sources: dict[str, CaptionSource]
) -> dict[str, str | None]:
    captions: dict[str, str | None] = {}
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected:
        key = row["image_id"].rsplit("|", 1)[-1]
        grouped.setdefault((row["source"], row["shard"]), {})[key] = row["image_id"]

    for (source_name, relative_tar), keys in grouped.items():
        source = caption_sources[source_name]
        tar_path = Path(source_roots[source_name]) / relative_tar
        if source.format == "parquet":
            found = read_parquet_captions(tar_path, Path(source.root), set(keys), source.caption_column)
        else:
            found = {}
            with tarfile.open(tar_path, mode="r:") as archive:
                for member in archive:
                    key = member.name.rsplit(".", 1)[0]
                    if not member.isfile() or key not in keys or Path(member.name).suffix.lower() != ".txt":
                        continue
                    content = archive.extractfile(member)
                    found[key] = content.read().decode("utf-8") if content is not None else None
        captions.update({keys[key]: found.get(key) for key in keys})
    return captions


def prepare_samples(  # noqa: PLR0913
    source_roots: dict[str, str],
    output_dir: str,
    source_sample_rates: dict[str, float] | None = None,
    batch_size: int = 32,
    seed: int = 42,
    samples_path: str | None = None,
    caption_sources: dict[str, CaptionSource] | None = None,
) -> dict:
    """Sample each source by its configured fraction using stable priorities."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rates = source_sample_rates or dict.fromkeys(source_roots, 1.0)
    if set(rates) != set(source_roots) or any(not 0 < rate <= 1 for rate in rates.values()):
        raise ValueError("source_sample_rates must contain every source with a value in (0, 1]")
    captions = caption_sources or {name: CaptionSource(root) for name, root in source_roots.items()}
    if set(captions) != set(source_roots):
        raise ValueError("caption_sources must contain every source")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifests = output / "manifests"
    manifests.mkdir()
    selected = []
    source_stats = {}
    for source, root_path in sorted(source_roots.items()):
        root = Path(root_path)
        shards = sorted(root.rglob("*.tar"), key=lambda path: path.relative_to(root).as_posix())
        available = 0
        for shard in shards:
            with tarfile.open(shard, mode="r:") as archive:
                available += sum(
                    member.isfile() and Path(member.name).suffix.lower() in _IMAGE_EXTENSIONS for member in archive
                )
        quota = int(available * rates[source])
        heap = []
        for shard in shards:
            relative = shard.relative_to(root).as_posix()
            with tarfile.open(shard, mode="r:") as archive:
                for member in archive:
                    if quota == 0 or not member.isfile() or Path(member.name).suffix.lower() not in _IMAGE_EXTENSIONS:
                        continue
                    row = {
                        "image_id": make_image_id(source, relative, member.name),
                        "source": source,
                        "shard": relative,
                        "member": member.name,
                        "offset": member.offset_data,
                        "size": member.size,
                    }
                    identity = json.dumps([source, relative, member.name, member.offset_data], ensure_ascii=False)
                    sampling_id = hashlib.sha256(identity.encode()).hexdigest()
                    priority = int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest(), "big")
                    item = (-priority, sampling_id, row)
                    if len(heap) < quota:
                        heapq.heappush(heap, item)
                    elif item[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, item)
        selected.extend(item[2] for item in heap)
        source_stats[source] = {
            "shards": [p.relative_to(root).as_posix() for p in shards],
            "available": available,
            "selected": len(heap),
            "sample_rate": rates[source],
        }
    if not selected:
        msg = "No supported image members found in the selected TAR files"
        raise ValueError(msg)
    samples = pd.DataFrame(selected).sort_values(["source", "shard", "offset"], ignore_index=True)
    caption_map = _resolve_captions(selected, source_roots, captions)
    samples["caption_raw"] = pd.Series(
        [caption_map.get(image_id) for image_id in samples["image_id"]], dtype="string"
    )
    sample_index = Path(samples_path) if samples_path is not None else output / "samples.parquet"
    sample_index.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(sample_index, index=False)
    for source in sorted(samples["source"].unique()):
        source_manifest = manifests / f"source={source}"
        source_manifest.mkdir()
        source_samples = samples[samples["source"] == source].reset_index(drop=True)
        for start in range(0, len(source_samples), batch_size):
            source_samples.iloc[start : start + batch_size].to_parquet(
                source_manifest / f"part-{start:08d}.parquet", index=False
            )
    config = {
        "source_roots": source_roots,
        "caption_sources": {
            name: {
                "format": source.format,
                "caption_column": source.caption_column,
            }
            for name, source in captions.items()
        },
        "selected": len(samples),
        "batch_size": batch_size,
        "source_sample_rates": rates,
        "seed": seed,
        "sources": source_stats,
    }
    (output / "samples.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True, help="NAME=/absolute/path, one per source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-sample-rate", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rates = {name: float(rate) for name, rate in (value.split("=", 1) for value in args.source_sample_rate)}
    result = prepare_samples(parse_sources(args.source), args.output, rates or None, args.batch_size, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
