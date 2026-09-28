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

"""Build a portable, offline review of pHash and native CLIP candidate pairs."""

import argparse
import hashlib
import json
import random
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

_CLIP_THRESHOLD = 0.99
_CLIP_BOUNDARY = 0.98


def _artifact_path(output: Path, current: str, legacy: str) -> Path:
    path = output / current
    return path if path.exists() else output / legacy


def read_parquets(directory: Path) -> pd.DataFrame:
    files = sorted(directory.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True) if files else pd.DataFrame()


def collect_candidates(output: Path) -> tuple[list[dict], dict]:
    threshold = json.loads((output / "run.json").read_text())["phash_threshold"]
    features = read_parquets(_artifact_path(output, "features/clip/schema-v1", "features")).set_index("image_id")
    hashes = {key: int(meta["phash"], 16) for key, meta in features["metadata"].items() if "phash" in meta}
    pairs = {}
    phash = read_parquets(output / "phash")
    for row in phash.to_dict("records"):
        a, b = sorted((row["id_a"], row["id_b"]))
        pairs[(a, b)] = {
            "id_a": a,
            "id_b": b,
            "phash_hit": bool(row["matched"]),
            "clip_hit": False,
            "hamming_distance": int(row["hamming_distance"]),
            "clip_score": None,
            "clip_from": None,
            "clip_to": None,
            "clip_diagnostic": None,
        }
    clip = read_parquets(
        _artifact_path(output, "dedup/clip/schema-v1/cache/pairwise_results", "clip/cache/pairwise_results")
    )
    for row in clip.to_dict("records"):
        if row["id"] == row["max_id"]:
            continue
        a, b = sorted((row["id"], row["max_id"]))
        if row["cosine_sim_score"] < _CLIP_BOUNDARY and (a, b) not in pairs:
            continue
        pair = pairs.setdefault(
            (a, b),
            {
                "id_a": a,
                "id_b": b,
                "phash_hit": False,
                "hamming_distance": None,
                "clip_diagnostic": None,
            },
        )
        pair.update(
            clip_hit=bool(row["cosine_sim_score"] >= _CLIP_THRESHOLD),
            clip_score=float(row["cosine_sim_score"]),
            clip_from=row["id"],
            clip_to=row["max_id"],
        )
        if hashes:
            pair["hamming_distance"] = (hashes[a] ^ hashes[b]).bit_count()
            pair["phash_hit"] = pair["hamming_distance"] <= threshold
    removed = annotate_candidates(output, pairs, features)
    return list(pairs.values()), {
        "decoded": len(features),
        "phash_hit_pairs": sum(pair["phash_hit"] for pair in pairs.values()),
        "clip_hit_pairs": sum(pair["clip_hit"] for pair in pairs.values()),
        "clip_duplicate_ids": len(removed),
        "candidate_pairs": len(pairs),
    }


def annotate_candidates(output: Path, pairs: dict[tuple[str, str], dict], features: pd.DataFrame) -> set[str]:
    clusters = {}
    kmeans_path = _artifact_path(output, "dedup/clip/schema-v1/cache/kmeans_results", "clip/cache/kmeans_results")
    for path in sorted(kmeans_path.glob("centroid=*/*.parquet")):
        clusters.update(dict.fromkeys(pd.read_parquet(path, columns=["image_id"])["image_id"], path.parent.name))
    for (a, b), pair in pairs.items():
        pair["clip_status"] = "not_run"
        if pair["clip_score"] is not None:
            pair["clip_status"] = "matched" if pair["clip_hit"] else "below_threshold"
        elif a in clusters and b in clusters:
            pair["clip_status"] = "cross_cluster" if clusters[a] != clusters[b] else "not_best_match"
        if "embedding" in features and pair["clip_score"] is None:
            pair["clip_diagnostic"] = float(
                np.dot(
                    np.asarray(features.loc[a, "embedding"], dtype=np.float32),
                    np.asarray(features.loc[b, "embedding"], dtype=np.float32),
                )
            )
    duplicates = read_parquets(_artifact_path(output, "dedup/clip/schema-v1/duplicates", "clip/duplicates"))
    removed = set(duplicates["id"]) if "id" in duplicates else set()
    groups = candidate_groups(list(pairs.values()))
    for (a, b), pair in pairs.items():
        pair.update(clip_removed_a=a in removed, clip_removed_b=b in removed, group_a=groups[a], group_b=groups[b])
    return removed


def candidate_groups(pairs: list[dict]) -> dict[str, str]:
    """Connected candidate groups provide browsing context, not transitive identity."""
    parents = {pair[key]: pair[key] for pair in pairs for key in ("id_a", "id_b")}

    def find(image_id: str) -> str:
        while parents[image_id] != image_id:
            parents[image_id] = parents[parents[image_id]]
            image_id = parents[image_id]
        return image_id

    for pair in pairs:
        if pair["phash_hit"] or pair["clip_hit"]:
            a, b = sorted((find(pair["id_a"]), find(pair["id_b"])))
            parents[b] = a
    return {image_id: find(image_id) for image_id in parents}


def sample_pairs(
    pairs: list[dict], budget: int = 200, seed: int = 42, mode: str = "both", max_distance: int = 8
) -> list[dict]:
    """Keep random-hit cohorts distinct from disagreement and boundary inspection."""
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    both = mode == "both"
    cohorts = {
        "phash_random": [i for i, pair in enumerate(pairs) if pair["phash_hit"]],
        "clip_random": [i for i, pair in enumerate(pairs) if pair["clip_hit"]],
        "phash_only": [i for i, pair in enumerate(pairs) if both and pair["phash_hit"] and not pair["clip_hit"]],
        "clip_only": [i for i, pair in enumerate(pairs) if both and pair["clip_hit"] and not pair["phash_hit"]],
        "phash_boundary": [
            i
            for i, pair in enumerate(pairs)
            if pair["hamming_distance"] is not None
            and pair["hamming_distance"] <= max_distance
            and not pair["phash_hit"]
        ],
        "clip_boundary": [
            i
            for i, pair in enumerate(pairs)
            if pair["clip_score"] is not None and _CLIP_BOUNDARY <= pair["clip_score"] < _CLIP_THRESHOLD
        ],
    }
    selected = {}
    weights = (0.3, 0.3, 0.1, 0.1, 0.1, 0.1)
    groups = candidate_groups(pairs)
    for (cohort, indices), weight in zip(cohorts.items(), weights, strict=True):
        shuffled = rng.sample(indices, len(indices))
        if not cohort.endswith("random"):
            counts = {}
            priorities = {}
            for index in shuffled:
                group = (groups[pairs[index]["id_a"]], groups[pairs[index]["id_b"]])
                priorities[index] = counts.get(group, 0)
                counts[group] = priorities[index] + 1
            shuffled.sort(key=priorities.__getitem__)
        for index in shuffled[: int(budget * weight)]:
            selected.setdefault(index, []).append(cohort)
    remainder = sorted(set(range(len(pairs))) - selected.keys())
    for index in rng.sample(remainder, min(len(remainder), budget - len(selected))):
        selected[index] = ["extra"]
    return [{**pairs[index], "cohorts": labels} for index, labels in selected.items()]


def export_assets(output: Path, pairs: list[dict]) -> dict:
    samples_root = _artifact_path(output, "samples/schema-v1", "input")
    config = json.loads((samples_root / "samples.json").read_text(encoding="utf-8"))
    records = pd.read_parquet(samples_root / ("manifest.parquet" if (samples_root / "manifest.parquet").exists() else "sampled_images.parquet")).set_index(
        "image_id"
    )
    decoded = read_parquets(_artifact_path(output, "annotations/decode/schema-v1", "annotations/decode/v1")).set_index(
        "image_id"
    )
    assets = output / "report/assets"
    assets.mkdir(parents=True, exist_ok=True)
    images = {}
    for image_id in sorted({pair[key] for pair in pairs for key in ("id_a", "id_b")}):
        row = records.loc[image_id]
        with (Path(config["source_roots"][row["source"]]) / row["shard"]).open("rb") as archive:
            archive.seek(int(row["offset"]))
            content = archive.read(int(row["size"]))
        asset_stem = hashlib.sha256(image_id.encode()).hexdigest()[:20]
        filename = asset_stem + Path(row["member"]).suffix.lower()
        (assets / filename).write_bytes(content)
        with Image.open(BytesIO(content)) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((640, 640))
            preview.save(assets / f"{asset_stem}.preview.jpg", quality=90)
        images[image_id] = {
            "source": row["source"],
            "member": row["member"],
            "shard": row["shard"],
            "width": int(decoded.loc[image_id, "width"]),
            "height": int(decoded.loc[image_id, "height"]),
            "original": f"assets/{filename}",
            "preview": f"assets/{asset_stem}.preview.jpg",
        }
    return images


def create_report(output: Path, budget: int = 200, seed: int = 42) -> Path:
    if budget < 1:
        msg = "Review pair budget must be positive"
        raise ValueError(msg)
    pairs, stats = collect_candidates(output)
    report = output / "report"
    report.mkdir(exist_ok=True)
    # Persist the full candidate union; HTML contains only the review sample.
    pd.DataFrame(
        pairs,
        columns=[
            "id_a",
            "id_b",
            "phash_hit",
            "clip_hit",
            "hamming_distance",
            "clip_score",
            "clip_from",
            "clip_to",
            "clip_diagnostic",
            "clip_status",
            "clip_removed_a",
            "clip_removed_b",
            "group_a",
            "group_b",
        ],
    ).to_parquet(report / "candidates.parquet", index=False)
    config = json.loads((output / "run.json").read_text(encoding="utf-8"))
    selected = sample_pairs(pairs, budget, seed, config["mode"], config["phash_max_distance"])
    records = read_parquets(_artifact_path(output, "annotations/decode/schema-v1", "annotations/decode/v1"))
    stats["decode_failures"] = int((records["status"] == "failed").sum())
    stats["review_pairs"] = len(selected)
    data = {
        "run": json.loads((output / "run.json").read_text(encoding="utf-8")),
        "stats": stats,
        "pairs": selected,
        "images": export_assets(output, selected),
    }
    (report / "data.js").write_text(
        "window.REVIEW = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8"
    )
    (report / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("review.html"), report / "index.html")
    print(f"Review: {report / 'index.html'}")
    return report / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Existing experiment directory")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_report(args.output, args.pairs, args.seed)
