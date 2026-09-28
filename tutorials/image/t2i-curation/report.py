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

"""Static visual review of persisted NSFW and HPSv3 scores."""

import argparse
import hashlib
import html
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from nemo_curator.stages.image.io.caption_lookup import resolve_captions
from nemo_curator.stages.image.io.caption_source import CaptionSource


def _sample_scores(input_run: Path, per_group: int, seed: int) -> tuple[dict, dict]:  # noqa: C901, PLR0912
    rng = random.Random(seed)  # noqa: S311 -- reproducible visual review
    summary = {}
    samples = defaultdict(list)
    for stage, field in (("nsfw", "nsfw_score"), ("hpsv3", "score_mu")):
        for path in sorted((input_run / f"annotations/{stage}/schema-v1").glob("source=*/*.parquet")):
            source = path.parent.name.removeprefix("source=")
            key = (stage, source)
            stats = summary.setdefault(
                key, {"statuses": defaultdict(int), "count": 0, "sum": 0.0, "min": None, "max": None}
            )
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    status = row["status"]
                    stats["statuses"][status] += 1
                    if status == "ok":
                        value = row[field]
                        stats["count"] += 1
                        stats["sum"] += value
                        stats["min"] = value if stats["min"] is None else min(stats["min"], value)
                        stats["max"] = value if stats["max"] is None else max(stats["max"], value)
                    bucket = samples[stage, source, status]
                    limit = max(256, per_group * 32) if status == "ok" else per_group
                    count = stats["statuses"][status]
                    if len(bucket) < limit:
                        bucket.append(row)
                    else:
                        slot = rng.randrange(count)
                        if slot < limit:
                            bucket[slot] = row
    groups = {}
    for (stage, source, status), rows in samples.items():
        if status != "ok":
            groups[stage, source, status] = rows
            continue
        field = "nsfw_score" if stage == "nsfw" else "score_mu"
        rows.sort(key=lambda row: row[field])
        for index, label in enumerate(("low", "middle", "high")):
            section = rows[index * len(rows) // 3 : (index + 1) * len(rows) // 3]
            groups[stage, source, label] = rng.sample(section, min(per_group, len(section)))
        values = [row[field] for row in rows]
        lower, upper = values[0], values[-1]
        bins = [0] * 10
        for value in values:
            index = min(9, int((value - lower) / (upper - lower) * 10)) if upper != lower else 0
            bins[index] += 1
        summary[stage, source].update(histogram=bins, histogram_range=[lower, upper], histogram_samples=len(values))
    for stats in summary.values():
        total = stats.pop("sum")
        stats["mean"] = total / stats["count"] if stats["count"] else None
    return groups, summary


def _histogram(stats: dict) -> str:
    bins = stats.get("histogram", [])
    if not bins:
        return ""
    peak = max(bins)
    bars = "".join(
        f'<rect x="{index * 30}" y="{80 - count / peak * 75:.1f}" width="25" '
        f'height="{count / peak * 75:.1f}"><title>{count}</title></rect>'
        for index, count in enumerate(bins)
    )
    lower, upper = stats["histogram_range"]
    return (
        f'<svg viewBox="0 0 300 80" class="histogram" role="img" aria-label="Sampled score distribution">{bars}</svg>'
        f'<p class="muted">{lower:.4g} - {upper:.4g} · distribution from {stats["histogram_samples"]} sampled scores</p>'
    )


def create_report(input_run: Path, output: Path, per_group: int = 8, seed: int = 42) -> Path:  # noqa: C901, PLR0912
    settings = json.loads((input_run / "run.json").read_text(encoding="utf-8"))
    sources = {name: CaptionSource(**source) for name, source in settings["sources"].items()}
    groups, summary = _sample_scores(input_run, per_group, seed)
    ids = {row["image_id"] for rows in groups.values() for row in rows}
    captions = resolve_captions(ids, sources)
    spec = importlib.util.spec_from_file_location(
        "caption_review", Path(__file__).parents[1] / "caption-dedup/report.py"
    )
    previews = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previews)
    assets = output / "assets"
    images = previews._export_images(ids, sources, assets)
    vlm = {}
    vlm_root = input_run / "annotations/vlm-basic/schema-v1"
    vlm_config = input_run / "configs/vlm.json"
    if vlm_config.exists():
        prompt_version = json.loads(vlm_config.read_text(encoding="utf-8")).get("prompt_version", "v2")
        vlm_paths = [vlm_root / f"prompt-{prompt_version}"]
    else:
        vlm_paths = sorted(vlm_root.glob("prompt-*"))
        if len(vlm_paths) > 1:
            msg = "Review requires configs/vlm.json to select one VLM prompt version"
            raise ValueError(msg)
    for path in sorted(path for directory in vlm_paths for path in directory.glob("source=*/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            vlm.update((row["image_id"], row) for row in batch.to_pylist() if row["image_id"] in ids)
    sections = []
    for (stage, source), stats in sorted(summary.items()):
        blocks = []
        for label in ("low", "middle", "high", *sorted(stats["statuses"].keys() - {"ok"})):
            cards = []
            for row in groups.get((stage, source, label), []):
                image_id = row["image_id"]
                caption = captions[image_id]
                if stage == "hpsv3" and row["caption_sha256"] is not None:
                    digest = hashlib.sha256(caption.encode("utf-8")).hexdigest() if isinstance(caption, str) else None
                    if digest != row["caption_sha256"]:
                        msg = f"Review caption differs from the scored caption: {image_id}"
                        raise ValueError(msg)
                details = html.escape(json.dumps(row, indent=2, ensure_ascii=False))
                if image_id in vlm:
                    annotation = {
                        key: value for key, value in vlm[image_id].items() if key not in ("raw_response", "image_path")
                    }
                    details += "\n\nVLM\n" + html.escape(json.dumps(annotation, indent=2, ensure_ascii=False))
                cards.append(
                    f'<article><img src="{images[image_id]}" loading="lazy" alt="">'
                    f'<div class="body"><code>{html.escape(image_id)}</code>'
                    f'<p class="caption">{html.escape(caption or "[no caption]")}</p>'
                    f"<details open><summary>Scores and annotations</summary><pre>{details}</pre></details></div></article>"
                )
            if cards:
                blocks.append(f'<h3>{html.escape(label)}</h3><div class="grid">{"".join(cards)}</div>')
        sections.append(
            f"<section><h2>{html.escape(stage)} / {html.escape(source)}</h2>"
            f"<p>{html.escape(json.dumps(stats['statuses']))} · mean {stats['mean']} · "
            f"min {stats['min']} · max {stats['max']}</p>{_histogram(stats)}{''.join(blocks)}</section>"
        )
    expected = defaultdict(int)
    for path in sorted((input_run / "annotations/decode/schema-v1").glob("source=*/*.parquet")):
        source = path.parent.name.removeprefix("source=")
        for batch in pq.ParquetFile(path).iter_batches(columns=["status"]):
            expected[source] += sum(status == "ok" for status in batch.column(0).to_pylist())
    result = {
        "decode_success": dict(expected),
        "scores": {f"{stage}/{source}": stats for (stage, source), stats in summary.items()},
        "seed": seed,
        "per_group": per_group,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>T2I score review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#25322c;background:#f8f9f7}*{box-sizing:border-box}
body{margin:0}main{max-width:1500px;margin:auto;padding:40px 28px}h1{font-size:38px;letter-spacing:-.04em}
header,section{padding-bottom:24px;border-bottom:1px solid #dce3df}section{margin:36px 0}h2{font-size:23px}
h3{font-size:15px;margin-top:28px}.muted,code{color:#6d7d73;font-size:12px}.histogram{width:300px;fill:#648e7b}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
article{background:white;border:1px solid #e0e6e2;border-radius:12px;overflow:hidden;min-width:0}
img{width:100%;height:260px;object-fit:contain;background:#f0f2f0}.body{padding:16px}code{overflow-wrap:anywhere}
.caption{font-size:13px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
summary{cursor:pointer;font-size:12px}pre{font-size:11px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:650px){main{padding:24px 14px}.grid{grid-template-columns:1fr}}
</style></head><body><main><header><h1>T2I score review</h1>
<p class="muted">Original captions · sampled low / middle / high groups · no filtering thresholds applied</p>
<details><summary>Coverage and statistics</summary><pre>"""
    page += (
        html.escape(json.dumps(result, indent=2))
        + "</pre></details></header>"
        + "".join(sections)
        + "</main></body></html>"
    )
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_report(args.input_run, args.output, args.per_group, args.seed)
