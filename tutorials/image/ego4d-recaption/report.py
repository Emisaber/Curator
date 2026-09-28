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


"""Review sampled frames, source annotations and generated captions in static HTML."""

import argparse
import html
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

import pyarrow.parquet as pq


def _preview_runtime() -> ModuleType:
    path = Path(__file__).parents[1] / "vlm-annotation/report.py"
    spec = importlib.util.spec_from_file_location("vlm_annotation_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_report(output: Path, caption_version: str = "v1", per_group: int | None = 20, seed: int = 42) -> Path:  # noqa: C901, PLR0912
    annotations = output / "annotations/recaption/schema-v1" / f"prompt-{caption_version}"
    rng = random.Random(seed)  # noqa: S311 - reproducible review samples
    counts = Counter()
    selected = defaultdict(list)
    for path in sorted(annotations.glob("source=*/*.parquet")):
        source = path.parent.name.removeprefix("source=")
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                key = (source, row["error_kind"] or row["status"])
                counts[key] += 1
                row["source"] = source
                bucket = selected[key]
                if per_group is None or len(bucket) < per_group:
                    bucket.append(row)
                else:
                    slot = rng.randrange(counts[key])
                    if slot < per_group:
                        bucket[slot] = row
    ids = {row["image_id"] for rows in selected.values() for row in rows}
    samples = {}
    for path in sorted((output / "samples/schema-v1/manifests").glob("source=*/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                if row["image_id"] in ids:
                    samples[row["image_id"]] = row
    report = output / "report" / f"recaption-{caption_version}"
    assets = report / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    videos = report / "videos"
    videos.mkdir(exist_ok=True)
    sampling = json.loads((output / "samples/schema-v1/run.json").read_text(encoding="utf-8"))
    data_root = Path(sampling["data_root"])
    preview_runtime = _preview_runtime()
    sections = []
    links = []
    for (source, status), rows in sorted(selected.items()):
        anchor = f"{source}-{status}"
        links.append(f'<a href="#{html.escape(anchor)}">{html.escape(source)} / {html.escape(status)}</a>')
        cards = []
        for row in rows:
            sample = samples[row["image_id"]]
            video = videos / sample["video"]
            if not video.exists():
                video.symlink_to((data_root / sample["video"]).resolve())
            video_url = f"videos/{video.name}#t={sample['timestamp_sec']:.3f}"
            if sample["decode_status"] == "ok" and row["error_kind"] != "decode_error":
                preview = preview_runtime._save_preview(row, assets)
                image = f'<img src="{html.escape(preview)}" loading="lazy" alt="Sampled video frame">'
            else:
                image = '<div class="missing">Frame decoding failed</div>'
            elements = json.dumps(row["prominent_elements"], indent=2, ensure_ascii=False)
            context = json.dumps(json.loads(sample["context_json"]), indent=2, ensure_ascii=False)
            original = json.dumps(json.loads(sample["annotation_json"]), indent=2, ensure_ascii=False)
            cards.append(
                f'<article>{image}<div class="body"><code>{html.escape(row["image_id"])}</code>'
                f'<p class="meta">Frame {sample["frame_number"]} · {sample["timestamp_sec"]:.3f} s'
                f" · {html.escape(row['error_kind'] or row['status'])}</p>"
                f'<p class="caption">{html.escape(row["comprehensive_description"] or row["error"] or "")}</p>'
                '<details class="video"><summary>Original video</summary>'
                f'<video controls preload="none" data-src="{html.escape(video_url)}"></video>'
                f'<p class="meta">Video {html.escape(sample["video_uid"])} · '
                f'Sampled at {sample["timestamp_sec"]:.3f} s</p>'
                f'<a href="{html.escape(video_url)}" target="_blank" rel="noopener">Open original video</a></details>'
                f"<details open><summary>Prominent elements</summary><pre>{html.escape(elements)}</pre></details>"
                f"<details><summary>Model input annotations</summary><pre>{html.escape(context)}</pre></details>"
                f"<details><summary>Original annotations and associations</summary><pre>{html.escape(original)}</pre></details>"
                f"<details><summary>Raw model response</summary><pre>{html.escape(row['raw_response'] or '')}</pre></details>"
                "</div></article>"
            )
        sections.append(
            f'<section id="{html.escape(anchor)}"><h2>{html.escape(source)} · {html.escape(status)} '
            f'<small>{counts[(source, status)]:,} results</small></h2><div class="grid">{"".join(cards)}</div></section>'
        )
    statistics = "".join(
        f"<span>{html.escape(source)} / {html.escape(status)}: <strong>{count:,}</strong></span>"
        for (source, status), count in sorted(counts.items())
    )
    planned = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in (output / "samples/schema-v1/plans").glob("source=*/*.parquet")
    )
    total = sum(counts.values())
    review = "all results shown" if per_group is None else f"up to {per_group} examples per source and status"
    run_path = annotations / "run.json"
    run_info = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    prompts = "".join(
        f"<details><summary>{html.escape(kind)} system prompt</summary><pre>{html.escape(info['prompt'])}</pre></details>"
        for kind, info in run_info.get("sources", {}).items()
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ego4D recaption review</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f7f8f5;color:#222b28;font-family:system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:36px 28px 80px}}h1{{font-size:36px;letter-spacing:-.04em}}
header{{border-bottom:1px solid #dce3de;padding-bottom:20px}}.stats{{display:flex;gap:12px;flex-wrap:wrap}}
.stats span{{padding:10px 14px;background:white;border:1px solid #dce3de;border-radius:8px}}
section{{margin-top:36px}}h2{{font-size:21px}}small,.meta{{font-size:12px;color:#68766e}}
nav{{position:sticky;top:0;z-index:2;display:flex;gap:8px;overflow:auto;padding:12px 0;background:#f7f8f5ee}}
nav a{{flex:none;color:#446a5e;text-decoration:none;font-size:12px;padding:7px 11px;border:1px solid #dce3de;border-radius:99px}}
section{{scroll-margin-top:65px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}}
article{{background:white;border:1px solid #dce3de;border-radius:12px;overflow:hidden}}
img{{display:block;width:100%;height:300px;object-fit:contain;background:#edf0eb}}.body{{padding:18px}}
video{{display:block;width:100%;margin-top:12px;background:#111}}.video a{{color:#446a5e;font-size:12px}}
code{{font-size:11px;overflow-wrap:anywhere}}.caption{{line-height:1.65}}summary{{cursor:pointer;font-size:13px}}
details{{border-top:1px solid #e7ebe5;padding-top:12px;margin-top:12px}}
pre{{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto}}
.missing{{padding:70px 20px;text-align:center;background:#edf0eb}}
@media(max-width:600px){{main{{padding:20px 12px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>Ego4D recaption</h1>
<p>{planned:,} planned · {total:,} written · {planned - total:,} pending · {review}</p>
<p class="meta">Model: {html.escape(run_info.get("model", ""))} · Caption version: {html.escape(caption_version)}</p>
<div class="stats">{statistics}</div>{prompts}</header><nav>{"".join(links)}</nav>{"".join(sections)}</main>
<script>
document.querySelectorAll('details.video').forEach(details => {{
    details.addEventListener('toggle', () => {{
        const video = details.querySelector('video');
        if (details.open && !video.getAttribute('src')) video.src = video.dataset.src;
        if (!details.open) video.pause();
    }});
}});
</script></body></html>"""
    path = report / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--caption-version", default="v1")
    parser.add_argument("--per-group", default=20, type=int)
    parser.add_argument("--all", action="store_true", help="Show every result, including failures")
    args = parser.parse_args()
    print(create_report(args.output, args.caption_version, None if args.all else args.per_group))
