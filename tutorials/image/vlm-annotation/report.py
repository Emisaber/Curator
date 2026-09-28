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

"""Build a static visual review from completed VLM annotation Parquets."""

import argparse
import hashlib
import html
import json
import random
import tarfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.source_reader import decode_vlv_image, read_vlv_rows


def _group_value(field: str, value: object) -> str:
    if field == "rendered_text":
        return "text_present" if value else "none"
    if isinstance(value, list):
        return ", ".join(value) if value else "none"
    return str(value)


def _sample_rows(annotations: Path, fields: list[str], per_group: int, seed: int) -> tuple[dict, dict, dict, dict]:
    rng = random.Random(seed)  # noqa: S311 - reproducible review sample
    counts = defaultdict(int)
    selected = defaultdict(list)
    statuses = defaultdict(int)
    failures = defaultdict(list)
    for path in sorted(annotations.glob("source=*/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                status = row["status"]
                statuses[status] += 1
                if status != "ok":
                    bucket = failures[status]
                    if len(bucket) < per_group:
                        bucket.append(row)
                    else:
                        slot = rng.randrange(statuses[status])
                        if slot < per_group:
                            bucket[slot] = row
                    continue
                for field in fields:
                    key = (field, _group_value(field, row[field]))
                    counts[key] += 1
                    bucket = selected[key]
                    if len(bucket) < per_group:
                        bucket.append(row)
                    else:
                        slot = rng.randrange(counts[key])
                        if slot < per_group:
                            bucket[slot] = row
    return selected, counts, statuses, failures


def _save_preview(row: dict, assets: Path, sources: dict[str, CaptionSource] | None = None) -> str:
    filename = hashlib.sha256(row["image_id"].encode()).hexdigest()[:20] + ".jpg"
    output = assets / filename
    if not output.exists():
        source = (sources or {}).get(row["image_id"].split("|", 1)[0])
        if source is not None and source.format == "vlv_parquet":
            archive_path, row_index = row["image_path"].rsplit(":", 1)
            index = int(row_index)
            values = read_vlv_rows(Path(archive_path), {index}, [source.image_column])
            preview = Image.fromarray(decode_vlv_image(values[index][source.image_column], source))
            preview.thumbnail((480, 480))
            preview.save(output, quality=88)
            return "assets/" + filename
        archive_path, offset, member = row["image_path"].rsplit(":", 2)
        with Path(archive_path).open("rb") as archive:
            archive.seek(int(offset) - tarfile.BLOCKSIZE)
            header = tarfile.TarInfo.frombuf(archive.read(tarfile.BLOCKSIZE), "utf-8", "surrogateescape")
            if header.name != member:
                msg = f"TAR member mismatch for {row['image_id']}: {header.name} != {member}"
                raise ValueError(msg)
            archive.seek(int(offset))
            image_bytes = archive.read(header.size)
        with Image.open(BytesIO(image_bytes)) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((480, 480))
            preview.save(output, quality=88)
    return "assets/" + filename


def _card(
    row: dict,
    assets: Path,
    fields: list[str],
    focus: str | None = None,
    sources: dict[str, CaptionSource] | None = None,
) -> str:
    preview = _save_preview(row, assets, sources)
    if focus is None:
        details = f"<p>{html.escape(str(row.get('error') or ''))}</p>"
        if row.get("raw_response"):
            details += f"<pre>{html.escape(row['raw_response'])}</pre>"
        label = row["status"]
        summary = "Response and error"
    else:
        details = (
            "<dl>"
            + "".join(
                f"<div><dt>{html.escape(name)}</dt><dd>{html.escape(str(row[name]))}</dd></div>" for name in fields
            )
            + "</dl>"
        )
        label = str(row[focus])
        summary = "All annotations"
    return (
        f'<article><img src="{preview}" loading="lazy" alt="">'
        f'<div class="card-body"><strong>{html.escape(label)}</strong>'
        f"<code>{html.escape(row['image_id'])}</code>"
        f"<details><summary>{summary}</summary>{details}</details></div></article>"
    )


def create_report(annotations: Path, report: Path, per_group: int = 8, seed: int = 42) -> Path:
    run_info = json.loads((annotations / "run.json").read_text(encoding="utf-8"))
    fields = run_info["fields"]
    source_data = run_info.get("sources", {})
    sources = {name: CaptionSource(**settings) for name, settings in source_data.items()}
    selected, counts, statuses, failures = _sample_rows(annotations, fields, per_group, seed)
    assets = report / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    total = sum(statuses.values())
    stat_cards = "".join(
        f'<div class="stat"><strong>{count}</strong><span>{html.escape(status)}</span></div>'
        for status, count in sorted(statuses.items(), key=lambda item: (item[0] != "ok", item[0]))
    )
    sections = []
    for field in fields:
        groups = []
        for (group_field, value), rows in sorted(selected.items()):
            if group_field == field:
                cards = "".join(_card(row, assets, fields, field, sources) for row in rows)
                groups.append(
                    f'<div class="group"><h3>{html.escape(value)} <small>{counts[(field, value)]}</small></h3>'
                    f'<div class="grid">{cards}</div></div>'
                )
        sections.append(f'<section id="{field}"><h2>{html.escape(field)}</h2>{"".join(groups)}</section>')
    if failures:
        groups = []
        for status, rows in sorted(failures.items()):
            cards = "".join(_card(row, assets, fields, sources=sources) for row in rows)
            groups.append(
                f'<div class="group"><h3>{html.escape(status)} <small>{statuses[status]}</small></h3>'
                f'<div class="grid">{cards}</div></div>'
            )
        sections.insert(0, f'<section id="failures"><h2>Failures</h2>{"".join(groups)}</section>')
    links = (['<a href="#failures">Failures</a>'] if failures else []) + [
        f'<a href="#{html.escape(field)}">{html.escape(field)}</a>' for field in fields
    ]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VLM annotation review</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#1d2523;background:#f8f9f7}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1600px;margin:auto;padding:40px 32px 90px}}
header{{border-bottom:1px solid #dce3df;padding-bottom:24px}}
.eyebrow{{color:#57716a;text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:700}}
h1{{font-size:clamp(30px,3vw,48px);letter-spacing:-.045em;margin:8px 0}}
.subtitle,small{{color:#697971;font-size:13px}}.stats{{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}}
.stat{{min-width:128px;padding:14px 18px;background:white;border:1px solid #e0e6e2;border-radius:12px}}
.stat strong{{display:block;font-size:25px}}.stat span{{font-size:12px;color:#697971}}
nav{{position:sticky;top:0;z-index:2;display:flex;gap:8px;overflow-x:auto;padding:12px 0;background:#f8f9f7ee;border-bottom:1px solid #e0e6e2}}
nav a{{flex:none;color:#446a5e;text-decoration:none;font-size:12px;padding:7px 11px;border:1px solid #d7e2da;border-radius:99px}}
section{{scroll-margin-top:65px;margin:44px 0 60px}}h2{{font-size:22px;letter-spacing:-.03em}}
.group{{margin-bottom:28px}}h3{{font-size:15px}}small{{font-weight:400;margin-left:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}
article{{min-width:0;overflow:hidden;background:white;border:1px solid #e0e6e2;border-radius:12px}}
img{{display:block;width:100%;height:245px;object-fit:contain;background:#f0f2f0}}
.card-body{{padding:13px 15px 15px}}code{{display:block;margin:7px 0 12px;color:#76827d;font-size:10px;overflow-wrap:anywhere}}
details{{border-top:1px solid #edf0ee;padding-top:10px;color:#586962;font-size:12px}}summary{{cursor:pointer}}
dl{{margin:10px 0 0}}dl div{{display:grid;grid-template-columns:100px 1fr;gap:10px;padding:5px 0;border-top:1px solid #f0f2f0}}
dt{{color:#798780}}dd{{margin:0;overflow-wrap:anywhere}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow-y:auto;font-size:11px}}
@media(max-width:680px){{main{{padding:24px 16px 60px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">T2I data curation / visual review</div><h1>VLM annotation review</h1>
<p class="subtitle">{total} written rows / up to {per_group} examples per value / seed {seed}</p></header>
<div class="stats"><div class="stat"><strong>{total}</strong><span>Written rows</span></div>{stat_cards}</div>
<nav>{"".join(links)}</nav>{"".join(sections)}</main></body></html>"""
    index = report / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(create_report(args.annotations, args.report, args.per_group, args.seed))
