# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Build a static, sampled visual review of caption-aware image candidates."""

import hashlib
import html
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from nemo_curator.stages.image.deduplication.caption import CaptionSource, read_webdataset_components
from nemo_curator.stages.image.io.source_reader import decode_vlv_image, find_ego4d_records, read_vlv_rows

_RELATIONS = ("same_caption", "different_caption", "missing_caption")


def _sample_decisions(decisions_dir: Path, budget: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)  # noqa: S311 - reproducible review sample
    per_relation = max(1, budget // len(_RELATIONS))
    reservoirs: dict[str, list[dict]] = {relation: [] for relation in _RELATIONS}
    counts = dict.fromkeys(_RELATIONS, 0)
    for path in sorted((decisions_dir / "pairs").glob("cluster_*_*.parquet")):
        if not path.stem.rsplit("_", 1)[-1].isdigit():
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for pair in batch.to_pylist():
                relation = pair["caption_relation"]
                counts[relation] += 1
                reservoir = reservoirs[relation]
                if len(reservoir) < per_relation:
                    reservoir.append(pair)
                else:
                    slot = rng.randrange(counts[relation])
                    if slot < per_relation:
                        reservoir[slot] = pair
    selected = [pair for relation in _RELATIONS for pair in reservoirs[relation]]
    return selected, counts


def _load_captions(decisions_dir: Path, image_ids: set[str]) -> dict[str, str | None]:
    captions = {}
    for path in (decisions_dir / "captions").glob("cluster_*_captions.parquet"):
        for batch in pq.ParquetFile(path).iter_batches(columns=["image_id", "caption_raw"], batch_size=4096):
            for row in batch.to_pylist():
                if row["image_id"] in image_ids:
                    captions[row["image_id"]] = row["caption_raw"]
    return captions


def _export_images(image_ids: set[str], sources: dict[str, CaptionSource], assets: Path) -> dict[str, str]:
    by_archive: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for image_id in image_ids:
        source_name, relative_tar, key = image_id.split("|", 2)
        by_archive[source_name, relative_tar][key] = image_id
    assets.mkdir(parents=True, exist_ok=True)
    images = {}
    for (source_name, relative_tar), keys in by_archive.items():
        source = sources[source_name]
        archive = Path(source.root) / relative_tar
        if source.format == "vlv_parquet":
            rows = read_vlv_rows(archive, {int(key) for key in keys}, [source.image_column])
            previews = {
                key: Image.fromarray(decode_vlv_image(rows[int(key)][source.image_column], source)) for key in keys
            }
        elif source.format == "ego4d_recaption":
            records = find_ego4d_records(set(keys.values()), source)
            previews = {}
            for key, image_id in keys.items():
                row = records[image_id]
                with (Path(source.root) / row["shard"]).open("rb") as stream:
                    stream.seek(row["offset"])
                    with Image.open(BytesIO(stream.read(row["size"]))) as image:
                        previews[key] = image.convert("RGB")
        else:
            components = read_webdataset_components(
                archive, set(keys), ("jpg", "jpeg", "png", "webp"), source.index_suffix
            )
            previews = {}
            for key in keys:
                with Image.open(BytesIO(components[key])) as image:
                    previews[key] = ImageOps.exif_transpose(image).convert("RGB")
        for key, image_id in keys.items():
            filename = hashlib.sha256(image_id.encode()).hexdigest()[:20] + ".jpg"
            preview = previews[key]
            preview.thumbnail((480, 480))
            preview.save(assets / filename, quality=88)
            images[image_id] = "assets/" + filename
    return images


def create_report(
    decisions_dir: Path, sources: dict[str, CaptionSource], report_dir: Path, budget: int = 240, seed: int = 42
) -> Path:
    """Sample each caption relation and save a self-contained static review folder."""
    if budget < len(_RELATIONS):
        msg = f"Review budget must be at least {len(_RELATIONS)}"
        raise ValueError(msg)
    selected, counts = _sample_decisions(decisions_dir, budget, seed)
    image_ids = {pair[field] for pair in selected for field in ("id_a", "id_b")}
    captions = _load_captions(decisions_dir, image_ids)
    report_dir.mkdir(parents=True, exist_ok=True)
    images = _export_images(image_ids, sources, report_dir / "assets")
    cards = []
    for pair in selected:
        relation = html.escape(pair["caption_relation"])
        items = []
        for field in ("id_a", "id_b"):
            image_id = pair[field]
            caption = captions[image_id]
            items.append(
                f'<div class="sample"><img src="{images[image_id]}" loading="lazy">'
                f"<code>{html.escape(image_id)}</code>"
                f"<p>{html.escape(caption) if caption is not None else '(missing)'}</p></div>"
            )
        cards.append(
            f'<article><header><span class="tag">{relation}</span>'
            f"<span>CLIP {pair['cosine_sim_score']:.5f}</span></header>"
            f"<div class='images'>{''.join(items)}</div></article>"
        )
    totals = " · ".join(f"{name}: {counts[name]}" for name in _RELATIONS)
    page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>Caption-aware image candidates</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{font-family:system-ui,sans-serif;color:#20262c;background:#f5f6f5}}
body{{max-width:1180px;margin:0 auto;padding:36px 22px}}
h1{{font-size:26px;letter-spacing:-.04em}}
.muted{{color:#637078}}
article{{background:white;border:1px solid #dfe4e2;border-radius:12px;padding:18px;margin:16px 0}}
header{{display:flex;justify-content:space-between;color:#657078;font-size:13px;margin-bottom:14px}}
.tag{{color:#185b52;font-weight:700}}
.images{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
.sample img{{width:100%;height:310px;object-fit:contain;background:#f4f5f4;border-radius:6px}}
.sample code{{display:block;overflow-wrap:anywhere;font-size:11px;color:#66737a;margin-top:10px}}
.sample p{{line-height:1.55;white-space:pre-wrap}}
@media(max-width:700px){{.images{{grid-template-columns:1fr}}}}
</style><main><h1>Caption-aware image candidates</h1>
<p class="muted">Image similarity is a candidate signal; caption relation is compared per pair.
No image was removed.</p>
<p class="muted">{html.escape(totals)} / sampled {len(selected)} pairs / seed {seed}</p>
{"".join(cards)}</main></html>"""
    index = report_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index
