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

"""Look up original captions for selected image IDs without normalizing text."""

from collections import defaultdict
from pathlib import Path

from nemo_curator.stages.image.io.caption_source import CaptionSource, read_parquet_captions
from nemo_curator.stages.image.io.source_reader import find_ego4d_records, read_vlv_rows
from nemo_curator.utils.image_id import sample_key_from_member, split_image_id


def read_webdataset_components(
    tar_path: Path, sample_keys: set[str], extensions: str | tuple[str, ...], index_suffix: str = ".idx"
) -> dict[str, bytes]:
    """Seek selected WebDataset components through DALI's v1.2 index."""
    if isinstance(extensions, str):
        extensions = (extensions,)
    accepted = {extension.lower() for extension in extensions}
    locations = {}
    with open(f"{tar_path}{index_suffix}", encoding="utf-8") as index_file:
        version = index_file.readline().split()[0]
        if version != "v1.2":
            msg = f"Caption lookup requires a v1.2 DALI index: {tar_path}{index_suffix}"
            raise ValueError(msg)
        for line in index_file:
            fields = line.split()
            for offset in range(0, len(fields), 4):
                component_ext, data_offset, size, filename = fields[offset : offset + 4]
                key = sample_key_from_member(filename)
                if component_ext.lower() in accepted and key in sample_keys:
                    locations[key] = (int(data_offset), int(size))
    components = {}
    with tar_path.open("rb") as archive:
        for key, (offset, size) in locations.items():
            archive.seek(offset)
            components[key] = archive.read(size)
    return components


def resolve_captions(image_ids: set[str], sources: dict[str, CaptionSource]) -> dict[str, str | None]:
    by_archive: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for image_id in image_ids:
        source_name, relative_tar, sample_key = split_image_id(image_id)
        by_archive[source_name, relative_tar][sample_key] = image_id

    captions = {}
    for (source_name, relative_tar), keys in by_archive.items():
        source = sources[source_name]
        root = Path(source.root)
        tar_path = root / relative_tar
        if source.format == "webdataset_txt":
            components = read_webdataset_components(tar_path, set(keys), "txt", source.index_suffix)
            found = {key: components[key].decode("utf-8") if key in components else None for key in keys}
        elif source.format == "parquet":
            found = read_parquet_captions(tar_path, root, set(keys), source.caption_column)
        elif source.format == "vlv_parquet":
            rows = read_vlv_rows(tar_path, {int(key) for key in keys}, [source.caption_column])
            found = {key: rows[int(key)][source.caption_column] for key in keys}
        elif source.format == "ego4d_recaption":
            records = find_ego4d_records(set(keys.values()), source)
            found = {key: records[image_id]["caption_raw"] for key, image_id in keys.items()}
        else:
            msg = f"Unsupported caption source format: {source.format}"
            raise ValueError(msg)
        captions.update({image_id: found[key] for key, image_id in keys.items()})
    return captions
