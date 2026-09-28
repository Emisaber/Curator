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

import io
import tarfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.sample_index import WebDatasetImageSampleIndexStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import FileGroupTask


def _write_indexed_tar(path: Path, captions: dict[str, str | None]) -> None:
    with tarfile.open(path, "w") as archive:
        for key, caption in captions.items():
            for extension, payload in (("jpg", b"not a decodable image"), ("txt", caption.encode() if caption else None)):
                if payload is None:
                    continue
                member = tarfile.TarInfo(f"{key}.{extension}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(path) as archive:
        by_key: dict[str, list[str]] = {}
        for member in archive:
            key, extension = member.name.rsplit(".", 1)
            by_key.setdefault(key, []).extend(
                [extension, str(member.offset_data), str(member.size), member.name]
            )
    with open(f"{path}.idx", "w", encoding="utf-8") as index_file:
        index_file.write(f"v1.2 {len(by_key)}\n")
        index_file.writelines(" ".join(fields) + "\n" for fields in by_key.values())


def test_index_and_parquet_writer_keep_all_source_samples(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    archive = root / "part.tar"
    _write_indexed_tar(archive, {"a": "original  caption", "b": None})
    task = FileGroupTask(dataset_name="images", data=[str(archive)], _metadata={"source_files": [str(archive)]})

    batch = WebDatasetImageSampleIndexStage("blip", CaptionSource(str(root))).process(task)
    output = ParquetWriter(
        path=str(tmp_path / "samples/v1/source=blip"),
        fields=["image_id", "shard", "member", "offset", "size", "caption_raw"],
    ).process(batch)
    records = pq.read_table(output.data[0]).to_pandas().set_index("image_id")

    assert set(records.index) == {"blip|part.tar|a", "blip|part.tar|b"}
    assert records.loc["blip|part.tar|a", "caption_raw"] == "original  caption"
    assert pd.isna(records.loc["blip|part.tar|b", "caption_raw"])
    assert set(records["source"]) == {"blip"}
    assert len(output.data) == 1


def test_index_resolves_parquet_caption_source(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    archive = root / "part.tar"
    _write_indexed_tar(archive, {"a": None, "b": None})
    pq.write_table(pa.table({"caption": ["first caption", "second caption"]}), root / "text.parquet")
    pq.write_table(
        pa.table(
            {
                "sample_key": ["a", "b"],
                "parquet_path": ["text.parquet", "text.parquet"],
                "row_group": [0, 0],
                "row_index": [0, 1],
            }
        ),
        f"{archive}.caption_refs.parquet",
    )

    batch = WebDatasetImageSampleIndexStage("relaion", CaptionSource(str(root), format="parquet")).process(
        FileGroupTask(dataset_name="images", data=[str(archive)])
    )
    records = batch.data.to_pandas().set_index("image_id")

    assert records.loc["relaion|part.tar|a", "caption_raw"] == "first caption"
    assert records.loc["relaion|part.tar|b", "caption_raw"] == "second caption"
