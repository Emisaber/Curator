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

import io
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.image.deduplication.caption import (
    CandidatePairPartitioningStage,
    CaptionAwareDeduplicationStage,
    CaptionSource,
)
from nemo_curator.tasks import EmptyTask, FileGroupTask


def _write_wds_tar(path: Path, captions: dict[str, str | None], caption_extension: str = "txt") -> None:
    with tarfile.open(path, "w") as archive:
        for key, caption in captions.items():
            for extension, payload in [
                ("jpg", b"fake image"),
                (caption_extension, caption.encode() if caption else None),
            ]:
                if payload is None:
                    continue
                info = tarfile.TarInfo(f"{key}.{extension}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(path) as archive:
        by_key = {}
        for info in archive:
            key, extension = info.name.rsplit(".", 1)
            by_key.setdefault(key, []).extend([extension, str(info.offset_data), str(info.size), info.name])
    with open(f"{path}.idx", "w", encoding="utf-8") as index_file:
        index_file.write(f"v1.2 {len(by_key)}\n")
        index_file.writelines(" ".join(fields) + "\n" for fields in by_key.values())


def test_caption_aware_stage_checks_every_edge_and_preserves_source_text(tmp_path: Path) -> None:
    source_root = tmp_path / "images"
    source_root.mkdir()
    tar_path = source_root / "part.tar"
    _write_wds_tar(tar_path, {"a": "café  at left", "b": "cafe\u0301 at  left", "c": "right side", "d": None})
    ids = {key: f"blip|part.tar|{key}" for key in "abcd"}

    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    pairs = pa.table(
        {
            "id_a": [ids["b"], ids["c"], ids["c"], ids["d"]],
            "id_b": [ids["a"], ids["b"], ids["a"], ids["a"]],
            "cosine_sim_score": [0.999, 0.997, 0.995, 0.994],
        }
    )
    candidate_path = candidate_dir / "cluster_0_00000000.parquet"
    pq.write_table(pairs, candidate_path)
    partition = CandidatePairPartitioningStage(str(candidate_dir))
    tasks = partition.process(EmptyTask(dataset_name="test"))
    assert len(tasks) == 1

    output = CaptionAwareDeduplicationStage(
        {"blip": CaptionSource(root=str(source_root))}, str(tmp_path / "decisions")
    ).process(tasks[0])
    decisions = pq.read_table(output.data[0]).to_pydict()
    assert Path(output.data[0]).parent.name == "pairs"
    assert Path(output._metadata["captions_path"]).parent.name == "captions"
    assert decisions["caption_relation"] == [
        "same_caption",
        "different_caption",
        "different_caption",
        "missing_caption",
    ]
    caption_table = pq.read_table(output._metadata["captions_path"]).to_pandas().set_index("image_id")
    assert caption_table.loc[ids["a"], "caption_raw"] == "café  at left"
    assert caption_table.loc[ids["a"], "caption_normalized"] == "café at left"
    assert caption_table.loc[ids["b"], "caption_normalized"] == "café at left"
    assert not (tmp_path / "decisions" / "duplicates").exists()


def test_uppercase_txt_caption_agrees_with_sample_index(tmp_path: Path) -> None:
    from nemo_curator.stages.image.io.sample_index import WebDatasetImageSampleIndexStage

    source_root = tmp_path / "images"
    source_root.mkdir()
    tar_path = source_root / "part.tar"
    _write_wds_tar(tar_path, {"a": "same caption", "b": "same caption"}, caption_extension="TXT")
    source = CaptionSource(root=str(source_root))
    sample_task = FileGroupTask(dataset_name="test", data=[str(tar_path)])
    samples = WebDatasetImageSampleIndexStage("blip", source).process(sample_task).data.to_pydict()
    assert samples["caption_raw"] == ["same caption", "same caption"]

    ids = ["blip|part.tar|a", "blip|part.tar|b"]
    candidate = tmp_path / "cluster_0_00000000.parquet"
    pq.write_table(pa.table({"id_a": [ids[0]], "id_b": [ids[1]], "cosine_sim_score": [0.99]}), candidate)
    task = FileGroupTask(dataset_name="test", data=[str(candidate)], _metadata={"centroid_id": 0})
    output = CaptionAwareDeduplicationStage({"blip": source}, str(tmp_path / "decisions")).process(task)
    assert pq.read_table(output.data[0])["caption_relation"].to_pylist() == ["same_caption"]


def test_parquet_caption_source_uses_selected_column_and_row_group(tmp_path: Path) -> None:
    source_root = tmp_path / "images"
    source_root.mkdir()
    parquet_path = source_root / "source.parquet"
    writer = pq.ParquetWriter(parquet_path, pa.schema([("caption", pa.string()), ("long_caption", pa.string())]))
    writer.write_table(pa.table({"caption": ["alt one"], "long_caption": ["object left"]}))
    writer.write_table(pa.table({"caption": ["alt two"], "long_caption": ["object right"]}))
    writer.close()
    tar_path = source_root / "part.tar"
    refs = pa.table(
        {
            "sample_key": ["a", "b"],
            "parquet_path": ["source.parquet", "source.parquet"],
            "row_group": [0, 1],
            "row_index": [0, 0],
        }
    )
    pq.write_table(refs, f"{tar_path}.caption_refs.parquet")
    ids = [f"relaion|part.tar|{key}" for key in ["a", "b"]]
    candidate = tmp_path / "cluster_1_00000000.parquet"
    pq.write_table(pa.table({"id_a": [ids[0]], "id_b": [ids[1]], "cosine_sim_score": [0.99]}), candidate)
    task = FileGroupTask(dataset_name="test", data=[str(candidate)], _metadata={"centroid_id": 1})
    output = CaptionAwareDeduplicationStage(
        {"relaion": CaptionSource(root=str(source_root), format="parquet", caption_column="long_caption")},
        str(tmp_path / "decisions"),
    ).process(task)
    captions = pq.read_table(output._metadata["captions_path"]).to_pydict()
    assert captions["caption_raw"] == ["object left", "object right"]
    assert pq.read_table(output.data[0])["caption_relation"].to_pylist() == ["different_caption"]
