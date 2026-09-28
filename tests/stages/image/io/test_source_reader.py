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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from nemo_curator.stages.image.deduplication.caption import CaptionAwareDeduplicationStage
from nemo_curator.stages.image.io.caption_source import CaptionSource
from nemo_curator.stages.image.io.source_reader import SourceShardReaderStage, find_ego4d_records, read_vlv_rows
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.hash_utils import get_deterministic_hash


@pytest.fixture(autouse=True)
def _cpu_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_vlv_batches_preserve_global_row_positions_and_pixels(tmp_path: Path) -> None:
    pixels = [np.full((3, 2, 2), index, dtype=np.uint8).tobytes() for index in range(5)]
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"image": pixels, "caption": [str(index) for index in range(5)]}), path, row_group_size=2)
    source = CaptionSource(str(tmp_path), format="vlv_parquet", image_shape=(3, 2, 2))
    reader = SourceShardReaderStage("vlv", source, image_batch_size=2)
    batches = reader.process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert [len(batch.data) for batch in batches] == [2, 2, 1]
    images = [image for batch in batches for image in batch.data]
    assert [image.image_id for image in images] == [f"vlv|part.parquet|{index}" for index in range(5)]
    for index, image in enumerate(images):
        np.testing.assert_array_equal(image.image_data, np.full((2, 2, 3), index, dtype=np.uint8))
    assert read_vlv_rows(path, {1, 4}, ["caption"]) == {1: {"caption": "1"}, 4: {"caption": "4"}}


def test_vlv_reader_accepts_configured_encoded_images(tmp_path: Path) -> None:
    payload = io.BytesIO()
    Image.new("RGB", (3, 2), "blue").save(payload, format="PNG")
    path = tmp_path / "encoded.parquet"
    pq.write_table(pa.table({"pixels": [{"bytes": payload.getvalue()}]}), path)
    source = CaptionSource(str(tmp_path), format="vlv_parquet", image_column="pixels", image_encoding="encoded")
    batches = SourceShardReaderStage("vlv", source).process(FileGroupTask(dataset_name="test", data=[str(path)]))
    np.testing.assert_array_equal(batches[0].data[0].image_data, np.full((2, 3, 3), [0, 0, 255], dtype=np.uint8))


def _ego4d_source(root: Path) -> tuple[CaptionSource, Path, str]:
    media, indices, captions = (root / name for name in ("media", "indices", "captions"))
    for directory in (media, indices, captions):
        directory.mkdir(parents=True)
    ids = [f"ego4d-fho|video|{frame}" for frame in (10, 20)]
    payload = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(payload, format="JPEG")
    archive_path = media / "video-000000.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("video/000000010.jpg")
        member.size = len(payload.getvalue())
        archive.addfile(member, io.BytesIO(payload.getvalue()))
    with tarfile.open(archive_path) as archive:
        offset = archive.getmember(member.name).offset_data
    index_path = indices / "video-000000.parquet"
    pq.write_table(
        pa.table(
            {
                "image_id": ids,
                "shard": [archive_path.name] * 2,
                "member": [member.name, "video/000000020.jpg"],
                "offset": [offset, None],
                "size": [member.size, None],
                "decode_status": ["ok", "failed"],
                "caption_raw": [None, None],
            }
        ),
        index_path,
    )
    pq.write_table(
        pa.table(
            {
                "image_id": ids,
                "status": ["ok", "failed"],
                "comprehensive_description": ["same caption", None],
            }
        ),
        captions / f"part-{get_deterministic_hash(ids)}.parquet",
    )
    source = CaptionSource(
        str(media), format="ego4d_recaption", sample_index_root=str(indices), annotations_root=str(captions)
    )
    return source, index_path, ids[0]


def test_ego4d_reader_reuses_ids_and_joins_captions_without_rewriting_manifest(tmp_path: Path) -> None:
    pytest.importorskip("nvidia.dali")
    source, path, image_id = _ego4d_source(tmp_path)
    original = path.read_bytes()
    batches = SourceShardReaderStage("ego4d-fho", source).process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert len(batches) == 1
    assert [image.image_id for image in batches[0].data] == [image_id]
    assert batches[0].data[0].image_data.shape == (4, 4, 3)
    assert find_ego4d_records({image_id}, source)[image_id]["caption_raw"] == "same caption"
    assert path.read_bytes() == original


def test_vlv_bad_rows_keep_original_positions_and_complete_decode_records(tmp_path: Path) -> None:
    payload = np.zeros((3, 2, 2), dtype=np.uint8).tobytes()
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"image": [payload, None, b"wrong size", payload, payload]}), path, row_group_size=2)
    source = CaptionSource(str(tmp_path), format="vlv_parquet", image_shape=(3, 2, 2))
    reader = SourceShardReaderStage("vlv", source, image_batch_size=2, records_dir=str(tmp_path / "decode"))
    batches = reader.process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert [image.image_id for batch in batches for image in batch.data] == [
        "vlv|part.parquet|0",
        "vlv|part.parquet|3",
        "vlv|part.parquet|4",
    ]
    files = list((tmp_path / "decode").glob("*.parquet"))
    assert len(files) == 1
    records = pq.ParquetFile(files[0]).read().to_pylist()
    assert [row["row_index"] for row in records] == list(range(5))
    assert [row["status"] for row in records] == ["ok", "failed", "failed", "ok", "ok"]
    assert records[1]["error"]
    assert records[2]["error"]
    assert "source" not in records[0]


def test_all_bad_vlv_rows_write_failures_without_empty_image_batches(tmp_path: Path) -> None:
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"image": pa.array([None, b"bad"], type=pa.binary())}), path)
    source = CaptionSource(str(tmp_path), format="vlv_parquet")
    reader = SourceShardReaderStage("vlv", source, records_dir=str(tmp_path / "decode"))
    assert reader.process(FileGroupTask(dataset_name="test", data=[str(path)])) == []
    records = pq.ParquetFile(next((tmp_path / "decode").glob("*.parquet"))).read().to_pylist()
    assert [row["status"] for row in records] == ["failed", "failed"]


def test_vlv_file_failure_is_not_an_image_failure(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"
    reader = SourceShardReaderStage(
        "vlv", CaptionSource(str(tmp_path), format="vlv_parquet"), records_dir=str(tmp_path / "decode")
    )
    with pytest.raises(FileNotFoundError):
        reader.process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert not (tmp_path / "decode").exists()


def test_ego4d_bad_member_falls_back_without_changing_ids_or_captions(tmp_path: Path) -> None:
    pytest.importorskip("nvidia.dali")
    source, path, image_id = _ego4d_source(tmp_path)
    archive_path = Path(source.root) / "video-000000.tar"
    with tarfile.open(archive_path, "a") as archive:
        member = tarfile.TarInfo("video/000000020.jpg")
        member.size = 6
        archive.addfile(member, io.BytesIO(b"broken"))
    with tarfile.open(archive_path) as archive:
        offset = archive.getmember(member.name).offset_data
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[1].update(offset=offset, size=6, decode_status="ok")
    pq.write_table(pa.Table.from_pylist(rows), path)
    captions_path = (
        Path(source.annotations_root) / f"part-{get_deterministic_hash([row['image_id'] for row in rows])}.parquet"
    )
    pq.write_table(
        pa.table(
            {
                "image_id": [row["image_id"] for row in rows],
                "status": ["ok", "ok"],
                "comprehensive_description": ["same caption", "bad frame caption"],
            }
        ),
        captions_path,
    )
    original_index = path.read_bytes()
    original_captions = captions_path.read_bytes()
    reader = SourceShardReaderStage("ego4d-fho", source, image_batch_size=1, records_dir=str(tmp_path / "decode"))
    batches = reader.process(FileGroupTask(dataset_name="test", data=[str(path)]))
    assert [image.image_id for batch in batches for image in batch.data] == [image_id]
    records = pq.ParquetFile(next((tmp_path / "decode").glob("*.parquet"))).read().to_pylist()
    assert [row["status"] for row in records] == ["ok", "failed"]
    assert find_ego4d_records({image_id}, source)[image_id]["caption_raw"] == "same caption"
    assert path.read_bytes() == original_index
    assert captions_path.read_bytes() == original_captions


def test_caption_decision_compares_vlv_and_ego4d_sources(tmp_path: Path) -> None:
    ego, _, ego_id = _ego4d_source(tmp_path / "ego")
    vlv_root = tmp_path / "vlv"
    vlv_root.mkdir()
    pq.write_table(pa.table({"caption": ["same  caption"]}), vlv_root / "part.parquet")
    pairs = tmp_path / "cluster_0_00000000.parquet"
    pq.write_table(
        pa.table(
            {
                "id_a": ["vlv|part.parquet|0"],
                "id_b": [ego_id],
                "cosine_sim_score": [0.999],
            }
        ),
        pairs,
    )
    stage = CaptionAwareDeduplicationStage(
        {"vlv": CaptionSource(str(vlv_root), format="vlv_parquet"), "ego4d-fho": ego}, str(tmp_path / "decisions")
    )
    output = stage.process(FileGroupTask(dataset_name="test", data=[str(pairs)], _metadata={"centroid_id": 0}))
    assert pq.read_table(output.data[0])["caption_relation"].to_pylist() == ["same_caption"]
    records = pq.ParquetFile(output._metadata["captions_path"]).read().to_pylist()
    assert {row["caption_raw"] for row in records} == {"same caption", "same  caption"}
