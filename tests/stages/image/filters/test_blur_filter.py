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

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from nemo_curator.stages.image.filters.blur_filter import ImageBlurFilterStage
from nemo_curator.tasks import ImageBatch, ImageObject


def test_scores_rgb_images_without_dropping_them() -> None:
    pixels = np.random.default_rng(7).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    expected = float(cv2.Laplacian(pixels, cv2.CV_64F).var())
    task = ImageBatch(
        dataset_name="images",
        data=[
            ImageObject(image_id="sharp", image_data=pixels, metadata={"source": "long"}),
            ImageObject(image_id="flat", image_data=np.full((32, 48, 3), 100, dtype=np.uint8)),
        ],
    )

    result = ImageBlurFilterStage(score_threshold=100.0).process(task)

    assert [image.image_id for image in result.data] == ["sharp", "flat"]
    assert result.data[0].metadata == {
        "source": "long",
        "laplacian_variance": pytest.approx(expected),
        "is_blurry": False,
    }
    assert result.data[1].metadata["laplacian_variance"] == 0.0
    assert result.data[1].metadata["is_blurry"] is True
    np.testing.assert_array_equal(result.data[0].image_data, pixels)


def test_explicit_filter_keeps_threshold_boundary_and_batch_metadata() -> None:
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    task = ImageBatch(
        dataset_name="images",
        data=[ImageObject(image_id="a", image_data=pixels), ImageObject(image_id="b", image_data=pixels)],
        _metadata={"source": "fixture"},
    )

    result = ImageBlurFilterStage(score_threshold=0.0, drop_blurry=True).process(task)
    assert [image.image_id for image in result.data] == ["a", "b"]
    assert result._metadata == task._metadata

    result = ImageBlurFilterStage(score_threshold=1.0, drop_blurry=True).process(task)
    assert result.data == []
    assert [image.metadata["is_blurry"] for image in task.data] == [True, True]


def test_missing_decoded_pixels_is_an_error() -> None:
    task = ImageBatch(dataset_name="images", data=[ImageObject(image_id="missing")])
    with pytest.raises(ValueError, match="missing has no decoded image_data"):
        ImageBlurFilterStage().process(task)


def test_annotation_file_retains_scores_before_optional_filter(tmp_path: Path) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    task = ImageBatch(dataset_name="images", data=[ImageObject(image_id="a", image_data=pixels)])
    stage = ImageBlurFilterStage(score_threshold=1.0, drop_blurry=True, annotations_dir=str(tmp_path))

    assert stage.process(task).data == []
    records = pd.read_parquet(tmp_path)
    assert records[["image_id", "status", "laplacian_variance", "is_blurry"]].to_dict("records") == [
        {"image_id": "a", "status": "ok", "laplacian_variance": 0.0, "is_blurry": True}
    ]
    assert pd.isna(records.loc[0, "error"])
