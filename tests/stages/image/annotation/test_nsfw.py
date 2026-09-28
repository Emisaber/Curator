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

import numpy as np
import pandas as pd
import pytest
import torch

from nemo_curator.stages.image.annotation import nsfw
from nemo_curator.tasks import DocumentBatch


def test_scores_persist_without_filtering(tmp_path: Path, monkeypatch) -> None:
    batch_sizes = []

    class Scorer:
        def __init__(self, model_dir: str) -> None:
            assert model_dir == str(tmp_path / "models")

        def setup(self) -> None:
            pass

        def __call__(self, embeddings: np.ndarray) -> torch.Tensor:
            batch_sizes.append(len(embeddings))
            return torch.tensor(embeddings[:, 0], dtype=torch.float32)

    monkeypatch.setattr(nsfw, "NSFWScorer", Scorer)
    stage = nsfw.ImageNSFWAnnotationStage(
        model_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "annotations"),
        source_name="sample",
        model_inference_batch_size=2,
    )
    stage.setup()
    feature_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "embedding": [np.full(768, score, dtype=np.float32) for score in (0.1, 0.2, 0.3)],
        }
    ).to_parquet(feature_path, index=False)
    task = DocumentBatch(
        dataset_name="sample",
        data=pd.read_parquet(feature_path, dtype_backend="pyarrow"),
        _metadata={"source_files": [str(feature_path)]},
    )

    result = stage.process(task)
    assert result is not None
    written = nsfw.NSFWAnnotationWriter().process(result)
    rows = pd.read_parquet(written.data[0])
    assert rows["image_id"].tolist() == ["a", "b", "c"]
    assert rows["status"].tolist() == ["ok", "ok", "ok"]
    assert rows["nsfw_score"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert batch_sizes == [2, 1]
    assert stage.process(task) is None
