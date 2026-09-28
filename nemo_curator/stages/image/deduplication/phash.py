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

"""Perceptual image hashes, stored alongside other image metadata."""

from dataclasses import dataclass

from PIL import Image

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import ImageBatch


@dataclass
class PerceptualHashStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Compute the 64-bit DCT pHash of each decoded RGB image."""

    name: str = "image_phash"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: ImageBatch) -> ImageBatch:
        import imagehash

        for image in task.data:
            image.metadata["phash"] = str(imagehash.phash(Image.fromarray(image.image_data), hash_size=8))
        return task
