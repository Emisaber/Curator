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

import importlib
import tarfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image, ImageFilter


@pytest.fixture
def tutorial_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "tutorials/image/dedup-comparison"))
    return {name: importlib.import_module(name) for name in ("prepare_samples", "report", "run")}


@pytest.fixture
def image_tars(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (96, 128, 3), dtype=np.uint8))
    for shard in range(2):
        with tarfile.open(root / f"{shard}.tar", "w") as archive:
            for index in range(6):
                buffer = BytesIO()
                variant = image if index < 4 else image.filter(ImageFilter.GaussianBlur(index - 3))
                variant.save(buffer, format="PNG")
                content = buffer.getvalue()
                member = tarfile.TarInfo(f"{index}.png")
                member.size = len(content)
                archive.addfile(member, BytesIO(content))
            member = tarfile.TarInfo("bad.jpg")
            member.size = 3
            archive.addfile(member, BytesIO(b"bad"))
    return root
