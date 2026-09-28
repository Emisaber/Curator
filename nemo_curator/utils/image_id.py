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

"""Stable source-sample identity shared by WebDataset image workflows."""


def sample_key_from_member(member: str) -> str:
    """Strip the first component extension while preserving an optional parent path."""
    dot = member.find(".", member.rfind("/") + 1)
    return member[:dot] if dot >= 0 else member


def make_image_id(source: str, relative_tar: str, member: str) -> str:
    """Identify a WebDataset sample independently of its mount point."""
    if not source or "|" in source or not relative_tar or "|" in relative_tar:
        raise ValueError("source and relative_tar must be nonempty and cannot contain '|'")
    return f"{source}|{relative_tar}|{sample_key_from_member(member)}"


def split_image_id(image_id: str) -> tuple[str, str, str]:
    """Recover the source location and WebDataset sample key."""
    parts = image_id.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid WebDataset image_id: {image_id!r}")
    return parts[0], parts[1], parts[2]
