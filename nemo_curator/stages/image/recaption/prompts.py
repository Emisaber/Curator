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


"""Confirmed Ego4D image recaption prompts."""

FHO_PROMPT = """Produce a structured English caption for the single supplied image.

Describe the whole visible scene and its main interaction, including
important objects, visible hands or people, appearance, positions,
current states, and observable relationships.

The image is the primary evidence. Dataset annotations are contextual
hints and may describe events outside this frame.

Return exactly:
{
  "comprehensive_description": string,
  "prominent_elements": [
    {
      "name": string,
      "appearance": string,
      "location": string,
      "state": string,
      "relationships": array of strings
    }
  ]
}

Use "" for unsupported attributes and [] for unsupported relationships.
Output valid JSON only, without Markdown, commentary, or additional keys.

This image comes from an Ego4D hand-object interaction record.

The narration, verb labels, and state-transition label describe the
annotated event, not necessarily what is visible at this frame.

The sampling roles indicate annotated event landmarks or uniform
sampling positions. They do not prove that contact, a state change,
or action completion is visible.

Object hints, when present, belong to this exact frame. Bounding boxes
provide localization hints; they do not establish an interaction.
An empty list does not mean that no objects are visible.

Describe visible posture, contact, placement, and object condition.
Do not infer movement direction, speed, intentions, hidden objects,
or earlier and later events from the action label alone.

Distinguish anatomical left/right hands from image-left/image-right.
If the record is partial, do not reconstruct missing event phases.
If annotations conflict with the image, describe the image.

Object hint boxes use normalized [x/W, y/H, width/W, height/H], with the origin at the image top-left."""

NARRATION_PROMPT = """Produce a structured English caption for the single supplied image.

Describe the whole visible scene and its main interaction, including
important objects, visible hands or people, appearance, positions,
current states, and observable relationships.

The image is the primary evidence. Dataset annotations are contextual
hints and may describe events outside this frame.

This image was sampled at an Ego4D narration timestamp.

The anchor_narration describes an event near the target frame.
It is not an exact action boundary and does not guarantee that the
entire narrated event is visible in this image.

#C refers to the camera wearer. #O refers to another person.
These tags do not establish whether that person is visible.
The camera wearer may be represented only by visible hands or arms.

The summary, when not null, describes a longer time interval.
Use it only to help interpret the visible setting.
Do not import its other activities, objects, or people into this frame.
A null summary means that no covering summary was provided.

Describe hand-object interactions only when supported by the image.
If the narrated action is not visible, describe what the image shows.

Do not infer movement direction, speed, intentions, hidden objects,
action completion, or earlier and later events from the narration alone.

Distinguish anatomical left/right hands from image-left/image-right.
If annotations conflict with the image, describe the image.

Return exactly:
{
  "comprehensive_description": string,
  "prominent_elements": [
    {
      "name": string,
      "appearance": string,
      "location": string,
      "state": string,
      "relationships": array of strings
    }
  ]
}

comprehensive_description must describe the current visible image
as a coherent English caption.

prominent_elements must describe important visible entities:
- name: the entity's name; use a general name if uncertain.
- appearance: visible attributes such as color, shape, and material.
- location: the entity's position in the image.
- state: its currently visible condition or posture.
- relationships: visible contact, holding, containment, support,
  or relative placement involving other entities.

Use "" for unsupported attributes and [] for unsupported relationships.
Do not invent entities to fill the fields.
Do not pad the caption with repeated details or exhaustively transcribe text.

Output valid JSON only, without Markdown, commentary, or additional keys."""
