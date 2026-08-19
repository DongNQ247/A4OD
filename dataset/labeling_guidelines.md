# Dataset Labeling Guidelines

This file defines the dataset-specific meaning of each class in `dataset/data.yaml`.
Use it together with `prompt/ai_annotation_prompt.md` before adding any YOLO label.

## Dataset Objective

Train a detector for visible, front-facing road traffic sign panels whose face is
usable for traffic control or road guidance from the camera view.

The dataset is not intended to detect every physical sign-shaped panel in the
scene. Back-facing panels, side-facing panels, shop boards, facility boards, and
advertising boards are negatives unless they satisfy the class-specific
conditions below.

## Global Decision Policy

- Label only classes listed in `dataset/data.yaml`.
- Exclusion rules override inclusion rules.
- If an object is still ambiguous after tight zoom, skip it.
- Use visible pixels only. Do not infer hidden, occluded, truncated, or blurred
  parts.
- Skip objects smaller than 10x10 pixels.
- Skip objects with less than 30% of the target panel visible.
- If an object is cut by the image boundary, clamp the box exactly to the image
  edge.
- If two target objects overlap, annotate each visible instance separately.

## Class 0: traffic_sign

### Annotate If All Conditions Hold

- The object is a physical road traffic sign panel installed for traffic control
  or road guidance.
- The sign front face is visible from the camera view.
- The visible face contains an identifiable official traffic-control cue, such
  as a symbol, number, arrow, text, border color, warning/prohibition/mandatory
  shape, or direction/instruction layout.
- At least 30% of the sign panel is visible.
- The visible panel boundary can be localized after zoom.
- The object is not excluded by any rule below.

### Orientation Policy

- Label front-facing traffic signs only.
- Do not label reverse-facing sign backs.
- Do not label side-facing signs when the front face content is not visible
  enough to identify the traffic sign type.
- Do not label a panel only because its shape, pole mounting, back plate, or
  nearby sign context suggests that it is a traffic sign.

### Label Examples

- Warning signs.
- Prohibition signs.
- Mandatory signs.
- Speed-limit signs.
- Direction signs.
- Road-instruction signs.
- Official supplemental traffic panels mounted with a main traffic sign, when
  the supplemental panel is front-facing and visually separable.

### Do Not Annotate If Any Condition Holds

- Reverse-facing traffic sign backs.
- Side-facing panels whose front content is not identifiable.
- Sign poles, brackets, frames, stickers, shadows, or support structures.
- Shop signs, facility name boards, billboards, banners, advertisements, house
  numbers, logos, decorative boards, and project or company boards.
- Generic rectangular boards near roads if they identify a building, facility,
  company, project, school, shop, or address rather than controlling or guiding
  road users.
- Traffic lights, road markings, vehicles, people, cones, barriers, lane
  markers, or other objects unless those classes are explicitly added to
  `dataset/data.yaml`.
- Tiny, blurred, far-away, or heavily occluded objects that cannot be confirmed
  as front-facing traffic signs after tight zoom.

## Bounding Box Rules

- Draw the box around the visible physical sign panel face only.
- Do not include poles, brackets, mounting hardware, frames outside the panel,
  shadows, background, or visualization text.
- For circular signs, box the visible outer circular sign border tightly.
- For triangular signs, box the visible triangular sign panel tightly.
- For rectangular signs, box the visible rectangular panel tightly.
- If a pole or another object occludes the sign panel, do not extend the box
  through the occluder to reconstruct the full sign.
- If the sign face is partially occluded but still satisfies the annotation
  conditions, box only the visible sign-face pixels.

## Ambiguity Tie-Breakers

- When inclusion and exclusion both seem plausible, skip the object.
- When orientation is uncertain, skip the object.
- When a sign-shaped object is installed near a road but appears to name a
  place, building, facility, company, or project, skip it.
- When only the back plate or side edge is visible, skip it.
