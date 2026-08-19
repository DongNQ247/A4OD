# Dataset Labeling Guidelines

This file defines the dataset-specific labeling rules for A4OD. AI annotation
agents must read this file together with `TOOL_CONTRACT.md`,
`.a4od/contract.yaml`, and `dataset/data.yaml` before adding labels.

## Dataset Objective

Train a detector for visible, front-facing road traffic sign panels whose face
is usable for traffic control or road guidance from the camera viewpoint.

The detector is not intended to detect every sign-shaped panel in the scene.
Back-facing panels, side-facing panels, shop boards, facility boards,
advertising boards, project boards, and decorative boards are background unless
they satisfy the class-specific positive definition below.

## Class List Source

The authoritative class id and class name mapping is `dataset/data.yaml`.
Every class section below must match one entry from `dataset/data.yaml`.
Do not introduce, rename, merge, or reinterpret classes in this file.

Current class list:

- Class 0: `traffic_sign`

## Global Annotation Rules

- Label only classes listed in `dataset/data.yaml`.
- Exclusion rules override inclusion rules.
- If an object remains ambiguous after tight zoom and inspection, skip it.
- Annotate visible object pixels only. Do not infer hidden, occluded, blurred,
  truncated, or background-covered parts.
- Skip objects smaller than 10x10 pixels.
- Skip objects with less than 30% of the target panel visible.
- Label truncated objects only when the visible part still satisfies the class
  definition; clamp the box exactly to the image boundary.
- If two target objects overlap, annotate each visible instance separately.
- Do not label an object only because its shape, mounting, or nearby context
  suggests it might be a traffic sign.

## Global Bounding Box Policy

- Use integer pixel coordinates in the original image coordinate frame.
- Use `xyxy` pixel format: `[xmin, ymin, xmax, ymax]`.
- Draw one bounding box per visible target instance.
- Include only the visible pixels of the target sign panel face.
- Exclude background, shadows, poles, brackets, support structures, mounting
  hardware, adjacent objects, and rendered visualization text.
- If the target is cut by the image boundary and truncation is allowed, clamp the
  box to the image edge.
- If another object occludes the sign, do not extend the box through the
  occluder to reconstruct the hidden area.

## Class 0: `traffic_sign`

### Positive Definition

Label this class only when all conditions hold:

- The object is a physical road traffic sign panel installed for traffic control
  or road guidance.
- The front face of the sign is visible from the camera viewpoint.
- The visible face contains identifiable official traffic-control evidence, such
  as a symbol, number, arrow, text, border color, warning/prohibition/mandatory
  shape, or direction/instruction layout.
- At least 30% of the sign panel is visible.
- The visible panel boundary can be localized after zoom or inspection.
- No global or class-specific negative rule applies.

### Negative Definition

Do not label this class if any condition holds:

- The object is the back side of a traffic sign.
- The object is side-facing and the front-face content is not identifiable.
- Only the pole, bracket, frame edge, back plate, side edge, sticker, shadow, or
  support structure is visible.
- The object is a shop sign, facility name board, billboard, banner,
  advertisement, house number, logo, decorative board, project board, company
  board, school board, address board, or building-identification board.
- The object is a generic rectangular board near a road but appears to name a
  place, building, facility, company, project, school, shop, or address rather
  than controlling or guiding road users.
- The object is a traffic light, road marking, vehicle, person, cone, barrier,
  lane marker, or other non-sign object.
- The object is too tiny, blurred, far away, occluded, or ambiguous to confirm as
  a front-facing road traffic sign after tight zoom.

### Bounding Box Rule

- Include: the visible physical sign panel face only.
- Exclude: poles, brackets, mounting hardware, outer support frames that are not
  part of the sign face, shadows, background, adjacent signs, and non-target
  objects.
- For circular signs: box the visible outer circular sign border tightly.
- For triangular signs: box the visible triangular sign panel tightly.
- For rectangular signs: box the visible rectangular sign panel tightly.
- For supplemental traffic panels: label the supplemental panel separately only
  if it is front-facing, visually separable, and satisfies the positive
  definition.
- For occlusion: box only visible sign-face pixels; do not infer the hidden full
  panel.
- For truncation: clamp to the image boundary if the remaining visible face still
  satisfies the positive definition.

### Examples To Label

- Front-facing warning signs.
- Front-facing prohibition signs.
- Front-facing mandatory signs.
- Front-facing speed-limit signs.
- Front-facing direction signs.
- Front-facing road-instruction signs.
- Official supplemental traffic panels mounted with a main traffic sign, when
  the panel is front-facing and visually separable.

### Examples To Skip

- Reverse-facing traffic sign backs.
- Side-facing signs whose front content is not identifiable.
- Sign poles, brackets, backs, frames, supports, and shadows.
- Shop signs, billboards, advertisements, banners, logos, and facility boards.
- Project, company, school, address, or building-name boards.
- Traffic lights, road markings, vehicles, people, cones, barriers, and lane
  markers.
- Tiny or blurred sign-like regions that cannot be confidently localized.

### Ambiguity Tie-Breakers

- If unsure whether the object is a traffic sign or background, skip it.
- If orientation is uncertain, skip it.
- If only the back plate or side edge is visible, skip it.
- If a sign-shaped object is installed near a road but appears to identify a
  place, building, facility, company, project, school, shop, or address, skip it.
- If the object might be an official traffic sign but the face content cannot be
  identified after tight zoom, skip it.

## Visual Example Policy

Visual examples are useful for AI agents, especially for hard positives and
near-miss negatives. Add them only when they are curated and named clearly.

Recommended structure:

```text
dataset/examples/
  positives/
    traffic_sign_front_facing_speed_limit.png
    traffic_sign_front_facing_direction_panel.png
  negatives/
    sign_back_do_not_label.png
    shop_board_do_not_label.png
    facility_board_do_not_label.png
  examples_manifest.md
```

Each example in `examples_manifest.md` should state:

- image path;
- class name or `negative`;
- bbox coordinates if it is a positive example;
- why it should be labeled or skipped;
- which rule in this guideline it demonstrates.

Do not use unreviewed examples as ground truth. If an example is ambiguous, put
it in the negative set or omit it.

## Final Checklist Before Adding Label

- The class name exists in `dataset/data.yaml`.
- The object satisfies the `traffic_sign` positive definition.
- No global or class-specific negative rule applies.
- The bbox includes only the visible sign panel face.
- The candidate was visually checked with `inspect`, `visual`, or `corners`.
- `verify` returned a fresh `verification_id`.
- The label is added only through `./a4od bbox --action add --verification-id`.
