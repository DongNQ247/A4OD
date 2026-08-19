# Dataset Labeling Guidelines

This file defines the dataset-specific labeling rules for A4OD. AI annotation
agents must read this file together with `TOOL_CONTRACT.md`,
`.a4od/contract.yaml`, and `dataset/data.yaml` before adding labels.

## Dataset Objective

Train a detector for visible dogs and balls in the image frame.

The detector is intended to localize the visible pixels of real dogs and balls.
Background objects, toys that are not balls, people, vehicles, signs, furniture,
and other animals are background unless they satisfy a class-specific positive
definition below.

## Class List Source

The authoritative class id and class name mapping is `dataset/data.yaml`.
Every class section below must match one entry from `dataset/data.yaml`.
Do not introduce, rename, merge, or reinterpret classes in this file.

Current class list:

- Class 0: `dog`
- Class 1: `ball`

## Global Annotation Rules

- Label only classes listed in `dataset/data.yaml`.
- Exclusion rules override inclusion rules.
- If an object remains ambiguous after tight zoom and inspection, skip it.
- Annotate visible object pixels only. Do not infer hidden, occluded, blurred,
  truncated, or background-covered parts.
- Skip objects smaller than 10x10 pixels.
- Skip objects with less than 30% of the target object visible.
- Label truncated objects only when the visible part still satisfies the class
  definition; clamp the box exactly to the image boundary.
- If two target objects overlap, annotate each visible instance separately.
- If a dog and ball overlap, annotate each visible instance separately.
- Do not label an object only because nearby context suggests it might be a dog
  or ball.

## Global Bounding Box Policy

- Use integer pixel coordinates in the original image coordinate frame.
- Use `xyxy` pixel format: `[xmin, ymin, xmax, ymax]`.
- Draw one bounding box per visible target instance.
- Include only the visible pixels of the target object.
- Exclude background, shadows, leashes, collars, harnesses, clothing, handlers,
  adjacent objects, and rendered visualization text.
- If the target is cut by the image boundary and truncation is allowed, clamp the
  box to the image edge.
- If another object occludes the target, do not extend the box through the
  occluder to reconstruct the hidden area.

## Class 0: `dog`

### Positive Definition

Label this class only when all conditions hold:

- The object is a real dog visible in the image.
- The visible body, head, legs, fur, or silhouette provides enough evidence to
  identify it as a dog.
- At least 30% of the dog is visible.
- The visible dog boundary can be localized after zoom or inspection.
- No global or class-specific negative rule applies.

### Negative Definition

Do not label this class if any condition holds:

- The object is a toy dog, statue, drawing, logo, poster, photo on a screen, or
  other non-real depiction of a dog.
- The object is another animal, such as a cat, fox, wolf, goat, sheep, cow,
  horse, or bird, unless it is clearly intended to be labeled as a dog in this
  dataset.
- Only a leash, collar, harness, clothing, shadow, reflection, or carried object
  is visible.
- The object is too tiny, blurred, far away, occluded, or ambiguous to confirm as
  a dog after tight zoom.

### Bounding Box Rule

- Include: the visible dog body pixels, including visible head, torso, legs,
  tail, ears, and fur.
- Exclude: leash, collar, harness, clothing, carried items, handler body parts,
  shadows, reflections, background, and non-target objects.
- For occlusion: box only visible dog pixels; do not infer the hidden full
  dog.
- For truncation: clamp to the image boundary if the remaining visible dog still
  satisfies the positive definition.

### Examples To Label

- Standing, sitting, running, or lying dogs.
- Partially occluded dogs when at least 30% is visible and the dog is
  identifiable.
- Truncated dogs at the image boundary when the visible portion remains
  identifiable.

### Examples To Skip

- Dog statues, plush toys, cartoons, posters, logos, and screen images.
- Other animals that are not clearly dogs.
- Leashes, collars, harnesses, clothing, handlers, shadows, and reflections.
- Tiny or blurred dog-like regions that cannot be confidently localized.

### Ambiguity Tie-Breakers

- If unsure whether the object is a dog or another animal/background, skip it.
- If only a very small body part is visible and the dog cannot be confidently
  identified, skip it.
- If the object might be a dog but cannot be identified after tight zoom, skip
  it.

## Class 1: `ball`

### Positive Definition

Label this class only when all conditions hold:

- The object is a physical ball visible in the image.
- The visible shape, contour, color, texture, or context provides enough evidence
  to identify it as a ball.
- At least 30% of the ball is visible.
- The visible ball boundary can be localized after zoom or inspection.
- No global or class-specific negative rule applies.

### Negative Definition

Do not label this class if any condition holds:

- The object is a drawing, logo, printed pattern, reflection, shadow, screen
  image, or other non-real depiction of a ball.
- The object is a wheel, helmet, bowl, plate, fruit, balloon, round sign, light,
  or other round object that is not clearly a ball.
- The object is too tiny, blurred, far away, occluded, or ambiguous to confirm as
  a ball after tight zoom.

### Bounding Box Rule

- Include: the visible ball pixels only.
- Exclude: background, shadows, hands, paws, nets, sticks, strings, motion blur
  outside the physical ball, and adjacent objects.
- For round or spherical balls: box the visible outer contour tightly.
- For partially occluded balls: box only the visible portion; do not infer the
  hidden full ball.
- For truncation: clamp to the image boundary if the remaining visible ball
  still satisfies the positive definition.

### Examples To Label

- Sports balls, toy balls, tennis balls, soccer balls, basketballs, rubber balls,
  and similar physical balls.
- Partially occluded balls when at least 30% is visible and the ball is
  identifiable.
- Truncated balls at the image boundary when the visible portion remains
  identifiable.

### Examples To Skip

- Wheels, round traffic signs, balloons, fruits, bowls, plates, lights, logos,
  printed graphics, and screen images.
- Shadows, reflections, motion trails, nets, rackets, sticks, paws, and hands.
- Tiny or blurred ball-like regions that cannot be confidently localized.

### Ambiguity Tie-Breakers

- If unsure whether the object is a ball or another round object, skip it.
- If a round object might be a ball but cannot be identified after tight zoom,
  skip it.

## Visual Example Policy

Visual examples are useful for AI agents, especially for hard positives and
near-miss negatives. Add them only when they are curated and named clearly.

Recommended structure:

```text
dataset/examples/
  positives/
    dog_standing_visible_body.png
    ball_soccer_visible_contour.png
  negatives/
    dog_plush_toy_do_not_label.png
    round_sign_do_not_label_as_ball.png
    leash_do_not_label.png
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
- The object satisfies the target class positive definition.
- No global or class-specific negative rule applies.
- The bbox includes only the visible target object pixels.
- The candidate was visually checked with `inspect`, `visual`, or `corners`.
- `verify` returned a fresh `verification_id`.
- The label is added only through `./a4od bbox --action add --verification-id`.
