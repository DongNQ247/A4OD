# Dataset Labeling Guidelines Example

Copy this file to `dataset/labeling_guidelines.md` when adapting A4OD to a new
object-detection dataset. Keep the headings stable so AI agents can parse the
rules without reading source code.

## Dataset Objective

Describe the detector's intended behavior in one short paragraph. State the
scene type, target object family, and what a correct detection should represent.

Example:
Train a detector for visible front-facing traffic signs used for road guidance
or traffic control.

## Class List Source

The authoritative class id and class name mapping is `dataset/data.yaml`.
Every class section below must match one entry from `dataset/data.yaml`.
Do not introduce class names here that are absent from `dataset/data.yaml`.

## Global Annotation Rules

- Label only classes listed in `dataset/data.yaml`.
- If an object remains ambiguous after tight zoom, skip it.
- State the minimum object size, for example: skip objects smaller than 10x10 pixels.
- State the occlusion threshold, for example: skip objects with less than 30% visible.
- State whether to annotate visible pixels only or full object extent.
- State whether truncated objects at image boundaries should be labeled.
- State how to handle overlapping instances.

## Global Bounding Box Policy

- Use integer pixel coordinates in the original image coordinate frame.
- Draw one bounding box per visible target instance.
- Include only the target object pixels defined by the class-specific rule.
- Exclude background, shadows, support structures, text overlays, and adjacent objects unless a class-specific rule says otherwise.
- If the object is cut by the image boundary and truncation is allowed, clamp the box to the image edge.

## Class 0: `<class_name>`

Replace `<class_name>` with the exact class name from `dataset/data.yaml`.

### Positive Definition

Label this class only when all conditions hold:

- The object is ...
- The visible evidence is ...
- The object is not excluded by any negative rule.

### Negative Definition

Do not label this class if any condition holds:

- The object is ...
- The object is too small, blurry, occluded, or ambiguous according to the global rules.
- The object belongs to a similar but excluded category.

### Bounding Box Rule

- Include: ...
- Exclude: ...
- For occlusion: ...
- For truncation: ...
- For unusual shapes: ...

### Examples To Label

- ...
- ...

### Examples To Skip

- ...
- ...

### Ambiguity Tie-Breakers

- If unsure between this class and background, skip.
- If unsure between two target classes, use the class-specific evidence rule or skip.
- If orientation, visibility, or category is uncertain after zoom, skip.

## Class 1: `<class_name>`

Repeat the same structure for every class in `dataset/data.yaml`.

### Positive Definition

- ...

### Negative Definition

- ...

### Bounding Box Rule

- ...

### Examples To Label

- ...

### Examples To Skip

- ...

### Ambiguity Tie-Breakers

- ...

## Final Checklist Before Adding Label

- The class name exists in `dataset/data.yaml`.
- The object satisfies the class positive definition.
- No global or class-specific negative rule applies.
- The bounding box follows the visible/full extent policy for this dataset.
- The candidate was verified with `corners` or `visual`.
- The label is added only through `annotation.py bbox --action add`.
