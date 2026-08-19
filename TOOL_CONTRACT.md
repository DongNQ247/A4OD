# A4OD Tool Contract

This contract is the source of truth for AI agents using A4OD. Agents should
read this file, `dataset/data.yaml`, and `dataset/labeling_guidelines.md`
instead of reading implementation source code.

Preferred CLI: `./a4od`.
Compatibility CLI: `.venv/bin/python annotation.py`.

Run `./a4od capabilities` for machine-readable discovery, including API
version, coordinate contract, mutation rules, and schema file paths.

## Stable Inputs

- `dataset/data.yaml`: authoritative class id and class name mapping.
- `dataset/labeling_guidelines.md`: authoritative semantic annotation rules.
- `data/<image_name>`: default staging location for images to annotate.
- `.a4od/contract.yaml`: machine-readable CLI/API contract.
- `schemas/*.v1.json`: JSON schemas for agent-critical command outputs.

## Stable Outputs

- Labels for `data/<image_name>` are written to `dataset/labels/<image_stem>.txt`.
- Rendered helper images are written to `tmp/<image_stem>/`.
- CLI commands print JSON to stdout.
- Exit code `0` means success. Non-zero means failure.

## Error Shape

Errors use this JSON shape:

```json
{
  "ok": false,
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable explanation",
    "details": {},
    "suggested_recovery": "Optional next step"
  }
}
```

Core error codes include `IMAGE_NOT_FOUND`, `UNKNOWN_CLASS`, `INVALID_BBOX`,
`INVALID_COORDINATE`, `INVALID_DATASET`, `LABEL_NOT_FOUND`,
`LABEL_WRITE_FAILED`, `VERIFICATION_REQUIRED`, and `VERIFICATION_MISMATCH`.

Success outputs include `ok: true` and keep `status: "success"` for
compatibility.

## Coordinate Contract

- Input bbox format: `xyxy`.
- Coordinate space: integer image pixels.
- Origin: top-left.
- Boundary semantics: `x1,y1` inclusive; `x2,y2` exclusive.
- Minimum bbox size: `1x1` pixel at the CLI contract level. Dataset-specific
  semantic minimums belong in `dataset/labeling_guidelines.md`.

## Agent Rules

- Never edit or create YOLO `.txt` label files directly.
- Use `./a4od bbox --action add` and `./a4od bbox --action delete` for label
  writes.
- Never use a class that is absent from `dataset/data.yaml`.
- Run `./a4od doctor --data dataset/data.yaml` before annotation work.
- Run `./a4od verify <image> <class> <x1> <y1> <x2> <y2>` before writing.
- Pass the returned `verification_id` to `bbox add` as `--verification-id`.
- `bbox add --force` bypasses the gate and returns a warning. Use it only when
  an external workflow has already verified the candidate.

## Core Commands

- `grid`: full-image coordinate grid.
- `zoom`: cropped ROI with global coordinates.
- `corners`: compact four-corner boundary check.
- `visual`: candidate bbox preview. Use `--crop-context` for token-efficient crops.
- `verify`: deterministic candidate validation and verification id.
- `bbox`: list, add, delete, or dry-run YOLO labels.
- `inspect`: compact candidate inspection sheet.
- `doctor`: dataset and contract checks.
- `schema`: machine-readable command/output summary.
- `capabilities`: machine-readable discovery for commands, schemas, coordinate
  contract, and mutation rules.
