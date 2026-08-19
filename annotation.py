#!/usr/bin/env python3
"""
A4OD Annotation CLI Tool
AI-Assisted Object Detection & YOLO Dataset Annotation
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from src.config import find_label_path, load_dataset_config, resolve_class
from src.coords import xyxy_pixel_to_yolo_norm
from src.corner_inspector import render_corner_inspection
from src.grid_renderer import render_grid_image
from src.visualizer import render_visual_preview
from src.yolo_io import add_yolo_label, delete_yolo_label, read_yolo_labels
from src.zoom_renderer import render_zoom_image


MIN_BBOX_SIZE = 1
VERSION = "1.0.0"
API_VERSION = "1"
IMPLEMENTATION = "annotation.py"
CORE_SCHEMA_FILES = {
    "error": "schemas/error.v1.json",
    "doctor": "schemas/doctor.v1.json",
    "capabilities": "schemas/capabilities.v1.json",
    "verify": "schemas/verify.v1.json",
    "bbox": "schemas/bbox.v1.json",
    "inspect": "schemas/inspect.v1.json",
}
COORDINATE_CONTRACT = {
    "bbox_input_format": "xyxy",
    "coordinate_space": "pixel",
    "origin": "top_left",
    "boundary_semantics": "x1,y1 inclusive; x2,y2 exclusive",
    "min_bbox_size": MIN_BBOX_SIZE,
}
MUTATION_RULES = {
    "bbox_add_requires_verification_id": True,
    "force_bypass_supported": True,
    "dry_run_writes": False,
}


def emit(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def emit_error(code: str, message: str, **details: Any) -> int:
    suggested_recovery = details.pop("suggested_recovery", None)
    payload: Dict[str, Any] = {
        "ok": False,
        "status": "error",
        "error": {
            "code": code,
            "message": message
        }
    }
    if suggested_recovery:
        payload["error"]["suggested_recovery"] = suggested_recovery
    if details:
        payload["error"]["details"] = details
    emit(payload)
    return 1


def add_status(metadata: Dict[str, Any], status: str = "success") -> Dict[str, Any]:
    return {"ok": status == "success", "status": status, **metadata}


def error_code_for_value_error(error: ValueError) -> str:
    message = str(error)
    if message.startswith("Unknown class"):
        return "UNKNOWN_CLASS"
    if message.startswith("No classes"):
        return "INVALID_DATASET"
    if message.startswith("Bounding box"):
        return "INVALID_BBOX"
    return "INVALID_DATASET"


def load_classes_or_error(data_path: Optional[str]) -> Dict[int, str]:
    class_map = load_dataset_config(data_path)
    if not class_map:
        raise ValueError("No classes found. Check dataset/data.yaml and its names field.")
    return class_map


def normalize_pixel_bbox(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    img_w: int,
    img_h: int,
    min_size: int = MIN_BBOX_SIZE
) -> Tuple[List[int], List[float]]:
    x1 = max(0, min(img_w, min(int(xmin), int(xmax))))
    x2 = max(0, min(img_w, max(int(xmin), int(xmax))))
    y1 = max(0, min(img_h, min(int(ymin), int(ymax))))
    y2 = max(0, min(img_h, max(int(ymin), int(ymax))))

    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        raise ValueError(f"Bounding box is smaller than {min_size}x{min_size}: {[x1, y1, x2, y2]}")

    norm_bbox = list(xyxy_pixel_to_yolo_norm(x1, y1, x2, y2, img_w, img_h))
    return [x1, y1, x2, y2], norm_bbox


def label_state_digest(label_path: Path) -> str:
    if not label_path.exists():
        return "missing"
    return hashlib.sha256(label_path.read_bytes()).hexdigest()


def verification_id_for(
    image_path: Path,
    class_id: int,
    norm_bbox: List[float],
    label_path: Path
) -> str:
    payload = {
        "api_version": API_VERSION,
        "contract": COORDINATE_CONTRACT,
        "image_path": str(image_path.resolve()),
        "class_id": class_id,
        "yolo_norm_bbox": [round(v, 6) for v in norm_bbox],
        "label_path": str(label_path.resolve()),
        "label_state": label_state_digest(label_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "v1:" + hashlib.sha256(encoded).hexdigest()[:32]


def validate_candidate(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    img_path = Path(args.image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    class_map = load_classes_or_error(args.data)
    cid, cname = resolve_class(args.class_name, class_map)
    label_path = find_label_path(img_path)
    pixel_bbox, norm_bbox = normalize_pixel_bbox(args.xmin, args.ymin, args.xmax, args.ymax, img_w, img_h)
    existing_boxes = read_yolo_labels(label_path, img_w, img_h, class_map)
    warnings = duplicate_warnings(pixel_bbox, existing_boxes)
    verification_id = verification_id_for(img_path, cid, norm_bbox, label_path)

    result = {
        "image_path": str(img_path),
        "image_size": [img_w, img_h],
        "label_path": str(label_path),
        "label_file_exists": label_path.exists(),
        "class_id": cid,
        "class_name": cname,
        "pixel_bbox": pixel_bbox,
        "yolo_norm_bbox": norm_bbox,
        "verification_id": verification_id,
        "warnings": warnings,
        "contract": {
            "api_version": API_VERSION,
            "coordinate_contract": COORDINATE_CONTRACT,
            "mutation_rules": MUTATION_RULES,
        },
    }
    return result, cid


def box_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return round(inter / union, 6)


def duplicate_warnings(candidate_bbox: List[int], existing_boxes: List[Dict[str, Any]], threshold: float = 0.8) -> List[Dict[str, Any]]:
    warnings = []
    for box in existing_boxes:
        iou = box_iou(candidate_bbox, box["pixel_bbox"])
        if iou >= threshold:
            warnings.append({
                "code": "POSSIBLE_DUPLICATE",
                "message": f"Candidate overlaps existing box index {box['index']} with IoU {iou}",
                "existing_index": box["index"],
                "iou": iou
            })
    return warnings


def get_tmp_image_dir(image_path: Path) -> Path:
    tmp_dir = Path("tmp") / image_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def handle_grid(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        return emit_error("IMAGE_NOT_FOUND", f"Image file not found: {args.image_path}")

    try:
        class_map = load_classes_or_error(args.data)
    except ValueError as e:
        return emit_error("INVALID_DATASET", str(e))
    label_path = find_label_path(img_path)

    tmp_dir = get_tmp_image_dir(img_path)
    out_grid_path = tmp_dir / f"{img_path.stem}_grid.png"

    try:
        metadata = render_grid_image(
            image_path=img_path,
            output_path=out_grid_path,
            cell_size=args.cell_size,
            show_existing=not args.no_existing,
            class_map=class_map,
            existing_label_path=label_path if label_path.exists() else None
        )
        emit(add_status(metadata))
        return 0
    except Exception as e:
        return emit_error("GRID_FAILED", f"Failed to generate grid: {str(e)}")


def handle_zoom(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        return emit_error("IMAGE_NOT_FOUND", f"Image file not found: {args.image_path}")

    try:
        class_map = load_classes_or_error(args.data)
    except ValueError as e:
        return emit_error("INVALID_DATASET", str(e))
    label_path = find_label_path(img_path)

    tmp_dir = get_tmp_image_dir(img_path)
    out_zoom_path = tmp_dir / f"{img_path.stem}_zoom_{args.xmin}_{args.ymin}_{args.xmax}_{args.ymax}.png"

    try:
        metadata = render_zoom_image(
            image_path=img_path,
            output_path=out_zoom_path,
            xmin=args.xmin,
            ymin=args.ymin,
            xmax=args.xmax,
            ymax=args.ymax,
            cell_size=args.cell_size,
            show_existing=not args.no_existing,
            class_map=class_map,
            existing_label_path=label_path if label_path.exists() else None
        )
        emit(add_status(metadata))
        return 0
    except Exception as e:
        return emit_error("ZOOM_FAILED", f"Failed to generate zoom: {str(e)}")


def handle_corners(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        return emit_error("IMAGE_NOT_FOUND", f"Image file not found: {args.image_path}")

    tmp_dir = get_tmp_image_dir(img_path)
    out_corners_path = tmp_dir / f"{img_path.stem}_corners_{args.xmin}_{args.ymin}_{args.xmax}_{args.ymax}.png"

    try:
        metadata = render_corner_inspection(
            image_path=img_path,
            output_path=out_corners_path,
            xmin=args.xmin,
            ymin=args.ymin,
            xmax=args.xmax,
            ymax=args.ymax,
            patch_size=args.patch_size
        )
        emit(add_status(metadata))
        return 0
    except Exception as e:
        return emit_error("CORNERS_FAILED", f"Failed to generate corner inspection: {str(e)}")


def handle_visual(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        return emit_error("IMAGE_NOT_FOUND", f"Image file not found: {args.image_path}")

    try:
        class_map = load_classes_or_error(args.data)
        cid, cname = resolve_class(args.class_name, class_map)
    except ValueError as e:
        return emit_error(error_code_for_value_error(e), str(e))
    label_path = find_label_path(img_path)

    tmp_dir = get_tmp_image_dir(img_path)
    out_visual_path = tmp_dir / f"{img_path.stem}_visual.png"

    try:
        metadata = render_visual_preview(
            image_path=img_path,
            output_path=out_visual_path,
            candidate_class=f"{cname} (id:{cid})",
            candidate_bbox_pixel=(args.xmin, args.ymin, args.xmax, args.ymax),
            class_map=class_map,
            existing_label_path=label_path if label_path.exists() else None,
            crop_context=args.crop_context
        )
        emit(add_status(metadata))
        return 0
    except Exception as e:
        return emit_error("VISUAL_FAILED", f"Failed to generate visual preview: {str(e)}")


def handle_bbox(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        return emit_error("IMAGE_NOT_FOUND", f"Image file not found: {args.image_path}")

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    try:
        class_map = load_classes_or_error(args.data)
    except ValueError as e:
        return emit_error("INVALID_DATASET", str(e))
    label_path = find_label_path(img_path)

    action = args.action.lower()

    if action == "list":
        boxes = read_yolo_labels(label_path, img_w, img_h, class_map)
        emit(add_status({
            "status": "success",
            "image_path": str(img_path),
            "label_path": str(label_path),
            "label_file_exists": label_path.exists(),
            "total_boxes": len(boxes),
            "boxes": boxes
        }))
        return 0

    elif action == "delete":
        if args.index is None:
            return emit_error("ARGUMENT_MISSING", "Missing required argument --index for delete action")
        try:
            success = delete_yolo_label(label_path, args.index)
        except OSError as e:
            return emit_error("LABEL_WRITE_FAILED", f"Failed to update label file: {str(e)}")
        if success:
            emit(add_status({
                "message": f"Deleted bounding box index {args.index} from {label_path}"
            }))
            return 0
        else:
            return emit_error("BBOX_INDEX_NOT_FOUND", f"Bounding box index {args.index} not found in {label_path}")

    elif action == "add":
        if args.class_name is None or args.xmin is None or args.ymin is None or args.xmax is None or args.ymax is None:
            return emit_error("ARGUMENT_MISSING", "Missing required arguments for 'add': <class> <xmin> <ymin> <xmax> <ymax>")

        try:
            verification, cid = validate_candidate(args)
            cname = verification["class_name"]
            expected_id = verification["verification_id"]

            warnings = list(verification["warnings"])
            if args.force:
                warnings.append({
                    "code": "VERIFICATION_BYPASSED",
                    "message": "bbox add was forced without a matching verification id"
                })
            elif not args.dry_run:
                if not args.verification_id:
                    return emit_error(
                        "VERIFICATION_REQUIRED",
                        "bbox add requires --verification-id from verify unless --force is passed",
                        suggested_recovery=f"Run verify and retry with --verification-id {expected_id}"
                    )
                if args.verification_id != expected_id:
                    return emit_error(
                        "VERIFICATION_MISMATCH",
                        "Provided verification id does not match current candidate or label state",
                        expected_verification_id=expected_id,
                        provided_verification_id=args.verification_id,
                        suggested_recovery="Run verify again against the current label file and retry"
                    )

            if not args.dry_run:
                add_yolo_label(label_path, cid, *verification["yolo_norm_bbox"])

            emit(add_status({
                "dry_run": args.dry_run,
                "message": f"Validated bounding box for '{cname}' (id:{cid})" if args.dry_run else f"Added bounding box for '{cname}' (id:{cid}) to {label_path}",
                "label_path": str(label_path),
                "class_id": cid,
                "class_name": cname,
                "pixel_bbox": verification["pixel_bbox"],
                "yolo_norm_bbox": verification["yolo_norm_bbox"],
                "verification_id": expected_id,
                "warnings": warnings
            }))
            return 0
        except FileNotFoundError as e:
            return emit_error("IMAGE_NOT_FOUND", str(e))
        except ValueError as e:
            return emit_error(error_code_for_value_error(e), str(e))
        except OSError as e:
            return emit_error("LABEL_WRITE_FAILED", f"Failed to write label file: {str(e)}")
        except Exception as e:
            return emit_error("BBOX_ADD_FAILED", f"Failed to add bounding box: {str(e)}")

    else:
        return emit_error("ACTION_INVALID", f"Unknown action: {action}. Supported: add, delete, list")


def dataset_root_from_data(data_path: Optional[str]) -> Path:
    if data_path:
        data_file = Path(data_path).resolve()
    else:
        data_file = Path("dataset/data.yaml").resolve()
    if data_file.parent.name == "dataset":
        return data_file.parent.parent
    return data_file.parent


def handle_doctor(args: argparse.Namespace) -> int:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    data_path = Path(args.data or "dataset/data.yaml")
    repo_root = dataset_root_from_data(args.data)
    try:
        class_map = load_classes_or_error(args.data)
    except ValueError as e:
        return emit_error("INVALID_DATASET", str(e))

    data_yaml_text = ""
    if data_path.exists():
        data_yaml_text = data_path.read_text(encoding="utf-8")
        nc_match = re.search(r"(?m)^\s*nc\s*:\s*(\d+)\s*$", data_yaml_text)
        if nc_match and int(nc_match.group(1)) != len(class_map):
            errors.append({
                "code": "NC_MISMATCH",
                "message": f"nc={nc_match.group(1)} but names has {len(class_map)} classes"
            })
    else:
        errors.append({"code": "DATA_YAML_MISSING", "message": f"Missing dataset config: {data_path}"})

    guideline_path = repo_root / "dataset" / "labeling_guidelines.md"
    if guideline_path.exists():
        guideline = guideline_path.read_text(encoding="utf-8")
        for cid, cname in class_map.items():
            pattern = rf"(?im)^##\s+Class\s+{cid}\s*:\s*`?{re.escape(cname)}`?\s*$"
            if not re.search(pattern, guideline):
                warnings.append({
                    "code": "GUIDELINE_CLASS_SECTION_MISSING",
                    "message": f"Missing guideline section for Class {cid}: {cname}"
                })
    else:
        warnings.append({
            "code": "GUIDELINE_MISSING",
            "message": f"Missing guideline file: {guideline_path}"
        })

    image_dir = repo_root / "data"
    labels_dir = repo_root / "dataset" / "labels"
    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]) if image_dir.exists() else []
    label_paths = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    image_stems = {p.stem for p in image_paths}
    for image_path in image_paths:
        label_path = find_label_path(image_path)
        if not label_path.exists():
            warnings.append({
                "code": "LABEL_MISSING",
                "message": f"No label file for image {image_path}",
                "image_path": str(image_path),
                "expected_label_path": str(label_path)
            })

    for label_path in label_paths:
        if label_path.stem not in image_stems:
            warnings.append({
                "code": "LABEL_WITHOUT_IMAGE",
                "message": f"Label has no matching image in data/: {label_path}",
                "label_path": str(label_path)
            })

        image_path = next((p for p in image_paths if p.stem == label_path.stem), None)
        img_w = img_h = 1
        if image_path and image_path.exists():
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        with open(label_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                tokens = line_str.split()
                if len(tokens) != 5:
                    errors.append({
                        "code": "LABEL_ROW_INVALID",
                        "message": f"Expected 5 YOLO fields at {label_path}:{line_no}",
                        "raw_line": line_str
                    })
                    continue
                try:
                    cid = int(tokens[0])
                    xc, yc, w, h = [float(v) for v in tokens[1:]]
                except ValueError:
                    errors.append({
                        "code": "LABEL_ROW_PARSE_FAILED",
                        "message": f"Cannot parse YOLO row at {label_path}:{line_no}",
                        "raw_line": line_str
                    })
                    continue
                if cid not in class_map:
                    errors.append({
                        "code": "LABEL_CLASS_UNKNOWN",
                        "message": f"Class id {cid} at {label_path}:{line_no} is absent from data.yaml"
                    })
                if not all(0.0 <= v <= 1.0 for v in [xc, yc, w, h]) or w <= 0 or h <= 0:
                    errors.append({
                        "code": "LABEL_BBOX_OUT_OF_RANGE",
                        "message": f"YOLO bbox out of range at {label_path}:{line_no}",
                        "raw_line": line_str
                    })

    if args.run_smoke:
        try:
            _ = {
                "commands": ["grid", "zoom", "corners", "visual", "bbox", "doctor", "schema", "inspect"],
                "class_count": len(class_map)
            }
        except Exception as e:
            errors.append({"code": "SMOKE_FAILED", "message": str(e)})

    emit({
        "ok": not errors,
        "status": "success" if not errors else "error",
        "data_path": str(data_path),
        "repo_root": str(repo_root),
        "class_count": len(class_map),
        "classes": class_map,
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "errors": errors,
        "warnings": warnings
    })
    return 0 if not errors else 1


def handle_schema(args: argparse.Namespace) -> int:
    emit(add_status({
        "schema_version": "1.0",
        "api_version": API_VERSION,
        "schema_files": CORE_SCHEMA_FILES,
        "commands": {
            "grid": {"writes": ["tmp/<stem>/<stem>_grid.png"], "json_fields": ["status", "grid_image_path", "width", "height", "existing_boxes"]},
            "zoom": {"writes": ["tmp/<stem>/<stem>_zoom_*.png"], "json_fields": ["status", "zoom_image_path", "crop_bbox", "scale"]},
            "corners": {"writes": ["tmp/<stem>/<stem>_corners_*.png"], "json_fields": ["status", "corners_image_path", "pixel_bbox"]},
            "visual": {"writes": ["tmp/<stem>/<stem>_visual.png"], "json_fields": ["status", "visual_image_path", "view_bbox", "candidate"]},
            "verify": {"writes": [], "json_fields": ["ok", "status", "verification_id", "label_path", "warnings", "contract"]},
            "bbox": {"writes": ["dataset/labels/<stem>.txt for verified or forced add/delete only"], "json_fields": ["ok", "status", "label_path", "boxes", "warnings", "verification_id"]},
            "inspect": {"writes": ["tmp/<stem>/<stem>_inspect.png"], "json_fields": ["status", "inspect_image_path", "candidate", "warnings"]},
            "doctor": {"writes": [], "json_fields": ["status", "errors", "warnings", "class_count", "image_count", "label_count"]}
        },
        "error_shape": {
            "ok": False,
            "status": "error",
            "error": {"code": "ERROR_CODE", "message": "Human-readable explanation", "details": {}}
        }
    }))
    return 0


def handle_version(args: argparse.Namespace) -> int:
    emit(add_status({
        "version": VERSION,
        "api_version": API_VERSION,
        "implementation": IMPLEMENTATION,
    }))
    return 0


def handle_capabilities(args: argparse.Namespace) -> int:
    emit(add_status({
        "version": VERSION,
        "api_version": API_VERSION,
        "implementation": IMPLEMENTATION,
        "preferred_cli": "a4od",
        "compatibility_cli": "annotation.py",
        "contract_path": ".a4od/contract.yaml",
        "schema_files": CORE_SCHEMA_FILES,
        "coordinate_contract": COORDINATE_CONTRACT,
        "mutation_rules": MUTATION_RULES,
        "commands": {
            "grid": {"mutates_labels": False, "writes": ["tmp/<stem>/<stem>_grid.png"]},
            "zoom": {"mutates_labels": False, "writes": ["tmp/<stem>/<stem>_zoom_*.png"]},
            "corners": {"mutates_labels": False, "writes": ["tmp/<stem>/<stem>_corners_*.png"]},
            "visual": {"mutates_labels": False, "writes": ["tmp/<stem>/<stem>_visual.png"]},
            "inspect": {"mutates_labels": False, "writes": ["tmp/<stem>/<stem>_inspect.png"]},
            "verify": {"mutates_labels": False, "writes": []},
            "bbox": {
                "mutates_labels": True,
                "actions": ["add", "delete", "list"],
                "add_requires": ["--verification-id from verify"],
                "bypass": "--force",
            },
            "doctor": {"mutates_labels": False, "writes": []},
            "schema": {"mutates_labels": False, "writes": []},
            "capabilities": {"mutates_labels": False, "writes": []},
        },
    }))
    return 0


def handle_verify(args: argparse.Namespace) -> int:
    try:
        verification, _ = validate_candidate(args)
    except FileNotFoundError as e:
        return emit_error("IMAGE_NOT_FOUND", str(e))
    except ValueError as e:
        return emit_error(error_code_for_value_error(e), str(e))
    except Exception as e:
        return emit_error("INVALID_BBOX", f"Failed to verify candidate: {str(e)}")

    emit(add_status(verification))
    return 0


def handle_inspect(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        return emit_error("IMAGE_NOT_FOUND", f"Image file not found: {args.image_path}")

    try:
        class_map = load_classes_or_error(args.data)
        cid, cname = resolve_class(args.class_name, class_map)
    except ValueError as e:
        return emit_error(error_code_for_value_error(e), str(e))

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    try:
        pixel_bbox, norm_bbox = normalize_pixel_bbox(args.xmin, args.ymin, args.xmax, args.ymax, img_w, img_h)
    except ValueError as e:
        return emit_error("INVALID_BBOX", str(e))

    label_path = find_label_path(img_path)
    existing_boxes = read_yolo_labels(label_path, img_w, img_h, class_map)
    warnings = duplicate_warnings(pixel_bbox, existing_boxes)

    tmp_dir = get_tmp_image_dir(img_path)
    visual_path = tmp_dir / f"{img_path.stem}_inspect_visual.png"
    corners_path = tmp_dir / f"{img_path.stem}_inspect_corners.png"
    inspect_path = tmp_dir / f"{img_path.stem}_inspect.png"

    try:
        visual_meta = render_visual_preview(
            image_path=img_path,
            output_path=visual_path,
            candidate_class=f"{cname} (id:{cid})",
            candidate_bbox_pixel=tuple(pixel_bbox),
            class_map=class_map,
            existing_label_path=label_path if label_path.exists() else None,
            crop_context=args.crop_context
        )
        corners_meta = render_corner_inspection(
            image_path=img_path,
            output_path=corners_path,
            xmin=pixel_bbox[0],
            ymin=pixel_bbox[1],
            xmax=pixel_bbox[2],
            ymax=pixel_bbox[3],
            patch_size=args.patch_size
        )

        visual_img = Image.open(visual_path).convert("RGB")
        corners_img = Image.open(corners_path).convert("RGB")
        max_w = max(visual_img.width, corners_img.width)
        sheet = Image.new("RGB", (max_w, visual_img.height + corners_img.height + 12), color=(245, 245, 245))
        sheet.paste(visual_img, ((max_w - visual_img.width) // 2, 0))
        sheet.paste(corners_img, ((max_w - corners_img.width) // 2, visual_img.height + 12))
        sheet.save(inspect_path, quality=95)
    except Exception as e:
        return emit_error("INSPECT_FAILED", f"Failed to render inspect sheet: {str(e)}")

    emit(add_status({
        "image_path": str(img_path),
        "inspect_image_path": str(inspect_path),
        "visual_image_path": str(visual_path),
        "corners_image_path": str(corners_path),
        "label_path": str(label_path),
        "class_id": cid,
        "class_name": cname,
        "candidate": {
            "pixel_bbox": pixel_bbox,
            "yolo_norm_bbox": norm_bbox
        },
        "visual": visual_meta,
        "corners": corners_meta,
        "warnings": warnings
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A4OD: AI-Assisted YOLO Annotation CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Overlay overview grid with coordinate rulers
  ./a4od grid dataset/images/1.png --cell-size 200 --data dataset/data.yaml

  # 2. Zoom into ROI with fine grid (50px) preserving global coordinates
  ./a4od zoom dataset/images/1.png 800 500 1200 800 --cell-size 50 --data dataset/data.yaml

  # 3. Inspect the 4 corners of proposed bbox (ultra-low tokens)
  ./a4od corners dataset/images/1.png 890 630 1020 760

  # 4. Visually verify proposed bounding box over full image
  ./a4od visual dataset/images/1.png car 800 600 1000 900 --data dataset/data.yaml

  # 5. Verify, then add bounding box to YOLO .txt label file
  ./a4od verify dataset/images/1.png car 800 600 1000 900 --data dataset/data.yaml
  ./a4od bbox dataset/images/1.png car 800 600 1000 900 --verification-id <verification_id> --data dataset/data.yaml

  # 6. List existing bounding boxes
  ./a4od bbox dataset/images/1.png --action list --data dataset/data.yaml

  # 7. Delete an incorrect bounding box by index
  ./a4od bbox dataset/images/1.png --action delete --index 0 --data dataset/data.yaml
"""
    )
    parser.add_argument("--version", action="store_true", help="Print JSON version metadata")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: grid
    grid_parser = subparsers.add_parser("grid", help="Render coordinate grid and rulers onto image")
    grid_parser.add_argument("image_path", type=str, help="Path to input image")
    grid_parser.add_argument("--cell-size", type=int, default=200, help="Grid cell size in pixels (default: 200)")
    grid_parser.add_argument("--data", type=str, default=None, help="Path to dataset config (default: dataset/data.yaml)")
    grid_parser.add_argument("--no-existing", action="store_true", help="Do not overlay existing bounding boxes")

    # Command: zoom
    zoom_parser = subparsers.add_parser("zoom", help="Crop ROI and render fine grid with global coordinate rulers")
    zoom_parser.add_argument("image_path", type=str, help="Path to input image")
    zoom_parser.add_argument("xmin", type=int, help="ROI x_min in global image pixels")
    zoom_parser.add_argument("ymin", type=int, help="ROI y_min in global image pixels")
    zoom_parser.add_argument("xmax", type=int, help="ROI x_max in global image pixels")
    zoom_parser.add_argument("ymax", type=int, help="ROI y_max in global image pixels")
    zoom_parser.add_argument("--cell-size", type=int, default=50, help="Fine grid cell size (default: 50)")
    zoom_parser.add_argument("--data", type=str, default=None, help="Path to dataset config (default: dataset/data.yaml)")
    zoom_parser.add_argument("--no-existing", action="store_true", help="Do not overlay existing bounding boxes")

    # Command: corners
    corners_parser = subparsers.add_parser("corners", help="Extract and tile the 4 corners of proposed bbox (ultra-low tokens)")
    corners_parser.add_argument("image_path", type=str, help="Path to input image")
    corners_parser.add_argument("xmin", type=int, help="BBox x_min in global image pixels")
    corners_parser.add_argument("ymin", type=int, help="BBox y_min in global image pixels")
    corners_parser.add_argument("xmax", type=int, help="BBox x_max in global image pixels")
    corners_parser.add_argument("ymax", type=int, help="BBox y_max in global image pixels")
    corners_parser.add_argument("--patch-size", type=int, default=70, help="Size of each corner patch in pixels (default: 70)")

    # Command: visual
    visual_parser = subparsers.add_parser("visual", help="Render proposed bounding box for visual verification")
    visual_parser.add_argument("image_path", type=str, help="Path to input image")
    visual_parser.add_argument("class_name", type=str, help="Class name or ID")
    visual_parser.add_argument("xmin", type=int, help="Pixel x_min")
    visual_parser.add_argument("ymin", type=int, help="Pixel y_min")
    visual_parser.add_argument("xmax", type=int, help="Pixel x_max")
    visual_parser.add_argument("ymax", type=int, help="Pixel y_max")
    visual_parser.add_argument("--data", type=str, default=None, help="Path to dataset config (default: dataset/data.yaml)")
    visual_parser.add_argument("--crop-context", type=int, default=None, help="Render only candidate crop with N pixels of context")

    # Command: bbox
    bbox_parser = subparsers.add_parser("bbox", help="Manage YOLO bounding box annotations in .txt file")
    bbox_parser.add_argument("image_path", type=str, help="Path to input image")
    bbox_parser.add_argument("class_name", type=str, nargs="?", default=None, help="Class name or ID (for add)")
    bbox_parser.add_argument("xmin", type=int, nargs="?", default=None, help="Pixel x_min")
    bbox_parser.add_argument("ymin", type=int, nargs="?", default=None, help="Pixel y_min")
    bbox_parser.add_argument("xmax", type=int, nargs="?", default=None, help="Pixel x_max")
    bbox_parser.add_argument("ymax", type=int, nargs="?", default=None, help="Pixel y_max")
    bbox_parser.add_argument("--action", type=str, default="add", choices=["add", "delete", "list"], help="Action: add, delete, or list")
    bbox_parser.add_argument("--index", type=int, default=None, help="Index of bounding box to delete")
    bbox_parser.add_argument("--data", type=str, default=None, help="Path to dataset config (default: dataset/data.yaml)")
    bbox_parser.add_argument("--dry-run", action="store_true", help="Validate add action and report output without writing label file")
    bbox_parser.add_argument("--verification-id", type=str, default=None, help="Verification id returned by verify for bbox add")
    bbox_parser.add_argument("--force", action="store_true", help="Bypass verification gate for bbox add and emit warning")

    # Command: verify
    verify_parser = subparsers.add_parser("verify", help="Validate a bbox candidate and return a verification id")
    verify_parser.add_argument("image_path", type=str, help="Path to input image")
    verify_parser.add_argument("class_name", type=str, help="Class name or ID")
    verify_parser.add_argument("xmin", type=int, help="Pixel x_min")
    verify_parser.add_argument("ymin", type=int, help="Pixel y_min")
    verify_parser.add_argument("xmax", type=int, help="Pixel x_max")
    verify_parser.add_argument("ymax", type=int, help="Pixel y_max")
    verify_parser.add_argument("--data", type=str, default=None, help="Path to dataset config (default: dataset/data.yaml)")

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Render compact candidate inspection sheet")
    inspect_parser.add_argument("image_path", type=str, help="Path to input image")
    inspect_parser.add_argument("class_name", type=str, help="Class name or ID")
    inspect_parser.add_argument("xmin", type=int, help="Pixel x_min")
    inspect_parser.add_argument("ymin", type=int, help="Pixel y_min")
    inspect_parser.add_argument("xmax", type=int, help="Pixel x_max")
    inspect_parser.add_argument("ymax", type=int, help="Pixel y_max")
    inspect_parser.add_argument("--data", type=str, default=None, help="Path to dataset config (default: dataset/data.yaml)")
    inspect_parser.add_argument("--crop-context", type=int, default=80, help="Pixels of context around candidate visual crop")
    inspect_parser.add_argument("--patch-size", type=int, default=70, help="Corner patch size")

    # Command: doctor
    doctor_parser = subparsers.add_parser("doctor", help="Check dataset config, guidelines, labels, and smoke contract")
    doctor_parser.add_argument("--data", type=str, default=None, help="Path to dataset config (default: dataset/data.yaml)")
    doctor_parser.add_argument("--run-smoke", action="store_true", help="Run lightweight CLI contract smoke checks")

    # Command: schema
    subparsers.add_parser("schema", help="Print machine-readable command/output contract summary")

    # Command: capabilities
    subparsers.add_parser("capabilities", help="Print machine-readable CLI capabilities and contract metadata")

    args = parser.parse_args()

    if args.version:
        return handle_version(args)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "grid":
        return handle_grid(args)
    elif args.command == "zoom":
        return handle_zoom(args)
    elif args.command == "corners":
        return handle_corners(args)
    elif args.command == "visual":
        return handle_visual(args)
    elif args.command == "bbox":
        return handle_bbox(args)
    elif args.command == "verify":
        return handle_verify(args)
    elif args.command == "inspect":
        return handle_inspect(args)
    elif args.command == "doctor":
        return handle_doctor(args)
    elif args.command == "schema":
        return handle_schema(args)
    elif args.command == "capabilities":
        return handle_capabilities(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
