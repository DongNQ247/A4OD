from pathlib import Path
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from src.coords import yolo_norm_to_xyxy_pixel


def atomic_write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip("\n") + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_yolo_labels(
    label_path: Union[str, Path],
    img_w: int,
    img_h: int,
    class_map: Optional[Dict[int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Read YOLO .txt label file and convert bounding boxes to pixel coordinates.
    Returns list of dicts with index, class_id, class_name, norm_bbox, pixel_bbox.
    """
    class_map = class_map or {}
    lbl_p = Path(label_path)
    if not lbl_p.exists():
        return []

    boxes: List[Dict[str, Any]] = []
    with open(lbl_p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            tokens = line_str.split()
            if len(tokens) < 5:
                continue
            try:
                cid = int(tokens[0])
                xc = float(tokens[1])
                yc = float(tokens[2])
                w = float(tokens[3])
                h = float(tokens[4])
            except ValueError:
                continue

            xmin, ymin, xmax, ymax = yolo_norm_to_xyxy_pixel(xc, yc, w, h, img_w, img_h)
            cname = class_map.get(cid, str(cid))

            boxes.append({
                "index": idx,
                "class_id": cid,
                "class_name": cname,
                "norm_bbox": [xc, yc, w, h],
                "pixel_bbox": [xmin, ymin, xmax, ymax],
                "raw_line": line_str
            })

    return boxes


def add_yolo_label(
    label_path: Union[str, Path],
    class_id: int,
    xc: float,
    yc: float,
    w: float,
    h: float
) -> None:
    """
    Append a bounding box to YOLO .txt label file.
    Creates parent directories and file if they do not exist.
    """
    lbl_p = Path(label_path)
    line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
    existing_lines: List[str] = []
    if lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
    existing_lines.append(line)
    atomic_write_lines(lbl_p, existing_lines)


def delete_yolo_label(
    label_path: Union[str, Path],
    index: int
) -> bool:
    """
    Delete a specific bounding box by 0-based line index.
    Returns True if deleted, False if index out of range or file not found.
    """
    lbl_p = Path(label_path)
    if not lbl_p.exists():
        return False

    with open(lbl_p, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not line.strip().startswith("#")]

    if index < 0 or index >= len(lines):
        return False

    lines.pop(index)

    atomic_write_lines(lbl_p, lines)

    return True
