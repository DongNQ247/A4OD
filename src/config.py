from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml


def load_dataset_config(yaml_path: Optional[Union[str, Path]] = None) -> Dict[int, str]:
    """
    Load class names mapping {class_id: class_name} from a YOLO data.yaml file.
    If yaml_path is None, checks standard locations, preferring ./dataset/data.yaml.
    """
    if yaml_path is None:
        candidates = [Path("dataset/data.yaml"), Path("data.yaml"), Path("dataset.yaml")]
        for cand in candidates:
            if cand.exists():
                yaml_path = cand
                break

    if yaml_path is None or not Path(yaml_path).exists():
        return {}

    yaml_file = Path(yaml_path)
    try:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Warning] Failed to parse YAML file {yaml_file}: {e}")
        return {}

    names = data.get("names", {})
    class_map: Dict[int, str] = {}

    if isinstance(names, list):
        for idx, name in enumerate(names):
            class_map[idx] = str(name)
    elif isinstance(names, dict):
        for k, v in names.items():
            try:
                class_map[int(k)] = str(v)
            except ValueError:
                pass

    return class_map


def resolve_class(
    class_input: Union[str, int],
    class_map: Dict[int, str]
) -> Tuple[int, str]:
    """
    Resolve a class input (string name or integer id) against class_map.
    Returns (class_id, class_name).
    """
    class_input_str = str(class_input).strip()

    if not class_map:
        raise ValueError("No classes loaded from dataset config")

    # Case 1: class_input is integer string (e.g. "0", "1")
    if class_input_str.isdigit():
        cid = int(class_input_str)
        if cid in class_map:
            return cid, class_map[cid]
        raise ValueError(f"Unknown class id '{cid}'. Allowed ids: {sorted(class_map.keys())}")

    # Case 2: class_input is class name string (e.g. "car")
    for cid, name in class_map.items():
        if name.lower() == class_input_str.lower():
            return cid, name

    allowed_names = ", ".join(class_map.values())
    raise ValueError(f"Unknown class name '{class_input_str}'. Allowed names: {allowed_names}")


def find_label_path(image_path: Union[str, Path]) -> Path:
    """
    Given an image path, determine the corresponding YOLO .txt label path.
    Supports:
    1. Standard YOLO directory structure: /images/.../foo.png -> /labels/.../foo.txt
    2. A4OD staging structure: /data/.../foo.png -> /dataset/labels/.../foo.txt
    3. Same directory: foo.png -> foo.txt
    """
    img_p = Path(image_path).resolve()

    # Check if image is in an 'images' directory
    parts = list(img_p.parts)
    if "images" in parts:
        # Find rightmost 'images' index and replace with 'labels'
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts_labels = list(parts)
        parts_labels[idx] = "labels"
        candidate_label = Path(*parts_labels).with_suffix(".txt")

        same_dir_label = img_p.with_suffix(".txt")

        # If existing label is in same dir, use it
        if same_dir_label.exists() and not candidate_label.exists():
            return same_dir_label

        # Otherwise prefer standard YOLO labels/ structure
        return candidate_label

    # A4OD stores images awaiting annotation in sibling data/ and labels in dataset/labels/.
    if "data" in parts:
        idx = len(parts) - 1 - parts[::-1].index("data")
        if idx < len(parts) - 1:
            dataset_root = Path(*parts[:idx]) / "dataset" / "labels"
            relative_image = Path(*parts[idx + 1:])
            return (dataset_root / relative_image).with_suffix(".txt")

    # Default to same folder with .txt extension
    return img_p.with_suffix(".txt")
