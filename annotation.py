#!/usr/bin/env python3
"""
A4OD Annotation CLI Tool
AI-Assisted Object Detection & YOLO Dataset Annotation
"""
import argparse
import json
import sys
from pathlib import Path
from PIL import Image

from src.config import find_label_path, load_dataset_config, resolve_class
from src.coords import xyxy_pixel_to_yolo_norm
from src.corner_inspector import render_corner_inspection
from src.grid_renderer import render_grid_image
from src.visualizer import render_visual_preview
from src.yolo_io import add_yolo_label, delete_yolo_label, read_yolo_labels
from src.zoom_renderer import render_zoom_image


def get_tmp_image_dir(image_path: Path) -> Path:
    tmp_dir = Path("tmp") / image_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def handle_grid(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(json.dumps({"error": f"Image file not found: {args.image_path}"}, indent=2))
        return 1

    class_map = load_dataset_config(args.data)
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
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"Failed to generate grid: {str(e)}"}, indent=2))
        return 1


def handle_zoom(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(json.dumps({"error": f"Image file not found: {args.image_path}"}, indent=2))
        return 1

    class_map = load_dataset_config(args.data)
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
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"Failed to generate zoom: {str(e)}"}, indent=2))
        return 1


def handle_corners(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(json.dumps({"error": f"Image file not found: {args.image_path}"}, indent=2))
        return 1

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
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"Failed to generate corner inspection: {str(e)}"}, indent=2))
        return 1


def handle_visual(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(json.dumps({"error": f"Image file not found: {args.image_path}"}, indent=2))
        return 1

    class_map = load_dataset_config(args.data)
    cid, cname = resolve_class(args.class_name, class_map)
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
            existing_label_path=label_path if label_path.exists() else None
        )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"Failed to generate visual preview: {str(e)}"}, indent=2))
        return 1


def handle_bbox(args: argparse.Namespace) -> int:
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(json.dumps({"error": f"Image file not found: {args.image_path}"}, indent=2))
        return 1

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    class_map = load_dataset_config(args.data)
    label_path = find_label_path(img_path)

    action = args.action.lower()

    if action == "list":
        boxes = read_yolo_labels(label_path, img_w, img_h, class_map)
        print(json.dumps({
            "image_path": str(img_path),
            "label_path": str(label_path),
            "label_file_exists": label_path.exists(),
            "total_boxes": len(boxes),
            "boxes": boxes
        }, indent=2, ensure_ascii=False))
        return 0

    elif action == "delete":
        if args.index is None:
            print(json.dumps({"error": "Missing required argument --index for delete action"}, indent=2))
            return 1
        success = delete_yolo_label(label_path, args.index)
        if success:
            print(json.dumps({
                "status": "success",
                "message": f"Deleted bounding box index {args.index} from {label_path}"
            }, indent=2))
            return 0
        else:
            print(json.dumps({
                "status": "error",
                "message": f"Bounding box index {args.index} not found in {label_path}"
            }, indent=2))
            return 1

    elif action == "add":
        if args.class_name is None or args.xmin is None or args.ymin is None or args.xmax is None or args.ymax is None:
            print(json.dumps({"error": "Missing required arguments for 'add': <class> <xmin> <ymin> <xmax> <ymax>"}, indent=2))
            return 1

        cid, cname = resolve_class(args.class_name, class_map)
        try:
            xc, yc, w, h = xyxy_pixel_to_yolo_norm(args.xmin, args.ymin, args.xmax, args.ymax, img_w, img_h)
            add_yolo_label(label_path, cid, xc, yc, w, h)
            print(json.dumps({
                "status": "success",
                "message": f"Added bounding box for '{cname}' (id:{cid}) to {label_path}",
                "label_path": str(label_path),
                "class_id": cid,
                "class_name": cname,
                "pixel_bbox": [args.xmin, args.ymin, args.xmax, args.ymax],
                "yolo_norm_bbox": [xc, yc, w, h]
            }, indent=2, ensure_ascii=False))
            return 0
        except Exception as e:
            print(json.dumps({"error": f"Failed to add bounding box: {str(e)}"}, indent=2))
            return 1

    else:
        print(json.dumps({"error": f"Unknown action: {action}. Supported: add, delete, list"}, indent=2))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A4OD: AI-Assisted YOLO Annotation CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Overlay overview grid with coordinate rulers
  python annotation.py grid dataset/images/1.png --cell-size 200 --data dataset/data.yaml

  # 2. Zoom into ROI with fine grid (50px) preserving global coordinates
  python annotation.py zoom dataset/images/1.png 800 500 1200 800 --cell-size 50 --data dataset/data.yaml

  # 3. Inspect the 4 corners of proposed bbox (ultra-low tokens)
  python annotation.py corners dataset/images/1.png 890 630 1020 760

  # 4. Visually verify proposed bounding box over full image
  python annotation.py visual dataset/images/1.png car 800 600 1000 900 --data dataset/data.yaml

  # 5. Add bounding box to YOLO .txt label file
  python annotation.py bbox dataset/images/1.png car 800 600 1000 900 --data dataset/data.yaml

  # 6. List existing bounding boxes
  python annotation.py bbox dataset/images/1.png --action list --data dataset/data.yaml

  # 7. Delete an incorrect bounding box by index
  python annotation.py bbox dataset/images/1.png --action delete --index 0 --data dataset/data.yaml
"""
    )
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

    args = parser.parse_args()

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
