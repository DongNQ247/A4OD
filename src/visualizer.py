from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from src.coords import xyxy_pixel_to_yolo_norm
from src.grid_renderer import get_font, draw_text_with_outline
from src.yolo_io import read_yolo_labels


def render_visual_preview(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    candidate_class: str,
    candidate_bbox_pixel: Tuple[int, int, int, int],
    class_map: Optional[Dict[int, str]] = None,
    existing_label_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Render visual verification image displaying:
    1. Existing annotations in green.
    2. Proposed candidate bounding box in bold red with label and coordinate details.
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(img_p).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)

    font_size = max(13, int(min(img_w, img_h) * 0.02))
    badge_font = get_font(font_size)
    coord_font = get_font(max(11, int(font_size * 0.85)))

    # 1. Draw existing boxes in Green
    existing_boxes: List[Dict[str, Any]] = []
    if existing_label_path:
        existing_boxes = read_yolo_labels(existing_label_path, img_w, img_h, class_map)
        for box in existing_boxes:
            bx1, by1, bx2, by2 = box["pixel_bbox"]
            draw.rectangle([bx1, by1, bx2, by2], outline=(0, 230, 80), width=2)
            cname = box["class_name"]
            lbl_txt = f"[{box['index']}] {cname}"

            bbox = badge_font.getbbox(lbl_txt) if hasattr(badge_font, 'getbbox') else (0, 0, 40, 14)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            by_top = max(0, by1 - th - 6)
            draw.rectangle([bx1, by_top, bx1 + tw + 8, by_top + th + 4], fill=(0, 200, 70))
            draw.text((bx1 + 4, by_top + 1), lbl_txt, font=badge_font, fill=(0, 0, 0))

    # 2. Draw Candidate Bounding Box in Bright Red
    cx1, cy1, cx2, cy2 = candidate_bbox_pixel
    # Clamp to image bounds
    cx1 = max(0, min(img_w, cx1))
    cy1 = max(0, min(img_h, cy1))
    cx2 = max(0, min(img_w, cx2))
    cy2 = max(0, min(img_h, cy2))

    # Bold red rectangle
    draw.rectangle([cx1, cy1, cx2, cy2], outline=(255, 30, 30), width=3)

    # Candidate label badge
    cand_label = f"★ [PROPOSED] {candidate_class} ({cx1}, {cy1}) -> ({cx2}, {cy2}) [w={cx2-cx1}, h={cy2-cy1}]"
    c_bbox = badge_font.getbbox(cand_label) if hasattr(badge_font, 'getbbox') else (0, 0, 60, 14)
    ctw = c_bbox[2] - c_bbox[0]
    cth = c_bbox[3] - c_bbox[1]

    cand_badge_y = max(0, cy1 - cth - 8)
    if cand_badge_y == 0 and cy1 < cth + 10:
        cand_badge_y = cy1 + 4  # Inside the box if at very top of image

    draw.rectangle([cx1, cand_badge_y, cx1 + ctw + 12, cand_badge_y + cth + 6], fill=(255, 30, 30))
    draw.text((cx1 + 6, cand_badge_y + 2), cand_label, font=badge_font, fill=(255, 255, 255))

    # Corner ticks & coordinate markers
    corner_txt_tl = f"({cx1}, {cy1})"
    corner_txt_br = f"({cx2}, {cy2})"
    draw_text_with_outline(draw, (cx1 + 4, cy1 + 4), corner_txt_tl, coord_font, fill_color="yellow", outline_color="black")
    draw_text_with_outline(draw, (max(0, cx2 - 80), max(0, cy2 - 20)), corner_txt_br, coord_font, fill_color="yellow", outline_color="black")

    # Compute YOLO normalized for metadata reporting
    try:
        norm_xc, norm_yc, norm_w, norm_h = xyxy_pixel_to_yolo_norm(cx1, cy1, cx2, cy2, img_w, img_h)
    except Exception:
        norm_xc, norm_yc, norm_w, norm_h = (0.0, 0.0, 0.0, 0.0)

    # Save preview image
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p, quality=95)

    return {
        "image_path": str(img_p),
        "visual_image_path": str(out_p),
        "width": img_w,
        "height": img_h,
        "candidate": {
            "class": candidate_class,
            "pixel_bbox": [cx1, cy1, cx2, cy2],
            "norm_bbox": [norm_xc, norm_yc, norm_w, norm_h]
        },
        "existing_boxes_count": len(existing_boxes)
    }
