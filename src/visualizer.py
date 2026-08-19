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
    crop_context: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Render visual verification image displaying:
    1. Existing annotations in green.
    2. Proposed candidate bounding box in bold red with label and coordinate details.
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_img = Image.open(img_p).convert("RGB")
    img_w, img_h = base_img.size

    cx1_raw, cy1_raw, cx2_raw, cy2_raw = candidate_bbox_pixel
    norm_xc, norm_yc, norm_w, norm_h = xyxy_pixel_to_yolo_norm(cx1_raw, cy1_raw, cx2_raw, cy2_raw, img_w, img_h)

    cx1 = max(0, min(img_w, min(cx1_raw, cx2_raw)))
    cx2 = max(0, min(img_w, max(cx1_raw, cx2_raw)))
    cy1 = max(0, min(img_h, min(cy1_raw, cy2_raw)))
    cy2 = max(0, min(img_h, max(cy1_raw, cy2_raw)))

    view_bbox = [0, 0, img_w, img_h]
    if crop_context is not None:
        context = max(0, int(crop_context))
        vx1 = max(0, cx1 - context)
        vy1 = max(0, cy1 - context)
        vx2 = min(img_w, cx2 + context)
        vy2 = min(img_h, cy2 + context)
        img = base_img.crop((vx1, vy1, vx2, vy2))
        view_bbox = [vx1, vy1, vx2, vy2]
        offset_x = vx1
        offset_y = vy1
    else:
        img = base_img.copy()
        offset_x = 0
        offset_y = 0

    view_w, view_h = img.size
    draw = ImageDraw.Draw(img)

    font_size = max(13, int(min(view_w, view_h) * 0.02))
    badge_font = get_font(font_size)
    coord_font = get_font(max(11, int(font_size * 0.85)))

    # 1. Draw existing boxes in Green
    existing_boxes: List[Dict[str, Any]] = []
    if existing_label_path:
        existing_boxes = read_yolo_labels(existing_label_path, img_w, img_h, class_map)
        for box in existing_boxes:
            bx1, by1, bx2, by2 = box["pixel_bbox"]
            ix1 = max(view_bbox[0], bx1)
            iy1 = max(view_bbox[1], by1)
            ix2 = min(view_bbox[2], bx2)
            iy2 = min(view_bbox[3], by2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            vx1, vy1 = ix1 - offset_x, iy1 - offset_y
            vx2, vy2 = ix2 - offset_x, iy2 - offset_y
            draw.rectangle([vx1, vy1, vx2, vy2], outline=(0, 230, 80), width=2)
            cname = box["class_name"]
            lbl_txt = f"[{box['index']}] {cname}"

            bbox = badge_font.getbbox(lbl_txt) if hasattr(badge_font, 'getbbox') else (0, 0, 40, 14)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            by_top = max(0, vy1 - th - 6)
            draw.rectangle([vx1, by_top, vx1 + tw + 8, by_top + th + 4], fill=(0, 200, 70))
            draw.text((vx1 + 4, by_top + 1), lbl_txt, font=badge_font, fill=(0, 0, 0))

    # 2. Draw Candidate Bounding Box in Bright Red
    vx1 = cx1 - offset_x
    vy1 = cy1 - offset_y
    vx2 = cx2 - offset_x
    vy2 = cy2 - offset_y

    # Bold red rectangle
    draw.rectangle([vx1, vy1, vx2, vy2], outline=(255, 30, 30), width=3)

    # Candidate label badge
    cand_label = f"★ [PROPOSED] {candidate_class} ({cx1}, {cy1}) -> ({cx2}, {cy2}) [w={cx2-cx1}, h={cy2-cy1}]"
    c_bbox = badge_font.getbbox(cand_label) if hasattr(badge_font, 'getbbox') else (0, 0, 60, 14)
    ctw = c_bbox[2] - c_bbox[0]
    cth = c_bbox[3] - c_bbox[1]

    cand_badge_y = max(0, vy1 - cth - 8)
    if cand_badge_y == 0 and vy1 < cth + 10:
        cand_badge_y = vy1 + 4

    badge_x2 = min(view_w, vx1 + ctw + 12)
    draw.rectangle([vx1, cand_badge_y, badge_x2, cand_badge_y + cth + 6], fill=(255, 30, 30))
    draw.text((vx1 + 6, cand_badge_y + 2), cand_label, font=badge_font, fill=(255, 255, 255))

    # Corner ticks & coordinate markers
    corner_txt_tl = f"({cx1}, {cy1})"
    corner_txt_br = f"({cx2}, {cy2})"
    draw_text_with_outline(draw, (vx1 + 4, vy1 + 4), corner_txt_tl, coord_font, fill_color="yellow", outline_color="black")
    draw_text_with_outline(draw, (max(0, vx2 - 80), max(0, vy2 - 20)), corner_txt_br, coord_font, fill_color="yellow", outline_color="black")

    # Save preview image
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p, quality=95)

    return {
        "image_path": str(img_p),
        "visual_image_path": str(out_p),
        "width": img_w,
        "height": img_h,
        "view_bbox": view_bbox,
        "view_size": [view_w, view_h],
        "candidate": {
            "class": candidate_class,
            "pixel_bbox": [cx1, cy1, cx2, cy2],
            "norm_bbox": [norm_xc, norm_yc, norm_w, norm_h]
        },
        "existing_boxes_count": len(existing_boxes)
    }
