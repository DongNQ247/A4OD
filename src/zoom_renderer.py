from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw

from src.grid_renderer import draw_text_with_outline, get_font
from src.yolo_io import read_yolo_labels


def render_zoom_image(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    cell_size: int = 50,
    show_existing: bool = True,
    class_map: Optional[Dict[int, str]] = None,
    existing_label_path: Optional[Union[str, Path]] = None,
    target_canvas_min: int = 500
) -> Dict[str, Any]:
    """
    Crop an ROI [xmin, ymin, xmax, ymax] from the base image and render a high-precision
    grid overlay with rulers preserving GLOBAL coordinates.
    Automatically upscales small crops for crystal-clear readability without text overlap.
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_img = Image.open(img_p).convert("RGB")
    orig_w, orig_h = base_img.size

    # Validate and clamp crop bounds
    x1 = max(0, min(orig_w, min(int(xmin), int(xmax))))
    x2 = max(0, min(orig_w, max(int(xmin), int(xmax))))
    y1 = max(0, min(orig_h, min(int(ymin), int(ymax))))
    y2 = max(0, min(orig_h, max(int(ymin), int(ymax))))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop dimensions: {crop_w}x{crop_h} from [{x1}, {y1}, {x2}, {y2}]")

    cropped_raw = base_img.crop((x1, y1, x2, y2))

    # Auto-scale factor to guarantee clear text without overlap
    max_crop_dim = max(crop_w, crop_h)
    if max_crop_dim < target_canvas_min:
        scale = max(1.0, round(target_canvas_min / max_crop_dim, 2))
    else:
        scale = 1.0

    scaled_w = int(round(crop_w * scale))
    scaled_h = int(round(crop_h * scale))
    if scale != 1.0:
        cropped_img = cropped_raw.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    else:
        cropped_img = cropped_raw

    # Canvas dimensions & ruler offsets
    ruler_top = max(45, int(scaled_h * 0.08))
    ruler_left = max(70, int(scaled_w * 0.10))
    margin_right = 15
    margin_bottom = 15

    canvas_w = scaled_w + ruler_left + margin_right
    canvas_h = scaled_h + ruler_top + margin_bottom

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    canvas.paste(cropped_img, (ruler_left, ruler_top))

    draw = ImageDraw.Draw(canvas)

    ruler_font = get_font(13)
    cell_font = get_font(12)

    # Draw border around cropped image
    draw.rectangle(
        [ruler_left, ruler_top, ruler_left + scaled_w, ruler_top + scaled_h],
        outline=(0, 0, 0),
        width=2
    )

    # Determine ruler tick step so text doesn't collide
    # A tick label takes ~40px width on canvas
    min_label_dist_px = 50
    step_multiplier = 1
    while (cell_size * step_multiplier * scale) < min_label_dist_px:
        step_multiplier *= 2
    label_step = cell_size * step_multiplier

    # 1. Top Ruler (X-axis) aligned to global cell_size multiples
    first_gx = ((x1 + cell_size - 1) // cell_size) * cell_size
    gx = first_gx
    while gx <= x2:
        lx = (gx - x1) * scale
        cx = ruler_left + int(round(lx))
        is_labeled = ((gx - first_gx) % label_step == 0)

        if is_labeled:
            # Major tick mark
            draw.line([(cx, ruler_top - 7), (cx, ruler_top)], fill=(0, 0, 0), width=2)
            txt = str(gx)
            bbox = ruler_font.getbbox(txt) if hasattr(ruler_font, 'getbbox') else (0, 0, 20, 10)
            tw = bbox[2] - bbox[0]
            draw.text((cx - tw // 2, ruler_top - 22), txt, font=ruler_font, fill=(0, 0, 0))
        else:
            # Minor tick mark
            draw.line([(cx, ruler_top - 4), (cx, ruler_top)], fill=(120, 120, 120), width=1)

        # Inner vertical grid line
        if 0 < lx < scaled_w:
            draw.line([(cx, ruler_top), (cx, ruler_top + scaled_h)], fill=(255, 255, 255), width=1)
        gx += cell_size

    # 2. Left Ruler (Y-axis) aligned to global cell_size multiples
    first_gy = ((y1 + cell_size - 1) // cell_size) * cell_size
    gy = first_gy
    while gy <= y2:
        ly = (gy - y1) * scale
        cy = ruler_top + int(round(ly))
        is_labeled = ((gy - first_gy) % label_step == 0)

        if is_labeled:
            # Major tick mark
            draw.line([(ruler_left - 7, cy), (ruler_left, cy)], fill=(0, 0, 0), width=2)
            txt = str(gy)
            bbox = ruler_font.getbbox(txt) if hasattr(ruler_font, 'getbbox') else (0, 0, 20, 10)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((ruler_left - tw - 10, cy - th // 2), txt, font=ruler_font, fill=(0, 0, 0))
        else:
            # Minor tick mark
            draw.line([(ruler_left - 4, cy), (ruler_left, cy)], fill=(120, 120, 120), width=1)

        # Inner horizontal grid line
        if 0 < ly < scaled_h:
            draw.line([(ruler_left, cy), (ruler_left + scaled_w, cy)], fill=(255, 255, 255), width=1)
        gy += cell_size

    # 3. Origin Point (x1, y1) in Red
    origin_txt = f"({x1}, {y1})"
    dot_radius = 4
    dot_x = ruler_left
    dot_y = ruler_top
    draw.ellipse(
        [(dot_x - dot_radius, dot_y - dot_radius), (dot_x + dot_radius, dot_y + dot_radius)],
        fill=(220, 20, 20),
        outline=(150, 0, 0)
    )
    draw.text((dot_x - 10, dot_y - 30), origin_txt, font=ruler_font, fill=(220, 20, 20))

    # 4. Cell Coordinates (gx, gy) at grid cell intersections (only if sufficient space)
    if (label_step * scale) >= 60:
        gy = first_gy
        while gy < y2:
            gx = first_gx
            while gx < x2:
                coord_str = f"({gx}, {gy})"
                pos_x = ruler_left + int(round((gx - x1) * scale)) + 5
                pos_y = ruler_top + int(round((gy - y1) * scale)) + 3
                draw_text_with_outline(
                    draw,
                    pos=(pos_x, pos_y),
                    text=coord_str,
                    font=cell_font,
                    fill_color="white",
                    outline_color="black",
                    outline_width=2
                )
                gx += label_step
            gy += label_step

    # 5. Overlay Existing Bounding Boxes in cropped view
    existing_boxes: List[Dict[str, Any]] = []
    if show_existing and existing_label_path:
        all_boxes = read_yolo_labels(existing_label_path, orig_w, orig_h, class_map)
        badge_font = get_font(12)

        for box in all_boxes:
            bx1, by1, bx2, by2 = box["pixel_bbox"]
            # Check if intersects crop region
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)

            if ix1 < ix2 and iy1 < iy2:
                existing_boxes.append(box)
                # Map to canvas coordinates
                cx_1 = ruler_left + int(round((bx1 - x1) * scale))
                cy_1 = ruler_top + int(round((by1 - y1) * scale))
                cx_2 = ruler_left + int(round((bx2 - x1) * scale))
                cy_2 = ruler_top + int(round((by2 - y1) * scale))

                # Clamp visual rectangle to inside crop frame
                vx1 = max(ruler_left, cx_1)
                vy1 = max(ruler_top, cy_1)
                vx2 = min(ruler_left + scaled_w, cx_2)
                vy2 = min(ruler_top + scaled_h, cy_2)

                draw.rectangle([vx1, vy1, vx2, vy2], outline=(0, 230, 80), width=3)

                cname = box["class_name"]
                lbl_txt = f"[{box['index']}] {cname}"
                lbbox = badge_font.getbbox(lbl_txt) if hasattr(badge_font, 'getbbox') else (0, 0, 30, 12)
                ltw = lbbox[2] - lbbox[0]
                lth = lbbox[3] - lbbox[1]

                badge_y = max(ruler_top, vy1 - lth - 5)
                draw.rectangle([vx1, badge_y, vx1 + ltw + 6, badge_y + lth + 4], fill=(0, 200, 70))
                draw.text((vx1 + 3, badge_y + 1), lbl_txt, font=badge_font, fill=(0, 0, 0))

    # Ensure output directory exists and save
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p, quality=95)

    return {
        "image_path": str(img_p),
        "zoom_image_path": str(out_p),
        "crop_bbox": [x1, y1, x2, y2],
        "crop_width": crop_w,
        "crop_height": crop_h,
        "scale": scale,
        "cell_size": cell_size,
        "existing_boxes_count": len(existing_boxes),
        "existing_boxes": existing_boxes
    }
