from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

from src.coords import yolo_norm_to_xyxy_pixel
from src.yolo_io import read_yolo_labels


def get_font(size: int = 14) -> ImageFont.ImageFont:
    """Load a truetype font or fallback to default."""
    font_candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    pos: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill_color: str = "white",
    outline_color: str = "black",
    outline_width: int = 1
) -> None:
    """Draw text with a strong outline/shadow for high visibility on any background."""
    x, y = pos
    # Draw outline around text
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill_color)


def render_grid_image(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    cell_size: int = 200,
    show_existing: bool = True,
    class_map: Optional[Dict[int, str]] = None,
    existing_label_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Render grid lines and coordinate markers onto the image, matching the visual
    layout of dataset/images/1.png with top/left pixel rulers and corner cell coordinates.
    Also overlays existing bounding boxes if show_existing=True.
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_img = Image.open(img_p).convert("RGB")
    img_w, img_h = base_img.size

    # Ruler dimensions
    ruler_top = max(40, int(img_h * 0.04))
    ruler_left = max(60, int(img_w * 0.04))
    margin_right = 10
    margin_bottom = 10

    canvas_w = img_w + ruler_left + margin_right
    canvas_h = img_h + ruler_top + margin_bottom

    # Create high-contrast canvas with white background
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    canvas.paste(base_img, (ruler_left, ruler_top))

    draw = ImageDraw.Draw(canvas)

    # Fonts
    ruler_font_size = max(13, int(min(img_w, img_h) * 0.016))
    cell_font_size = max(14, int(min(img_w, img_h) * 0.018))
    ruler_font = get_font(ruler_font_size)
    cell_font = get_font(cell_font_size)

    # Draw inner image border
    draw.rectangle(
        [ruler_left, ruler_top, ruler_left + img_w, ruler_top + img_h],
        outline=(0, 0, 0),
        width=2
    )

    # 1. Top Ruler (x-axis) & Vertical Grid Lines
    sub_tick = 50
    sx = 0
    while sx <= img_w:
        cx = ruler_left + sx
        is_major = (sx % cell_size == 0)
        if is_major:
            # Top major tick mark
            draw.line([(cx, ruler_top - 7), (cx, ruler_top)], fill=(0, 0, 0), width=2)
            # Top ruler number
            if sx > 0:
                txt = str(sx)
                bbox = ruler_font.getbbox(txt) if hasattr(ruler_font, 'getbbox') else (0, 0, 20, 10)
                tw = bbox[2] - bbox[0]
                draw.text((cx - tw // 2, ruler_top - ruler_font_size - 10), txt, font=ruler_font, fill=(0, 0, 0))
            # Inner vertical grid line
            if 0 < sx < img_w:
                draw.line([(cx, ruler_top), (cx, ruler_top + img_h)], fill=(255, 255, 255), width=1)
        else:
            # Top minor sub-tick mark
            draw.line([(cx, ruler_top - 3), (cx, ruler_top)], fill=(120, 120, 120), width=1)
        sx += sub_tick

    # 2. Left Ruler (y-axis) & Horizontal Grid Lines
    sy = 0
    while sy <= img_h:
        cy = ruler_top + sy
        is_major = (sy % cell_size == 0)
        if is_major:
            # Left major tick mark
            draw.line([(ruler_left - 7, cy), (ruler_left, cy)], fill=(0, 0, 0), width=2)
            # Left ruler number
            if sy > 0:
                txt = str(sy)
                bbox = ruler_font.getbbox(txt) if hasattr(ruler_font, 'getbbox') else (0, 0, 20, 10)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((ruler_left - tw - 12, cy - th // 2), txt, font=ruler_font, fill=(0, 0, 0))
            # Inner horizontal grid line
            if 0 < sy < img_h:
                draw.line([(ruler_left, cy), (ruler_left + img_w, cy)], fill=(255, 255, 255), width=1)
        else:
            # Left minor sub-tick mark
            draw.line([(ruler_left - 3, cy), (ruler_left, cy)], fill=(120, 120, 120), width=1)
        sy += sub_tick

    # 3. Origin Point (0, 0) in Red
    origin_txt = "(0, 0)"
    dot_radius = 5
    dot_x = ruler_left
    dot_y = ruler_top
    draw.ellipse(
        [(dot_x - dot_radius, dot_y - dot_radius), (dot_x + dot_radius, dot_y + dot_radius)],
        fill=(220, 20, 20),
        outline=(150, 0, 0)
    )
    draw.text((dot_x - 10, dot_y - ruler_font_size - 14), origin_txt, font=ruler_font, fill=(220, 20, 20))

    # 4. Cell Coordinates (x, y) at top-left of each grid cell
    for cy_val in range(0, img_h, cell_size):
        for cx_val in range(0, img_w, cell_size):
            coord_str = f"({cx_val}, {cy_val})"
            pos_x = ruler_left + cx_val + 8
            pos_y = ruler_top + cy_val + 6
            draw_text_with_outline(
                draw,
                pos=(pos_x, pos_y),
                text=coord_str,
                font=cell_font,
                fill_color="white",
                outline_color="black",
                outline_width=2
            )

    # 5. Overlay Existing Bounding Boxes if requested
    existing_boxes: List[Dict[str, Any]] = []
    if show_existing and existing_label_path:
        existing_boxes = read_yolo_labels(existing_label_path, img_w, img_h, class_map)
        badge_font = get_font(max(12, int(ruler_font_size * 0.9)))

        for box in existing_boxes:
            bx1, by1, bx2, by2 = box["pixel_bbox"]
            gx1 = ruler_left + bx1
            gy1 = ruler_top + by1
            gx2 = ruler_left + bx2
            gy2 = ruler_top + by2

            # Draw green bounding box outline
            draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 230, 80), width=3)

            # Draw label badge
            cname = box["class_name"]
            lbl_txt = f"[{box['index']}] {cname}"
            lbbox = badge_font.getbbox(lbl_txt) if hasattr(badge_font, 'getbbox') else (0, 0, 40, 14)
            ltw = lbbox[2] - lbbox[0]
            lth = lbbox[3] - lbbox[1]

            badge_y1 = max(ruler_top, gy1 - lth - 6)
            badge_y2 = badge_y1 + lth + 4
            draw.rectangle([gx1, badge_y1, gx1 + ltw + 8, badge_y2], fill=(0, 200, 70))
            draw.text((gx1 + 4, badge_y1 + 1), lbl_txt, font=badge_font, fill=(0, 0, 0))

    # Ensure output directory exists and save
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p, quality=95)

    return {
        "image_path": str(img_p),
        "grid_image_path": str(out_p),
        "width": img_w,
        "height": img_h,
        "cell_size": cell_size,
        "existing_boxes_count": len(existing_boxes),
        "existing_boxes": existing_boxes
    }
