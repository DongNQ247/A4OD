from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from PIL import Image, ImageDraw

from src.grid_renderer import draw_text_with_outline, get_font


def extract_patch(
    base_img: Image.Image,
    center_x: int,
    center_y: int,
    patch_size: int
) -> Tuple[Image.Image, int, int]:
    """
    Extract a patch of size patch_size x patch_size around (center_x, center_y).
    Handles boundaries by padding with neutral gray background if out of bounds.
    Returns (patch_image, local_center_x, local_center_y).
    """
    orig_w, orig_h = base_img.size
    half = patch_size // 2

    src_x1 = center_x - half
    src_y1 = center_y - half
    src_x2 = src_x1 + patch_size
    src_y2 = src_y1 + patch_size

    # Target patch canvas
    patch = Image.new("RGB", (patch_size, patch_size), color=(50, 50, 50))

    # Calculate intersecting coordinates with base_img
    clamped_x1 = max(0, min(orig_w, src_x1))
    clamped_y1 = max(0, min(orig_h, src_y1))
    clamped_x2 = max(0, min(orig_w, src_x2))
    clamped_y2 = max(0, min(orig_h, src_y2))

    if clamped_x1 < clamped_x2 and clamped_y1 < clamped_y2:
        cropped = base_img.crop((clamped_x1, clamped_y1, clamped_x2, clamped_y2))
        dst_x = clamped_x1 - src_x1
        dst_y = clamped_y1 - src_y1
        patch.paste(cropped, (dst_x, dst_y))

    local_cx = half
    local_cy = half
    return patch, local_cx, local_cy


def render_corner_inspection(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    patch_size: int = 70
) -> Dict[str, Any]:
    """
    Extract 4 corner patches (TL, TR, BL, BR) around proposed bounding box
    and combine them into a single high-contrast composite inspection tile.
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    base_img = Image.open(img_p).convert("RGB")
    orig_w, orig_h = base_img.size

    x1 = min(int(xmin), int(xmax))
    x2 = max(int(xmin), int(xmax))
    y1 = min(int(ymin), int(ymax))
    y2 = max(int(ymin), int(ymax))

    # Extract 4 patches
    tl_patch, tlc_x, tlc_y = extract_patch(base_img, x1, y1, patch_size)
    tr_patch, trc_x, trc_y = extract_patch(base_img, x2, y1, patch_size)
    bl_patch, blc_x, blc_y = extract_patch(base_img, x1, y2, patch_size)
    br_patch, brc_x, brc_y = extract_patch(base_img, x2, y2, patch_size)

    # Draw corner boundaries on each patch
    # 1. Top-Left: border goes right from tlc_x and down from tlc_y
    draw_tl = ImageDraw.Draw(tl_patch)
    draw_tl.line([(tlc_x, tlc_y), (patch_size, tlc_y)], fill=(255, 30, 30), width=2)
    draw_tl.line([(tlc_x, tlc_y), (tlc_x, patch_size)], fill=(255, 30, 30), width=2)
    draw_tl.ellipse([(tlc_x - 3, tlc_y - 3), (tlc_x + 3, tlc_y + 3)], fill=(255, 255, 0))

    # 2. Top-Right: border goes left from trc_x and down from trc_y
    draw_tr = ImageDraw.Draw(tr_patch)
    draw_tr.line([(0, trc_y), (trc_x, trc_y)], fill=(255, 30, 30), width=2)
    draw_tr.line([(trc_x, trc_y), (trc_x, patch_size)], fill=(255, 30, 30), width=2)
    draw_tr.ellipse([(trc_x - 3, trc_y - 3), (trc_x + 3, trc_y + 3)], fill=(255, 255, 0))

    # 3. Bottom-Left: border goes right from blc_x and up from blc_y
    draw_bl = ImageDraw.Draw(bl_patch)
    draw_bl.line([(blc_x, blc_y), (patch_size, blc_y)], fill=(255, 30, 30), width=2)
    draw_bl.line([(blc_x, 0), (blc_x, blc_y)], fill=(255, 30, 30), width=2)
    draw_bl.ellipse([(blc_x - 3, blc_y - 3), (blc_x + 3, blc_y + 3)], fill=(255, 255, 0))

    # 4. Bottom-Right: border goes left from brc_x and up from brc_y
    draw_br = ImageDraw.Draw(br_patch)
    draw_br.line([(0, brc_y), (brc_x, brc_y)], fill=(255, 30, 30), width=2)
    draw_br.line([(brc_x, 0), (brc_x, brc_y)], fill=(255, 30, 30), width=2)
    draw_br.ellipse([(brc_x - 3, brc_y - 3), (brc_x + 3, brc_y + 3)], fill=(255, 255, 0))

    # Layout parameters - ensure sufficient width for label strings
    header_h = 24
    footer_h = 24
    border = 8
    cell_w = max(patch_size, 110)
    cell_h = patch_size + header_h

    composite_w = cell_w * 2 + border * 3
    composite_h = cell_h * 2 + border * 3 + footer_h

    composite = Image.new("RGB", (composite_w, composite_h), color=(30, 30, 35))
    comp_draw = ImageDraw.Draw(composite)
    font = get_font(12)

    # Column/Row coordinates
    col1_x = border
    col2_x = border * 2 + cell_w
    row1_y = border
    row2_y = border * 2 + cell_h

    # Center patches in their cell columns if cell_w > patch_size
    p_offset_x = (cell_w - patch_size) // 2

    composite.paste(tl_patch, (col1_x + p_offset_x, row1_y + header_h))
    composite.paste(tr_patch, (col2_x + p_offset_x, row1_y + header_h))
    composite.paste(bl_patch, (col1_x + p_offset_x, row2_y + header_h))
    composite.paste(br_patch, (col2_x + p_offset_x, row2_y + header_h))

    # Labels for each corner
    comp_draw.text((col1_x + 2, row1_y + 3), f"[TL] ({x1}, {y1})", font=font, fill=(100, 230, 255))
    comp_draw.text((col2_x + 2, row1_y + 3), f"[TR] ({x2}, {y1})", font=font, fill=(100, 230, 255))
    comp_draw.text((col1_x + 2, row2_y + 3), f"[BL] ({x1}, {y2})", font=font, fill=(100, 230, 255))
    comp_draw.text((col2_x + 2, row2_y + 3), f"[BR] ({x2}, {y2})", font=font, fill=(100, 230, 255))

    # Footer summary
    footer_y = composite_h - footer_h
    comp_draw.text(
        (border + 2, footer_y),
        f"BBox: {x2-x1}x{y2-y1}px (Red=Edge, Yellow=Vertex)",
        font=font,
        fill=(255, 220, 100)
    )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_p, quality=95)

    return {
        "image_path": str(img_p),
        "corners_image_path": str(out_p),
        "pixel_bbox": [x1, y1, x2, y2],
        "box_width": x2 - x1,
        "box_height": y2 - y1,
        "patch_size": patch_size,
        "composite_size": [composite_w, composite_h]
    }
