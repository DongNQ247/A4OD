from typing import Tuple


def xyxy_pixel_to_yolo_norm(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    """
    Convert absolute pixel coordinates (xmin, ymin, xmax, ymax)
    to YOLO normalized coordinates (x_center, y_center, width, height) in [0.0, 1.0].
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

    # Ensure min <= max
    x1 = min(float(xmin), float(xmax))
    x2 = max(float(xmin), float(xmax))
    y1 = min(float(ymin), float(ymax))
    y2 = max(float(ymin), float(ymax))

    # Clamp to image boundaries
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Bounding box has zero or negative area: width={box_w}, height={box_h}")

    xc_norm = (x1 + x2) / (2.0 * img_w)
    yc_norm = (y1 + y2) / (2.0 * img_h)
    w_norm = box_w / img_w
    h_norm = box_h / img_h

    return (
        round(xc_norm, 6),
        round(yc_norm, 6),
        round(w_norm, 6),
        round(h_norm, 6)
    )


def yolo_norm_to_xyxy_pixel(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized coordinates (xc, yc, w, h)
    to absolute pixel coordinates (xmin, ymin, xmax, ymax).
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")

    xmin = int(round((xc - w / 2.0) * img_w))
    ymin = int(round((yc - h / 2.0) * img_h))
    xmax = int(round((xc + w / 2.0) * img_w))
    ymax = int(round((yc + h / 2.0) * img_h))

    # Clamp to [0, img_w], [0, img_h]
    xmin = max(0, min(img_w, xmin))
    ymin = max(0, min(img_h, ymin))
    xmax = max(0, min(img_w, xmax))
    ymax = max(0, min(img_h, ymax))

    return xmin, ymin, xmax, ymax
