"""
Single source of truth for detection area calculations.

Uses Shoelace formula on polygon_points if available (3+ points).
Falls back to bounding box (pixel_width * pixel_height) otherwise.
"""

import logging

logger = logging.getLogger(__name__)


def compute_detection_area_sf(detection, scale_ratio, dpi=None):
    """Single source of truth for detection area calculation.

    Uses Shoelace formula on polygon_points if available (3+ points).
    Falls back to bounding box (pixel_width * pixel_height).
    Returns area in square feet.

    Args:
        detection: dict with pixel_width, pixel_height, polygon_points (optional)
        scale_ratio: pixels per foot (e.g., 48 for 1/4"=1')
        dpi: optional, not used in current formula but reserved

    Returns:
        Area in square feet (float)
    """
    polygon_points = detection.get('polygon_points')

    # Try Shoelace first
    if polygon_points and isinstance(polygon_points, (list, dict)) and _has_valid_polygon(polygon_points):
        try:
            # Handle nested format {outer: [...], holes: [...]}
            if isinstance(polygon_points, dict) and 'outer' in polygon_points:
                points = polygon_points['outer']
            else:
                points = polygon_points

            area_pixels = _shoelace_area(points)
            if area_pixels > 0:
                return area_pixels / (scale_ratio * scale_ratio)
        except Exception as e:
            logger.warning(f"Shoelace failed, falling back to bbox: {e}")

    # Fallback to bounding box
    pw = float(detection.get('pixel_width', 0) or 0)
    ph = float(detection.get('pixel_height', 0) or 0)

    if pw <= 0 or ph <= 0:
        logger.warning(
            f"Detection has zero/null dimensions: "
            f"w={detection.get('pixel_width')}, h={detection.get('pixel_height')}"
        )
        return 0.0

    return (pw / scale_ratio) * (ph / scale_ratio)


def compute_detection_perimeter_lf(detection, scale_ratio):
    """Calculate perimeter in linear feet.

    Uses polygon perimeter if polygon_points available (3+ points).
    Falls back to bounding box perimeter (2 * (width + height)).

    Args:
        detection: dict with pixel_width, pixel_height, polygon_points (optional)
        scale_ratio: pixels per foot

    Returns:
        Perimeter in linear feet (float)
    """
    polygon_points = detection.get('polygon_points')

    # Try polygon perimeter first
    if polygon_points and isinstance(polygon_points, (list, dict)) and _has_valid_polygon(polygon_points):
        try:
            # Handle nested format {outer: [...], holes: [...]}
            if isinstance(polygon_points, dict) and 'outer' in polygon_points:
                points = polygon_points['outer']
            else:
                points = polygon_points

            perimeter_pixels = _polygon_perimeter(points)
            if perimeter_pixels > 0:
                return perimeter_pixels / scale_ratio
        except Exception as e:
            logger.warning(f"Polygon perimeter failed, falling back to bbox: {e}")

    # Fallback to bounding box perimeter
    pw = float(detection.get('pixel_width', 0) or 0)
    ph = float(detection.get('pixel_height', 0) or 0)

    width_ft = pw / scale_ratio
    height_ft = ph / scale_ratio
    return 2 * (width_ft + height_ft)


def _has_valid_polygon(polygon_points):
    """Check if polygon_points has at least 3 valid points."""
    if isinstance(polygon_points, dict) and 'outer' in polygon_points:
        points = polygon_points['outer']
    else:
        points = polygon_points

    if not isinstance(points, list):
        return False

    return len(points) >= 3


def _shoelace_area(points):
    """Shoelace formula for polygon area in pixel coordinates.

    Points should be list of dicts with 'x' and 'y' keys,
    or list of [x, y] tuples/lists.

    Returns:
        Area in square pixels (always positive)
    """
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        xi, yi = _get_xy(points[i])
        xj, yj = _get_xy(points[j])
        area += xi * yj
        area -= xj * yi

    return abs(area) / 2.0


def _polygon_perimeter(points):
    """Calculate polygon perimeter in pixels.

    Args:
        points: List of points as [{x, y}, ...] or [[x, y], ...]

    Returns:
        Perimeter in pixels
    """
    n = len(points)
    if n < 2:
        return 0.0

    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        xi, yi = _get_xy(points[i])
        xj, yj = _get_xy(points[j])
        # Distance formula
        perimeter += ((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5

    return perimeter


def _get_xy(point):
    """Extract x, y from dict or list/tuple."""
    if isinstance(point, dict):
        return float(point.get('x', 0)), float(point.get('y', 0))
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    return 0.0, 0.0
