"""
Read-side normalization for detection rows.

Detections in extraction_detections_draft come from three writers that
historically disagreed on class naming and on which measurement columns
they populate:

- Bluebeam fresh import writes 'corner_outside' / 'corner_inside'
  (CLASS_MAPPING in bluebeam_fresh_import_service.py)
- Intelligent analysis floor-plan corners write 'corner_outside' /
  'corner_inside' (store_corner_detections in intelligent_analysis_service.py)
- Bluebeam roundtrip import writes 'corner' / 'outside_corner' /
  'inside_corner' (CLASS_NAME_MAPPING in bluebeam_import_service.py)

None of the import paths write real_width_ft / real_height_ft, so any
aggregation that reads those columns directly silently gets zero.

This module is the single choke point both problems route through:
every consumer that aggregates detections by class must read class names
via normalize_detection_class(), and read real-world dimensions via
derive_real_dimensions_ft(). The database is NOT migrated — both corner
spellings remain on disk; they are mapped on read.

Canonical corner class names (chosen to match the existing aggregation
branch names): 'outside_corner' and 'inside_corner'.

Relationship to utils/scale.py: DPI fallback routes through
get_safe_dpi() (the single DPI fallback implementation). scale_ratio
deliberately does NOT route through get_safe_scale_ratio() here — that
helper substitutes the default 48 when scale is missing, which is right
for the AI pipeline but wrong for dimension derivation: a missing scale
must fall through to the Bluebeam-content fallback and then a warning,
never silently assume a drawing scale.
"""

import logging
import re
from typing import Dict, Optional, Tuple

from utils.scale import get_safe_dpi

logger = logging.getLogger(__name__)

# Canonical class names for corners
OUTSIDE_CORNER = 'outside_corner'
INSIDE_CORNER = 'inside_corner'
CORNER_CLASSES = {OUTSIDE_CORNER, INSIDE_CORNER}

# Alias → canonical. Keys are post-cleanup (lowercased, underscored), so
# 'Corner Outside' and 'corner_outside' both resolve through one entry.
# A bare 'corner' is treated as an outside corner — the fresh import made
# the same call (CLASS_MAPPING maps 'corner' → outside).
_CLASS_ALIASES = {
    'corner': OUTSIDE_CORNER,
    'corner_outside': OUTSIDE_CORNER,
    'corner_inside': INSIDE_CORNER,
}

# Markup types whose pixel geometry is a marker location, not a measured
# shape (Bluebeam Count markups, floor-plan corner markers).
POINT_MARKUP_TYPES = {'point'}

# Boxes at or below this size in BOTH dimensions are treated as point
# markers even when markup_type is missing: fresh-import stamps are 16px,
# intelligence-analysis corner markers are 20px. No real building element
# renders this small at any plausible scale/DPI.
MIN_MEANINGFUL_PIXELS = 24

# Sanity bounds for pixel×scale derivation. Page scale_ratio values are
# not all in architectural units (elevation pages calibrated in the
# Detection Editor store values like 13.978 whose unit convention
# differs), so a derived dimension outside these bounds means the scale
# convention didn't match — reject and fall through rather than poison
# the totals.
MIN_PLAUSIBLE_FT = 0.5
MAX_PLAUSIBLE_FT = 40.0


def normalize_detection_class(raw_class) -> str:
    """
    Normalize a detection class name read from the database.

    Lowercases, trims, converts spaces to underscores, then collapses
    known aliases (currently the corner spellings) to one canonical name.
    Unknown classes pass through cleaned but unmapped.
    """
    if not raw_class:
        return ''
    cleaned = str(raw_class).strip().lower().replace(' ', '_')
    return _CLASS_ALIASES.get(cleaned, cleaned)


def _to_float(value) -> float:
    """Coerce a DB value (None, str, int, float) to float, defaulting to 0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# Dimension content parsing: "10' x 8'", "3'-6\" x 5'-0\"", "36\" x 60\"",
# "12 ft x 8 ft". Sides without an explicit foot/inch marker are rejected —
# a bare "36 x 60" is ambiguous (feet vs inches) and guessing wrong by 12x
# is worse than falling through to the warning.
_DIM_SPLIT_RE = re.compile(r'\s*[x×]\s*', re.IGNORECASE)
_SIDE_RE = re.compile(
    r"""^\s*
    (?:(?P<feet>\d+(?:\.\d+)?)\s*(?:'|ft\b|feet\b))?
    \s*-?\s*
    (?:(?P<inches>\d+(?:\.\d+)?)\s*(?:"|in\b|inch\b|inches\b))?
    \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def _parse_side_ft(text: str) -> Optional[float]:
    match = _SIDE_RE.match(text or '')
    if not match:
        return None
    feet, inches = match.group('feet'), match.group('inches')
    if feet is None and inches is None:
        return None
    return float(feet or 0) + float(inches or 0) / 12.0


def parse_content_dimensions_ft(content) -> Optional[Tuple[float, float]]:
    """
    Parse a "W x H" dimension expression from a Bluebeam content string.

    Returns (width_ft, height_ft) or None if the content does not contain
    a two-sided dimension with explicit foot/inch units.
    """
    if not content:
        return None
    parts = _DIM_SPLIT_RE.split(str(content).strip())
    if len(parts) != 2:
        return None
    width_ft = _parse_side_ft(parts[0])
    height_ft = _parse_side_ft(parts[1])
    if width_ft is None or height_ft is None:
        return None
    return round(width_ft, 2), round(height_ft, 2)


def _default_warn(message: str):
    # Unified logging style (see a58db0f "guardrails and logging for
    # silent failures") — WARNING level, module logger
    logger.warning(message)


def _dims_from_area_perimeter(area_sf: float, perimeter_lf: float,
                              pixel_width: float, pixel_height: float):
    """
    Recover rectangle width/height (ft) from the stored area and
    perimeter columns: w + h = P/2 and w·h = A, so w and h are the roots
    of x² − (P/2)x + A = 0. Exact and unit-free — both inputs were
    computed with the original (trusted) import scale, so this is immune
    to page scale_ratio convention drift. Orientation (which root is the
    width) comes from the pixel aspect ratio.

    Returns (width_ft, height_ft) or None when not solvable.
    """
    if area_sf <= 0 or perimeter_lf <= 0:
        return None
    half = perimeter_lf / 2.0
    disc = half * half - 4.0 * area_sf
    if disc < 0:
        return None
    root = disc ** 0.5
    longer, shorter = (half + root) / 2.0, (half - root) / 2.0
    if shorter <= 0:
        return None
    if pixel_height > pixel_width > 0:
        return round(shorter, 2), round(longer, 2)
    return round(longer, 2), round(shorter, 2)


def _dims_from_area_aspect(area_sf: float, pixel_width: float, pixel_height: float):
    """
    Recover width/height (ft) from the stored area and the pixel aspect
    ratio: w = √(A · pxw/pxh), h = A / w. Used when perimeter is missing
    or inconsistent. Unit-free, like the quadratic.
    """
    if area_sf <= 0 or pixel_width <= 0 or pixel_height <= 0:
        return None
    width = (area_sf * pixel_width / pixel_height) ** 0.5
    return round(width, 2), round(area_sf / width, 2)


def derive_real_dimensions_ft(detection: Dict, page: Optional[Dict] = None,
                              warn=_default_warn) -> Tuple[float, float, Optional[str]]:
    """
    Resolve a detection's real-world width/height in feet, in order:

    1. Stored real_width_ft / real_height_ft columns (detection_details rows)
    2. Area + perimeter quadratic — exact, unit-free, from trusted columns
    3. Area + pixel aspect ratio — unit-free, from trusted area
    4. Pixel geometry × the page's scale_ratio (skipped for point markers;
       results outside MIN/MAX_PLAUSIBLE_FT are rejected with a log naming
       the page and scale, since page scale conventions are inconsistent)
    5. A "W x H" dimension parsed from the Bluebeam content text
    6. Give up: log a warning naming the detection id and return zeros

    Returns:
        (width_ft, height_ft, source) where source is one of
        'stored' | 'area_perimeter' | 'area_aspect' | 'pixel_scale'
        | 'content' | None
    """
    page = page or {}

    # 1. Stored columns win when present
    width_ft = _to_float(detection.get('real_width_ft'))
    height_ft = _to_float(detection.get('real_height_ft'))
    if width_ft > 0 or height_ft > 0:
        return width_ft, height_ft, 'stored'

    pixel_width = _to_float(detection.get('pixel_width'))
    pixel_height = _to_float(detection.get('pixel_height'))
    area_sf = _to_float(detection.get('area_sf'))
    perimeter_lf = _to_float(detection.get('perimeter_lf'))
    markup_type = (detection.get('markup_type') or '').strip().lower()
    is_point_marker = (
        markup_type in POINT_MARKUP_TYPES
        or (0 < pixel_width <= MIN_MEANINGFUL_PIXELS
            and 0 < pixel_height <= MIN_MEANINGFUL_PIXELS)
    )

    # 2. Trusted area + perimeter (exact rectangle recovery)
    dims = _dims_from_area_perimeter(area_sf, perimeter_lf, pixel_width, pixel_height)
    if dims:
        return dims[0], dims[1], 'area_perimeter'

    # 3. Trusted area + pixel aspect
    dims = _dims_from_area_aspect(area_sf, pixel_width, pixel_height)
    if dims:
        return dims[0], dims[1], 'area_aspect'

    # 4. Pixel geometry × page scale, sanity-bounded
    scale_ratio = _to_float(page.get('scale_ratio'))
    if scale_ratio > 0 and not is_point_marker and (pixel_width > 0 or pixel_height > 0):
        dpi = get_safe_dpi(page.get('dpi'),
                           context=f"derive_real_dimensions detection {detection.get('id')}")
        ft_per_pixel = scale_ratio / dpi / 12.0
        width_ft = round(pixel_width * ft_per_pixel, 2)
        height_ft = round(pixel_height * ft_per_pixel, 2)
        implausible = any(
            d > 0 and not (MIN_PLAUSIBLE_FT <= d <= MAX_PLAUSIBLE_FT)
            for d in (width_ft, height_ft)
        )
        if not implausible:
            return width_ft, height_ft, 'pixel_scale'
        logger.warning(
            f"[DimensionDerivation] pixel-scale result rejected for detection "
            f"{detection.get('id')} (class={detection.get('class')}): "
            f"{width_ft}x{height_ft} ft outside [{MIN_PLAUSIBLE_FT}, {MAX_PLAUSIBLE_FT}] — "
            f"page {page.get('page_number')} scale_ratio={scale_ratio} is likely "
            f"in a different unit convention; falling through"
        )

    # 5. Dimensions written into the Bluebeam content text
    content = detection.get('bluebeam_content')
    dims = parse_content_dimensions_ft(content)
    if dims:
        return dims[0], dims[1], 'content'

    # 6. Nothing usable — warn instead of silently contributing zero
    warn(
        f"[DimensionDerivation] WARNING: detection {detection.get('id')} "
        f"(class={detection.get('class')}) has no derivable real dimensions: "
        f"real_* columns empty, no usable area/perimeter, "
        f"{'point marker' if is_point_marker else 'page scale_ratio missing' if scale_ratio <= 0 else 'no pixel geometry'}, "
        f"no parseable 'W x H' in content ({content!r}) — treating as 0 LF"
    )
    return 0.0, 0.0, None
