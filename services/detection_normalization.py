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
# intelligent-analysis corner markers are 20px. No real building element
# renders this small at any plausible scale/DPI.
MIN_MEANINGFUL_PIXELS = 24


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


def derive_real_dimensions_ft(detection: Dict, page: Optional[Dict] = None,
                              warn=_default_warn) -> Tuple[float, float, Optional[str]]:
    """
    Resolve a detection's real-world width/height in feet, in order:

    1. Stored real_width_ft / real_height_ft columns (detection_details rows)
    2. Pixel geometry × the page's scale_ratio (skipped for point markers,
       whose pixel box is a marker size, not a measurement)
    3. A "W x H" dimension parsed from the Bluebeam content text
    4. Give up: log a warning naming the detection id and return zeros

    Returns:
        (width_ft, height_ft, source) where source is one of
        'stored' | 'pixel_scale' | 'content' | None
    """
    page = page or {}

    # 1. Stored columns win when present
    width_ft = _to_float(detection.get('real_width_ft'))
    height_ft = _to_float(detection.get('real_height_ft'))
    if width_ft > 0 or height_ft > 0:
        return width_ft, height_ft, 'stored'

    pixel_width = _to_float(detection.get('pixel_width'))
    pixel_height = _to_float(detection.get('pixel_height'))
    markup_type = (detection.get('markup_type') or '').strip().lower()
    is_point_marker = (
        markup_type in POINT_MARKUP_TYPES
        or (0 < pixel_width <= MIN_MEANINGFUL_PIXELS
            and 0 < pixel_height <= MIN_MEANINGFUL_PIXELS)
    )

    # 2. Pixel geometry × page scale
    scale_ratio = _to_float(page.get('scale_ratio'))
    if scale_ratio > 0 and not is_point_marker and (pixel_width > 0 or pixel_height > 0):
        dpi = get_safe_dpi(page.get('dpi'),
                           context=f"derive_real_dimensions detection {detection.get('id')}")
        ft_per_pixel = scale_ratio / dpi / 12.0
        return (round(pixel_width * ft_per_pixel, 2),
                round(pixel_height * ft_per_pixel, 2),
                'pixel_scale')

    # 3. Dimensions written into the Bluebeam content text
    content = detection.get('bluebeam_content')
    dims = parse_content_dimensions_ft(content)
    if dims:
        return dims[0], dims[1], 'content'

    # 4. Nothing usable — warn instead of silently contributing zero
    warn(
        f"[DimensionDerivation] WARNING: detection {detection.get('id')} "
        f"(class={detection.get('class')}) has no derivable real dimensions: "
        f"real_* columns empty, "
        f"{'point marker' if is_point_marker else 'page scale_ratio missing' if scale_ratio <= 0 else 'no pixel geometry'}, "
        f"no parseable 'W x H' in content ({content!r}) — treating as 0 LF"
    )
    return 0.0, 0.0, None
