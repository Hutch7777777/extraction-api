"""
Safe scale ratio and DPI handling with fallback and logging.
"""

import logging
from config import config

logger = logging.getLogger(__name__)


def get_safe_scale_ratio(value, context="unknown"):
    """
    Returns validated scale_ratio. Logs WARNING on fallback.

    Args:
        value: The scale ratio value (may be None, 0, or invalid)
        context: Description of where this scale is being used (for logging)

    Returns:
        float: Validated scale ratio, or DEFAULT_SCALE_RATIO (48) if invalid
    """
    try:
        ratio = float(value) if value else 0
    except (TypeError, ValueError):
        ratio = 0

    if ratio <= 0:
        logger.warning(
            f"Null/zero scale_ratio in {context}, "
            f"falling back to {config.DEFAULT_SCALE_RATIO} (1/4\"=1')"
        )
        return float(config.DEFAULT_SCALE_RATIO)
    return ratio


def get_safe_dpi(value, context="unknown"):
    """
    Returns validated DPI. Logs WARNING on fallback.

    Args:
        value: The DPI value (may be None, 0, or invalid)
        context: Description of where this DPI is being used (for logging)

    Returns:
        int: Validated DPI, or DEFAULT_DPI (200) if invalid
    """
    try:
        dpi = int(value) if value else 0
    except (TypeError, ValueError):
        dpi = 0

    if dpi <= 0:
        logger.warning(
            f"Null/zero DPI in {context}, "
            f"falling back to {config.DEFAULT_DPI}"
        )
        return config.DEFAULT_DPI
    return dpi
