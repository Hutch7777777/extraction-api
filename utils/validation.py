"""
Validation utilities
"""

import logging

from config import config

logger = logging.getLogger(__name__)


def normalize_page_type(raw_type, page_id=None):
    """
    Normalize page type to valid enum value.

    Args:
        raw_type: Raw page type string from Claude
        page_id: Optional page ID for logging context

    Returns:
        Normalized page type string
    """
    if not raw_type or not isinstance(raw_type, str):
        logger.warning(
            f"Page classification failed for page {page_id or 'unknown'}, "
            f"raw_type={raw_type!r}, defaulting to 'unknown'"
        )
        return 'unknown'

    cleaned = raw_type.lower().strip()

    if cleaned in config.VALID_PAGE_TYPES:
        return cleaned

    mappings = {
        'floorplan': 'floor_plan',
        'floor plan': 'floor_plan',
        'siteplan': 'site_plan',
        'site plan': 'site_plan',
        'roofplan': 'roof_plan',
        'roof plan': 'roof_plan',
        'roof': 'roof_plan',
        'title': 'cover',
        'title sheet': 'cover',
        'other': 'unknown',  # Map old 'other' to 'unknown'
    }

    result = mappings.get(cleaned)
    if result:
        return result

    logger.warning(
        f"Unrecognized page type '{raw_type}' for page {page_id or 'unknown'}, "
        f"defaulting to 'unknown'"
    )
    return 'unknown'


def validate_job_id(job_id):
    """Validate job ID format (UUID)"""
    if not job_id:
        return False
    
    import re
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, str(job_id).lower()))


def validate_page_type(page_type):
    """Check if page type is valid"""
    return page_type in config.VALID_PAGE_TYPES


def validate_scale_ratio(scale_ratio):
    """Check if scale ratio is reasonable"""
    if scale_ratio is None:
        return False
    
    try:
        ratio = float(scale_ratio)
        # Reasonable range: 1:4 (3" scale) to 1:200 (site plans)
        return 4 <= ratio <= 200
    except (TypeError, ValueError):
        return False
