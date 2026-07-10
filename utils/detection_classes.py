"""
Canonical detection class names and aliases.

This module intentionally lives under utils so low-level clients such as
core.roboflow_client can normalize classes without importing the services
package and triggering app startup cycles.
"""

import re

OUTSIDE_CORNER = 'outside_corner'
INSIDE_CORNER = 'inside_corner'
CORNER_CLASSES = {OUTSIDE_CORNER, INSIDE_CORNER}

_CLASS_ALIASES = {
    'window': 'window',
    'windows': 'window',
    'door': 'door',
    'doors': 'door',
    'garage': 'garage',
    'garage_door': 'garage',
    'garage_doors': 'garage',
    'building': 'building',
    'buildings': 'building',
    'exterior_wall': 'exterior_wall',
    'exterior_walls': 'exterior_wall',
    'exteriorwall': 'exterior_wall',
    'wall': 'exterior_wall',
    'walls': 'exterior_wall',
    'facade': 'exterior_wall',
    # Keep siding distinct from gross exterior-wall geometry. Jobs can carry
    # both layers; collapsing them double-counts the facade (MN568 regression).
    'siding': 'siding',
    'roof': 'roof',
    'roofs': 'roof',
    'gable': 'gable',
    'gables': 'gable',
    'corner': OUTSIDE_CORNER,
    'corner_outside': OUTSIDE_CORNER,
    'corner_inside': INSIDE_CORNER,
}


def normalize_detection_class(raw_class) -> str:
    """
    Normalize a detection class name to the canonical value used on disk.
    """
    if not raw_class:
        return ''
    cleaned = re.sub(r'[\s-]+', '_', str(raw_class).strip().lower())
    return _CLASS_ALIASES.get(cleaned, cleaned)
