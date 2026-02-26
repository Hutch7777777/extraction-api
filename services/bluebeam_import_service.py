"""
Bluebeam PDF import service.

Imports edited PDFs from Bluebeam back into EstimatePros.ai,
matching annotations to original detections via embedded UUID metadata.

Key features:
- Parses EST:{...} JSON from annotation NM (Name) field
- Transforms PDF coordinates back to pixel coordinates
- Runs 4-way diff: unchanged, modified, deleted, new
- Updates extraction_detections_draft with changes
- Preserves history via soft deletes (is_deleted flag)
"""

import io
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("WARNING: pymupdf not installed. Install with: pip install pymupdf", flush=True)

from config import config
from database import (
    supabase_request,
    get_detection,
    update_detection,
    get_detections_by_page,
)


# Coordinate tolerance for "unchanged" detection
# PDF coordinate round-trip loses precision: pixel → PDF points → pixel
# Use generous tolerance to avoid false "modified" detections
COORDINATE_TOLERANCE_PX = 10.0  # Absolute pixel tolerance
COORDINATE_TOLERANCE_PCT = 0.03  # 3% relative tolerance for dimensions

# Border inflation: PDF annotations include stroke in their rect
# A 52x105 detection becomes ~62x115 in the PDF (+5px per side)
ANNOTATION_BORDER_PX = 5.0  # Estimated border/stroke width per side

# Class name normalization mapping (Bluebeam Subject → our class names)
CLASS_NAME_MAPPING = {
    'building': 'building',
    'exterior wall': 'exterior_wall',
    'exterior_wall': 'exterior_wall',
    'exteriorwall': 'exterior_wall',
    'window': 'window',
    'windows': 'window',
    'door': 'door',
    'doors': 'door',
    'garage': 'garage',
    'garage door': 'garage',
    'garage_door': 'garage',
    'garagedoor': 'garage',
    'gable': 'gable',
    'gables': 'gable',
    'roof': 'roof',
    'roofs': 'roof',
    'soffit': 'soffit',
    'fascia': 'fascia',
    'trim': 'trim',
    'corner': 'corner',
    'inside corner': 'inside_corner',
    'inside_corner': 'inside_corner',
    'outside corner': 'outside_corner',
    'outside_corner': 'outside_corner',
    'stone': 'stone',
    'stucco': 'stucco',
    'brick': 'brick',
}

# Classes skipped during export (should not be marked as deleted on import)
# Must match SKIP_CLASSES in bluebeam_service.py
SKIP_CLASSES = {'building'}

# Color-to-class fallback mapping (RGB 0-1 range)
COLOR_CLASS_MAPPING = {
    (0, 0.47, 1): 'building',        # Blue
    (1, 0.65, 0): 'window',          # Orange
    (1, 0, 1): 'door',               # Magenta
    (0, 1, 0): 'garage',             # Green
    (1, 1, 0): 'gable',              # Yellow
    (0.55, 0.27, 0.07): 'soffit',    # Brown
    (0.86, 0.08, 0.24): 'roof',      # Crimson
}


def normalize_class_name(raw_name: str) -> str:
    """
    Normalize class name from Bluebeam Subject field.

    Handles variations like "Exterior Wall", "exterior_wall", "WINDOW", etc.
    """
    if not raw_name:
        return 'unknown'

    normalized = raw_name.lower().strip()
    return CLASS_NAME_MAPPING.get(normalized, normalized.replace(' ', '_'))


def parse_roundtrip_metadata(nm_field: str) -> Optional[Dict[str, Any]]:
    """
    Parse round-trip metadata from annotation NM field.

    Expected format: "EST:{\"v\":1,\"det_id\":\"uuid\",...}"

    Returns:
        Parsed JSON dict or None if not our annotation
    """
    if not nm_field:
        return None

    # Handle PDF string escaping
    cleaned = nm_field.replace('\\(', '(').replace('\\)', ')').replace('\\\\', '\\')

    # Look for our EST: prefix
    if not cleaned.startswith('EST:'):
        return None

    json_str = cleaned[4:]  # Skip "EST:"

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[Bluebeam Import] Failed to parse roundtrip JSON: {e}", flush=True)
        return None


def infer_class_from_color(color: Tuple[float, float, float]) -> Optional[str]:
    """
    Infer detection class from annotation color (fallback when Subject missing).

    Args:
        color: RGB tuple in 0-1 range (from fitz)

    Returns:
        Class name or None if no match
    """
    if not color:
        return None

    # Round to 2 decimal places for matching
    rounded = tuple(round(c, 2) for c in color)

    for color_key, class_name in COLOR_CLASS_MAPPING.items():
        color_key_rounded = tuple(round(c, 2) for c in color_key)
        # Allow some tolerance in color matching
        if all(abs(a - b) < 0.1 for a, b in zip(rounded, color_key_rounded)):
            return class_name

    return None


def transform_pdf_to_pixel(
    pdf_rect: Any,
    pdf_page_rect: Any,
    image_width: int,
    image_height: int
) -> Dict[str, float]:
    """
    Transform PDF annotation rectangle to pixel coordinates.

    This reverses the export transformation:
    - Export: pixel → PDF via (pixel * scale_factor)
    - Import: PDF → pixel via (pdf / scale_factor)

    Args:
        pdf_rect: fitz.Rect annotation bounding box
        pdf_page_rect: fitz.Rect page dimensions
        image_width: Original image width in pixels
        image_height: Original image height in pixels

    Returns:
        Dict with center-based pixel coordinates (pixel_x, pixel_y, pixel_width, pixel_height)
    """
    # Calculate scale factors (reverse of export)
    scale_x = image_width / pdf_page_rect.width
    scale_y = image_height / pdf_page_rect.height

    # Transform PDF corners to pixel corners
    x1 = pdf_rect.x0 * scale_x
    y1 = pdf_rect.y0 * scale_y
    x2 = pdf_rect.x1 * scale_x
    y2 = pdf_rect.y1 * scale_y

    # Convert to center-based format (how we store detections)
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    return {
        'pixel_x': round(center_x, 2),
        'pixel_y': round(center_y, 2),
        'pixel_width': round(width, 2),
        'pixel_height': round(height, 2)
    }


def compensate_border_inflation(
    coords: Dict[str, float],
    border_px: float = ANNOTATION_BORDER_PX
) -> Dict[str, float]:
    """
    Compensate for PDF annotation border/stroke inflation.

    PDF annotations include the stroke width in their rect, making them
    larger than the original detection. Subtract the border to get the
    actual content area.

    Args:
        coords: Imported coordinates with border inflation
        border_px: Border width per side (default 5px)

    Returns:
        Adjusted coordinates with border subtracted
    """
    return {
        'pixel_x': coords.get('pixel_x', 0),  # Center-based, no adjustment needed
        'pixel_y': coords.get('pixel_y', 0),  # Center-based, no adjustment needed
        'pixel_width': max(0, coords.get('pixel_width', 0) - (border_px * 2)),
        'pixel_height': max(0, coords.get('pixel_height', 0) - (border_px * 2))
    }


def coords_within_tolerance(
    original: Dict[str, float],
    imported: Dict[str, float],
    abs_tolerance: float = COORDINATE_TOLERANCE_PX,
    pct_tolerance: float = COORDINATE_TOLERANCE_PCT,
    is_point_marker: bool = False
) -> bool:
    """
    Check if two coordinate sets are within tolerance.

    Uses BOTH absolute (pixels) and percentage-based tolerance.
    A coordinate is "unchanged" if it's within EITHER threshold.
    This handles both small detections (where % is too tight) and
    large detections (where absolute px is too tight).

    Args:
        original: Original detection coordinates
        imported: Imported annotation coordinates
        abs_tolerance: Absolute pixel tolerance
        pct_tolerance: Percentage tolerance (0.03 = 3%)
        is_point_marker: If True, only compare position, ignore dimensions

    Returns:
        True if coordinates are effectively unchanged
    """
    # For point markers (width=0 or height=0), only compare position
    # These get exported as small rectangles but their "size" is meaningless
    if is_point_marker:
        keys_to_check = ['pixel_x', 'pixel_y']
    else:
        keys_to_check = ['pixel_x', 'pixel_y', 'pixel_width', 'pixel_height']

    for key in keys_to_check:
        orig_val = float(original.get(key, 0) or 0)
        imp_val = float(imported.get(key, 0) or 0)
        diff = abs(orig_val - imp_val)

        # Calculate percentage-based threshold (use larger of the two values)
        max_val = max(abs(orig_val), abs(imp_val), 1)  # Avoid div by zero
        pct_threshold = max_val * pct_tolerance

        # Pass if within EITHER absolute OR percentage tolerance
        if diff > abs_tolerance and diff > pct_threshold:
            return False
    return True


def extract_annotations_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract all annotations from a PDF.

    Returns:
        List of annotation dicts with:
        - roundtrip: parsed EST:{} metadata (or None for new annotations)
        - subject: annotation Subject (class name)
        - contents: annotation Contents (notes/comments)
        - rect: fitz.Rect bounding box
        - color: stroke color tuple
        - page_number: 1-based page number
        - annot_type: annotation type (e.g., 'Square', 'Polygon')
    """
    if fitz is None:
        return []

    annotations = []

    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype='pdf')

        for page_idx, pdf_page in enumerate(pdf_doc):
            page_number = page_idx + 1
            page_rect = pdf_page.rect

            for annot in pdf_page.annots():
                if annot is None:
                    continue

                # Skip text annotations (not shapes)
                annot_type = annot.type[1]  # e.g., 'Square', 'Polygon', 'FreeText'
                if annot_type in ['FreeText', 'Text', 'Stamp', 'Highlight', 'Underline', 'StrikeOut']:
                    # These are labels/text, not detection shapes
                    continue

                # Get NM field for round-trip metadata
                # xref_get_key is the reliable method - it auto-decodes hex strings
                nm_field = None
                try:
                    nm_raw = pdf_doc.xref_get_key(annot.xref, "NM")
                    # Debug: log what we're reading
                    if nm_raw:
                        print(f"[Bluebeam Import] NM raw: type={nm_raw[0]}, value={str(nm_raw[1])[:80]}...", flush=True)

                    # nm_raw is a tuple like ('string', 'EST:{...}') - PyMuPDF auto-decodes hex
                    if nm_raw and len(nm_raw) > 1 and nm_raw[0] == 'string':
                        nm_field = nm_raw[1]
                        # Strip parentheses if present (literal string format)
                        if nm_field.startswith('(') and nm_field.endswith(')'):
                            nm_field = nm_field[1:-1]
                except Exception as e:
                    print(f"[Bluebeam Import] Error reading NM: {e}", flush=True)

                # Get Subject (class name)
                try:
                    subj_raw = pdf_doc.xref_get_key(annot.xref, "Subj")
                    subject = subj_raw[1] if subj_raw and len(subj_raw) > 1 else None
                    if subject and subject.startswith('(') and subject.endswith(')'):
                        subject = subject[1:-1]
                except:
                    subject = None

                # Get annotation info
                annot_info = annot.info

                annotations.append({
                    'roundtrip': parse_roundtrip_metadata(nm_field),
                    'subject': subject or annot_info.get('subject'),
                    'contents': annot_info.get('content') or annot.get_text(),
                    'rect': annot.rect,
                    'color': annot.colors.get('stroke'),
                    'page_number': page_number,
                    'page_rect': page_rect,
                    'annot_type': annot_type,
                })

        pdf_doc.close()

    except Exception as e:
        print(f"[Bluebeam Import] Error reading PDF: {e}", flush=True)
        return []

    return annotations


def import_bluebeam_pdf(
    job_id: str,
    pdf_bytes: bytes,
    apply_changes: bool = True
) -> Dict[str, Any]:
    """
    Import edited Bluebeam PDF and sync changes to database.

    This is the main entry point for the import process.

    Args:
        job_id: UUID of the extraction job
        pdf_bytes: Raw PDF file bytes
        apply_changes: If True, apply changes to database. If False, just report diff.

    Returns:
        Dict with:
        - success: bool
        - summary: {unchanged, modified, deleted, added}
        - changes: List of individual change records
        - error: str (if failed)
    """
    if fitz is None:
        return {
            'success': False,
            'error': 'pymupdf not installed. Run: pip install pymupdf'
        }

    print(f"[Bluebeam Import] Starting import for job {job_id}", flush=True)

    # 1. Verify job exists and get page data
    jobs = supabase_request('GET', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
    if not jobs:
        return {'success': False, 'error': 'Job not found'}

    job = jobs[0]
    project_name = job.get('project_name', 'Unknown')
    print(f"[Bluebeam Import] Job: {project_name}", flush=True)

    # 2. Get all pages for this job (need image dimensions for coord transform)
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'order': 'page_number.asc'
    })

    if not pages:
        return {'success': False, 'error': 'No pages found for this job'}

    # Build page lookup by page_number
    page_lookup = {p['page_number']: p for p in pages}
    page_id_lookup = {p['id']: p for p in pages}

    print(f"[Bluebeam Import] Found {len(pages)} pages", flush=True)

    # 3. Get all current detections for this job
    current_detections = supabase_request('GET', 'extraction_detections_draft', filters={
        'job_id': f'eq.{job_id}',
        'status': 'neq.deleted',
        'order': 'page_id,class,detection_index'
    })

    # Build detection lookup by ID
    # IMPORTANT: Ensure keys are strings for matching against JSON det_id
    detection_lookup = {str(d['id']): d for d in (current_detections or [])}
    print(f"[Bluebeam Import] Found {len(detection_lookup)} current detections", flush=True)

    # Debug: show sample DB IDs and their types
    if current_detections:
        sample_ids = [d.get('id') for d in current_detections[:3]]
        print(f"[Bluebeam Import] Sample DB IDs: {sample_ids}", flush=True)
        print(f"[Bluebeam Import] DB ID types: {[type(id).__name__ for id in sample_ids]}", flush=True)
        print(f"[Bluebeam Import] DB fields available: {list(current_detections[0].keys())[:10]}", flush=True)

    # 4. Extract annotations from PDF
    annotations = extract_annotations_from_pdf(pdf_bytes)
    with_roundtrip = sum(1 for a in annotations if a.get('roundtrip'))
    print(f"[Bluebeam Import] Extracted {len(annotations)} annotations from PDF ({with_roundtrip} with roundtrip metadata)", flush=True)

    # Debug: show sample annotation det_ids
    sample_rt = [a.get('roundtrip', {}).get('det_id') for a in annotations if a.get('roundtrip')][:3]
    print(f"[Bluebeam Import] Sample PDF det_ids: {sample_rt}", flush=True)
    print(f"[Bluebeam Import] PDF det_id types: {[type(id).__name__ for id in sample_rt if id]}", flush=True)

    # 5. Run diff algorithm
    changes = []
    seen_detection_ids = set()
    pages_with_annotations = set()  # Track which pages had annotations in the PDF

    for annot in annotations:
        roundtrip = annot.get('roundtrip')
        pdf_page_number = annot['page_number']  # 0-indexed from PyMuPDF

        # Get page data for coordinate transformation
        # Use roundtrip page_id for accurate page matching (exact DB page ID)
        page_data = None
        if roundtrip and roundtrip.get('page_id'):
            page_data = page_id_lookup.get(str(roundtrip['page_id']))

        if not page_data:
            # New annotation or missing roundtrip - try page_number
            # PyMuPDF is 0-indexed, DB is 1-indexed, so try both
            page_data = page_lookup.get(pdf_page_number) or page_lookup.get(pdf_page_number + 1)

        if not page_data:
            print(f"[Bluebeam Import] Warning: No page data for PDF page {pdf_page_number}", flush=True)
            continue

        # Track DB page_number (1-indexed) for pages_with_annotations
        db_page_number = page_data.get('page_number')
        if db_page_number:
            pages_with_annotations.add(db_page_number)

        image_width = page_data.get('original_width') or page_data.get('image_width') or page_data.get('width')
        image_height = page_data.get('original_height') or page_data.get('image_height') or page_data.get('height')
        has_dimensions = bool(image_width and image_height)

        # Determine class name
        raw_class = annot.get('subject')
        if raw_class:
            detected_class = normalize_class_name(raw_class)
        else:
            # Try to infer from color
            detected_class = infer_class_from_color(annot.get('color')) or 'unknown'

        if roundtrip and roundtrip.get('det_id'):
            # This is one of our annotations - check for modifications
            det_id = str(roundtrip['det_id'])  # Ensure string for matching
            seen_detection_ids.add(det_id)

            original = detection_lookup.get(det_id)
            # Debug first few lookups
            if len(seen_detection_ids) <= 3:
                print(f"[Bluebeam Import] Looking up det_id='{det_id}' (type={type(det_id).__name__}), found={original is not None}", flush=True)
                if not original and detection_lookup:
                    # Show what keys look like
                    sample_key = list(detection_lookup.keys())[0]
                    print(f"[Bluebeam Import] Sample lookup key: '{sample_key}' (type={type(sample_key).__name__})", flush=True)

            if not original:
                # Detection was deleted from DB but still in PDF - treat as re-add
                # But we need dimensions to create the new coords
                if not has_dimensions:
                    print(f"[Bluebeam Import] Warning: Cannot re-add det {det_id} - no page dimensions", flush=True)
                    continue
                new_coords = transform_pdf_to_pixel(
                    annot['rect'],
                    annot['page_rect'],
                    image_width,
                    image_height
                )
                changes.append({
                    'change_type': 'readded',
                    'detection_id': det_id,
                    'class': detected_class,
                    'page_id': page_data['id'],
                    'page_number': db_page_number,
                    'new_coords': new_coords,
                    'imported_bbox': {
                        'x': new_coords['pixel_x'],
                        'y': new_coords['pixel_y'],
                        'w': new_coords['pixel_width'],
                        'h': new_coords['pixel_height']
                    },
                    'notes': annot.get('contents')
                })
                continue

            # Check for geometry changes
            # Use roundtrip bbox if available (exact original coords from export)
            # This avoids precision issues from coordinate transformations
            rt_bbox = roundtrip.get('bbox', {})
            if rt_bbox and rt_bbox.get('x') is not None:
                original_coords = {
                    'pixel_x': float(rt_bbox.get('x', 0) or 0),
                    'pixel_y': float(rt_bbox.get('y', 0) or 0),
                    'pixel_width': float(rt_bbox.get('w', 0) or 0),
                    'pixel_height': float(rt_bbox.get('h', 0) or 0)
                }
            else:
                original_coords = {
                    'pixel_x': float(original.get('pixel_x', 0) or 0),
                    'pixel_y': float(original.get('pixel_y', 0) or 0),
                    'pixel_width': float(original.get('pixel_width', 0) or 0),
                    'pixel_height': float(original.get('pixel_height', 0) or 0)
                }

            # Detect point markers (original width=0 or height=0)
            # These are brackets, corbels, etc. - only compare position
            is_point_marker = original_coords['pixel_width'] == 0 or original_coords['pixel_height'] == 0

            # If we have page dimensions, transform PDF coords and compare
            # If not, we can still use the roundtrip bbox for comparison
            if has_dimensions:
                new_coords = transform_pdf_to_pixel(
                    annot['rect'],
                    annot['page_rect'],
                    image_width,
                    image_height
                )
                # Compensate for PDF annotation border inflation before comparison
                adjusted_coords = compensate_border_inflation(new_coords)

                geometry_changed = not coords_within_tolerance(
                    original_coords,
                    adjusted_coords,
                    is_point_marker=is_point_marker
                )
            else:
                # No dimensions - can't transform PDF coords, but if we have roundtrip bbox
                # we can still compare against original detection to see if class changed
                # Assume geometry unchanged since we can't measure it
                print(f"[Bluebeam Import] No dimensions for page {db_page_number or pdf_page_number}, assuming geometry unchanged for det {det_id}", flush=True)
                geometry_changed = False
                new_coords = original_coords  # Use original coords if no dimensions

            # Check for class change
            original_class = original.get('class', 'unknown').lower().replace(' ', '_')
            class_changed = detected_class != original_class

            if geometry_changed or class_changed:
                change = {
                    'change_type': 'modified',
                    'detection_id': det_id,
                    'page_id': page_data['id'],
                    'page_number': db_page_number,
                }

                if geometry_changed:
                    change['field'] = 'geometry'
                    change['old_coords'] = original_coords
                    change['new_coords'] = new_coords
                    change['imported_bbox'] = {
                        'x': new_coords['pixel_x'],
                        'y': new_coords['pixel_y'],
                        'w': new_coords['pixel_width'],
                        'h': new_coords['pixel_height']
                    }

                    # Calculate area change if it's an area detection
                    old_area = original.get('area_sf', 0)
                    if old_area:
                        change['old_sf'] = round(old_area, 2)

                if class_changed:
                    if 'field' in change:
                        change['field'] = 'geometry,class'
                    else:
                        change['field'] = 'class'
                    change['old_class'] = original_class
                    change['new_class'] = detected_class

                changes.append(change)
            # else: unchanged - no action needed

        else:
            # New annotation added by contractor (no roundtrip metadata)
            # Require dimensions to create new detection
            if not has_dimensions:
                print(f"[Bluebeam Import] Warning: Cannot add new annotation - no page dimensions for page {db_page_number or pdf_page_number}", flush=True)
                continue
            new_coords = transform_pdf_to_pixel(
                annot['rect'],
                annot['page_rect'],
                image_width,
                image_height
            )
            changes.append({
                'change_type': 'added',
                'class': detected_class,
                'page_id': page_data['id'],
                'page_number': db_page_number,
                'coords': new_coords,
                'imported_bbox': {
                    'x': new_coords['pixel_x'],
                    'y': new_coords['pixel_y'],
                    'w': new_coords['pixel_width'],
                    'h': new_coords['pixel_height']
                },
                'notes': annot.get('contents'),
                'annot_type': annot.get('annot_type')
            })

    # 6. Find deleted detections (in DB but not in returned PDF)
    # Only mark as deleted if:
    # - The detection was not seen in the imported PDF
    # - The detection's page actually had annotations in the exported PDF
    #   (pages without annotations were not exported, so missing detections
    #    on those pages should remain "unchanged", not "deleted")
    # - The detection's class is NOT in SKIP_CLASSES (those were never exported)
    print(f"[Bluebeam Import] Pages with annotations in PDF: {sorted(pages_with_annotations)}", flush=True)

    for det_id, detection in detection_lookup.items():
        if det_id not in seen_detection_ids:
            det_class = (detection.get('class') or '').lower()
            # Skip classes that were never exported
            if det_class in SKIP_CLASSES:
                continue

            # Check if this detection's page was actually in the exported PDF
            # (had annotations - meaning it was included in the export)
            page_data = page_id_lookup.get(detection.get('page_id'))
            if page_data and page_data['page_number'] in pages_with_annotations:
                changes.append({
                    'change_type': 'deleted',
                    'detection_id': det_id,
                    'class': detection.get('class'),
                    'page_id': detection.get('page_id'),
                    'page_number': page_data['page_number'],
                    'area_sf': detection.get('area_sf')
                })

    # 7. Calculate summary
    summary = {
        'unchanged': len(detection_lookup) - sum(1 for c in changes if c['change_type'] in ['modified', 'deleted', 'readded']),
        'modified': sum(1 for c in changes if c['change_type'] == 'modified'),
        'deleted': sum(1 for c in changes if c['change_type'] == 'deleted'),
        'added': sum(1 for c in changes if c['change_type'] == 'added'),
        'readded': sum(1 for c in changes if c['change_type'] == 'readded'),
    }

    print(f"[Bluebeam Import] Diff complete: {summary}", flush=True)

    # 8. Apply changes to database (if requested)
    if apply_changes and changes:
        applied = _apply_changes_to_database(changes, job_id, pages)
        summary['applied'] = applied

    return {
        'success': True,
        'job_id': job_id,
        'project_name': project_name,
        'summary': summary,
        'changes': changes
    }


def _apply_changes_to_database(
    changes: List[Dict[str, Any]],
    job_id: str,
    pages: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Apply diff changes to the database.

    Args:
        changes: List of change records from diff
        job_id: Job UUID
        pages: Page data list

    Returns:
        Dict with counts of applied changes
    """
    applied = {'modified': 0, 'deleted': 0, 'added': 0, 'readded': 0, 'errors': 0}

    # Build page lookup for new detections
    page_id_lookup = {p['id']: p for p in pages}

    for change in changes:
        change_type = change.get('change_type') or change.get('action')  # Support both for backward compat

        try:
            if change_type == 'modified':
                det_id = change['detection_id']
                updates = {}

                if 'new_coords' in change:
                    updates.update(change['new_coords'])
                    updates['geometry_modified_at'] = datetime.now(timezone.utc).isoformat()
                    updates['geometry_modified_source'] = 'bluebeam_import'

                if 'new_class' in change:
                    updates['class'] = change['new_class']
                    updates['class_modified_at'] = datetime.now(timezone.utc).isoformat()
                    updates['class_modified_source'] = 'bluebeam_import'

                if updates:
                    result = update_detection(det_id, updates)
                    if result:
                        applied['modified'] += 1
                    else:
                        applied['errors'] += 1

            elif change_type == 'deleted':
                det_id = change['detection_id']
                # Soft delete - set is_deleted flag
                result = update_detection(det_id, {
                    'status': 'deleted',
                    'deleted_at': datetime.now(timezone.utc).isoformat(),
                    'deleted_source': 'bluebeam_import'
                })
                if result:
                    applied['deleted'] += 1
                else:
                    applied['errors'] += 1

            elif change_type == 'added':
                page_id = change['page_id']
                page_data = page_id_lookup.get(page_id, {})

                # Get next detection index for this page
                existing = supabase_request('GET', 'extraction_detections_draft', filters={
                    'page_id': f'eq.{page_id}',
                    'order': 'detection_index.desc',
                    'limit': '1'
                })
                next_index = (existing[0]['detection_index'] + 1) if existing else 1

                new_detection = {
                    'id': str(uuid.uuid4()),
                    'job_id': job_id,
                    'page_id': page_id,
                    'class': change['class'],
                    'detection_index': next_index,
                    'confidence': 1.0,  # User-created
                    'status': 'manual',
                    'is_user_created': True,
                    'created_source': 'bluebeam_import',
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    **change['coords']
                }

                # Calculate real-world measurements if page has scale
                scale_ratio = page_data.get('scale_ratio')
                dpi = page_data.get('dpi') or 200
                if scale_ratio:
                    from geometry.calculations import calculate_real_dimensions
                    real_dims = calculate_real_dimensions(
                        change['coords']['pixel_width'],
                        change['coords']['pixel_height'],
                        scale_ratio,
                        dpi
                    )
                    new_detection.update(real_dims)

                result = supabase_request('POST', 'extraction_detections_draft', new_detection)
                if result:
                    applied['added'] += 1
                    # Add the new ID to the change record for response
                    change['new_detection_id'] = new_detection['id']
                else:
                    applied['errors'] += 1

            elif change_type == 'readded':
                det_id = change['detection_id']
                # Re-enable a deleted detection
                updates = {
                    'status': 'manual',
                    'deleted_at': None,
                    'readded_at': datetime.now(timezone.utc).isoformat(),
                    'readded_source': 'bluebeam_import',
                    **change['new_coords']
                }
                result = update_detection(det_id, updates)
                if result:
                    applied['readded'] += 1
                else:
                    applied['errors'] += 1

        except Exception as e:
            print(f"[Bluebeam Import] Error applying {change_type}: {e}", flush=True)
            applied['errors'] += 1

    print(f"[Bluebeam Import] Applied changes: {applied}", flush=True)
    return applied


def get_import_preview(job_id: str, pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Preview what would be imported without applying changes.

    Args:
        job_id: Job UUID
        pdf_bytes: PDF file bytes

    Returns:
        Same as import_bluebeam_pdf but with apply_changes=False
    """
    return import_bluebeam_pdf(job_id, pdf_bytes, apply_changes=False)


# =============================================================================
# RECALCULATION AGGREGATION
# =============================================================================

# Point marker classes (count-based, not area/linear)
POINT_MARKER_CLASSES = {
    'corbel', 'belly_band', 'bracket', 'shutter', 'louver',
    'address_block', 'hose_bib', 'vent', 'light_fixture', 'outlet'
}


def aggregate_detections_for_recalc(job_id: str) -> Dict[str, Any]:
    """
    Re-aggregate detection data into the format expected by
    the 'Approve from Detection Editor' n8n webhook.

    This function queries all active detections for a job and builds
    the payload that triggers the Multi-Trade Coordinator pipeline.

    Args:
        job_id: UUID of the extraction job

    Returns:
        Dict matching the webhook payload format with:
        - facade, windows, doors, garages, trim, corners, gables
        - products, material_assignments, detection_counts
        - project info (job_id, project_id, project_name, etc.)
    """
    print(f"[Bluebeam Recalc] Aggregating detections for job {job_id}", flush=True)

    # 1. Get all active detections for this job
    detections = supabase_request('GET', 'extraction_detections_draft', filters={
        'job_id': f'eq.{job_id}',
        'status': 'neq.deleted'
    })

    if not detections:
        detections = []

    print(f"[Bluebeam Recalc] Found {len(detections)} active detections", flush=True)

    # 2. Initialize aggregation buckets
    facade = {
        'gross_area_sf': 0.0,
        'net_siding_sf': 0.0,
        'perimeter_lf': 0.0,
        'level_starter_lf': 0.0
    }
    windows = {
        'count': 0,
        'area_sf': 0.0,
        'perimeter_lf': 0.0,
        'head_lf': 0.0,
        'jamb_lf': 0.0,
        'sill_lf': 0.0
    }
    doors = {
        'count': 0,
        'area_sf': 0.0,
        'perimeter_lf': 0.0,
        'head_lf': 0.0,
        'jamb_lf': 0.0
    }
    garages = {
        'count': 0,
        'area_sf': 0.0,
        'perimeter_lf': 0.0,
        'head_lf': 0.0,
        'jamb_lf': 0.0
    }
    corners = {
        'outside_count': 0,
        'outside_lf': 0.0,
        'inside_count': 0,
        'inside_lf': 0.0
    }
    gables = {
        'count': 0,
        'area_sf': 0.0,
        'rake_lf': 0.0
    }
    roof_area_sf = 0.0  # Track roof to subtract from facade
    detection_counts = {}  # For point markers
    material_assignments = []

    # 3. Aggregate by class
    for det in detections:
        cls = (det.get('class') or '').lower().replace(' ', '_')
        area = float(det.get('area_sf') or 0)
        perim = float(det.get('perimeter_lf') or 0)
        width_ft = float(det.get('real_width_ft') or 0)
        height_ft = float(det.get('real_height_ft') or 0)

        if cls in ('building', 'exterior_wall'):
            facade['gross_area_sf'] += area
            facade['perimeter_lf'] += width_ft  # Bottom edge = starter
            facade['level_starter_lf'] += width_ft

        elif cls == 'window':
            windows['count'] += 1
            windows['area_sf'] += area
            windows['perimeter_lf'] += perim
            windows['head_lf'] += width_ft
            windows['jamb_lf'] += height_ft * 2
            windows['sill_lf'] += width_ft

        elif cls == 'door':
            doors['count'] += 1
            doors['area_sf'] += area
            doors['perimeter_lf'] += perim
            doors['head_lf'] += width_ft
            doors['jamb_lf'] += height_ft * 2

        elif cls in ('garage', 'garage_door'):
            garages['count'] += 1
            garages['area_sf'] += area
            garages['perimeter_lf'] += perim
            garages['head_lf'] += width_ft
            garages['jamb_lf'] += height_ft * 2

        elif cls == 'gable':
            gables['count'] += 1
            gables['area_sf'] += area
            # Gable perimeter approximates rake length (two sloped sides)
            gables['rake_lf'] += perim

        elif cls == 'outside_corner':
            corners['outside_count'] += 1
            corners['outside_lf'] += height_ft

        elif cls == 'inside_corner':
            corners['inside_count'] += 1
            corners['inside_lf'] += height_ft

        elif cls == 'roof':
            roof_area_sf += area

        elif cls in POINT_MARKER_CLASSES:
            if cls not in detection_counts:
                detection_counts[cls] = {
                    'count': 0,
                    'display_name': cls.replace('_', ' ').title(),
                    'measurement_type': 'count',
                    'unit': 'EA'
                }
            detection_counts[cls]['count'] += 1

        # Collect material assignments
        if det.get('assigned_material_id'):
            material_assignments.append({
                'detection_id': det['id'],
                'detection_class': det.get('class', ''),
                'pricing_item_id': det['assigned_material_id'],
                'quantity': area if area > 0 else 1,
                'unit': 'SF' if area > 0 else 'EA'
            })

    # 4. Calculate net siding (gross facade - roof - openings + gables)
    total_opening_sf = windows['area_sf'] + doors['area_sf'] + garages['area_sf']
    facade['net_siding_sf'] = facade['gross_area_sf'] - roof_area_sf - total_opening_sf + gables['area_sf']

    # 5. Calculate trim totals
    trim = {
        'total_head_lf': windows['head_lf'] + doors['head_lf'] + garages['head_lf'],
        'total_jamb_lf': windows['jamb_lf'] + doors['jamb_lf'] + garages['jamb_lf'],
        'total_sill_lf': windows['sill_lf'],
        'total_trim_lf': 0.0
    }
    trim['total_trim_lf'] = trim['total_head_lf'] + trim['total_jamb_lf'] + trim['total_sill_lf']

    # 6. Get job info
    job_info = supabase_request('GET', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
    job = job_info[0] if job_info else {}
    project_id = job.get('project_id') or job_id

    # 7. Get project info
    project = {}
    if project_id:
        project_info = supabase_request('GET', 'projects', filters={'id': f'eq.{project_id}'})
        project = project_info[0] if project_info else {}

    # 8. Get product selections from existing cad_hover_measurements or project_configurations
    products = _get_product_selections(job_id, project_id)

    # 9. Calculate total point count
    total_point_count = sum(d['count'] for d in detection_counts.values())

    # 10. Round all numeric values
    for bucket in [facade, windows, doors, garages, corners, gables, trim]:
        for key in bucket:
            if isinstance(bucket[key], float):
                bucket[key] = round(bucket[key], 2)

    payload = {
        'job_id': job_id,
        'project_id': project_id,
        'project_name': project.get('name') or job.get('project_name') or 'Bluebeam Import',
        'client_name': project.get('customer_name') or '',
        'address': project.get('address') or '',
        'selected_trades': ['siding'],  # Default to siding

        'facade': facade,
        'windows': windows,
        'doors': doors,
        'garages': garages,
        'trim': trim,
        'corners': corners,
        'gables': gables,
        'products': products,

        'material_assignments': material_assignments,
        'organization_id': project.get('organization_id'),
        'detection_counts': detection_counts,
        'total_point_count': total_point_count
    }

    print(f"[Bluebeam Recalc] Aggregation complete: facade={facade['net_siding_sf']} SF, "
          f"windows={windows['count']}, doors={doors['count']}, garages={garages['count']}", flush=True)

    return payload


def _get_product_selections(job_id: str, project_id: str) -> Dict[str, str]:
    """
    Get product selections from existing measurements or project config.

    Tries in order:
    1. cad_hover_measurements for this job
    2. project_configurations for this project
    3. Default values

    Returns:
        Dict with siding_product, siding_color, trim_product, trim_color
    """
    # Defaults
    products = {
        'siding_product': 'HardiePlank 8.25" Cedarmill',
        'siding_color': 'Arctic White',
        'trim_product': 'HardieTrim 4/4',
        'trim_color': 'Arctic White'
    }

    # Try to get from existing cad_hover_measurements
    measurements = supabase_request('GET', 'cad_hover_measurements', filters={
        'job_id': f'eq.{job_id}',
        'order': 'created_at.desc',
        'limit': '1'
    })

    if measurements and measurements[0].get('products'):
        saved_products = measurements[0]['products']
        if isinstance(saved_products, dict):
            products.update({k: v for k, v in saved_products.items() if v})
            return products

    # Try project_configurations
    if project_id:
        configs = supabase_request('GET', 'project_configurations', filters={
            'project_id': f'eq.{project_id}',
            'order': 'created_at.desc',
            'limit': '1'
        })

        if configs and configs[0].get('configuration'):
            config = configs[0]['configuration']
            if isinstance(config, dict):
                # Map config fields to product fields
                if config.get('siding_product'):
                    products['siding_product'] = config['siding_product']
                if config.get('siding_color'):
                    products['siding_color'] = config['siding_color']
                if config.get('trim_product'):
                    products['trim_product'] = config['trim_product']
                if config.get('trim_color'):
                    products['trim_color'] = config['trim_color']

    return products


def trigger_recalculation_webhook(job_id: str) -> Dict[str, Any]:
    """
    Trigger the recalculation pipeline via n8n webhook.

    Aggregates detection data and calls the same webhook that the
    Detection Editor frontend uses when clicking "Approve & Calculate".

    Args:
        job_id: UUID of the extraction job

    Returns:
        Dict with success status and webhook response or error
    """
    import requests

    APPROVE_WEBHOOK = 'https://n8n-production-293e.up.railway.app/webhook/approve-detection-editor'

    print(f"[Bluebeam Recalc] Triggering recalculation for job {job_id}", flush=True)

    try:
        # Aggregate detections into webhook payload format
        payload = aggregate_detections_for_recalc(job_id)

        # Call the n8n webhook
        response = requests.post(
            APPROVE_WEBHOOK,
            json=payload,
            timeout=120  # Match the 120s timeout in the n8n workflow
        )

        if response.ok:
            result = response.json() if response.content else {}
            print(f"[Bluebeam Recalc] Webhook success: {response.status_code}", flush=True)
            return {
                'success': True,
                'status_code': response.status_code,
                'response': result
            }
        else:
            error_text = response.text[:500] if response.text else 'No error details'
            print(f"[Bluebeam Recalc] Webhook failed: {response.status_code} - {error_text}", flush=True)
            return {
                'success': False,
                'status_code': response.status_code,
                'error': f"Webhook returned {response.status_code}: {error_text}"
            }

    except requests.Timeout:
        print(f"[Bluebeam Recalc] Webhook timeout after 120s", flush=True)
        return {
            'success': False,
            'error': 'Recalculation webhook timed out after 120 seconds'
        }

    except Exception as e:
        print(f"[Bluebeam Recalc] Webhook error: {e}", flush=True)
        return {
            'success': False,
            'error': f"Failed to call recalculation webhook: {str(e)}"
        }
