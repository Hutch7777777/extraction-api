"""
Bluebeam Fresh Import Service.

Imports Bluebeam-annotated PDFs directly - no prior extraction job required.
Creates a new extraction job, converts PDF pages to images, parses all
annotations as detections, and prepares for Detection Editor review.

This is a PARALLEL flow to the existing AI detection pipeline:
- Existing flow: Upload PDF → AI detection (Roboflow) → Claude Vision → Review
- Fresh import:  Upload annotated PDF → Parse annotations → Review

Key features:
- Creates extraction_job with status='importing', stage='bluebeam_import'
- Converts PDF pages to images using PyMuPDF at 150 DPI
- Uploads images to Supabase Storage
- Parses ALL PDF annotations (polygons, polylines, rectangles, stamps)
- Creates extraction_detections_draft records with true geometry
- Infers class from annotation subject, color, or label
- Sets final status to 'classified' for Detection Editor review
"""

import io
import re
import uuid
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("WARNING: pymupdf not installed. Install with: pip install pymupdf", flush=True)

from config import config
from database import supabase_request, upload_to_storage, create_job, update_job, create_page


# =============================================================================
# CLASS MAPPINGS
# =============================================================================

# Subject field → detection class mapping (Bluebeam Subject → our class names)
CLASS_MAPPING = {
    'exterior wall': 'exterior_wall',
    'wall': 'exterior_wall',
    'siding': 'siding',
    'window': 'window',
    'windows': 'window',
    'door': 'door',
    'doors': 'door',
    'garage': 'garage',
    'garage door': 'garage',
    'garage_door': 'garage',
    'gable': 'gable',
    'gables': 'gable',
    'roof': 'roof',
    'roofs': 'roof',
    'soffit': 'soffit',
    'fascia': 'fascia',
    'trim': 'trim',
    'corner': 'corner_outside',
    'outside corner': 'corner_outside',
    'outside_corner': 'corner_outside',
    'inside corner': 'corner_inside',
    'inside_corner': 'corner_inside',
    'belly band': 'belly_band',
    'belly_band': 'belly_band',
    'corbel': 'corbel',
    'shutter': 'shutter',
    'column': 'column',
    'post': 'post',
    'gutter': 'gutter',
    'downspout': 'downspout',
    'flashing': 'flashing',
    'vent': 'vent',
    'eave': 'eave',
    'rake': 'rake',
    'ridge': 'ridge',
    'valley': 'valley',
    'stone': 'stone',
    'stucco': 'stucco',
    'brick': 'brick',
    'building': 'building',
}

# Color → class fallback mapping (RGB hex lowercase)
COLOR_TO_CLASS = {
    '#ff0000': 'siding',         # Red = siding/walls
    '#00ff00': 'window',         # Green = windows
    '#0000ff': 'door',           # Blue = doors
    '#ffff00': 'gable',          # Yellow = gables
    '#ff00ff': 'roof',           # Magenta = roof
    '#00ffff': 'trim',           # Cyan = trim
    '#ff8000': 'garage',         # Orange = garage
    '#800080': 'fascia',         # Purple = fascia
    '#008000': 'soffit',         # Dark green = soffit
    '#804000': 'corner_outside', # Brown = corners
}

# Keyword-based class suggestion for Bluebeam subjects
# NOTE: Keywords are matched in order within each class - longer/more specific first
# IMPORTANT: 'corner' alone is too generic (matches I/S Corner wrong), so use explicit variants
# IMPORTANT: 'deduct' should map to deduction class, not window!
SUBJECT_KEYWORDS = {
    'siding': ['lap siding', 'board & batten', 'board and batten', 'shingle siding',
               'panel siding', 'fiber cement panel', 'shake siding', 'horizontal siding',
               'vertical siding', 'reveal', 'hardie'],
    'trim': ['trim count', 'trim'],
    'soffit': ['soffit'],
    'fascia': ['fascia'],
    'flashing': ['flashing count', 'head flashing', 'flashing'],
    'corbel': ['corbel'],
    'gutter': ['gutter'],
    'downspout': ['downspout'],
    'wrb': ['wrb', 'house wrap', 'weather barrier'],
    'deduction': ['lap deduct', 'panel deduct', 'window deduct', 'deduct'],  # Deductions, NOT windows
    'window': ['window'],  # Removed 'deduct' keywords - they belong to deduction class
    'door': ['door', 'sgd', 'swing'],
    'garage': ['garage door', 'garage'],
    'corner_inside': ['i/s corner', 'inside corner', 'is corner'],  # Check BEFORE corner_outside
    'corner_outside': ['o/s corner', 'outside corner', 'os corner'],  # Removed generic 'corner'
    'belly_band': ['belly band'],
    'column': ['column'],
    'post': ['post'],
    'roof': ['roof'],
    'gable': ['gable'],
    'eave': ['eave'],
    'rake': ['rake'],
    'ridge': ['ridge'],
    'valley': ['valley'],
    'shutter': ['shutter'],
    'vent': ['vent'],
}

# Subjects to skip by default (non-construction measurement/annotation tools)
# Keep this list MINIMAL - only skip obvious non-elements
SKIP_SUBJECTS = [
    'length measurement',
    'arrow',
    'legend',
]


def get_class_from_mapping(mapping: Optional[Dict]) -> Optional[str]:
    """
    Determine detection class from bluebeam_subject_mappings record.

    Uses material_category + sub_category to determine the appropriate
    detection class for the takeoff system.

    Args:
        mapping: A bluebeam_subject_mappings record dict, or None

    Returns:
        Detection class name, or None if no mapping/no match
    """
    if not mapping:
        return None

    cat = mapping.get('material_category', '') or ''
    sub = mapping.get('sub_category', '') or ''

    # Siding types
    if cat in ('lap_siding', 'panel_siding'):
        return 'siding'

    # Corner types (check sub_category first)
    if sub == 'outside_corner':
        return 'corner_outside'
    if sub == 'inside_corner':
        return 'corner_inside'

    # Decorative elements
    if sub == 'decorative':
        return 'corbel'

    # Accessories by sub_category
    if cat == 'accessories':
        if sub == 'flashing':
            return 'flashing'
        if sub == 'wrb':
            return 'wrb'
        if sub == 'vent':
            return 'vent'

    # Direct category mappings
    if cat == 'soffit':
        return 'soffit'
    if cat == 'deduction':
        return 'deduction'
    if cat == 'trim':
        return 'trim'

    # Openings
    if cat == 'opening':
        return 'window' if sub == 'window' else 'door'

    return None


def suggest_class_from_subject(subject: str) -> str:
    """
    Suggest a detection class based on the Bluebeam subject using keyword matching.

    This is the FALLBACK when no bluebeam_subject_mappings record exists.

    Args:
        subject: The annotation subject string from Bluebeam

    Returns:
        Suggested class name, 'SKIP' for non-construction elements, or 'unknown'
    """
    if not subject:
        return 'unknown'

    subject_lower = subject.lower().strip()

    # Handle "(No Subject)" annotations - suggest unknown, not SKIP
    if subject_lower.startswith('(no subject'):
        return 'unknown'

    # Check keyword matches FIRST (longer matches first for specificity)
    # This ensures "Trim Count" matches "trim" before checking skip patterns
    for class_name, keywords in SUBJECT_KEYWORDS.items():
        for keyword in sorted(keywords, key=len, reverse=True):
            if keyword in subject_lower:
                return class_name

    # Only skip if no construction keyword was found
    for skip_pattern in SKIP_SUBJECTS:
        if skip_pattern in subject_lower:
            return 'SKIP'

    return 'unknown'


def parse_bluebeam_content(content: str, subject: str) -> dict:
    """
    Parse the Bluebeam Content/Comments field to extract measurement values.

    Examples:
      "141 sf" → {'area_sf': 141.0}
      "502 sf" → {'area_sf': 502.0}
      "0'-6\"" → (length measurement, skip)
      "6" (with "Trim Count" subject) → {'item_count': 6}
      "3" (with "Flashing Count" subject) → {'item_count': 3}
      "10" (with "Corbel Count" subject) → {'item_count': 10}
    """
    if not content:
        return {}

    content = content.strip()
    result = {}

    # Check for SF area: "141 sf", "502 sf", "30 sf"
    sf_match = re.match(r'^([\d,.]+)\s*sf\s*$', content, re.IGNORECASE)
    if sf_match:
        result['area_sf'] = float(sf_match.group(1).replace(',', ''))
        return result

    # Check for LF length: "12.5 lf", "8 lf"
    lf_match = re.match(r'^([\d,.]+)\s*lf\s*$', content, re.IGNORECASE)
    if lf_match:
        result['perimeter_lf'] = float(lf_match.group(1).replace(',', ''))
        return result

    # Check for pure count (number only) when subject contains "Count"
    subject_lower = (subject or '').lower()
    if 'count' in subject_lower:
        count_match = re.match(r'^(\d+)\s*$', content)
        if count_match:
            result['item_count'] = int(count_match.group(1))
            return result

    # Check for pure number (could be sf if subject suggests area)
    num_match = re.match(r'^([\d,.]+)\s*$', content)
    if num_match and any(kw in subject_lower for kw in ['siding', 'soffit', 'wrb', 'panel', 'batten', 'shingle']):
        result['area_sf'] = float(num_match.group(1).replace(',', ''))
        return result

    return {}


def get_annotation_type_name(annot_type: int) -> str:
    """Convert PyMuPDF annotation type code to human-readable name."""
    type_names = {
        0: 'Text',
        1: 'Link',
        2: 'FreeText',
        3: 'Line',
        4: 'Square',
        5: 'Circle',
        6: 'Polygon',
        7: 'PolyLine',
        8: 'Highlight',
        9: 'Underline',
        10: 'Squiggly',
        11: 'StrikeOut',
        12: 'Redact',
        13: 'Stamp',
        14: 'Caret',
        15: 'Ink',
        16: 'Popup',
        17: 'FileAttachment',
        18: 'Sound',
        19: 'Movie',
        20: 'Widget',
        21: 'Screen',
        22: 'PrinterMark',
        23: 'TrapNet',
        24: 'Watermark',
        25: '3D',
    }
    return type_names.get(annot_type, f'Type-{annot_type}')


# =============================================================================
# PREVIEW FUNCTION (No DB writes)
# =============================================================================

def preview_bluebeam_fresh(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Scan a Bluebeam-annotated PDF and return unique subjects with counts.
    Does NOT create any database records - for preview/mapping step only.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        Dict with:
        - success: bool
        - total_pages: int
        - total_annotations: int
        - subjects: list of {subject, count, annotation_type, sample_content, suggested_class}
        - error: str (if failed)
    """
    if fitz is None:
        return {
            'success': False,
            'error': 'pymupdf not installed. Run: pip install pymupdf'
        }

    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        total_pages = len(doc)

        # Collect subject info: {subject: {count, annotation_types, sample_contents}}
        subject_data = {}
        total_annotations = 0

        for page_num in range(total_pages):
            page = doc[page_num]

            for annot in page.annots():
                if annot is None:
                    continue

                annot_type = annot.type[0]

                # Skip non-markup annotations
                if annot_type in (0, 1, 19, 20):  # Link, Text, Widget, Popup
                    continue

                total_annotations += 1

                # Get subject
                subject = ''
                try:
                    subject = (annot.info.get('subject') or '').strip()
                except:
                    pass

                if not subject:
                    subject = f'(No Subject - {get_annotation_type_name(annot_type)})'

                # Get content (may contain measurement values)
                content = ''
                try:
                    content = (annot.info.get('content') or '').strip()
                except:
                    pass

                # Track this subject
                if subject not in subject_data:
                    subject_data[subject] = {
                        'count': 0,
                        'annotation_types': set(),
                        'sample_contents': [],
                    }

                subject_data[subject]['count'] += 1
                subject_data[subject]['annotation_types'].add(get_annotation_type_name(annot_type))

                # Store up to 3 sample contents
                if content and len(subject_data[subject]['sample_contents']) < 3:
                    if content not in subject_data[subject]['sample_contents']:
                        subject_data[subject]['sample_contents'].append(content)

        doc.close()

        # Build response list sorted by count descending
        subjects = []
        for subject, data in subject_data.items():
            # Get the most common annotation type
            annotation_types = list(data['annotation_types'])
            primary_type = annotation_types[0] if len(annotation_types) == 1 else ', '.join(sorted(annotation_types))

            # Get sample content
            sample_content = data['sample_contents'][0] if data['sample_contents'] else None

            subjects.append({
                'subject': subject,
                'count': data['count'],
                'annotation_type': primary_type,
                'sample_content': sample_content,
                'suggested_class': suggest_class_from_subject(subject),
            })

        # Sort by count descending
        subjects.sort(key=lambda x: x['count'], reverse=True)

        return {
            'success': True,
            'total_pages': total_pages,
            'total_annotations': total_annotations,
            'subjects': subjects,
        }

    except Exception as e:
        print(f"[Bluebeam Preview] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_class_name(raw_name: str) -> str:
    """
    Normalize class name from Bluebeam Subject field.
    Handles variations like "Exterior Wall", "exterior_wall", "WINDOW", etc.
    """
    if not raw_name:
        return 'unknown'

    normalized = raw_name.lower().strip()
    return CLASS_MAPPING.get(normalized, normalized.replace(' ', '_'))


def classify_page_from_annotations(annotations_on_page: List[Dict], page_num: int) -> str:
    """
    Classify a page based on what Bluebeam annotations are present.

    Rules:
    - Pages with siding/trim/window/flashing/soffit/wrb annotations → 'elevation'
    - Pages with roof annotations → 'roof_plan'
    - Pages with only measurement lines or no annotations → 'detail'
    - Page 1 with no annotations → 'cover'
    """
    if not annotations_on_page:
        return 'cover' if page_num == 1 else 'detail'

    # Check what classes are present
    classes = set(a.get('class', '') for a in annotations_on_page)

    # Elevation indicators - typical siding/exterior elements
    elevation_classes = {
        'siding', 'trim', 'window', 'door', 'garage', 'soffit',
        'fascia', 'flashing', 'corbel', 'wrb', 'belly_band',
        'corner_outside', 'corner_inside', 'column', 'post',
        'gutter', 'downspout', 'shutter', 'exterior_wall',
        'eave', 'rake', 'stone', 'stucco', 'brick', 'vent'
    }

    if classes & elevation_classes:
        return 'elevation'

    # Roof plan indicators
    roof_classes = {'roof', 'ridge', 'valley', 'gable'}
    if classes & roof_classes:
        return 'roof_plan'

    return 'detail'


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

    # Convert to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(color[0] * 255),
        int(color[1] * 255),
        int(color[2] * 255)
    )

    return COLOR_TO_CLASS.get(hex_color.lower())


def infer_class(annot) -> str:
    """
    Infer detection class from Bluebeam annotation properties.
    Priority: subject field > color > label > default based on type
    """
    # 1. Check subject field (Bluebeam lets users set this)
    subject = None
    try:
        annot_info = annot.info
        subject = (annot_info.get('subject') or '').strip().lower()
    except:
        pass

    if subject:
        mapped = CLASS_MAPPING.get(subject)
        if mapped:
            return mapped

    # 2. Check annotation color
    try:
        colors = annot.colors
        if colors and colors.get('stroke'):
            stroke = colors['stroke']
            color_class = infer_class_from_color(stroke)
            if color_class:
                return color_class
    except:
        pass

    # 3. Check content/label
    try:
        content = (annot.info.get('content') or '').strip().lower()
        if content:
            mapped = CLASS_MAPPING.get(content)
            if mapped:
                return mapped
    except:
        pass

    # 4. Default based on annotation type
    try:
        annot_type = annot.type[0]
        if annot_type in (4, 5, 6):  # Square, Circle, Polygon - area markups
            return 'siding'  # Default area class
        elif annot_type == 7:  # Polyline
            return 'trim'  # Default linear class
        elif annot_type == 13:  # Stamp
            return 'window'  # Default count class
    except:
        pass

    return 'unknown'


def deduplicate_count_detections(detections: List[Dict]) -> List[Dict]:
    """
    Deduplicate Bluebeam Count markup detections to prevent N×N overcounting.

    Bluebeam Count markups export each click-point as a separate annotation,
    but each carries the GROUP TOTAL as the value. So 6 clicks = 6 annotations
    each saying "6" = 36 instead of correct 6.

    FIX: Group by (marker_label, page_id, bluebeam_content) and keep only one.

    Args:
        detections: List of detection dicts from parse_page_annotations()

    Returns:
        Deduplicated list of detections
    """
    count_detections = []
    non_count_detections = []

    for det in detections:
        marker_label = (det.get('marker_label') or '').lower()
        # Check if this is a Count markup (subject contains "count")
        if 'count' in marker_label:
            count_detections.append(det)
        else:
            non_count_detections.append(det)

    # Deduplicate count detections by (marker_label, page_id, bluebeam_content)
    seen = set()
    deduped_counts = []

    for det in count_detections:
        key = (
            det.get('marker_label'),
            det.get('page_id'),
            det.get('bluebeam_content')
        )
        if key not in seen:
            seen.add(key)
            deduped_counts.append(det)

    original_count = len(count_detections)
    deduped_count = len(deduped_counts)
    if original_count != deduped_count:
        print(f"[Bluebeam Fresh] Deduplicated Count markups: {original_count} → {deduped_count} (removed {original_count - deduped_count} duplicates)", flush=True)

    return non_count_detections + deduped_counts


def shoelace_area(vertices: List[Tuple[float, float]]) -> float:
    """Calculate polygon area using Shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2


def calculate_polyline_length(vertices: List[Tuple[float, float]]) -> float:
    """Calculate total length of a polyline."""
    total_length = 0
    for i in range(len(vertices) - 1):
        dx = vertices[i + 1][0] - vertices[i][0]
        dy = vertices[i + 1][1] - vertices[i][1]
        total_length += math.sqrt(dx * dx + dy * dy)
    return total_length


# =============================================================================
# PAGE CONVERSION
# =============================================================================

def convert_and_upload_pages(doc, job_id: str, dpi: int = 150) -> List[Dict[str, Any]]:
    """
    Convert PDF pages to images and upload to Supabase Storage.

    Renders both clean (no annotations) and annotated versions for Bluebeam imports.

    Args:
        doc: PyMuPDF document object
        job_id: UUID of the extraction job
        dpi: Resolution for image conversion (default 150)

    Returns:
        List of page info dicts with image_url, thumbnail_url, annotated_image_url, width, height, dpi
    """
    pages = []

    for i in range(len(doc)):
        page = doc[i]
        page_num = i + 1

        print(f"[Bluebeam Fresh] Converting page {page_num}/{len(doc)}", flush=True)

        mat = fitz.Matrix(dpi / 72, dpi / 72)

        # Render clean image (without Bluebeam markup overlays)
        pix = page.get_pixmap(matrix=mat, annots=False)
        img_bytes = pix.tobytes("png")

        # Upload clean image
        path = f"extraction-pages/{job_id}/page-{page_num}.png"
        image_url = upload_to_storage(img_bytes, path, 'image/png', bucket='extraction-markups')

        # Render annotated image (WITH Bluebeam markup overlays visible)
        pix_annotated = page.get_pixmap(matrix=mat, annots=True)
        annotated_bytes = pix_annotated.tobytes("png")

        # Upload annotated image
        annotated_path = f"extraction-pages/{job_id}/page-{page_num}_annotated.png"
        annotated_image_url = upload_to_storage(annotated_bytes, annotated_path, 'image/png', bucket='extraction-markups')

        # Generate and upload thumbnail (25% size, clean version only)
        thumb_mat = fitz.Matrix(dpi / 72 * 0.25, dpi / 72 * 0.25)
        thumb_pix = page.get_pixmap(matrix=thumb_mat, annots=False)
        thumb_bytes = thumb_pix.tobytes("png")
        thumb_path = f"extraction-pages/{job_id}/thumb-{page_num}.png"
        thumb_url = upload_to_storage(thumb_bytes, thumb_path, 'image/png', bucket='extraction-markups')

        pages.append({
            'page_number': page_num,
            'image_url': image_url,
            'annotated_image_url': annotated_image_url,
            'thumbnail_url': thumb_url,
            'width': pix.width,
            'height': pix.height,
            'dpi': dpi
        })

    return pages


# =============================================================================
# ANNOTATION PARSING
# =============================================================================

def parse_rect_annotation(annot, scale_x: float, scale_y: float, page_record: Dict) -> Optional[Dict]:
    """Extract rectangle annotation as detection."""
    rect = annot.rect

    # Convert PDF points to pixel coordinates
    x1 = rect.x0 * scale_x
    y1 = rect.y0 * scale_y
    x2 = rect.x1 * scale_x
    y2 = rect.y1 * scale_y

    width = x2 - x1
    height = y2 - y1

    # Center-based coordinates
    pixel_x = x1 + width / 2
    pixel_y = y1 + height / 2

    # Calculate area in PDF points first, then convert
    area_points = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
    # PDF points to inches: divide by 72
    area_sq_inches = area_points / (72 * 72)
    area_sf = area_sq_inches / 144  # inches² to ft²

    # Apply scale ratio if set
    scale_ratio = page_record.get('scale_ratio') or page_record.get('dpi', 72)
    if page_record.get('scale_ratio'):
        area_sf = area_sq_inches * (scale_ratio ** 2) / 144

    # Perimeter calculation
    perimeter_points = 2 * ((rect.x1 - rect.x0) + (rect.y1 - rect.y0))
    perimeter_inches = perimeter_points / 72
    perimeter_lf = perimeter_inches / 12
    if page_record.get('scale_ratio'):
        perimeter_lf = perimeter_inches * scale_ratio / 12

    return {
        'pixel_x': round(pixel_x, 2),
        'pixel_y': round(pixel_y, 2),
        'pixel_width': round(width, 2),
        'pixel_height': round(height, 2),
        'markup_type': 'polygon',
        'area_sf': round(area_sf, 2),
        'perimeter_lf': round(perimeter_lf, 2),
    }


def parse_polygon_annotation(annot, scale_x: float, scale_y: float, page_record: Dict) -> Optional[Dict]:
    """Extract true polygon vertices and calculate area."""
    vertices = annot.vertices
    if not vertices or len(vertices) < 3:
        return parse_rect_annotation(annot, scale_x, scale_y, page_record)

    # Convert PDF points to pixel coordinates
    pixel_vertices = [
        {'x': round(v[0] * scale_x, 2), 'y': round(v[1] * scale_y, 2)}
        for v in vertices
    ]

    # Calculate area using Shoelace formula (in PDF points)
    area_points = shoelace_area(vertices)

    # Convert to real-world SF
    area_sq_inches = area_points / (72 * 72)
    area_sf = area_sq_inches / 144

    scale_ratio = page_record.get('scale_ratio')
    if scale_ratio:
        area_sf = area_sq_inches * (scale_ratio ** 2) / 144

    # Calculate perimeter
    perimeter_points = 0
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        dx = vertices[j][0] - vertices[i][0]
        dy = vertices[j][1] - vertices[i][1]
        perimeter_points += math.sqrt(dx * dx + dy * dy)

    perimeter_inches = perimeter_points / 72
    perimeter_lf = perimeter_inches / 12
    if scale_ratio:
        perimeter_lf = perimeter_inches * scale_ratio / 12

    # Bounding box for pixel coordinates
    xs = [v['x'] for v in pixel_vertices]
    ys = [v['y'] for v in pixel_vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return {
        'pixel_x': round((min_x + max_x) / 2, 2),  # Center X
        'pixel_y': round((min_y + max_y) / 2, 2),  # Center Y
        'pixel_width': round(max_x - min_x, 2),
        'pixel_height': round(max_y - min_y, 2),
        'polygon_points': pixel_vertices,
        'markup_type': 'polygon',
        'area_sf': round(area_sf, 2),
        'perimeter_lf': round(perimeter_lf, 2),
    }


def parse_polyline_annotation(annot, scale_x: float, scale_y: float, page_record: Dict) -> Optional[Dict]:
    """Extract polyline vertices and calculate linear footage."""
    vertices = annot.vertices
    if not vertices or len(vertices) < 2:
        return None

    # Convert to pixel coordinates
    pixel_vertices = [
        {'x': round(v[0] * scale_x, 2), 'y': round(v[1] * scale_y, 2)}
        for v in vertices
    ]

    # Calculate total length in PDF points
    total_length_pts = calculate_polyline_length(vertices)

    # Convert to real-world LF
    length_inches = total_length_pts / 72
    perimeter_lf = length_inches / 12

    scale_ratio = page_record.get('scale_ratio')
    if scale_ratio:
        perimeter_lf = length_inches * scale_ratio / 12

    xs = [v['x'] for v in pixel_vertices]
    ys = [v['y'] for v in pixel_vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return {
        'pixel_x': round((min_x + max_x) / 2, 2),
        'pixel_y': round((min_y + max_y) / 2, 2),
        'pixel_width': round(max_x - min_x, 2),
        'pixel_height': round(max_y - min_y, 2),
        'polygon_points': pixel_vertices,
        'markup_type': 'line',
        'perimeter_lf': round(perimeter_lf, 2),
    }


def parse_stamp_annotation(annot, scale_x: float, scale_y: float, page_record: Dict) -> Optional[Dict]:
    """Parse stamp/point annotation as a count marker."""
    rect = annot.rect
    cx = (rect.x0 + rect.x1) / 2 * scale_x
    cy = (rect.y0 + rect.y1) / 2 * scale_y

    # Get label from annotation
    marker_label = None
    try:
        marker_label = annot.info.get('subject', '') or annot.info.get('title', '')
    except:
        pass

    return {
        'pixel_x': round(cx, 2),
        'pixel_y': round(cy, 2),
        'pixel_width': 16,  # Small marker box
        'pixel_height': 16,
        'markup_type': 'point',
        'marker_label': marker_label,
    }


def parse_page_annotations(
    pdf_page,
    page_record: Dict,
    page_index: int,
    subject_class_map: Dict[str, str] = None,
    mapping_dict: Dict[str, Dict] = None,
    pricing_by_sku: Dict[str, str] = None
) -> List[Dict]:
    """
    Extract all annotations from a PDF page.
    Handles: rectangles, polygons, polylines, points/stamps.

    Args:
        pdf_page: PyMuPDF page object
        page_record: Page record dict with id, job_id, dimensions
        page_index: Zero-based page index
        subject_class_map: Optional mapping of Bluebeam subjects to class names
                           (from frontend UI). If a subject maps to 'SKIP', the
                           annotation is skipped.
        mapping_dict: Dict of bluebeam_subject_mappings keyed by bluebeam_subject.
                      Used for DB-driven class determination and material assignment.
        pricing_by_sku: Dict mapping SKU → pricing_items.id for material assignment.
    """
    detections = []
    page_id = page_record['id']
    page_width = page_record.get('original_width') or page_record.get('width')
    page_height = page_record.get('original_height') or page_record.get('height')

    if not page_width or not page_height:
        print(f"[Bluebeam Fresh] Warning: No page dimensions for page {page_index + 1}", flush=True)
        return detections

    # PDF page dimensions (points, 72 per inch)
    pdf_rect = pdf_page.rect
    pdf_width = pdf_rect.width
    pdf_height = pdf_rect.height

    # Scale factor: PDF points → image pixels
    scale_x = page_width / pdf_width
    scale_y = page_height / pdf_height

    detection_index = 0

    for annot in pdf_page.annots():
        if annot is None:
            continue

        annot_type = annot.type[0]  # Integer type code

        # Skip non-markup annotations (links, widgets, popups, free text)
        if annot_type in (0, 1, 2, 19, 20):  # Link, Text, FreeText, Widget, Popup
            continue

        # Get the annotation subject (used for class mapping and marker_label)
        annot_subject = ''
        annot_content = ''
        try:
            annot_subject = (annot.info.get('subject') or '').strip()
            annot_content = (annot.info.get('content') or '').strip()
        except:
            pass

        # Look up subject in bluebeam_subject_mappings (for class + material)
        db_mapping = mapping_dict.get(annot_subject) if mapping_dict and annot_subject else None
        # Resolve suggested_sku → pricing_items.id (downstream n8n joins against pricing_items, not product_catalog)
        sku = db_mapping.get('suggested_sku') if db_mapping else None
        assigned_material_id = pricing_by_sku.get(sku) if sku and pricing_by_sku else None

        # Determine detection class using priority:
        # 1. Frontend subject_class_map (user override from UI)
        # 2. DB mapping via get_class_from_mapping()
        # 3. Fallback to keyword-based infer_class()
        det_class = None

        # Priority 1: Frontend UI mapping (allows user to override/skip)
        if subject_class_map and annot_subject:
            mapped_class = subject_class_map.get(annot_subject)
            if mapped_class == 'SKIP':
                # Skip this annotation entirely
                continue
            elif mapped_class:
                det_class = mapped_class

        # Priority 2: Database mapping from bluebeam_subject_mappings
        if det_class is None and db_mapping:
            det_class = get_class_from_mapping(db_mapping)

        # Priority 3: Fallback to keyword inference
        if det_class is None:
            det_class = infer_class(annot)

        # Determine geometry type and extract coordinates
        detection = None

        if annot_type in (4, 5):  # Square/Circle → area markup
            detection = parse_rect_annotation(annot, scale_x, scale_y, page_record)

        elif annot_type == 6:  # Polygon
            detection = parse_polygon_annotation(annot, scale_x, scale_y, page_record)

        elif annot_type == 7:  # Polyline
            detection = parse_polyline_annotation(annot, scale_x, scale_y, page_record)

        elif annot_type in (3, 8, 9, 10, 11):  # Line, Highlight, Underline, etc.
            # Treat as line annotation
            try:
                if hasattr(annot, 'vertices') and annot.vertices:
                    detection = parse_polyline_annotation(annot, scale_x, scale_y, page_record)
                else:
                    detection = parse_rect_annotation(annot, scale_x, scale_y, page_record)
            except:
                detection = parse_rect_annotation(annot, scale_x, scale_y, page_record)

        elif annot_type == 13:  # Stamp (count marker)
            detection = parse_stamp_annotation(annot, scale_x, scale_y, page_record)

        else:
            # Fallback: use bounding rect
            detection = parse_rect_annotation(annot, scale_x, scale_y, page_record)

        if detection:
            detection_index += 1
            detection.update({
                'id': str(uuid.uuid4()),
                'job_id': page_record['job_id'],
                'page_id': page_id,
                'class': det_class,
                'detection_index': detection_index,
                'confidence': 1.0,
                'is_user_created': True,
                'is_deleted': False,
                'status': 'complete',
                'marker_label': annot_subject if annot_subject else None,  # Store original Bluebeam subject
                'bluebeam_content': annot_subject if annot_subject else None,  # Store subject name for takeoff grouping
                'assigned_material_id': assigned_material_id,  # pricing_items.id resolved from suggested_sku
            })

            # Parse Bluebeam content field for real measurement values
            if annot_content:
                parsed_values = parse_bluebeam_content(annot_content, annot_subject)
                if 'area_sf' in parsed_values:
                    detection['area_sf'] = parsed_values['area_sf']
                if 'perimeter_lf' in parsed_values:
                    detection['perimeter_lf'] = parsed_values['perimeter_lf']
                if 'item_count' in parsed_values:
                    detection['item_count'] = parsed_values['item_count']

            detections.append(detection)

    return detections


# =============================================================================
# MAIN IMPORT FUNCTION
# =============================================================================

def import_bluebeam_fresh(
    pdf_bytes: bytes,
    project_id: str,
    project_name: str = None,
    organization_id: str = None,
    subject_class_map: Dict[str, str] = None,
    bluebeam_project_id: str = None
) -> Dict[str, Any]:
    """
    Full fresh import: PDF → job + pages + detections.
    No prior extraction job required.

    Args:
        pdf_bytes: Raw PDF file bytes
        project_id: UUID of the project to associate with
        project_name: Optional project name for display
        organization_id: Optional organization ID for multi-tenant
        subject_class_map: Optional mapping of Bluebeam subjects to class names.
                           If a subject maps to 'SKIP', the annotation is skipped.
        bluebeam_project_id: Optional UUID of bluebeam_projects record to link.
                             If provided, updates bluebeam_projects.cad_extraction_id.

    Returns:
        Dict with:
        - success: bool
        - job_id: UUID of created job
        - total_pages: int
        - total_detections: int
        - detection_summary: dict by class with counts and totals
        - pages: list of page info
        - error: str (if failed)
    """
    if fitz is None:
        return {
            'success': False,
            'error': 'pymupdf not installed. Run: pip install pymupdf'
        }

    print(f"[Bluebeam Fresh] Starting fresh import for project {project_id}", flush=True)

    try:
        # 0. Load bluebeam_subject_mappings from database (one query for all subjects)
        # This provides class mapping and material assignment for each Bluebeam subject
        print("[Bluebeam Fresh] Loading bluebeam_subject_mappings...", flush=True)
        mappings_raw = supabase_request('GET', 'bluebeam_subject_mappings?active=eq.true')
        mapping_dict = {}
        if mappings_raw:
            for m in mappings_raw:
                subject = m.get('bluebeam_subject')
                if subject:
                    mapping_dict[subject] = m
            print(f"[Bluebeam Fresh] Loaded {len(mapping_dict)} subject mappings", flush=True)
        else:
            print("[Bluebeam Fresh] No subject mappings found, will use keyword fallback", flush=True)

        # Load pricing_items by SKU to resolve suggested_sku → pricing_items.id
        # The downstream n8n "Query Assigned Materials" joins against pricing_items, not product_catalog
        pricing_by_sku = {}
        all_skus = [m.get('suggested_sku') for m in mapping_dict.values() if m.get('suggested_sku')]
        if all_skus:
            unique_skus = list(set(all_skus))
            sku_filter = ','.join(f'"{s}"' for s in unique_skus)
            pricing_items = supabase_request('GET', f'pricing_items?sku=in.({sku_filter})&active=eq.true&select=id,sku')
            if pricing_items:
                pricing_by_sku = {p['sku']: p['id'] for p in pricing_items}
                print(f"[Bluebeam Fresh] Loaded {len(pricing_by_sku)} pricing_items for SKU lookup", flush=True)
            else:
                print("[Bluebeam Fresh] No pricing_items found for suggested SKUs", flush=True)
        else:
            print("[Bluebeam Fresh] No suggested_sku values in mappings", flush=True)

        # 1. Create extraction job
        job_data = {
            'project_id': project_id,
            'project_name': project_name or 'Bluebeam Import',
            'status': 'importing',
            'stage': 'uploaded',
            'source_pdf_url': None,  # No source PDF URL for fresh imports
        }


        job = create_job(job_data)

        if not job:
            return {'success': False, 'error': 'Failed to create extraction job'}

        job_id = job['id']
        print(f"[Bluebeam Fresh] Created job {job_id}", flush=True)

        # 1b. Link bluebeam_projects.cad_extraction_id if bluebeam_project_id provided
        if bluebeam_project_id:
            link_result = supabase_request(
                'PATCH',
                f'bluebeam_projects?id=eq.{bluebeam_project_id}',
                {'cad_extraction_id': job_id}
            )
            if link_result:
                print(f"[Bluebeam Fresh] Linked bluebeam_project {bluebeam_project_id} → job {job_id}", flush=True)
            else:
                print(f"[Bluebeam Fresh] Warning: Failed to link bluebeam_project {bluebeam_project_id}", flush=True)

        # 2. Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        total_pages = len(doc)
        print(f"[Bluebeam Fresh] PDF has {total_pages} pages", flush=True)

        # 3. Convert pages to images and upload to storage
        page_images = convert_and_upload_pages(doc, job_id, dpi=150)

        # 4. Create extraction_pages records
        page_records = []
        for page_img in page_images:
            page_data = {
                'job_id': job_id,
                'page_number': page_img['page_number'],
                'image_url': page_img['image_url'],
                'annotated_image_url': page_img.get('annotated_image_url'),
                'thumbnail_url': page_img.get('thumbnail_url'),
                'original_image_url': page_img['image_url'],
                'original_width': page_img['width'],
                'original_height': page_img['height'],
                'page_type': 'elevation',  # Default; user can reclassify
                'status': 'complete',
                'dpi': page_img.get('dpi', 150),
            }

            page_record = create_page(page_data)
            if isinstance(page_record, list): page_record = page_record[0]
            if page_record:
                # Add dimensions back for annotation parsing
                page_record['width'] = page_img['width']
                page_record['height'] = page_img['height']
                page_records.append(page_record)

        print(f"[Bluebeam Fresh] Created {len(page_records)} page records", flush=True)

        # 5. Parse annotations from each page and auto-classify page types
        all_detections = []
        page_annotation_counts = []
        elevation_count = 0

        for page_num in range(total_pages):
            pdf_page = doc[page_num]
            page_record = page_records[page_num] if page_num < len(page_records) else None

            if not page_record:
                page_annotation_counts.append({
                    'page_number': page_num + 1,
                    'annotation_count': 0
                })
                continue

            detections = parse_page_annotations(pdf_page, page_record, page_num, subject_class_map, mapping_dict, pricing_by_sku)
            all_detections.extend(detections)

            # Auto-classify page based on annotation content
            page_type = classify_page_from_annotations(detections, page_num + 1)

            # Generate elevation name for elevation pages
            elevation_name = None
            if page_type == 'elevation':
                elevation_count += 1
                elevation_name = f"Elevation {elevation_count}"

            # Update page record with classification
            update_data = {'page_type': page_type}
            if elevation_name:
                update_data['elevation_name'] = elevation_name

            supabase_request('PATCH', f'extraction_pages?id=eq.{page_record["id"]}', update_data)

            page_annotation_counts.append({
                'page_number': page_num + 1,
                'annotation_count': len(detections),
                'page_type': page_type
            })

        print(f"[Bluebeam Fresh] Parsed {len(all_detections)} annotations, classified {elevation_count} elevations", flush=True)

        # 5b. Deduplicate Count markups to prevent N×N overcounting
        # Bluebeam Count markups export each click-point as separate annotation with GROUP TOTAL
        all_detections = deduplicate_count_detections(all_detections)

        # 6. Batch insert detections into extraction_detections_draft
        if all_detections:
            # Strip keys that don't exist in extraction_detections_draft table
            ALLOWED_COLUMNS = {
                'id', 'job_id', 'page_id', 'source_detection_id', 'class',
                'pixel_x', 'pixel_y', 'pixel_width', 'pixel_height',
                'confidence', 'detection_index', 'matched_tag', 'is_triangle',
                'assigned_material_id', 'material_notes', 'is_deleted',
                'is_user_created', 'polygon_points', 'markup_type',
                'marker_label', 'material_cost_override', 'color_override', 'status',
                'area_sf', 'perimeter_lf', 'item_count', 'bluebeam_content'
            }
            for det in all_detections:
                keys_to_remove = [k for k in det if k not in ALLOWED_COLUMNS]
                for k in keys_to_remove:
                    del det[k]

            # Normalize all detections to have consistent keys for PostgREST batch insert
            all_keys = set()
            for det in all_detections:
                all_keys.update(det.keys())
            for det in all_detections:
                for key in all_keys:
                    if key not in det:
                        det[key] = None

            # Insert in batches of 50 to avoid timeouts
            batch_size = 50
            for i in range(0, len(all_detections), batch_size):
                batch = all_detections[i:i + batch_size]
                result = supabase_request('POST', 'extraction_detections_draft', batch)
                if result:
                    print(f"[Bluebeam Fresh] Inserted batch {i // batch_size + 1} ({len(batch)} detections)", flush=True)
                else:
                    print(f"[Bluebeam Fresh] FAILED batch {i // batch_size + 1}. Sample detection keys: {list(batch[0].keys())}", flush=True)
                    print(f"[Bluebeam Fresh] Sample detection: {batch[0]}", flush=True)

        # 7. Build detection summary by class
        detection_summary = {}
        for det in all_detections:
            cls = det.get('class', 'unknown')
            if cls not in detection_summary:
                detection_summary[cls] = {
                    'count': 0,
                    'total_sf': 0,
                    'total_lf': 0,
                }
            detection_summary[cls]['count'] += 1
            detection_summary[cls]['total_sf'] += det.get('area_sf') or 0
            detection_summary[cls]['total_lf'] += det.get('perimeter_lf') or 0

        # Round totals
        for cls in detection_summary:
            detection_summary[cls]['total_sf'] = round(detection_summary[cls]['total_sf'], 2)
            detection_summary[cls]['total_lf'] = round(detection_summary[cls]['total_lf'], 2)

        # 7b. Calculate and insert extraction_job_totals (required for Approve button)
        # Aggregate item_count by class (for corners, flashing counts, etc.)
        item_counts_by_class = {}
        for det in all_detections:
            cls = det.get('class', 'unknown')
            item_count = det.get('item_count') or 0
            if cls not in item_counts_by_class:
                item_counts_by_class[cls] = 0
            item_counts_by_class[cls] += item_count

        # Calculate totals for extraction_job_totals
        # Siding SF = all siding-type area_sf values
        siding_sf = detection_summary.get('siding', {}).get('total_sf', 0)
        # Deductions = opening deductions
        deduction_sf = detection_summary.get('deduction', {}).get('total_sf', 0)
        # Net siding = gross siding - deductions
        net_siding_sf = max(0, siding_sf - deduction_sf)

        # Corner counts from item_count aggregation
        outside_corners = item_counts_by_class.get('corner_outside', 0)
        inside_corners = item_counts_by_class.get('corner_inside', 0)

        # Window/door/garage counts from flashing item_counts
        # (Bluebeam uses "Window Head Flashing Count" etc. which map to 'flashing' class)
        # For now, estimate from flashing counts or detection counts
        window_count = item_counts_by_class.get('window', 0)
        door_count = item_counts_by_class.get('door', 0)
        garage_count = item_counts_by_class.get('garage', 0)

        # Gable and roof SF
        gable_sf = detection_summary.get('gable', {}).get('total_sf', 0)
        roof_sf = detection_summary.get('roof', {}).get('total_sf', 0)
        soffit_sf = detection_summary.get('soffit', {}).get('total_sf', 0)

        # Build totals record - ONLY include columns that exist on extraction_job_totals table
        # Schema: job_id, elevation_count, total_windows, total_doors, total_garages, total_gables,
        #         total_gross_facade_sf, total_openings_sf, total_net_siding_sf, total_gable_sf,
        #         total_roof_sf, siding_squares, outside_corners_count, inside_corners_count,
        #         outside_corners_lf, inside_corners_lf, corner_source, detection_counts_by_class,
        #         calculation_version, calculated_at
        job_totals = {
            'job_id': job_id,
            'elevation_count': elevation_count,
            'total_gross_facade_sf': round(siding_sf, 2),
            'total_openings_sf': round(deduction_sf, 2),
            'total_net_siding_sf': round(net_siding_sf, 2),
            'total_windows': window_count,
            'total_doors': door_count,
            'total_garages': garage_count,
            'total_gables': detection_summary.get('gable', {}).get('count', 0),
            'total_gable_sf': round(gable_sf, 2),
            'total_roof_sf': round(roof_sf, 2),
            'outside_corners_count': outside_corners,
            'inside_corners_count': inside_corners,
            'outside_corners_lf': 0,  # Not available from Bluebeam count markups
            'inside_corners_lf': 0,   # Not available from Bluebeam count markups
            'siding_squares': round(net_siding_sf / 100, 2),  # SF to squares
            'calculation_version': 'bluebeam_import_v1',
            'corner_source': 'bluebeam_markup',
            'detection_counts_by_class': detection_summary,
            'calculated_at': datetime.now(timezone.utc).isoformat(),
        }

        # Insert totals record with detailed logging
        print(f"[Bluebeam Fresh] Inserting extraction_job_totals: {job_totals}", flush=True)
        try:
            totals_result = supabase_request('POST', 'extraction_job_totals', job_totals)
            if totals_result:
                print(f"[Bluebeam Fresh] Created extraction_job_totals: {net_siding_sf:.0f} net SF, {outside_corners} O/S corners, {inside_corners} I/S corners", flush=True)
            else:
                print(f"[Bluebeam Fresh] WARNING: extraction_job_totals INSERT returned None/empty", flush=True)
        except Exception as totals_err:
            print(f"[Bluebeam Fresh] ERROR inserting extraction_job_totals: {totals_err}", flush=True)
            import traceback
            traceback.print_exc()

        # 8. Update job status to 'complete' (ready for Detection Editor)
        update_job(job_id, {
            'status': 'complete',
            'total_pages': total_pages,
            'elevation_count': elevation_count,  # Actual count from auto-classification
            'total_detections': len(all_detections),
            'completed_at': datetime.now(timezone.utc).isoformat(),
        })

        doc.close()

        print(f"[Bluebeam Fresh] Import complete: {len(all_detections)} detections from {total_pages} pages", flush=True)

        return {
            'success': True,
            'job_id': job_id,
            'total_pages': total_pages,
            'total_detections': len(all_detections),
            'detection_summary': detection_summary,
            'pages': page_annotation_counts,
        }

    except Exception as e:
        print(f"[Bluebeam Fresh] Import failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'error': str(e)
        }
