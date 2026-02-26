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

    Args:
        doc: PyMuPDF document object
        job_id: UUID of the extraction job
        dpi: Resolution for image conversion (default 150)

    Returns:
        List of page info dicts with image_url, thumbnail_url, width, height, dpi
    """
    pages = []

    for i in range(len(doc)):
        page = doc[i]
        page_num = i + 1

        print(f"[Bluebeam Fresh] Converting page {page_num}/{len(doc)}", flush=True)

        # Render at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # Upload full image
        path = f"extraction-pages/{job_id}/page-{page_num}.png"
        image_url = upload_to_storage(img_bytes, path, 'image/png', bucket='extraction-markups')

        # Generate and upload thumbnail (25% size)
        thumb_mat = fitz.Matrix(dpi / 72 * 0.25, dpi / 72 * 0.25)
        thumb_pix = page.get_pixmap(matrix=thumb_mat)
        thumb_bytes = thumb_pix.tobytes("png")
        thumb_path = f"extraction-pages/{job_id}/thumb-{page_num}.png"
        thumb_url = upload_to_storage(thumb_bytes, thumb_path, 'image/png', bucket='extraction-markups')

        pages.append({
            'page_number': page_num,
            'image_url': image_url,
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


def parse_page_annotations(pdf_page, page_record: Dict, page_index: int) -> List[Dict]:
    """
    Extract all annotations from a PDF page.
    Handles: rectangles, polygons, polylines, points/stamps.
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

        # Infer class from annotation properties
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
                'status': 'imported',
            })
            detections.append(detection)

    return detections


# =============================================================================
# MAIN IMPORT FUNCTION
# =============================================================================

def import_bluebeam_fresh(
    pdf_bytes: bytes,
    project_id: str,
    project_name: str = None,
    organization_id: str = None
) -> Dict[str, Any]:
    """
    Full fresh import: PDF → job + pages + detections.
    No prior extraction job required.

    Args:
        pdf_bytes: Raw PDF file bytes
        project_id: UUID of the project to associate with
        project_name: Optional project name for display
        organization_id: Optional organization ID for multi-tenant

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
                'thumbnail_url': page_img.get('thumbnail_url'),
                'original_image_url': page_img['image_url'],
                'original_width': page_img['width'],
                'original_height': page_img['height'],
                'page_type': 'elevation',  # Default; user can reclassify
                'status': 'imported',
                'dpi': page_img.get('dpi', 150),
            }

            page_record = create_page(page_data)
            if page_record:
                # Add dimensions back for annotation parsing
                page_record['width'] = page_img['width']
                page_record['height'] = page_img['height']
                page_records.append(page_record)

        print(f"[Bluebeam Fresh] Created {len(page_records)} page records", flush=True)

        # 5. Parse annotations from each page
        all_detections = []
        page_annotation_counts = []

        for page_num in range(total_pages):
            pdf_page = doc[page_num]
            page_record = page_records[page_num] if page_num < len(page_records) else None

            if not page_record:
                page_annotation_counts.append({
                    'page_number': page_num + 1,
                    'annotation_count': 0
                })
                continue

            detections = parse_page_annotations(pdf_page, page_record, page_num)
            all_detections.extend(detections)

            page_annotation_counts.append({
                'page_number': page_num + 1,
                'annotation_count': len(detections)
            })

        print(f"[Bluebeam Fresh] Parsed {len(all_detections)} annotations", flush=True)

        # 6. Batch insert detections into extraction_detections_draft
        if all_detections:
            # Insert in batches of 50 to avoid timeouts
            batch_size = 50
            for i in range(0, len(all_detections), batch_size):
                batch = all_detections[i:i + batch_size]
                result = supabase_request('POST', 'extraction_detections_draft', batch)
                if not result:
                    print(f"[Bluebeam Fresh] Warning: Failed to insert batch {i // batch_size + 1}", flush=True)

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

        # 8. Update job status to 'classified' (ready for Detection Editor)
        update_job(job_id, {
            'status': 'classified',
            'total_pages': total_pages,
            'elevation_count': total_pages,  # Assume all pages are elevations
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
