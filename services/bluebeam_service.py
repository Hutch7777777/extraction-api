"""
Bluebeam PDF export service.

Generates annotated PDFs with detection overlays that Bluebeam reads as native markups.
Uses PyMuPDF (fitz) to create standard PDF annotations.
"""

import io
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("WARNING: pymupdf not installed. Install with: pip install pymupdf", flush=True)

try:
    from PIL import Image
except ImportError:
    Image = None

from config import config
from database import supabase_request, upload_to_storage


# Detection class categorization for measurement types
# Include both underscore and space variants since DB uses both formats
AREA_CLASSES = {
    'building',
    'exterior_wall', 'exterior wall',
    'gable',
    'soffit',
    'door',
    'window',
    'garage_door', 'garage door', 'garage',
    'siding',
    'roof',
}
LINEAR_CLASSES = {
    'fascia',
    'trim',
    'belly_band', 'belly band',
    'rake',
    'frieze',
    'flashing',
    'eave',
}
POINT_CLASSES = {
    'hose_bib', 'hose bib',
    'vent',
    'light_fixture', 'light fixture',
    'outlet',
    'corbel',
    # Corner markers are points, not linear measurements
    'corner',
    'corner_inside', 'corner inside', 'inside_corner', 'inside corner',
    'corner_outside', 'corner outside', 'outside_corner', 'outside corner',
}

# Classes to skip in Bluebeam export (helper markups for detection, not estimation)
SKIP_CLASSES = {'building'}


# =============================================================================
# PIXEL-TO-REAL-WORLD CONVERSION FUNCTIONS
# =============================================================================
# IMPORTANT: scale_ratio from extraction_pages is PIXELS PER FOOT
# (already accounts for image resizing from original scan)
# To convert: feet = pixels / scale_ratio

def pixels_to_feet(pixels: float, scale_ratio: float) -> float:
    """
    Convert pixel measurement to real-world feet.

    Args:
        pixels: Measurement in pixels
        scale_ratio: Pixels per foot (from extraction_pages.scale_ratio)

    Returns:
        Real-world measurement in feet
    """
    if not scale_ratio or float(scale_ratio) == 0:
        return 0.0
    return float(pixels) / float(scale_ratio)


def calculate_area_sqft(pixel_width: float, pixel_height: float, scale_ratio: float) -> float:
    """Calculate area in square feet from pixel dimensions."""
    width_ft = pixels_to_feet(pixel_width, scale_ratio)
    height_ft = pixels_to_feet(pixel_height, scale_ratio)
    return width_ft * height_ft


def calculate_perimeter_lf(pixel_width: float, pixel_height: float, scale_ratio: float) -> float:
    """Calculate perimeter in linear feet from pixel dimensions."""
    width_ft = pixels_to_feet(pixel_width, scale_ratio)
    height_ft = pixels_to_feet(pixel_height, scale_ratio)
    return 2 * (width_ft + height_ft)


def polygon_area_pixels(points: List) -> float:
    """
    Calculate polygon area in pixel space using the shoelace formula.

    Args:
        points: List of points as [{x, y}, ...] or [[x, y], ...]

    Returns:
        Area in square pixels
    """
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        # Handle both {x,y} dict format and [x,y] array format
        if isinstance(points[i], dict):
            xi, yi = float(points[i].get('x', 0)), float(points[i].get('y', 0))
            xj, yj = float(points[j].get('x', 0)), float(points[j].get('y', 0))
        else:
            xi, yi = float(points[i][0]), float(points[i][1])
            xj, yj = float(points[j][0]), float(points[j][1])
        area += xi * yj - xj * yi

    return abs(area) / 2.0


def calculate_polygon_area_sqft(polygon_points: List, scale_ratio: float) -> float:
    """
    Calculate real-world area from polygon pixel coordinates.

    Args:
        polygon_points: List of points as [{x, y}, ...] or {outer: [...], holes: [...]}
        scale_ratio: Pixels per foot

    Returns:
        Area in square feet
    """
    # Handle nested format {outer: [...], holes: [...]}
    if isinstance(polygon_points, dict) and 'outer' in polygon_points:
        points = polygon_points['outer']
    else:
        points = polygon_points

    if not points or not isinstance(points, list):
        return 0.0

    if not scale_ratio or float(scale_ratio) == 0:
        return 0.0

    pixel_area = polygon_area_pixels(points)
    ft_per_pixel = 1.0 / float(scale_ratio)
    return pixel_area * (ft_per_pixel ** 2)


def calculate_polygon_perimeter_lf(polygon_points: List, scale_ratio: float) -> float:
    """
    Calculate perimeter of a polygon in linear feet.

    Args:
        polygon_points: List of points as [{x, y}, ...]
        scale_ratio: Pixels per foot

    Returns:
        Perimeter in linear feet
    """
    # Handle nested format
    if isinstance(polygon_points, dict) and 'outer' in polygon_points:
        points = polygon_points['outer']
    else:
        points = polygon_points

    if not points or not isinstance(points, list) or len(points) < 2:
        return 0.0

    perimeter_pixels = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        if isinstance(points[i], dict):
            xi, yi = float(points[i].get('x', 0)), float(points[i].get('y', 0))
            xj, yj = float(points[j].get('x', 0)), float(points[j].get('y', 0))
        else:
            xi, yi = float(points[i][0]), float(points[i][1])
            xj, yj = float(points[j][0]), float(points[j][1])

        # Distance formula
        perimeter_pixels += ((xj - xi) ** 2 + (yj - yi) ** 2) ** 0.5

    return pixels_to_feet(perimeter_pixels, scale_ratio)


def calculate_measure_conversion(page_data: Dict, pdf_page_width_pts: float) -> float:
    """
    Calculate the feet-per-PDF-point conversion factor for Bluebeam Measure dictionaries.

    PDF uses "points" where 72 points = 1 inch.
    We need to tell Bluebeam how many real-world feet each PDF point represents.

    We know:
    - scale_ratio = pixels per foot (from extraction_pages)
    - original_width = image width in pixels (or image_width)
    - pdf_page_width_pts = PDF page width in points (from fitz page.rect.width)

    Calculation:
    - real_width_ft = original_width / scale_ratio
    - ft_per_pdf_point = real_width_ft / pdf_page_width_pts

    Args:
        page_data: Page data dict with scale_ratio and original_width/image_width
        pdf_page_width_pts: PDF page width in points from fitz

    Returns:
        Feet per PDF point conversion factor
    """
    scale_ratio = float(page_data.get('scale_ratio', 0) or 0)
    # Try multiple field names for original image width
    original_width = float(
        page_data.get('original_width', 0) or
        page_data.get('image_width', 0) or
        page_data.get('width', 0) or 0
    )

    if not scale_ratio or not original_width or not pdf_page_width_pts:
        # Default fallback for 1/4" = 1'-0" scale on 36" wide sheet
        # 1/4" = 1' means 1 inch = 4 feet, so ft_per_inch = 4
        # ft_per_point = 4 / 72 = 0.05556
        print(f"[Bluebeam Export] WARNING: Using default scale conversion (1/4\"=1'-0\")", flush=True)
        return 0.05556

    real_width_ft = original_width / scale_ratio
    ft_per_point = real_width_ft / pdf_page_width_pts

    print(f"[Bluebeam Export] Measure conversion: {original_width}px / {scale_ratio}ppf = {real_width_ft:.2f}ft / {pdf_page_width_pts:.0f}pts = {ft_per_point:.6f} ft/pt", flush=True)

    return ft_per_point


def get_image_dimensions(image_url: str) -> tuple:
    """
    Fetch an image and return its dimensions (width, height).
    Returns (None, None) if unable to fetch or parse.
    """
    if not image_url:
        return None, None

    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            if Image:
                # Use PIL for accurate dimensions
                img = Image.open(io.BytesIO(response.content))
                return img.size  # (width, height)
            elif fitz:
                # Fallback to fitz
                img_doc = fitz.open(stream=response.content, filetype='png')
                if len(img_doc) > 0:
                    rect = img_doc[0].rect
                    img_doc.close()
                    return rect.width, rect.height
    except Exception as e:
        print(f"[Bluebeam Export] Failed to get image dimensions: {e}", flush=True)

    return None, None


# Extended color mapping for Bluebeam annotations
# RGB tuples (0-255 range for PIL, but fitz uses 0-1 range)
BLUEBEAM_COLORS = {
    'building': (0, 120, 255),      # Blue
    'exterior_wall': (0, 120, 255), # Blue
    'exterior wall': (0, 120, 255), # Blue
    'window': (255, 165, 0),        # Orange
    'door': (255, 0, 255),          # Magenta
    'garage': (0, 255, 0),          # Green
    'gable': (255, 255, 0),         # Yellow
    'soffit': (139, 69, 19),        # Brown
    'fascia': (0, 255, 255),        # Cyan
    'corner': (255, 0, 0),          # Red
    'trim': (128, 0, 128),          # Purple
    'roof': (220, 20, 60),          # Crimson
    'gutter': (0, 255, 255),        # Cyan
    'siding': (34, 139, 34),        # Forest Green
    'eave': (100, 149, 237),        # Cornflower Blue
    'rake': (218, 112, 214),        # Orchid
}

# Fallback to config colors if available
for cls_name, color in config.MARKUP_COLORS.items():
    if cls_name not in BLUEBEAM_COLORS:
        BLUEBEAM_COLORS[cls_name] = color


def rgb_to_fitz(rgb_tuple):
    """Convert RGB (0-255) to fitz color (0-1)"""
    return tuple(c / 255.0 for c in rgb_tuple)


def get_detection_color(class_name: str) -> tuple:
    """Get color for a detection class"""
    normalized = class_name.lower().replace(' ', '_')
    return BLUEBEAM_COLORS.get(normalized, (200, 200, 200))  # Gray default


def get_measurement_type(det_class: str) -> tuple:
    """
    Determine the measurement type (unit and intent) based on detection class.

    Returns:
        tuple: (unit, intent) where intent is the PDF annotation intent
    """
    det_class_normalized = det_class.lower().replace(' ', '_')

    if det_class_normalized in AREA_CLASSES:
        return 'SF', '/PolygonDimension'
    elif det_class_normalized in LINEAR_CLASSES:
        return 'LF', '/PolyLineDimension'
    else:
        return 'EA', '/PolygonDimension'


def calculate_detection_measurement(
    detection: Dict[str, Any],
    scale_ratio: float
) -> tuple:
    """
    Calculate the real-world measurement for a detection from pixel coordinates.

    Args:
        detection: Detection dict with pixel_x, pixel_y, pixel_width, pixel_height, polygon_points
        scale_ratio: Pixels per foot (from extraction_pages.scale_ratio)

    Returns:
        tuple: (value, unit, intent) - the calculated measurement
    """
    det_class = detection.get('class', 'unknown')
    # Normalize: lowercase and handle both space and underscore variants
    det_class_normalized = det_class.lower()

    # Get pixel dimensions
    pixel_w = float(detection.get('pixel_width', 0) or 0)
    pixel_h = float(detection.get('pixel_height', 0) or 0)
    polygon_points = detection.get('polygon_points')

    unit, intent = get_measurement_type(det_class)

    # Calculate based on class type
    if det_class_normalized in AREA_CLASSES or det_class_normalized.replace(' ', '_') in AREA_CLASSES:
        # For area classes, prefer polygon area if available
        if polygon_points:
            value = calculate_polygon_area_sqft(polygon_points, scale_ratio)
        else:
            value = calculate_area_sqft(pixel_w, pixel_h, scale_ratio)

    elif det_class_normalized in LINEAR_CLASSES or det_class_normalized.replace(' ', '_') in LINEAR_CLASSES:
        # For linear classes, calculate from polygon perimeter or use height as primary
        if polygon_points:
            value = calculate_polygon_perimeter_lf(polygon_points, scale_ratio)
        else:
            # For linear elements like fascia, trim, use the longer dimension
            value = pixels_to_feet(max(pixel_w, pixel_h), scale_ratio)

    else:
        # Point/count items
        value = 1

    return value, unit, intent


def set_annotation_info_before_update(
    annot,
    det_class: str,
    value: float,
    unit: str,
    material_name: str = ''
):
    """
    Set annotation info using annot.set_info() BEFORE calling update().

    Args:
        annot: The PyMuPDF annotation object
        det_class: Detection class name
        value: Calculated measurement value
        unit: Unit string (SF, LF, EA)
        material_name: Optional material name from product_catalog
    """
    # Format display class name
    display_class = det_class.replace('_', ' ').title()

    # Build content/comments with measurement and material info
    parts = [f"{value:.1f} {unit}"]
    if material_name:
        parts.append(material_name)
    contents = ' | '.join(parts)

    # Use PyMuPDF's built-in method
    annot.set_info(
        title="EstimatePros.ai",  # Shows as Author in Bluebeam
        subject=display_class,    # Shows as Subject in Bluebeam (groups by this)
        content=contents          # Shows as Comments in Bluebeam
    )


def set_annotation_metadata_after_update(
    doc,
    annot,
    det_class: str,
    value: float,
    unit: str,
    intent: str,
    ft_per_point: float = 0.05556
):
    """
    Set PDF annotation xref keys AFTER calling update().

    Args:
        doc: The PyMuPDF document
        annot: The annotation object
        det_class: Detection class name
        value: Calculated measurement value
        unit: Unit string
        intent: PDF annotation intent string
        ft_per_point: Feet per PDF point conversion factor (from calculate_measure_conversion)
    """
    xref = annot.xref
    det_class_normalized = det_class.lower().replace(' ', '_')

    # Set Intent - tells Bluebeam this is a measurement annotation
    doc.xref_set_key(xref, "IT", intent)

    # Calculate conversion factors for Measure dictionary
    # /C value tells Bluebeam: 1 PDF point = C feet
    area_conversion = ft_per_point ** 2  # For square feet

    # Calculate the scale ratio for the /R description string
    # ft_per_point * 72 = feet per inch on paper
    ft_per_inch = ft_per_point * 72
    if ft_per_inch > 0:
        # Express as "1 in = X ft" (e.g., "1 in = 4.00 ft" for 1/4" = 1'-0")
        scale_desc = f"1 in = {ft_per_inch:.2f} ft"
    else:
        scale_desc = "1 ft = 1 ft"

    # Add Measure dictionary for area measurements
    if det_class_normalized in AREA_CLASSES and value > 0:
        measure_dict = (
            f"<< /Type /Measure /Subtype /RL "
            f"/R ({scale_desc}) "
            f"/X [ << /Type /NumberFormat /U (ft) /C {ft_per_point:.8f} /F /D /D 100 /FD false >> ] "
            f"/D [ << /Type /NumberFormat /U (ft) /C {ft_per_point:.8f} /F /D /D 100 /FD false >> ] "
            f"/A [ << /Type /NumberFormat /U (SF) /C {area_conversion:.10f} /F /D /D 100 /FD false >> ] "
            f">>"
        )
        doc.xref_set_key(xref, "Measure", measure_dict)

    # For linear measurements
    elif det_class_normalized in LINEAR_CLASSES and value > 0:
        measure_dict = (
            f"<< /Type /Measure /Subtype /RL "
            f"/R ({scale_desc}) "
            f"/X [ << /Type /NumberFormat /U (ft) /C {ft_per_point:.8f} /F /D /D 100 /FD false >> ] "
            f"/D [ << /Type /NumberFormat /U (LF) /C {ft_per_point:.8f} /F /D /D 100 /FD false >> ] "
            f">>"
        )
        doc.xref_set_key(xref, "Measure", measure_dict)


def build_label_text(
    det_class: str,
    value: float,
    unit: str,
    material_name: str = ''
) -> str:
    """
    Build the visual label text that appears on the PDF drawing.

    Format: "Class | Value Unit | Material" (material only if assigned)
    Examples:
        - "Exterior Wall | 333.96 SF | HardiePlank 8.25 ColorPlus"
        - "Window | 19.0 SF"
        - "Fascia | 45.2 LF | HardieTrim 5/4"
        - "Corbel | 1.0 EA"
    """
    display_class = det_class.replace('_', ' ').title()

    # Build label: Class | Value Unit
    label = f"{display_class} | {value:.1f} {unit}"

    # Add material name if assigned
    if material_name:
        label += f" | {material_name}"

    return label


def lookup_materials(detections: List[Dict], supabase_request_fn) -> Dict[str, Dict]:
    """
    Batch lookup material names from pricing_items table for all detections.

    Note: assigned_material_id references pricing_items, NOT product_catalog.

    Args:
        detections: List of detection dicts with assigned_material_id
        supabase_request_fn: The supabase_request function

    Returns:
        Dict mapping material_id -> material info dict
    """
    # Collect unique material IDs
    assigned_ids = set(
        det.get('assigned_material_id') for det in detections
        if det.get('assigned_material_id')
    )

    if not assigned_ids:
        return {}

    materials_lookup = {}
    for mid in assigned_ids:
        try:
            result = supabase_request_fn('GET', 'pricing_items', filters={'id': f'eq.{mid}'})
            if result and len(result) > 0:
                materials_lookup[mid] = result[0]
        except Exception as e:
            print(f"[Bluebeam Export] Error looking up material {mid}: {e}", flush=True)

    return materials_lookup


def get_material_name(detection: Dict, materials_lookup: Dict) -> str:
    """
    Get the display name for a detection's assigned material.

    Args:
        detection: Detection dict with assigned_material_id
        materials_lookup: Dict from lookup_materials() (pricing_items data)

    Returns:
        Material name string or empty string
    """
    mid = detection.get('assigned_material_id')
    if not mid or mid not in materials_lookup:
        return ''

    mat = materials_lookup[mid]
    # pricing_items uses 'product_name' and 'manufacturer' fields
    name = mat.get('product_name', '')
    manufacturer = mat.get('manufacturer', '')

    if manufacturer and name:
        return f"{manufacturer} - {name}"
    return name or ''


def export_bluebeam_pdf(job_id: str, include_materials: bool = True) -> Dict[str, Any]:
    """
    Export detections to a Bluebeam-compatible annotated PDF.

    Args:
        job_id: The extraction job UUID
        include_materials: Whether to include material assignment labels

    Returns:
        Dict with success status and download_url or error message
    """
    if fitz is None:
        return {
            'success': False,
            'error': 'pymupdf not installed. Run: pip install pymupdf'
        }

    print(f"[Bluebeam Export] Starting export for job {job_id}", flush=True)

    # 1. Get job details
    jobs = supabase_request('GET', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
    if not jobs:
        return {'success': False, 'error': 'Job not found'}

    job = jobs[0]
    project_name = job.get('project_name', 'Untitled')
    source_pdf_url = job.get('source_pdf_url')

    print(f"[Bluebeam Export] Job: {project_name}, PDF URL: {source_pdf_url}", flush=True)

    # 2. Get all pages for this job
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'order': 'page_number.asc'
    })

    if not pages:
        return {'success': False, 'error': 'No pages found for this job'}

    print(f"[Bluebeam Export] Found {len(pages)} pages", flush=True)

    # 3. Try to download the original PDF, or create from page images
    pdf_doc = None
    created_from_images = False

    if source_pdf_url:
        try:
            print(f"[Bluebeam Export] Downloading original PDF...", flush=True)
            response = requests.get(source_pdf_url, timeout=60)
            if response.status_code == 200:
                pdf_doc = fitz.open(stream=response.content, filetype='pdf')
                print(f"[Bluebeam Export] Original PDF loaded: {len(pdf_doc)} pages", flush=True)
        except Exception as e:
            print(f"[Bluebeam Export] Failed to download PDF: {e}", flush=True)

    # If no PDF available, create one from page images
    if pdf_doc is None:
        print(f"[Bluebeam Export] Creating PDF from page images...", flush=True)
        pdf_doc = fitz.open()
        created_from_images = True

        for page_data in pages:
            image_url = page_data.get('original_image_url') or page_data.get('image_url')
            if not image_url:
                continue

            try:
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    img = fitz.open(stream=img_response.content, filetype='png')
                    # Create a page with the image dimensions
                    rect = img[0].rect
                    pdf_page = pdf_doc.new_page(width=rect.width, height=rect.height)
                    pdf_page.insert_image(rect, stream=img_response.content)
                    img.close()
            except Exception as e:
                print(f"[Bluebeam Export] Error loading image for page {page_data.get('page_number')}: {e}", flush=True)

        print(f"[Bluebeam Export] Created PDF with {len(pdf_doc)} pages from images", flush=True)

    if len(pdf_doc) == 0:
        return {'success': False, 'error': 'Could not create PDF - no pages or images available'}

    # 4. Add annotations for each page
    total_annotations = 0

    for page_idx, page_data in enumerate(pages):
        page_id = page_data.get('id')
        page_number = page_data.get('page_number', page_idx + 1)

        # Try to get image dimensions from multiple sources
        image_width = page_data.get('image_width') or page_data.get('width')
        image_height = page_data.get('image_height') or page_data.get('height')

        # Get PDF page (handle page number vs index)
        pdf_page_idx = page_number - 1 if page_number else page_idx
        if pdf_page_idx >= len(pdf_doc):
            pdf_page_idx = page_idx
        if pdf_page_idx >= len(pdf_doc):
            continue

        pdf_page = pdf_doc[pdf_page_idx]
        pdf_rect = pdf_page.rect

        # If we created PDF from images, dimensions already match (scale = 1.0)
        if created_from_images:
            scale_x = 1.0
            scale_y = 1.0
        elif image_width and image_height:
            # Have image dimensions from database - calculate scale
            scale_x = pdf_rect.width / image_width
            scale_y = pdf_rect.height / image_height
        else:
            # Try to fetch image dimensions from the stored image
            image_url = page_data.get('original_image_url') or page_data.get('image_url')
            fetched_width, fetched_height = get_image_dimensions(image_url)

            if fetched_width and fetched_height:
                image_width = fetched_width
                image_height = fetched_height
                scale_x = pdf_rect.width / image_width
                scale_y = pdf_rect.height / image_height
                print(f"[Bluebeam Export] Fetched image dimensions: {image_width}x{image_height}", flush=True)
            else:
                # Last resort: assume common rendering dimensions (2048px wide, aspect ratio from PDF)
                # This is a fallback - the aspect ratio should match the PDF
                print(f"[Bluebeam Export] WARNING: Could not get image dimensions, using PDF dimensions", flush=True)
                # Use PDF dimensions directly (scale = 1.0) - annotations may not align perfectly
                scale_x = 1.0
                scale_y = 1.0

        print(f"[Bluebeam Export] Processing page {page_number}, PDF: {pdf_rect.width:.0f}x{pdf_rect.height:.0f}, Image: {image_width}x{image_height}, scale: {scale_x:.3f}x{scale_y:.3f}", flush=True)

        # Get page scale info for real-world measurement calculations
        # scale_ratio is pixels per foot (already accounts for image resolution)
        page_scale_ratio = float(page_data.get('scale_ratio', 0) or 0)

        print(f"[Bluebeam Export] Page scale: ratio={page_scale_ratio} pixels/ft", flush=True)

        # Calculate conversion factor for Bluebeam Measure dictionary
        # This tells Bluebeam how many feet each PDF point represents
        ft_per_point = calculate_measure_conversion(page_data, pdf_rect.width)

        # Get detections for this page - PRIORITIZE DRAFTS (user edits)
        # First check extraction_detections_draft (where user edits are saved)
        detections = supabase_request('GET', 'extraction_detections_draft', filters={
            'page_id': f'eq.{page_id}',
            'is_deleted': 'eq.false',
            'order': 'class,detection_index'
        })

        if detections:
            print(f"[Bluebeam Export] Found {len(detections)} DRAFT detections (user-edited)", flush=True)
        else:
            # No drafts - fall back to original AI detections
            print(f"[Bluebeam Export] No drafts found, loading original detections", flush=True)
            detections = supabase_request('GET', 'extraction_detection_details', filters={
                'page_id': f'eq.{page_id}',
                'status': 'neq.deleted',
                'order': 'class,detection_index'
            })

            if not detections:
                # Try without status filter (older schema)
                detections = supabase_request('GET', 'extraction_detection_details', filters={
                    'page_id': f'eq.{page_id}',
                    'order': 'class'
                })

        if not detections:
            print(f"[Bluebeam Export] No detections for page {page_number}", flush=True)
            continue

        # Batch lookup materials for this page's detections
        materials_lookup = lookup_materials(detections, supabase_request) if include_materials else {}
        if materials_lookup:
            print(f"[Bluebeam Export] Loaded {len(materials_lookup)} material assignments", flush=True)

        print(f"[Bluebeam Export] Adding {len(detections)} annotations to page {page_number}", flush=True)

        # Add annotations for each detection
        for det in detections:
            cls = det.get('class', 'unknown')

            # Skip helper classes that are used for detection only, not estimation
            if cls.lower() in SKIP_CLASSES:
                continue

            color_rgb = get_detection_color(cls)
            color_fitz = rgb_to_fitz(color_rgb)

            # Check for polygon points (user-drawn polygons)
            polygon_points = det.get('polygon_points')
            markup_type = det.get('markup_type', 'polygon')

            # Get pixel coordinates (center-based for bounding box)
            px = float(det.get('pixel_x', 0))
            py = float(det.get('pixel_y', 0))
            pw = float(det.get('pixel_width', 0))
            ph = float(det.get('pixel_height', 0))

            # Calculate bounding box corners (used for label positioning and fallback)
            x1 = (px - pw / 2) * scale_x
            y1 = (py - ph / 2) * scale_y
            x2 = (px + pw / 2) * scale_x
            y2 = (py + ph / 2) * scale_y

            try:
                annot = None

                # Handle different shape types
                # IMPORTANT: Always use polygon annotations (not rect) so Bluebeam
                # recognizes them as measurement markups with SF/LF values
                if polygon_points and isinstance(polygon_points, (list, dict)):
                    # User-drawn polygon - extract outer points
                    # polygon_points can be [{x, y}, ...] or {outer: [...], holes: [...]}
                    points_list = polygon_points
                    if isinstance(polygon_points, dict) and 'outer' in polygon_points:
                        points_list = polygon_points['outer']

                    if isinstance(points_list, list) and len(points_list) >= 3:
                        # Convert polygon points to PDF coordinates
                        pdf_points = []
                        for pt in points_list:
                            if isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                                pdf_x = float(pt['x']) * scale_x
                                pdf_y = float(pt['y']) * scale_y
                                pdf_points.append(fitz.Point(pdf_x, pdf_y))

                        if len(pdf_points) >= 3:
                            # Add polygon annotation
                            annot = pdf_page.add_polygon_annot(pdf_points)
                            annot.set_colors(stroke=color_fitz)
                            annot.set_border(width=2)
                            annot.set_opacity(0.8)
                        else:
                            # Fallback to bbox as polygon (not rect!)
                            rect_as_polygon = [
                                fitz.Point(x1, y1),
                                fitz.Point(x2, y1),
                                fitz.Point(x2, y2),
                                fitz.Point(x1, y2)
                            ]
                            annot = pdf_page.add_polygon_annot(rect_as_polygon)
                            annot.set_colors(stroke=color_fitz)
                            annot.set_border(width=2)
                            annot.set_opacity(0.8)
                    else:
                        # Invalid polygon, use bbox as polygon
                        rect_as_polygon = [
                            fitz.Point(x1, y1),
                            fitz.Point(x2, y1),
                            fitz.Point(x2, y2),
                            fitz.Point(x1, y2)
                        ]
                        annot = pdf_page.add_polygon_annot(rect_as_polygon)
                        annot.set_colors(stroke=color_fitz)
                        annot.set_border(width=2)
                        annot.set_opacity(0.8)

                elif markup_type == 'line':
                    # Line markup - draw as line from top-left to bottom-right of bbox
                    line_start = fitz.Point(x1, y1)
                    line_end = fitz.Point(x2, y2)
                    annot = pdf_page.add_line_annot(line_start, line_end)
                    annot.set_colors(stroke=color_fitz)
                    annot.set_border(width=2)
                    annot.set_opacity(0.8)

                elif markup_type == 'point':
                    # Point markup - draw as small circle at center
                    center_x = px * scale_x
                    center_y = py * scale_y
                    radius = 8  # Small circle
                    point_rect = fitz.Rect(center_x - radius, center_y - radius,
                                          center_x + radius, center_y + radius)
                    annot = pdf_page.add_circle_annot(point_rect)
                    annot.set_colors(stroke=color_fitz, fill=color_fitz)
                    annot.set_border(width=2)
                    annot.set_opacity(0.8)

                else:
                    # Default: Use polygon annotation (not rect!) so Bluebeam reads as area
                    rect_as_polygon = [
                        fitz.Point(x1, y1),
                        fitz.Point(x2, y1),
                        fitz.Point(x2, y2),
                        fitz.Point(x1, y2)
                    ]
                    annot = pdf_page.add_polygon_annot(rect_as_polygon)
                    annot.set_colors(stroke=color_fitz)
                    annot.set_border(width=2)
                    annot.set_opacity(0.8)

                # Calculate real-world measurement for this detection
                meas_value, meas_unit, meas_intent = calculate_detection_measurement(
                    det, page_scale_ratio
                )

                # Get material name if assigned
                material_name = get_material_name(det, materials_lookup) if include_materials else ''

                # Apply metadata and finalize annotation
                if annot:
                    # Step 1: Set info BEFORE update (subject, title, content)
                    set_annotation_info_before_update(
                        annot, cls, meas_value, meas_unit, material_name
                    )

                    # Step 2: Call update to apply changes
                    annot.update()

                    # Step 3: Set xref keys AFTER update (IT, Measure)
                    set_annotation_metadata_after_update(
                        pdf_doc, annot, cls, meas_value, meas_unit, meas_intent, ft_per_point
                    )

                    total_annotations += 1

                # Build visual label text with measurement and material
                label_text = build_label_text(cls, meas_value, meas_unit, material_name)

                # Add label as page content text (NOT an annotation)
                # This way it appears on the drawing but NOT in the Markup List
                label_point = fitz.Point(x1 + 2, y1 - 4)
                pdf_page.insert_text(
                    label_point,
                    label_text,
                    fontsize=7,
                    fontname="helv",
                    color=color_fitz
                )

            except Exception as e:
                print(f"[Bluebeam Export] Error adding annotation: {e}", flush=True)

        # Add page summary legend
        try:
            _add_page_legend(pdf_page, detections, page_scale_ratio)
        except Exception as e:
            print(f"[Bluebeam Export] Error adding legend: {e}", flush=True)

    print(f"[Bluebeam Export] Added {total_annotations} total annotations", flush=True)

    # Debug: Check what annotations are actually in the PDF
    print(f"[Bluebeam Export] === ANNOTATION DEBUG (first page) ===", flush=True)
    if len(pdf_doc) > 0:
        debug_page = pdf_doc[0]
        annot_count = 0
        for annot in debug_page.annots():
            annot_count += 1
            annot_type = annot.type  # Returns (type_int, type_name)
            xref = annot.xref

            # Read back the keys we set
            try:
                subj = pdf_doc.xref_get_key(xref, "Subj")
                it = pdf_doc.xref_get_key(xref, "IT")
                measure = pdf_doc.xref_get_key(xref, "Measure")
                contents = pdf_doc.xref_get_key(xref, "Contents")
                title = pdf_doc.xref_get_key(xref, "T")
                print(f"[DEBUG] Annot #{annot_count} | Type: {annot_type} | Subj: {subj} | IT: {it} | T: {title}", flush=True)
                print(f"[DEBUG]   Contents: {contents[:100] if contents else 'None'}...", flush=True)
                print(f"[DEBUG]   Measure: {'SET' if measure and measure[0] != 'null' else 'NOT SET'}", flush=True)
            except Exception as e:
                print(f"[DEBUG] Annot #{annot_count} | Type: {annot_type} | Error reading keys: {e}", flush=True)

            if annot_count >= 5:  # Only debug first 5 annotations
                print(f"[DEBUG] ... (showing first 5 of {len(list(debug_page.annots()))} annotations)", flush=True)
                break
    print(f"[Bluebeam Export] === END DEBUG ===", flush=True)

    # 5. Save to bytes
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    # 6. Upload to Supabase storage
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = project_name.replace(' ', '_').replace('/', '-')[:50]
    filename = f"exports/{job_id}/bluebeam_{safe_name}_{timestamp}.pdf"

    print(f"[Bluebeam Export] Uploading to storage: {filename}", flush=True)

    download_url = upload_to_storage(
        pdf_bytes,
        filename,
        content_type='application/pdf',
        bucket='extraction-markups'
    )

    if not download_url:
        return {'success': False, 'error': 'Failed to upload PDF to storage'}

    print(f"[Bluebeam Export] Export complete: {download_url}", flush=True)

    return {
        'success': True,
        'download_url': download_url,
        'filename': f"bluebeam_{safe_name}_{timestamp}.pdf",
        'pages': len(pages),
        'annotations': total_annotations,
        'created_from_images': created_from_images
    }


def _add_page_legend(pdf_page, detections: List[Dict], scale_ratio: float):
    """Add a summary legend to the page corner with calculated measurements."""

    # Count detections by class and aggregate measurements
    class_counts = {}
    class_measurements = {}  # Stores (total_value, unit) per class

    for det in detections:
        cls = det.get('class', 'unknown')

        # Skip helper classes that are used for detection only, not estimation
        if cls.lower() in SKIP_CLASSES:
            continue

        value, unit, _ = calculate_detection_measurement(det, scale_ratio)

        if cls not in class_counts:
            class_counts[cls] = 0
            class_measurements[cls] = {'value': 0, 'unit': unit}
        class_counts[cls] += 1
        class_measurements[cls]['value'] += value

    # Build legend text
    legend_lines = ["DETECTION SUMMARY"]
    legend_lines.append("-" * 30)

    for cls, count in sorted(class_counts.items()):
        measurement = class_measurements.get(cls, {'value': 0, 'unit': 'EA'})
        value = measurement['value']
        unit = measurement['unit']
        legend_lines.append(f"{cls.title()}: {count} ({value:.1f} {unit})")

    legend_lines.append("-" * 30)
    total_count = sum(class_counts.values())

    # Calculate total area (only for area classes)
    total_area = sum(
        class_measurements[cls]['value']
        for cls in class_counts
        if cls.lower().replace(' ', '_') in AREA_CLASSES
    )
    # Calculate total linear (only for linear classes)
    total_linear = sum(
        class_measurements[cls]['value']
        for cls in class_counts
        if cls.lower().replace(' ', '_') in LINEAR_CLASSES
    )

    legend_lines.append(f"Total: {total_count} detections")
    if total_area > 0:
        legend_lines.append(f"Total Area: {total_area:.1f} SF")
    if total_linear > 0:
        legend_lines.append(f"Total Linear: {total_linear:.1f} LF")

    # Position in top-left corner
    page_rect = pdf_page.rect
    legend_rect = fitz.Rect(10, 10, 250, 10 + len(legend_lines) * 14)

    # Add white background rectangle (page content, not annotation)
    shape = pdf_page.new_shape()
    shape.draw_rect(legend_rect)
    shape.finish(fill=(1, 1, 1), color=(0, 0, 0), width=1)
    shape.commit()

    # Add legend text as page content (NOT annotation - won't appear in Markup List)
    y_pos = 22
    for line in legend_lines:
        pdf_page.insert_text(
            fitz.Point(15, y_pos),
            line,
            fontsize=9,
            fontname="cour",  # Courier for aligned text
            color=(0, 0, 0)
        )
        y_pos += 12
