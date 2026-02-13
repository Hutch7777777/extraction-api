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
AREA_CLASSES = {
    'building', 'exterior_wall', 'exterior wall', 'gable', 'soffit',
    'door', 'window', 'garage_door', 'garage'
}
LINEAR_CLASSES = {
    'fascia', 'trim', 'corner', 'inside_corner', 'outside_corner',
    'belly_band', 'rake', 'frieze', 'flashing', 'eave'
}
POINT_CLASSES = {
    'hose_bib', 'vent', 'light_fixture', 'outlet', 'corbel'
}


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


def get_measurement_value_and_unit(detection: Dict[str, Any]) -> tuple:
    """
    Determine the measurement value and unit based on detection class.

    Returns:
        tuple: (value, unit, intent) where intent is the PDF annotation intent
    """
    det_class = detection.get('class', 'unknown').lower().replace(' ', '_')
    dimensions = detection.get('dimensions', {}) or {}

    # Also check top-level fields for backwards compatibility
    area_sqft = dimensions.get('area_sqft') or detection.get('area_sf', 0)
    perimeter_lf = dimensions.get('perimeter_lf', 0)
    length_lf = dimensions.get('length_lf', 0)
    height_ft = dimensions.get('height_ft') or detection.get('real_height_ft', 0)

    if det_class in AREA_CLASSES:
        value = float(area_sqft or 0)
        unit = 'SF'
        intent = '/PolygonDimension'
    elif det_class in LINEAR_CLASSES:
        # Prefer perimeter, then length, then height
        value = float(perimeter_lf or length_lf or height_ft or 0)
        unit = 'LF'
        intent = '/PolyLineDimension'
    else:
        # Point/count items
        value = 1
        unit = 'EA'
        intent = '/PolygonDimension'

    return value, unit, intent


def set_annotation_metadata(doc, annot, detection: Dict[str, Any]):
    """
    Set PDF annotation metadata for Bluebeam measurement display.

    This modifies the PDF dictionary using xref keys so Bluebeam's
    Markup List shows useful estimation data instead of "1 Count".
    """
    xref = annot.xref
    det_class = detection.get('class', 'unknown')
    material = detection.get('material_assignment', {}) or {}

    # Get measurement value and unit
    value, unit, intent = get_measurement_value_and_unit(detection)

    # Format display class name
    display_class = det_class.replace('_', ' ').title()

    # Build Subject - Bluebeam groups by this
    doc.xref_set_key(xref, "Subj", f"({display_class})")

    # Set Intent - tells Bluebeam this is a measurement annotation
    doc.xref_set_key(xref, "IT", intent)

    # Build Contents/Comments with measurement and material info
    parts = [f"{value:.1f} {unit}"]
    if material.get('product_name'):
        parts.append(material['product_name'])
    if material.get('manufacturer'):
        parts.append(f"Mfr: {material['manufacturer']}")
    contents = ' | '.join(parts)
    doc.xref_set_key(xref, "Contents", f"({contents})")

    # Set Author for branding
    doc.xref_set_key(xref, "T", "(EstimatePros.ai)")

    # Add Measure dictionary for area measurements
    det_class_normalized = det_class.lower().replace(' ', '_')

    if det_class_normalized in AREA_CLASSES and value > 0:
        measure_dict = (
            "<<"
            "/Type /Measure "
            "/Subtype /RL "
            "/R (1 in = 1 ft) "
            "/X [<< /Type /NumberFormat /U (ft) /C 1.0 /F /D /D 100 >>] "
            "/A [<< /Type /NumberFormat /U (SF) /C 1.0 /F /D /D 100 >>] "
            "/D [<< /Type /NumberFormat /U (ft) /C 1.0 /F /D /D 100 >>] "
            ">>"
        )
        doc.xref_set_key(xref, "Measure", measure_dict)

    # For linear measurements
    elif det_class_normalized in LINEAR_CLASSES and value > 0:
        measure_dict = (
            "<<"
            "/Type /Measure "
            "/Subtype /RL "
            "/R (1 in = 1 ft) "
            "/X [<< /Type /NumberFormat /U (ft) /C 1.0 /F /D /D 100 >>] "
            "/D [<< /Type /NumberFormat /U (LF) /C 1.0 /F /D /D 100 >>] "
            ">>"
        )
        doc.xref_set_key(xref, "Measure", measure_dict)


def build_label_text(detection: Dict[str, Any]) -> str:
    """
    Build the visual label text that appears on the PDF drawing.

    Format: "Class | Value Unit | Material" (material only if assigned)
    Examples:
        - "Exterior Wall | 333.96 SF | HardiePlank 8.25 ColorPlus"
        - "Window | 19 SF"
        - "Fascia | 45.2 LF | HardieTrim 5/4"
        - "Corbel | 1 EA"
    """
    det_class = detection.get('class', 'unknown')
    display_class = det_class.replace('_', ' ').title()
    material = detection.get('material_assignment', {}) or {}

    # Get measurement value and unit
    value, unit, _ = get_measurement_value_and_unit(detection)

    # Build label: Class | Value Unit
    label = f"{display_class} | {value:.1f} {unit}"

    # Add material name if assigned (keep concise - no SKU/pricing/manufacturer)
    if material.get('product_name'):
        label += f" | {material['product_name']}"

    return label


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

        print(f"[Bluebeam Export] Adding {len(detections)} annotations to page {page_number}", flush=True)

        # Add annotations for each detection
        for det in detections:
            cls = det.get('class', 'unknown')
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
                # Handle different shape types
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
                            annot.update()  # MUST call update() BEFORE modifying xref keys
                            set_annotation_metadata(pdf_doc, annot, det)
                            total_annotations += 1
                        else:
                            # Fallback to rectangle if polygon invalid
                            rect = fitz.Rect(x1, y1, x2, y2)
                            annot = pdf_page.add_rect_annot(rect)
                            annot.set_colors(stroke=color_fitz)
                            annot.set_border(width=2)
                            annot.set_opacity(0.8)
                            annot.update()
                            set_annotation_metadata(pdf_doc, annot, det)
                            total_annotations += 1
                    else:
                        # Invalid polygon, use rectangle
                        rect = fitz.Rect(x1, y1, x2, y2)
                        annot = pdf_page.add_rect_annot(rect)
                        annot.set_colors(stroke=color_fitz)
                        annot.set_border(width=2)
                        annot.set_opacity(0.8)
                        annot.update()
                        set_annotation_metadata(pdf_doc, annot, det)
                        total_annotations += 1

                elif markup_type == 'line':
                    # Line markup - draw as line from top-left to bottom-right of bbox
                    line_start = fitz.Point(x1, y1)
                    line_end = fitz.Point(x2, y2)
                    annot = pdf_page.add_line_annot(line_start, line_end)
                    annot.set_colors(stroke=color_fitz)
                    annot.set_border(width=2)
                    annot.set_opacity(0.8)
                    annot.update()
                    set_annotation_metadata(pdf_doc, annot, det)
                    total_annotations += 1

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
                    annot.update()
                    set_annotation_metadata(pdf_doc, annot, det)
                    total_annotations += 1

                else:
                    # Default: rectangle annotation (Bluebeam reads these as markups)
                    rect = fitz.Rect(x1, y1, x2, y2)
                    annot = pdf_page.add_rect_annot(rect)
                    annot.set_colors(stroke=color_fitz)
                    annot.set_border(width=2)
                    annot.set_opacity(0.8)
                    annot.update()
                    set_annotation_metadata(pdf_doc, annot, det)
                    total_annotations += 1

                # Build visual label text with measurement and material
                label_text = build_label_text(det) if include_materials else build_label_text({
                    **det,
                    'material_assignment': None
                })

                # Position label at top-left of detection
                # Adjust width based on label length
                label_width = min(max(len(label_text) * 6, 150), 350)
                label_rect = fitz.Rect(x1, y1 - 18, x1 + label_width, y1)

                # Add freetext annotation for label
                text_annot = pdf_page.add_freetext_annot(
                    label_rect,
                    label_text,
                    fontsize=8,
                    fontname="helv",
                    text_color=color_fitz,
                    fill_color=(1, 1, 1),  # White background
                    border_color=color_fitz
                )
                text_annot.set_opacity(0.9)
                text_annot.update()
                total_annotations += 1

            except Exception as e:
                print(f"[Bluebeam Export] Error adding annotation: {e}", flush=True)

        # Add page summary legend
        try:
            _add_page_legend(pdf_page, detections, include_materials)
        except Exception as e:
            print(f"[Bluebeam Export] Error adding legend: {e}", flush=True)

    print(f"[Bluebeam Export] Added {total_annotations} total annotations", flush=True)

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


def _add_page_legend(pdf_page, detections: List[Dict], include_materials: bool):
    """Add a summary legend to the page corner"""

    # Count detections by class and aggregate measurements
    class_counts = {}
    class_measurements = {}  # Stores (total_value, unit) per class

    for det in detections:
        cls = det.get('class', 'unknown')
        value, unit, _ = get_measurement_value_and_unit(det)

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

    legend_text = "\n".join(legend_lines)

    # Position in top-left corner
    page_rect = pdf_page.rect
    legend_rect = fitz.Rect(10, 10, 250, 10 + len(legend_lines) * 14)

    # Add white background rectangle
    shape = pdf_page.new_shape()
    shape.draw_rect(legend_rect)
    shape.finish(fill=(1, 1, 1), color=(0, 0, 0), width=1)
    shape.commit()

    # Add text annotation
    text_annot = pdf_page.add_freetext_annot(
        legend_rect,
        legend_text,
        fontsize=9,
        fontname="cour",  # Courier for aligned text
        text_color=(0, 0, 0),
        fill_color=(1, 1, 1, 0.95)
    )
    text_annot.update()
