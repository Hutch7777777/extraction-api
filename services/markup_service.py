"""
Markup generation service - Enhanced with shape-aware rendering

Changes in this version:
- Gables render as triangles (not rectangles)
- Roofs render with diagonal hatching (to show "excluded" area)
- Gable labels include rake LF
- Legend shows shape indicators
"""

import io
import math
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

from config import config
from database import (
    get_page, update_page, get_elevation_pages,
    upload_to_storage, supabase_request
)


# ============================================================
# SHAPE DRAWING HELPERS
# ============================================================

def draw_detection_shape(draw, x1, y1, x2, y2, class_name, outline_color, 
                         fill_color=None, width=3, draw_hatching=True):
    """
    Draw appropriate shape for detection class.
    
    - Gables: Triangle (apex at top center, base at bottom)
    - Roof: Rectangle with diagonal hatching (indicates excluded area)
    - Everything else: Rectangle
    
    Args:
        draw: PIL ImageDraw object
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
        class_name: Detection class (e.g., 'gable', 'roof', 'window')
        outline_color: Color for outline (RGB tuple or hex string)
        fill_color: Color for fill (RGBA tuple for transparency) or None
        width: Line width for outline
        draw_hatching: Whether to draw hatching on roofs
    
    Returns:
        dict with shape info for labels (e.g., {'type': 'triangle', 'apex': (x, y)})
    """
    class_lower = class_name.lower() if class_name else ''
    
    if class_lower == 'gable':
        # Triangle: apex at top center, base along bottom
        apex = ((x1 + x2) / 2, y1)  # Top center
        bottom_left = (x1, y2)
        bottom_right = (x2, y2)
        points = [apex, bottom_left, bottom_right]
        
        if fill_color:
            draw.polygon(points, fill=fill_color, outline=outline_color, width=width)
        else:
            draw.polygon(points, outline=outline_color, width=width)
        
        return {
            'type': 'triangle',
            'apex': apex,
            'bottom_left': bottom_left,
            'bottom_right': bottom_right
        }
    
    elif class_lower == 'roof':
        # Rectangle with crosshatch to indicate "excluded from siding"
        if fill_color:
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=width)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=width)
        
        # Add diagonal lines for visual distinction (hatching)
        if draw_hatching:
            spacing = 15
            # Ensure we're working with ints for range
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            height = y2_int - y1_int
            
            # Draw diagonal lines from top-left to bottom-right direction
            for i in range(x1_int - height, x2_int + 1, spacing):
                # Line from (i, y1) going down-right
                start_x = max(i, x1_int)
                start_y = y1_int + (start_x - i)
                end_x = min(i + height, x2_int)
                end_y = y1_int + (end_x - i)
                
                if start_y <= y2_int and end_y >= y1_int:
                    start_y = max(start_y, y1_int)
                    end_y = min(end_y, y2_int)
                    if start_x < end_x:
                        draw.line([(start_x, start_y), (end_x, end_y)], 
                                  fill=outline_color, width=1)
        
        return {'type': 'rectangle_hatched'}
    
    else:
        # Standard rectangle for all other classes
        if fill_color:
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=width)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=width)
        
        return {'type': 'rectangle'}


def calculate_gable_rake_lf(width_ft, height_ft):
    """
    Calculate total rake length for a gable (both sloped edges).
    
    Formula: 2 × sqrt((width/2)² + height²)
    
    Args:
        width_ft: Gable width in feet
        height_ft: Gable height in feet
    
    Returns:
        Total rake length in linear feet
    """
    half_width = width_ft / 2
    rake_per_side = math.sqrt(half_width ** 2 + height_ft ** 2)
    return rake_per_side * 2


# ============================================================
# LEGEND HELPERS
# ============================================================

# Shape indicators for legend
SHAPE_INDICATORS = {
    'building': '■',
    'exterior_wall': '■',
    'window': '■',
    'door': '■',
    'garage': '■',
    'roof': '▤',      # Hatched (excluded)
    'gable': '△',     # Triangle (added to siding)
}


def get_shape_indicator(class_name):
    """Get the shape indicator symbol for a class."""
    return SHAPE_INDICATORS.get(class_name.lower(), '■')


# ============================================================
# BASIC MARKUP GENERATION
# ============================================================

def generate_markup_image(image_data, predictions, scale_ratio, dpi=None,
                         trade_filter=None, show_dimensions=True, show_labels=True):
    """
    Generate marked-up image with bounding boxes and measurements.
    
    Args:
        image_data: Raw image bytes
        predictions: List of Roboflow predictions
        scale_ratio: Drawing scale ratio
        dpi: Image DPI
        trade_filter: List of classes to include
        show_dimensions: Show dimension labels
        show_labels: Show class labels
    
    Returns:
        Tuple of (PIL Image, totals dict)
    """
    if dpi is None:
        dpi = config.DEFAULT_DPI
    
    img = Image.open(BytesIO(image_data)).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    if not scale_ratio or scale_ratio <= 0:
        scale_ratio = 48
    
    inches_per_pixel = (1.0 / dpi) * scale_ratio
    
    # Filter predictions if trade_filter specified
    if trade_filter:
        filtered_preds = [
            p for p in predictions 
            if p.get('class', '').lower() in trade_filter
        ]
    else:
        filtered_preds = predictions
    
    totals = {}
    
    for pred in filtered_preds:
        class_name = pred.get('class', '').lower().replace(' ', '_')
        x = pred.get('x', 0)
        y = pred.get('y', 0)
        width = pred.get('width', 0)
        height = pred.get('height', 0)
        
        # Calculate bounding box
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        
        # Get color
        color = config.MARKUP_COLORS.get(class_name, (128, 128, 128))
        
        # Calculate real measurements
        real_width_in = width * inches_per_pixel
        real_height_in = height * inches_per_pixel
        real_width_ft = real_width_in / 12
        real_height_ft = real_height_in / 12
        
        # Calculate area (gables are triangles)
        if class_name == 'gable':
            area_sqft = (real_width_in * real_height_in) / 144 / 2
        else:
            area_sqft = (real_width_in * real_height_in) / 144
        
        # Track totals
        if class_name not in totals:
            totals[class_name] = {'count': 0, 'area_sqft': 0, 'rake_lf': 0}
        totals[class_name]['count'] += 1
        totals[class_name]['area_sqft'] += area_sqft
        
        # Calculate rake for gables
        if class_name == 'gable':
            rake_lf = calculate_gable_rake_lf(real_width_ft, real_height_ft)
            totals[class_name]['rake_lf'] += rake_lf
        
        # Draw shape (triangle for gable, hatched rect for roof, rect for others)
        draw_detection_shape(draw, x1, y1, x2, y2, class_name, color, 
                            fill_color=None, width=3, draw_hatching=True)
        
        # Draw label
        if show_labels and class_name in ['window', 'door', 'garage', 'building', 'roof', 'gable']:
            label = class_name.upper()[:3]
            if show_dimensions:
                if class_name == 'gable':
                    # Show area and rake for gables
                    rake_lf = calculate_gable_rake_lf(real_width_ft, real_height_ft)
                    label += f" {area_sqft:.0f}SF R:{rake_lf:.1f}'"
                else:
                    label += f" {real_width_ft:.1f}x{real_height_ft:.1f}"
            
            # Position label - for gables, place below apex
            if class_name == 'gable':
                label_x = x1 + (x2 - x1) // 4
                label_y = y1 + (y2 - y1) // 3
            else:
                label_x = x1 + 2
                label_y = y1 + 2
            
            try:
                draw.text((label_x, label_y), label, fill=color, font=small_font)
            except:
                draw.text((label_x, label_y), label, fill=color)
    
    # Draw legend with shape indicators
    legend_y = 10
    try:
        legend_height = 25 + len(totals) * 20
        draw.rectangle([5, 5, 280, legend_height], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((10, legend_y), "LEGEND", fill=(0, 0, 0), font=font)
        legend_y += 20
        
        for class_name, data in totals.items():
            color = config.MARKUP_COLORS.get(class_name, (128, 128, 128))
            shape = get_shape_indicator(class_name)
            
            # Draw color swatch
            draw.rectangle([10, legend_y, 25, legend_y + 12], fill=color)
            
            # Build label text
            if class_name == 'gable':
                label_text = f"{shape} {class_name}: {data['count']} ({data['area_sqft']:.0f}SF, {data['rake_lf']:.1f}' rake)"
            else:
                label_text = f"{shape} {class_name}: {data['count']} ({data['area_sqft']:.0f}SF)"
            
            draw.text((30, legend_y), label_text, fill=(0, 0, 0), font=small_font)
            legend_y += 18
    except:
        pass
    
    return img, totals


def generate_markups_for_page(page_id, trades=None):
    """
    Generate markup images for a single page.
    
    Args:
        page_id: UUID of the extraction_page
        trades: List of trades to generate ['all', 'siding', 'roofing', 'windows', 'doors', 'gutters']
    
    Returns:
        Dict with markup URLs
    """
    if trades is None:
        trades = ['all']
    
    # Get page data
    page = get_page(page_id)
    if not page:
        return {"error": "Page not found"}
    
    image_url = page.get('image_url')
    extraction_data = page.get('extraction_data', {})
    predictions = extraction_data.get('raw_predictions', [])
    scale_ratio = float(page.get('scale_ratio') or 48)
    dpi = int(page.get('dpi') or config.DEFAULT_DPI)
    job_id = page.get('job_id')
    page_num = page.get('page_number')
    
    if not predictions:
        return {"error": "No detection data for this page"}
    
    # Download original image
    try:
        response = requests.get(image_url, timeout=30)
        image_data = response.content
    except Exception as e:
        return {"error": f"Failed to download image: {e}"}
    
    markup_urls = {}
    errors = []
    
    for trade in trades:
        try:
            trade_filter = config.TRADE_GROUPS.get(trade, config.TRADE_GROUPS['all'])
            print(f"Generating {trade} markup, filter={trade_filter}, preds={len(predictions)}", flush=True)
            
            # Generate markup
            marked_img, totals = generate_markup_image(
                image_data, predictions, scale_ratio, dpi,
                trade_filter=trade_filter, show_dimensions=True, show_labels=True
            )
            
            # Save to buffer
            buffer = BytesIO()
            marked_img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            
            # Upload to Supabase
            filename = f"{job_id}/markup_{page_num:03d}_{trade}.png"
            markup_url = upload_to_storage(buffer.getvalue(), filename, 'image/png')
            
            if markup_url:
                markup_urls[trade] = {
                    'url': markup_url,
                    'totals': totals
                }
            else:
                errors.append(f"Upload failed for {trade}")
        
        except Exception as e:
            import traceback
            print(f"Markup error: {traceback.format_exc()}", flush=True)
            errors.append(f"{trade}: {str(e)}")
    
    if errors:
        print(f"Markup errors: {errors}", flush=True)
    
    # Update page with markup URLs
    update_page(page_id, {'markup_urls': markup_urls})
    
    return {"success": True, "page_id": page_id, "markups": markup_urls}


def generate_markups_for_job(job_id, trades=None):
    """
    Generate markups for all elevation pages in a job.
    
    Args:
        job_id: Job UUID
        trades: List of trades
    
    Returns:
        Dict with results
    """
    if trades is None:
        trades = ['all', 'siding', 'roofing']
    
    print(f"[{job_id}] Generating markups...", flush=True)
    
    # Get all elevation pages with extraction data
    pages = get_elevation_pages(job_id, status='complete')
    
    if not pages:
        return {"error": "No elevation pages found"}
    
    results = []
    for page in pages:
        page_id = page.get('id')
        result = generate_markups_for_page(page_id, trades)
        results.append(result)
    
    successful = len([r for r in results if r.get('success')])
    
    return {
        "success": True,
        "job_id": job_id,
        "pages_processed": len(results),
        "successful": successful,
        "results": results
    }


# ============================================================
# COMPREHENSIVE MARKUP GENERATION
# ============================================================

def generate_comprehensive_markup(page_id):
    """
    Generate a comprehensive markup showing all detections with calculations.
    
    Features:
    - Shape-aware rendering (triangles for gables, hatched rectangles for roofs)
    - Transparent fills for overlapping visibility
    - Dimension labels with area/rake calculations
    - Summary legend with formula breakdown
    
    Args:
        page_id: UUID of the extraction_page
    
    Returns:
        Dict with success status and markup URL
    """
    # Get page data
    page = get_page(page_id)
    if not page:
        return {"error": "Page not found"}
    
    job_id = page.get('job_id')
    page_number = page.get('page_number', 0)
    elevation_name = page.get('elevation_name', 'unknown')
    
    # Get detection details for this page
    detections = supabase_request('GET', 'extraction_detection_details', filters={
        'page_id': f'eq.{page_id}',
        'order': 'class,detection_index'
    })
    
    if not detections:
        return {"error": "No detections for this page"}
    
    # Get elevation calculations
    calcs = supabase_request('GET', 'extraction_elevation_calcs', filters={
        'page_id': f'eq.{page_id}'
    })
    calc = calcs[0] if calcs else {}
    
    # Download base image
    image_url = page.get('image_url')
    if not image_url:
        return {"error": "No image URL for page"}
    
    response = requests.get(image_url)
    if response.status_code != 200:
        return {"error": f"Failed to download image: {response.status_code}"}
    
    base_img = Image.open(io.BytesIO(response.content)).convert('RGBA')
    
    # Create overlay for transparent fills
    overlay = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Try to load fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_large = font
    
    # Color scheme with RGB tuples for fills
    COLORS = {
        'building': {'outline': '#22C55E', 'fill': (34, 197, 94, 60), 'rgb': (34, 197, 94)},
        'exterior_wall': {'outline': '#22C55E', 'fill': (34, 197, 94, 60), 'rgb': (34, 197, 94)},
        'window': {'outline': '#3B82F6', 'fill': (59, 130, 246, 80), 'rgb': (59, 130, 246)},
        'door': {'outline': '#F97316', 'fill': (249, 115, 22, 80), 'rgb': (249, 115, 22)},
        'garage': {'outline': '#A855F7', 'fill': (168, 85, 247, 80), 'rgb': (168, 85, 247)},
        'roof': {'outline': '#EF4444', 'fill': (239, 68, 68, 50), 'rgb': (239, 68, 68)},
        'gable': {'outline': '#EC4899', 'fill': (236, 72, 153, 70), 'rgb': (236, 72, 153)},
    }
    
    # Count by class for legend
    class_counts = {}
    class_areas = {}
    class_rake_lf = {}  # Track rake LF for gables
    
    # Sort detections: draw larger items first (building, then roof, then gable, then others)
    def sort_key(det):
        cls = det.get('class', '')
        if cls in ['building', 'exterior_wall']:
            return 0
        elif cls == 'roof':
            return 1
        elif cls == 'gable':
            return 2
        else:
            return 3
    
    sorted_detections = sorted(detections, key=sort_key)
    
    # Draw each detection with appropriate shape
    for det in sorted_detections:
        cls = det.get('class', 'unknown')
        colors = COLORS.get(cls, {'outline': '#888888', 'fill': (136, 136, 136, 50), 'rgb': (136, 136, 136)})
        
        px = float(det.get('pixel_x', 0))
        py = float(det.get('pixel_y', 0))
        pw = float(det.get('pixel_width', 0))
        ph = float(det.get('pixel_height', 0))
        
        x1 = px - pw / 2
        y1 = py - ph / 2
        x2 = px + pw / 2
        y2 = py + ph / 2
        
        area_sf = float(det.get('area_sf', 0))
        width_ft = float(det.get('real_width_ft', 0))
        height_ft = float(det.get('real_height_ft', 0))
        
        # Draw shape using helper function
        draw_detection_shape(
            overlay_draw, x1, y1, x2, y2, cls,
            outline_color=colors['rgb'],
            fill_color=colors['fill'],
            width=2,
            draw_hatching=(cls == 'roof')
        )
        
        # Track counts and areas
        if cls not in class_counts:
            class_counts[cls] = 0
            class_areas[cls] = 0
            class_rake_lf[cls] = 0
        class_counts[cls] += 1
        class_areas[cls] += area_sf
        
        # Calculate rake for gables
        if cls == 'gable':
            rake_lf = calculate_gable_rake_lf(width_ft, height_ft)
            class_rake_lf[cls] += rake_lf
    
    # Composite overlay onto base image
    img = Image.alpha_composite(base_img, overlay)
    
    # Create another overlay for labels
    label_overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_overlay)
    
    # Draw labels on top
    for det in sorted_detections:
        cls = det.get('class', 'unknown')
        colors = COLORS.get(cls, {'outline': '#888888', 'fill': (136, 136, 136, 50), 'rgb': (136, 136, 136)})
        
        px = float(det.get('pixel_x', 0))
        py = float(det.get('pixel_y', 0))
        pw = float(det.get('pixel_width', 0))
        ph = float(det.get('pixel_height', 0))
        
        x1 = px - pw / 2
        y1 = py - ph / 2
        x2 = px + pw / 2
        y2 = py + ph / 2
        
        width_ft = float(det.get('real_width_ft', 0))
        height_ft = float(det.get('real_height_ft', 0))
        area_sf = float(det.get('area_sf', 0))
        
        # Draw dimension label with background
        if cls in ['window', 'door', 'garage']:
            label = f"{width_ft:.1f}'x{height_ft:.1f}'"
            label_x = x1 + 3
            label_y = y1 + 2
            bbox = label_draw.textbbox((label_x, label_y), label, font=font_small)
            label_draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 200))
            label_draw.text((label_x, label_y), label, fill=colors['rgb'], font=font_small)
        
        elif cls == 'gable':
            # Special label for gables: show area and rake
            rake_lf = calculate_gable_rake_lf(width_ft, height_ft)
            label = f"△ {area_sf:.0f} SF"
            label2 = f"Rake: {rake_lf:.1f}'"
            
            # Position in center of triangle
            label_x = x1 + pw / 4
            label_y = y1 + ph / 3
            
            bbox = label_draw.textbbox((label_x, label_y), label, font=font)
            label_draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 220))
            label_draw.text((label_x, label_y), label, fill=colors['rgb'], font=font)
            
            # Second line for rake
            label_y2 = label_y + 18
            bbox2 = label_draw.textbbox((label_x, label_y2), label2, font=font_small)
            label_draw.rectangle([bbox2[0] - 2, bbox2[1] - 1, bbox2[2] + 2, bbox2[3] + 1], fill=(255, 255, 255, 200))
            label_draw.text((label_x, label_y2), label2, fill=colors['rgb'], font=font_small)
        
        elif cls in ['building', 'exterior_wall', 'roof']:
            label = f"{area_sf:.0f} SF"
            label_x = x1 + 5
            label_y = y1 + 5
            bbox = label_draw.textbbox((label_x, label_y), label, font=font)
            label_draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 220))
            label_draw.text((label_x, label_y), label, fill=colors['rgb'], font=font)
    
    # Composite labels
    img = Image.alpha_composite(img, label_overlay)
    
    # Draw legend box
    legend_x = 10
    legend_y = 10
    legend_width = 300
    legend_height = 260
    
    legend_overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    legend_draw = ImageDraw.Draw(legend_overlay)
    legend_draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                         fill=(255, 255, 255, 240), outline=(0, 0, 0, 255), width=2)
    
    img = Image.alpha_composite(img, legend_overlay)
    draw = ImageDraw.Draw(img)
    
    # Legend title
    title = f"ELEVATION: {elevation_name.upper() if elevation_name else 'UNKNOWN'}"
    draw.text((legend_x + 10, legend_y + 8), title, fill='#000000', font=font_large)
    
    # Legend items with shape indicators
    y_offset = legend_y + 35
    line_height = 20
    
    for cls_name, colors in COLORS.items():
        if cls_name in class_counts:
            count = class_counts[cls_name]
            area = class_areas[cls_name]
            shape = get_shape_indicator(cls_name)
            
            # Build label with rake for gables
            if cls_name == 'gable':
                rake = class_rake_lf.get(cls_name, 0)
                label = f"{shape} {cls_name.replace('_', ' ').title()}: {count} ({area:.0f} SF, {rake:.1f}' rake)"
            elif cls_name == 'roof':
                label = f"{shape} {cls_name.replace('_', ' ').title()}: {count} ({area:.0f} SF) [EXCLUDED]"
            else:
                label = f"{shape} {cls_name.replace('_', ' ').title()}: {count} ({area:.0f} SF)"
            
            # Draw color swatch (triangle for gable, hatched for roof)
            swatch_x1 = legend_x + 10
            swatch_y1 = y_offset
            swatch_x2 = legend_x + 28
            swatch_y2 = y_offset + 14
            
            if cls_name == 'gable':
                # Draw triangle swatch
                apex = ((swatch_x1 + swatch_x2) / 2, swatch_y1)
                bl = (swatch_x1, swatch_y2)
                br = (swatch_x2, swatch_y2)
                draw.polygon([apex, bl, br], fill=colors['outline'], outline='#000000')
            else:
                draw.rectangle([swatch_x1, swatch_y1, swatch_x2, swatch_y2],
                              fill=colors['outline'], outline='#000000')
            
            draw.text((legend_x + 35, y_offset - 2), label, fill='#000000', font=font_small)
            y_offset += line_height
    
    # Summary calculations
    y_offset += 8
    draw.line([(legend_x + 10, y_offset), (legend_x + legend_width - 10, y_offset)], fill='#000000', width=1)
    y_offset += 10
    
    gross_facade = float(calc.get('gross_facade_sf', 0))
    openings = float(calc.get('total_openings_sf', 0))
    net_siding = float(calc.get('net_siding_sf', 0))
    gable = float(calc.get('gable_area_sf', 0))
    gable_rake = float(calc.get('gable_rake_lf', 0))
    
    draw.text((legend_x + 10, y_offset), f"Gross Facade: {gross_facade:.0f} SF", fill='#000000', font=font_small)
    y_offset += line_height
    draw.text((legend_x + 10, y_offset), f"- Openings: {openings:.0f} SF", fill='#000000', font=font_small)
    y_offset += line_height
    draw.text((legend_x + 10, y_offset), f"+ Gables: {gable:.0f} SF (△ {gable_rake:.1f}' rake)", fill='#000000', font=font_small)
    y_offset += line_height
    draw.text((legend_x + 10, y_offset), f"= NET SIDING: {net_siding:.0f} SF", fill='#000000', font=font_large)
    
    # Convert to RGB for saving
    final_img = img.convert('RGB')
    
    # Save to buffer
    buffer = io.BytesIO()
    final_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Upload to Supabase
    filename = f"comprehensive_{page_number:03d}_{elevation_name or 'elevation'}.png"
    filepath = f"{job_id}/{filename}"
    
    markup_url = upload_to_storage(buffer.getvalue(), filepath, 'image/png')
    
    if markup_url:
        return {
            "success": True,
            "page_id": page_id,
            "elevation": elevation_name,
            "markup_url": markup_url,
            "summary": {
                "gross_facade_sf": gross_facade,
                "openings_sf": openings,
                "gable_sf": gable,
                "gable_rake_lf": gable_rake,
                "net_siding_sf": net_siding,
                "detections": class_counts
            }
        }
    else:
        return {"error": "Upload failed"}
