"""
Markup generation service
"""

import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

from config import config
from database import (
    get_page, update_page, get_elevation_pages,
    upload_to_storage, supabase_request
)


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
        area_sqft = (real_width_in * real_height_in) / 144
        
        # Track totals
        if class_name not in totals:
            totals[class_name] = {'count': 0, 'area_sqft': 0}
        totals[class_name]['count'] += 1
        totals[class_name]['area_sqft'] += area_sqft
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        if show_labels and class_name in ['window', 'door', 'garage', 'building', 'roof']:
            label = class_name.upper()[:3]
            if show_dimensions:
                real_width_ft = real_width_in / 12
                real_height_ft = real_height_in / 12
                label += f" {real_width_ft:.1f}x{real_height_ft:.1f}"
            try:
                draw.text((x1 + 2, y1 + 2), label, fill=color, font=small_font)
            except:
                draw.text((x1 + 2, y1 + 2), label, fill=color)
    
    # Draw legend
    legend_y = 10
    try:
        draw.rectangle([5, 5, 220, 25 + len(totals) * 20], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((10, legend_y), "LEGEND", fill=(0, 0, 0), font=font)
        legend_y += 20
        for class_name, data in totals.items():
            color = config.MARKUP_COLORS.get(class_name, (128, 128, 128))
            draw.rectangle([10, legend_y, 25, legend_y + 12], fill=color)
            draw.text((30, legend_y), f"{class_name}: {data['count']} ({data['area_sqft']:.0f}SF)", fill=(0, 0, 0), font=small_font)
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
        page_num = page.get('page_number')
        print(f"[{job_id}] Marking up page {page_num}...", flush=True)
        
        result = generate_markups_for_page(page_id, trades)
        results.append({
            'page_number': page_num,
            'page_id': page_id,
            'result': result
        })
    
    print(f"[{job_id}] Markup generation complete!", flush=True)
    return {"success": True, "job_id": job_id, "pages_marked": len(results), "results": results}


def generate_comprehensive_markup(page_id):
    """
    Generate a single comprehensive markup image for an elevation.
    Shows all detections with tinted fills, dimensions and a summary legend.
    """
    import io
    
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
        'building': {'outline': '#22C55E', 'fill': (34, 197, 94, 60)},
        'exterior_wall': {'outline': '#22C55E', 'fill': (34, 197, 94, 60)},
        'window': {'outline': '#3B82F6', 'fill': (59, 130, 246, 80)},
        'door': {'outline': '#F97316', 'fill': (249, 115, 22, 80)},
        'garage': {'outline': '#A855F7', 'fill': (168, 85, 247, 80)},
        'roof': {'outline': '#EF4444', 'fill': (239, 68, 68, 50)},
        'gable': {'outline': '#EC4899', 'fill': (236, 72, 153, 70)},
    }
    
    # Count by class for legend
    class_counts = {}
    class_areas = {}
    
    # Sort detections: draw larger items first
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
    
    # Draw each detection
    for det in sorted_detections:
        cls = det.get('class', 'unknown')
        colors = COLORS.get(cls, {'outline': '#888888', 'fill': (136, 136, 136, 50)})
        
        px = float(det.get('pixel_x', 0))
        py = float(det.get('pixel_y', 0))
        pw = float(det.get('pixel_width', 0))
        ph = float(det.get('pixel_height', 0))
        
        x1 = px - pw / 2
        y1 = py - ph / 2
        x2 = px + pw / 2
        y2 = py + ph / 2
        
        area_sf = float(det.get('area_sf', 0))
        
        # Draw filled rectangle with transparency
        overlay_draw.rectangle([x1, y1, x2, y2], fill=colors['fill'], outline=colors['outline'], width=2)
        
        # Track counts and areas
        if cls not in class_counts:
            class_counts[cls] = 0
            class_areas[cls] = 0
        class_counts[cls] += 1
        class_areas[cls] += area_sf
    
    # Composite overlay onto base image
    img = Image.alpha_composite(base_img, overlay)
    
    # Create another overlay for labels
    label_overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_overlay)
    
    # Draw labels on top
    for det in sorted_detections:
        cls = det.get('class', 'unknown')
        colors = COLORS.get(cls, {'outline': '#888888', 'fill': (136, 136, 136, 50)})
        
        px = float(det.get('pixel_x', 0))
        py = float(det.get('pixel_y', 0))
        pw = float(det.get('pixel_width', 0))
        ph = float(det.get('pixel_height', 0))
        
        x1 = px - pw / 2
        y1 = py - ph / 2
        
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
            label_draw.text((label_x, label_y), label, fill=colors['outline'], font=font_small)
        elif cls in ['building', 'exterior_wall', 'roof', 'gable']:
            label = f"{area_sf:.0f} SF"
            label_x = x1 + 5
            label_y = y1 + 5
            bbox = label_draw.textbbox((label_x, label_y), label, font=font)
            label_draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 220))
            label_draw.text((label_x, label_y), label, fill=colors['outline'], font=font)
    
    # Composite labels
    img = Image.alpha_composite(img, label_overlay)
    
    # Draw legend box
    legend_x = 10
    legend_y = 10
    legend_width = 280
    legend_height = 230
    
    legend_overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    legend_draw = ImageDraw.Draw(legend_overlay)
    legend_draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                         fill=(255, 255, 255, 240), outline=(0, 0, 0, 255), width=2)
    
    img = Image.alpha_composite(img, legend_overlay)
    draw = ImageDraw.Draw(img)
    
    # Legend title
    title = f"ELEVATION: {elevation_name.upper() if elevation_name else 'UNKNOWN'}"
    draw.text((legend_x + 10, legend_y + 8), title, fill='#000000', font=font_large)
    
    # Legend items
    y_offset = legend_y + 35
    line_height = 20
    
    for cls_name, colors in COLORS.items():
        if cls_name in class_counts:
            count = class_counts[cls_name]
            area = class_areas[cls_name]
            label = f"{cls_name.replace('_', ' ').title()}: {count} ({area:.0f} SF)"
            
            draw.rectangle([legend_x + 10, y_offset, legend_x + 28, y_offset + 14],
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
    
    draw.text((legend_x + 10, y_offset), f"Gross Facade: {gross_facade:.0f} SF", fill='#000000', font=font_small)
    y_offset += line_height
    draw.text((legend_x + 10, y_offset), f"- Openings: {openings:.0f} SF", fill='#000000', font=font_small)
    y_offset += line_height
    draw.text((legend_x + 10, y_offset), f"+ Gables: {gable:.0f} SF", fill='#000000', font=font_small)
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
                "net_siding_sf": net_siding,
                "detections": class_counts
            }
        }
    else:
        return {"error": "Upload failed"}
