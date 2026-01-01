"""
Extraction API v3.2 - Visual Markups for All Trades
"""

import os
import json
import base64
import requests
import time
import threading
import tempfile
import re
from io import BytesIO
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

app = Flask(__name__)

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://okwtyttfqbfmcqtenize.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

MAX_CONCURRENT_CLAUDE = 3
BATCH_DELAY_SECONDS = 0.5
PDF_CHUNK_SIZE = 5
DEFAULT_DPI = 100

ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/infer/workflows/exterior-finishes/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables"

VALID_PAGE_TYPES = {'elevation', 'schedule', 'floor_plan', 'section', 'detail', 'cover', 'site_plan', 'other'}

# Markup colors (RGB)
MARKUP_COLORS = {
    'window': (0, 120, 255),      # Blue
    'door': (255, 140, 0),        # Orange
    'garage': (148, 0, 211),      # Purple
    'building': (34, 139, 34),    # Green (siding area)
    'roof': (220, 20, 60),        # Red
    'gable': (255, 105, 180),     # Pink
    'gutter': (0, 255, 255),      # Cyan
}

TRADE_GROUPS = {
    'siding': ['building', 'window', 'door', 'garage'],
    'roofing': ['roof', 'gable'],
    'windows': ['window'],
    'doors': ['door', 'garage'],
    'gutters': ['roof'],  # Gutters follow roof edges
    'all': ['window', 'door', 'garage', 'building', 'roof', 'gable']
}


def parse_scale_notation(notation):
    if not notation:
        return None
    notation = notation.strip().upper().replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
    
    frac_pattern = r'(\d+)/(\d+)\s*["\"]?\s*=\s*1\s*[\'\']?\s*-?\s*0?\s*["\"]?'
    match = re.search(frac_pattern, notation)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        paper_inches = numerator / denominator
        return 12 / paper_inches
    
    whole_pattern = r'(\d+)\s*["\"]?\s*=\s*(\d+)\s*[\'\']?\s*-?\s*0?\s*["\"]?'
    match = re.search(whole_pattern, notation)
    if match:
        paper_inches = float(match.group(1))
        real_feet = float(match.group(2))
        return (real_feet * 12) / paper_inches
    
    ratio_pattern = r'1\s*:\s*(\d+)'
    match = re.search(ratio_pattern, notation)
    if match:
        return float(match.group(1))
    
    return None


def normalize_page_type(raw_type):
    if not raw_type or not isinstance(raw_type, str):
        return 'other'
    cleaned = raw_type.lower().strip()
    if cleaned in VALID_PAGE_TYPES:
        return cleaned
    mappings = {
        'floorplan': 'floor_plan', 'floor plan': 'floor_plan',
        'siteplan': 'site_plan', 'site plan': 'site_plan',
        'unknown': 'other', 'title': 'cover', 'title sheet': 'cover',
    }
    return mappings.get(cleaned, 'other')


def calculate_derived_measurements(width_in, height_in, qty, element_type='window'):
    measurements = {
        'head_trim_lf': round((width_in * qty) / 12, 2),
        'jamb_trim_lf': round((height_in * 2 * qty) / 12, 2),
        'casing_lf': round(((width_in * 2) + (height_in * 2)) * qty / 12, 2),
        'rough_opening_width': width_in + 1,
        'rough_opening_height': height_in + 1,
        'head_flashing_lf': round((width_in + 4) * qty / 12, 2),
        'area_sf': round((width_in * height_in * qty) / 144, 2),
        'perimeter_lf': round(((width_in * 2) + (height_in * 2)) * qty / 12, 2),
    }
    if element_type == 'window':
        measurements['sill_trim_lf'] = round((width_in * qty) / 12, 2)
        measurements['sill_pan_lf'] = round((width_in + 4) * qty / 12, 2)
    else:
        measurements['sill_trim_lf'] = 0
        measurements['sill_pan_lf'] = 0
    return measurements


def generate_markup_image(image_data, predictions, scale_ratio, dpi=DEFAULT_DPI, 
                          trade_filter=None, show_dimensions=True, show_labels=True):
    """Generate marked-up image with bounding boxes and measurements"""
    img = Image.open(BytesIO(image_data)).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    if not scale_ratio or scale_ratio <= 0:
        scale_ratio = 48
    inches_per_pixel = (1.0 / dpi) * scale_ratio
    
    if trade_filter:
        filtered_preds = [p for p in predictions if p.get('class', '').lower() in trade_filter]
    else:
        filtered_preds = predictions
    
    totals = {}
    
    for pred in filtered_preds:
        class_name = pred.get('class', '').lower().replace(' ', '_')
        x = pred.get('x', 0)
        y = pred.get('y', 0)
        width = pred.get('width', 0)
        height = pred.get('height', 0)
        
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        
        color = MARKUP_COLORS.get(class_name, (128, 128, 128))
        
        real_width_in = width * inches_per_pixel
        real_height_in = height * inches_per_pixel
        area_sqft = (real_width_in * real_height_in) / 144
        
        if class_name not in totals:
            totals[class_name] = {'count': 0, 'area_sqft': 0}
        totals[class_name]['count'] += 1
        totals[class_name]['area_sqft'] += area_sqft
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        if show_labels and class_name in ['window', 'door', 'garage', 'building', 'roof']:
            label = class_name.upper()[:3]
            real_width_ft = real_width_in / 12
            real_height_ft = real_height_in / 12
            if show_dimensions:
                label += f" {real_width_ft:.1f}x{real_height_ft:.1f}"
            try:
                draw.text((x1+2, y1+2), label, fill=color, font=small_font)
            except:
                draw.text((x1+2, y1+2), label, fill=color)
    
    legend_y = 10
    try:
        draw.rectangle([5, 5, 220, 25 + len(totals) * 20], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((10, legend_y), "LEGEND", fill=(0, 0, 0), font=font)
        legend_y += 20
        for class_name, data in totals.items():
            color = MARKUP_COLORS.get(class_name, (128, 128, 128))
            draw.rectangle([10, legend_y, 25, legend_y + 12], fill=color)
            draw.text((30, legend_y), f"{class_name}: {data['count']} ({data['area_sqft']:.0f}SF)", fill=(0, 0, 0), font=small_font)
            legend_y += 18
    except:
        pass
    
    return img, totals


def supabase_request(method, endpoint, data=None, filters=None):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    if filters:
        filter_parts = [f"{k}={v}" for k, v in filters.items()]
        url += "?" + "&".join(filter_parts)
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'apikey': SUPABASE_KEY,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method == 'PATCH':
            response = requests.patch(url, headers=headers, json=data)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            return None
        if response.status_code >= 400:
            print(f"Supabase {method} error: {response.status_code} - {response.text}", flush=True)
            return None
        return response.json() if response.content else []
    except Exception as e:
        print(f"Supabase error: {e}", flush=True)
        return None


def upload_to_supabase(image_data, filename, content_type='image/jpeg'):
    if not SUPABASE_KEY:
        return None
    try:
        upload_url = f"{SUPABASE_URL}/storage/v1/object/extraction-markups/{filename}"
        headers = {'Authorization': f'Bearer {SUPABASE_KEY}', 'Content-Type': content_type, 'x-upsert': 'true'}
        response = requests.post(upload_url, headers=headers, data=image_data)
        if response.status_code in [200, 201]:
            return f"{SUPABASE_URL}/storage/v1/object/public/extraction-markups/{filename}"
        return None
    except:
        return None


def update_job(job_id, updates):
    return supabase_request('PATCH', 'extraction_jobs', updates, {'id': f'eq.{job_id}'})


def update_page(page_id, updates):
    result = supabase_request('PATCH', 'extraction_pages', updates, {'id': f'eq.{page_id}'})
    if not result:
        print(f"FAILED update page {page_id}", flush=True)
    return result


def detect_with_roboflow(image_url):
    payload = {"api_key": ROBOFLOW_API_KEY, "inputs": {"image": {"type": "url", "value": image_url}}}
    try:
        response = requests.post(ROBOFLOW_WORKFLOW_URL, json=payload, timeout=120)
        if response.status_code != 200:
            return {"error": f"Roboflow error: {response.status_code}"}
        result = response.json()
        predictions = []
        if 'outputs' in result and len(result['outputs']) > 0:
            output = result['outputs'][0]
            if 'predictions' in output:
                pred_data = output['predictions']
                if isinstance(pred_data, dict) and 'predictions' in pred_data:
                    predictions = pred_data['predictions']
                elif isinstance(pred_data, list):
                    predictions = pred_data
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}


def classify_page_with_claude(image_url):
    if not ANTHROPIC_API_KEY:
        return {"page_type": "other", "confidence": 0}
    try:
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        prompt = """Analyze this architectural drawing. Return ONLY valid JSON:

{
  "page_type": "elevation" | "schedule" | "floor_plan" | "section" | "detail" | "cover" | "site_plan" | "other",
  "confidence": 0.0 to 1.0,
  "elevation_name": "front" | "rear" | "left" | "right" | "north" | "south" | "east" | "west" | null,
  "scale_notation": "exact scale text from drawing or null",
  "scale_location": "where scale was found",
  "contains_schedule": true | false,
  "schedule_type": "window" | "door" | "window_and_door" | null,
  "notes": "brief description"
}"""

        api_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 500, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}}, {"type": "text", "text": prompt}]}]},
            timeout=60
        )
        if api_response.status_code == 200:
            text = api_response.json()['content'][0]['text']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result['page_type'] = normalize_page_type(result.get('page_type'))
                scale_notation = result.get('scale_notation')
                if scale_notation:
                    result['scale_ratio'] = parse_scale_notation(scale_notation)
                return result
        return {"page_type": "other", "confidence": 0}
    except Exception as e:
        return {"page_type": "other", "confidence": 0, "error": str(e)}


def ocr_schedule_with_claude(image_url):
    if not ANTHROPIC_API_KEY:
        return {"error": "No API key"}
    try:
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        prompt = """Extract the window and door schedule from this architectural drawing.

Return ONLY valid JSON:
{
  "windows": [{"tag": "W1", "width_inches": 36, "height_inches": 48, "type": "DH", "qty": 4, "notes": ""}],
  "doors": [{"tag": "D1", "width_inches": 36, "height_inches": 80, "type": "Entry", "qty": 1, "notes": ""}],
  "raw_text": "additional text"
}"""

        api_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 4000, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}}, {"type": "text", "text": prompt}]}]},
            timeout=120
        )
        if api_response.status_code == 200:
            text = api_response.json()['content'][0]['text']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        return {"error": "API error"}
    except Exception as e:
        return {"error": str(e)}


def calculate_real_measurements(predictions, scale_ratio, dpi=DEFAULT_DPI):
    if not scale_ratio or scale_ratio <= 0:
        scale_ratio = 48
        scale_warning = "Using default scale"
    else:
        scale_warning = None
    
    inches_per_pixel = 1.0 / dpi
    real_inches_per_pixel = inches_per_pixel * scale_ratio
    sqft_per_sq_pixel = (real_inches_per_pixel ** 2) / 144
    
    results = {
        'scale_used': scale_ratio, 'scale_warning': scale_warning, 'dpi': dpi,
        'items': {'windows': [], 'doors': [], 'garages': [], 'buildings': [], 'roofs': [], 'gables': []},
        'counts': {'window': 0, 'door': 0, 'garage': 0, 'building': 0, 'roof': 0, 'gable': 0},
        'areas': {'window_sqft': 0, 'door_sqft': 0, 'garage_sqft': 0, 'building_sqft': 0, 'roof_sqft': 0, 'gable_sqft': 0}
    }
    
    for pred in predictions:
        class_name = pred.get('class', '').lower()
        width_px = pred.get('width', 0)
        height_px = pred.get('height', 0)
        
        width_inches = width_px * real_inches_per_pixel
        height_inches = height_px * real_inches_per_pixel
        area_sqft = width_px * height_px * sqft_per_sq_pixel
        
        item = {
            'width_inches': round(width_inches, 1), 'height_inches': round(height_inches, 1),
            'area_sqft': round(area_sqft, 1), 'pixel_x': pred.get('x', 0), 'pixel_y': pred.get('y', 0),
            'pixel_width': width_px, 'pixel_height': height_px, 'confidence': pred.get('confidence', 0)
        }
        
        if class_name in results['items']:
            results['items'][class_name + 's'].append(item)
        if class_name in results['counts']:
            results['counts'][class_name] += 1
            results['areas'][class_name + '_sqft'] += area_sqft
    
    results['areas']['gross_wall_sqft'] = round(results['areas']['building_sqft'], 1)
    results['areas']['openings_sqft'] = round(sum(results['areas'][k] for k in ['window_sqft', 'door_sqft', 'garage_sqft']), 1)
    results['areas']['net_siding_sqft'] = round(results['areas']['gross_wall_sqft'] - results['areas']['openings_sqft'], 1)
    
    return results


def generate_markups_for_page(page_id, trades=None):
    """
    Generate markup images for a single page
    
    Args:
        page_id: UUID of the extraction_page
        trades: List of trades to generate ['all', 'siding', 'roofing', 'windows', 'doors', 'gutters']
    
    Returns:
        Dict with markup URLs
    """
    if trades is None:
        trades = ['all']
    
    # Get page data
    page = supabase_request('GET', 'extraction_pages', filters={'id': f'eq.{page_id}'})
    if not page:
        return {"error": "Page not found"}
    page = page[0]
    
    image_url = page.get('image_url')
    extraction_data = page.get('extraction_data', {})
    predictions = extraction_data.get('raw_predictions', [])
    scale_ratio = page.get('scale_ratio') or 48
    dpi = page.get('dpi') or DEFAULT_DPI
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
    
    for trade in trades:
        try:
            trade_filter = TRADE_GROUPS.get(trade, TRADE_GROUPS['all'])
            
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
            markup_url = upload_to_supabase(buffer.getvalue(), filename, 'image/png')
            
            if markup_url:
                markup_urls[trade] = {
                    'url': markup_url,
                    'totals': totals
                }
            else:
                print(f"Failed to upload markup for {trade}", flush=True)
        except Exception as e:
            print(f"Error generating markup for {trade}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    # Update page with markup URLs
    update_page(page_id, {
        'markup_urls': markup_urls
    })
    
    return {"success": True, "page_id": page_id, "markups": markup_urls}


def generate_markups_for_job(job_id, trades=None):
    """Generate markups for all elevation pages in a job"""
    if trades is None:
        trades = ['all', 'siding', 'roofing']
    
    print(f"[{job_id}] Generating markups...", flush=True)
    
    # Get all elevation pages with extraction data
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.elevation',
        'status': 'eq.complete',
        'order': 'page_number'
    })
    
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


# Background processing functions
def convert_pdf_background(job_id, pdf_url):
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
        print(f"[{job_id}] Downloading PDF...", flush=True)
        update_job(job_id, {'status': 'converting'})
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            response = requests.get(pdf_url, timeout=300, stream=True)
            if response.status_code != 200:
                update_job(job_id, {'status': 'failed', 'error_message': 'Download failed'})
                return
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        try:
            info = pdfinfo_from_path(tmp_path)
            total_pages = info['Pages']
        except:
            total_pages = 100
        update_job(job_id, {'total_pages': total_pages, 'plan_dpi': DEFAULT_DPI})
        pages_converted = 0
        for start_page in range(1, total_pages + 1, PDF_CHUNK_SIZE):
            end_page = min(start_page + PDF_CHUNK_SIZE - 1, total_pages)
            try:
                images = convert_from_path(tmp_path, dpi=DEFAULT_DPI, fmt='png', first_page=start_page, last_page=end_page, thread_count=1)
                for i, img in enumerate(images):
                    page_num = start_page + i
                    if img.width > 2000 or img.height > 2000:
                        img.thumbnail((2000, 2000))
                    buffer = BytesIO()
                    img.save(buffer, format='PNG', optimize=True)
                    buffer.seek(0)
                    filename = f"{job_id}/page_{page_num:03d}.png"
                    image_url = upload_to_supabase(buffer.getvalue(), filename, 'image/png')
                    thumb = img.copy()
                    thumb.thumbnail((200, 200))
                    thumb_buffer = BytesIO()
                    thumb.save(thumb_buffer, format='PNG')
                    thumb_filename = f"{job_id}/thumb_{page_num:03d}.png"
                    thumb_url = upload_to_supabase(thumb_buffer.getvalue(), thumb_filename, 'image/png')
                    supabase_request('POST', 'extraction_pages', {
                        'job_id': job_id, 'page_number': page_num,
                        'image_url': image_url, 'thumbnail_url': thumb_url,
                        'status': 'pending', 'dpi': DEFAULT_DPI
                    })
                    pages_converted += 1
                    del img
                del images
                update_job(job_id, {'pages_converted': pages_converted})
            except Exception as e:
                print(f"[{job_id}] Chunk error: {e}", flush=True)
        try:
            os.unlink(tmp_path)
        except:
            pass
        update_job(job_id, {'status': 'converted'})
    except Exception as e:
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def classify_job_background(job_id):
    try:
        update_job(job_id, {'status': 'classifying'})
        pages = supabase_request('GET', 'extraction_pages', filters={'job_id': f'eq.{job_id}', 'status': 'eq.pending', 'order': 'page_number'})
        if not pages:
            return
        
        elevation_count = schedule_count = floor_plan_count = other_count = 0
        scales_found = []
        
        for i, page in enumerate(pages):
            classification = classify_page_with_claude(page.get('image_url'))
            page_type = normalize_page_type(classification.get('page_type'))
            
            if page_type == 'elevation': elevation_count += 1
            elif page_type == 'schedule': schedule_count += 1
            elif page_type == 'floor_plan': floor_plan_count += 1
            else: other_count += 1
            
            scale_ratio = classification.get('scale_ratio')
            if scale_ratio:
                scales_found.append(scale_ratio)
            
            update_page(page['id'], {
                'page_type': page_type,
                'page_type_confidence': classification.get('confidence', 0),
                'status': 'classified',
                'scale_notation': classification.get('scale_notation'),
                'scale_ratio': scale_ratio,
                'elevation_name': classification.get('elevation_name')
            })
            update_job(job_id, {'pages_classified': i + 1})
            
            if (i + 1) % MAX_CONCURRENT_CLAUDE == 0:
                time.sleep(BATCH_DELAY_SECONDS)
        
        from collections import Counter
        default_scale = Counter(scales_found).most_common(1)[0][0] if scales_found else None
        
        update_job(job_id, {
            'status': 'classified',
            'elevation_count': elevation_count,
            'schedule_count': schedule_count,
            'floor_plan_count': floor_plan_count,
            'other_count': other_count,
            'default_scale_ratio': default_scale
        })
    except Exception as e:
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def process_job_background(job_id, scale_override=None, generate_markups=True):
    try:
        update_job(job_id, {'status': 'processing'})
        
        job = supabase_request('GET', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
        default_scale = job[0].get('default_scale_ratio') if job else None
        job_dpi = job[0].get('plan_dpi', DEFAULT_DPI) if job else DEFAULT_DPI
        
        pages = supabase_request('GET', 'extraction_pages', filters={'job_id': f'eq.{job_id}', 'status': 'eq.classified', 'order': 'page_number'})
        if not pages:
            return
        
        elevation_pages = [p for p in pages if p.get('page_type') == 'elevation']
        schedule_pages = [p for p in pages if p.get('page_type') == 'schedule']
        
        totals = {'total_net_siding_sqft': 0, 'total_gross_wall_sqft': 0, 'total_windows': 0, 'total_doors': 0}
        processed = 0
        
        for page in elevation_pages:
            scale_ratio = scale_override or page.get('scale_ratio') or default_scale or 48
            dpi = page.get('dpi') or job_dpi
            
            detection = detect_with_roboflow(page['image_url'])
            if 'error' not in detection:
                measurements = calculate_real_measurements(detection['predictions'], scale_ratio, dpi)
                totals['total_net_siding_sqft'] += measurements['areas'].get('net_siding_sqft', 0)
                totals['total_gross_wall_sqft'] += measurements['areas'].get('gross_wall_sqft', 0)
                totals['total_windows'] += measurements['counts'].get('window', 0)
                totals['total_doors'] += measurements['counts'].get('door', 0)
                
                update_page(page['id'], {
                    'status': 'complete',
                    'extraction_data': {'measurements': measurements, 'raw_predictions': detection['predictions']}
                })
            else:
                update_page(page['id'], {'status': 'failed'})
            processed += 1
            update_job(job_id, {'pages_processed': processed})
        
        for page in schedule_pages:
            schedule_data = ocr_schedule_with_claude(page['image_url'])
            update_page(page['id'], {
                'status': 'complete' if 'error' not in schedule_data else 'failed',
                'extraction_data': schedule_data
            })
            processed += 1
            update_job(job_id, {'pages_processed': processed})
        
        for page in pages:
            if page.get('page_type') not in ['elevation', 'schedule']:
                update_page(page['id'], {'status': 'skipped'})
        
        update_job(job_id, {'status': 'complete', 'results_summary': totals})
        
        # Auto-generate markups
        if generate_markups:
            generate_markups_for_job(job_id, trades=['all', 'siding', 'roofing'])
        
        # Auto-run cross-reference
        build_cross_references(job_id)
        
    except Exception as e:
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def build_cross_references(job_id):
    """Build cross-references between schedule and detections"""
    schedule_pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}', 'page_type': 'eq.schedule', 'status': 'eq.complete'
    })
    
    if not schedule_pages:
        return {"error": "No schedule data"}
    
    windows_by_tag = {}
    doors_by_tag = {}
    
    for page in schedule_pages:
        extraction_data = page.get('extraction_data', {})
        page_id = page.get('id')
        
        for window in extraction_data.get('windows', []):
            tag = window.get('tag', 'UNKNOWN')
            if tag not in windows_by_tag or window.get('qty', 1) > windows_by_tag[tag]['qty']:
                windows_by_tag[tag] = {**window, 'schedule_page_id': page_id}
        
        for door in extraction_data.get('doors', []):
            tag = door.get('tag', 'UNKNOWN')
            if tag not in doors_by_tag or door.get('qty', 1) > doors_by_tag[tag]['qty']:
                doors_by_tag[tag] = {**door, 'schedule_page_id': page_id}
    
    elevation_pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}', 'page_type': 'eq.elevation', 'status': 'eq.complete'
    })
    
    elevation_page_ids = [p.get('id') for p in elevation_pages or []]
    total_detected_windows = sum(p.get('extraction_data', {}).get('measurements', {}).get('counts', {}).get('window', 0) for p in elevation_pages or [])
    total_detected_doors = sum(p.get('extraction_data', {}).get('measurements', {}).get('counts', {}).get('door', 0) for p in elevation_pages or [])
    
    supabase_request('DELETE', 'extraction_cross_refs', filters={'job_id': f'eq.{job_id}'})
    
    for tag, window in windows_by_tag.items():
        derived = calculate_derived_measurements(window.get('width_inches', 0), window.get('height_inches', 0), window.get('qty', 1), 'window')
        supabase_request('POST', 'extraction_cross_refs', {
            'job_id': job_id, 'element_type': 'window', 'tag': tag,
            'schedule_width': window.get('width_inches'), 'schedule_height': window.get('height_inches'),
            'schedule_qty': window.get('qty', 1), 'schedule_type': window.get('type', ''),
            'schedule_page_id': window.get('schedule_page_id'),
            'elevation_page_ids': [str(x) for x in elevation_page_ids],
            'head_trim_lf': derived['head_trim_lf'], 'jamb_trim_lf': derived['jamb_trim_lf'],
            'sill_trim_lf': derived['sill_trim_lf'], 'casing_lf': derived['casing_lf'],
            'rough_opening_width': derived['rough_opening_width'], 'rough_opening_height': derived['rough_opening_height'],
            'head_flashing_lf': derived['head_flashing_lf'], 'sill_pan_lf': derived['sill_pan_lf'],
            'needs_review': False
        })
    
    for tag, door in doors_by_tag.items():
        derived = calculate_derived_measurements(door.get('width_inches', 0), door.get('height_inches', 0), door.get('qty', 1), 'door')
        supabase_request('POST', 'extraction_cross_refs', {
            'job_id': job_id, 'element_type': 'door', 'tag': tag,
            'schedule_width': door.get('width_inches'), 'schedule_height': door.get('height_inches'),
            'schedule_qty': door.get('qty', 1), 'schedule_type': door.get('type', ''),
            'schedule_page_id': door.get('schedule_page_id'),
            'elevation_page_ids': [str(x) for x in elevation_page_ids],
            'head_trim_lf': derived['head_trim_lf'], 'jamb_trim_lf': derived['jamb_trim_lf'],
            'sill_trim_lf': derived['sill_trim_lf'], 'casing_lf': derived['casing_lf'],
            'rough_opening_width': derived['rough_opening_width'], 'rough_opening_height': derived['rough_opening_height'],
            'head_flashing_lf': derived['head_flashing_lf'], 'sill_pan_lf': derived['sill_pan_lf'],
            'needs_review': False
        })
    
    summary = {
        'job_id': job_id,
        'total_windows': sum(w.get('qty', 1) for w in windows_by_tag.values()),
        'total_window_sqft': round(sum((w.get('width_inches', 0) * w.get('height_inches', 0) * w.get('qty', 1)) / 144 for w in windows_by_tag.values()), 2),
        'total_window_head_trim_lf': round(sum((w.get('width_inches', 0) * w.get('qty', 1)) / 12 for w in windows_by_tag.values()), 2),
        'total_window_jamb_trim_lf': round(sum((w.get('height_inches', 0) * 2 * w.get('qty', 1)) / 12 for w in windows_by_tag.values()), 2),
        'total_window_sill_trim_lf': round(sum((w.get('width_inches', 0) * w.get('qty', 1)) / 12 for w in windows_by_tag.values()), 2),
        'total_doors': sum(d.get('qty', 1) for d in doors_by_tag.values()),
        'total_door_sqft': round(sum((d.get('width_inches', 0) * d.get('height_inches', 0) * d.get('qty', 1)) / 144 for d in doors_by_tag.values()), 2),
        'total_door_head_trim_lf': round(sum((d.get('width_inches', 0) * d.get('qty', 1)) / 12 for d in doors_by_tag.values()), 2),
        'total_door_jamb_trim_lf': round(sum((d.get('height_inches', 0) * 2 * d.get('qty', 1)) / 12 for d in doors_by_tag.values()), 2),
        'windows_by_tag': {t: {**w, 'schedule_page_id': str(w['schedule_page_id'])} for t, w in windows_by_tag.items()},
        'doors_by_tag': {t: {**d, 'schedule_page_id': str(d['schedule_page_id'])} for t, d in doors_by_tag.items()},
        'discrepancies': {'schedule_windows': sum(w.get('qty', 1) for w in windows_by_tag.values()), 'detected_windows': total_detected_windows,
                         'schedule_doors': sum(d.get('qty', 1) for d in doors_by_tag.values()), 'detected_doors': total_detected_doors}
    }
    
    existing = supabase_request('GET', 'extraction_takeoff_summary', filters={'job_id': f'eq.{job_id}'})
    if existing:
        supabase_request('PATCH', 'extraction_takeoff_summary', summary, {'job_id': f'eq.{job_id}'})
    else:
        supabase_request('POST', 'extraction_takeoff_summary', summary)
    
    return {"success": True, "windows_count": len(windows_by_tag), "doors_count": len(doors_by_tag)}


# === ENDPOINTS ===

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "version": "3.2", "features": ["markups", "scale_extraction", "cross_reference"]})

@app.route('/start-job', methods=['POST'])
def start_job():
    data = request.json
    if not data.get('pdf_url') or not data.get('project_id'):
        return jsonify({"error": "pdf_url and project_id required"}), 400
    job_result = supabase_request('POST', 'extraction_jobs', {
        'project_id': data['project_id'], 'project_name': data.get('project_name', ''),
        'source_pdf_url': data['pdf_url'], 'status': 'pending', 'plan_dpi': DEFAULT_DPI
    })
    if not job_result:
        return jsonify({"error": "Failed to create job"}), 500
    job_id = job_result[0]['id']
    threading.Thread(target=convert_pdf_background, args=(job_id, data['pdf_url'])).start()
    return jsonify({"success": True, "job_id": job_id})

@app.route('/classify-job', methods=['POST'])
def classify_job():
    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400
    threading.Thread(target=classify_job_background, args=(data['job_id'],)).start()
    return jsonify({"success": True, "job_id": data['job_id'], "status": "classifying"})

@app.route('/process-job', methods=['POST'])
def process_job():
    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400
    threading.Thread(target=process_job_background, args=(data['job_id'], data.get('scale_ratio'), data.get('generate_markups', True))).start()
    return jsonify({"success": True, "job_id": data['job_id'], "status": "processing"})

@app.route('/generate-markups', methods=['POST'])
def generate_markups():
    """Generate markups for a job or single page"""
    data = request.json
    trades = data.get('trades', ['all', 'siding', 'roofing'])
    
    if data.get('page_id'):
        result = generate_markups_for_page(data['page_id'], trades)
    elif data.get('job_id'):
        result = generate_markups_for_job(data['job_id'], trades)
    else:
        return jsonify({"error": "job_id or page_id required"}), 400
    
    return jsonify(result)

@app.route('/cross-reference', methods=['POST'])
def cross_reference():
    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400
    result = build_cross_references(data['job_id'])
    return jsonify(result)

@app.route('/takeoff-summary', methods=['GET'])
def takeoff_summary():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    summary = supabase_request('GET', 'extraction_takeoff_summary', filters={'job_id': f'eq.{job_id}'})
    if not summary:
        return jsonify({"error": "No summary found"}), 404
    return jsonify(summary[0])

@app.route('/cross-refs', methods=['GET'])
def get_cross_refs():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    refs = supabase_request('GET', 'extraction_cross_refs', filters={'job_id': f'eq.{job_id}', 'order': 'element_type,tag'})
    return jsonify({"cross_refs": refs or []})

@app.route('/job-status', methods=['GET'])
def job_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    job = supabase_request('GET', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
    if not job:
        return jsonify({"error": "Job not found"}), 404
    result = {"job": job[0]}
    if request.args.get('include_pages', 'false').lower() == 'true':
        result["pages"] = supabase_request('GET', 'extraction_pages', filters={'job_id': f'eq.{job_id}', 'order': 'page_number'})
    return jsonify(result)

@app.route('/list-jobs', methods=['GET'])
def list_jobs():
    return jsonify({"jobs": supabase_request('GET', 'extraction_jobs', filters={'order': 'created_at.desc', 'limit': '50'}) or []})

@app.route('/parse-scale', methods=['POST'])
def parse_scale_endpoint():
    data = request.json
    return jsonify({"notation": data.get('notation', ''), "scale_ratio": parse_scale_notation(data.get('notation', ''))})


@app.route('/debug-markup', methods=['POST'])
def debug_markup():
    """Debug markup with full error return"""
    data = request.json
    page_id = data.get('page_id')
    
    try:
        page = supabase_request('GET', 'extraction_pages', filters={'id': f'eq.{page_id}'})
        if not page:
            return jsonify({"step": "get_page", "error": "Page not found"})
        page = page[0]
        
        image_url = page.get('image_url')
        extraction_data = page.get('extraction_data', {})
        predictions = extraction_data.get('raw_predictions', [])
        scale_ratio = float(page.get('scale_ratio') or 48)
        dpi = page.get('dpi') or 100
        job_id = page.get('job_id')
        page_num = page.get('page_number')
        
        if not predictions:
            return jsonify({"step": "check_predictions", "error": "No predictions"})
        
        # Download image
        response = requests.get(image_url, timeout=30)
        image_data = response.content
        
        # Simple markup - just draw boxes
        from PIL import Image, ImageDraw
        img = Image.open(BytesIO(image_data)).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        count = 0
        for pred in predictions[:10]:  # Just first 10
            x = pred.get('x', 0)
            y = pred.get('y', 0)
            w = pred.get('width', 0)
            h = pred.get('height', 0)
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            count += 1
        
        # Save
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Upload
        filename = f"{job_id}/debug_markup_{page_num:03d}.png"
        markup_url = upload_to_supabase(buffer.getvalue(), filename, 'image/png')
        
        return jsonify({
            "success": True,
            "boxes_drawn": count,
            "url": markup_url
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5050)), debug=False)

@app.route('/test-markup', methods=['POST'])
def test_markup():
    """Debug markup generation"""
    data = request.json
    page_id = data.get('page_id')
    
    try:
        # Get page data
        page = supabase_request('GET', 'extraction_pages', filters={'id': f'eq.{page_id}'})
        if not page:
            return jsonify({"error": "Page not found"})
        page = page[0]
        
        image_url = page.get('image_url')
        extraction_data = page.get('extraction_data', {})
        predictions = extraction_data.get('raw_predictions', [])
        scale_ratio = page.get('scale_ratio') or 48
        
        # Download image
        response = requests.get(image_url, timeout=30)
        image_data = response.content
        
        # Try to open image
        from PIL import Image
        img = Image.open(BytesIO(image_data))
        
        return jsonify({
            "image_url": image_url,
            "image_size": img.size,
            "predictions_count": len(predictions),
            "scale_ratio": scale_ratio,
            "first_prediction": predictions[0] if predictions else None
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})
