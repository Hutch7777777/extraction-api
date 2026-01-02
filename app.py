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
    'exterior wall': (34, 139, 34),  # Green (same as building)
    'exterior_wall': (34, 139, 34),  # Green (normalized)
    'roof': (220, 20, 60),        # Red
    'gable': (255, 105, 180),     # Pink
    'gutter': (0, 255, 255),      # Cyan
}

TRADE_GROUPS = {
    'siding': ['building', 'exterior wall', 'window', 'door', 'garage'],
    'roofing': ['roof', 'gable'],
    'windows': ['window'],
    'doors': ['door', 'garage'],
    'gutters': ['roof'],
    'all': ['window', 'door', 'garage', 'building', 'exterior wall', 'roof', 'gable']
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
    scale_ratio = float(page.get('scale_ratio') or 48)
    dpi = int(page.get('dpi') or DEFAULT_DPI)
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
            trade_filter = TRADE_GROUPS.get(trade, TRADE_GROUPS['all'])
            print(f"Generating {trade} markup, filter={trade_filter}, preds={len(predictions)}", flush=True)
            
            # Generate markup
            marked_img, totals = generate_markup_image(
                image_data, predictions, scale_ratio, dpi,
                trade_filter=trade_filter, show_dimensions=True, show_labels=True
            )
            print(f"Generated image, totals={totals}", flush=True)
            
            # Save to buffer
            buffer = BytesIO()
            marked_img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            print(f"Saved to buffer, size={len(buffer.getvalue())}", flush=True)
            
            # Upload to Supabase
            filename = f"{job_id}/markup_{page_num:03d}_{trade}.png"
            markup_url = upload_to_supabase(buffer.getvalue(), filename, 'image/png')
            print(f"Upload result: {markup_url}", flush=True)
            
            if markup_url:
                markup_urls[trade] = {
                    'url': markup_url,
                    'totals': totals
                }
            else:
                errors.append(f"Upload failed for {trade}")
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"Error: {err}", flush=True)
            errors.append(f"{trade}: {str(e)}")
    
    if errors:
        print(f"Markup errors: {errors}", flush=True)
    
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
    return jsonify({"status": "healthy", "version": "3.6", "features": ["markups", "scale_extraction", "cross_reference"]})

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



# ============================================================
# TAKEOFF CALCULATION ENGINE v2.0
# ============================================================

def calculate_takeoff_for_page(page_id):
    """
    Calculate all takeoff measurements for a single elevation page.
    Stores results in:
    - extraction_detection_details (individual detections)
    - extraction_elevation_calcs (aggregated per elevation)
    """
    # Get page data
    page = supabase_request('GET', 'extraction_pages', filters={'id': f'eq.{page_id}'})
    if not page:
        return {"error": "Page not found"}
    page = page[0]
    
    job_id = page.get('job_id')
    elevation_name = page.get('elevation_name', 'unknown')
    extraction_data = page.get('extraction_data', {})
    predictions = extraction_data.get('raw_predictions', [])
    scale_ratio = float(page.get('scale_ratio') or 48)
    dpi = int(page.get('dpi') or 100)
    
    if not predictions:
        return {"error": "No predictions for this page"}
    
    # Conversion factor: pixels to real inches
    inches_per_pixel = (1.0 / dpi) * scale_ratio
    
    # Delete existing detection details for this page
    supabase_request('DELETE', 'extraction_detection_details', filters={'page_id': f'eq.{page_id}'})
    
    # Initialize counters
    counts = {
        'building': 0, 'window': 0, 'door': 0, 'garage': 0,
        'gable': 0, 'roof': 0, 'exterior_wall': 0
    }
    areas = {
        'gross_facade_sf': 0, 'window_area_sf': 0, 'door_area_sf': 0,
        'garage_area_sf': 0, 'gable_area_sf': 0, 'roof_area_sf': 0
    }
    linear = {
        'window_perimeter_lf': 0, 'window_head_lf': 0, 'window_jamb_lf': 0, 'window_sill_lf': 0,
        'door_perimeter_lf': 0, 'door_head_lf': 0, 'door_jamb_lf': 0,
        'garage_head_lf': 0, 'gable_rake_lf': 0,
        'roof_eave_lf': 0, 'roof_rake_lf': 0
    }
    confidences = []
    
    # Process each detection
    for idx, pred in enumerate(predictions):
        class_name = pred.get('class', '').lower().replace(' ', '_')
        px_width = pred.get('width', 0)
        px_height = pred.get('height', 0)
        confidence = pred.get('confidence', 0)
        
        # Calculate real dimensions
        real_width_in = px_width * inches_per_pixel
        real_height_in = px_height * inches_per_pixel
        real_width_ft = real_width_in / 12
        real_height_ft = real_height_in / 12
        
        # Calculate area (gables are triangles)
        is_triangle = (class_name == 'gable')
        if is_triangle:
            area_sf = (real_width_in * real_height_in) / 144 / 2  # Triangle = base * height / 2
        else:
            area_sf = (real_width_in * real_height_in) / 144
        
        # Calculate perimeter
        perimeter_lf = (real_width_in * 2 + real_height_in * 2) / 12
        
        # Store detection detail
        detail = {
            'job_id': job_id,
            'page_id': page_id,
            'class': class_name,
            'detection_index': idx,
            'confidence': confidence,
            'pixel_x': pred.get('x', 0),
            'pixel_y': pred.get('y', 0),
            'pixel_width': px_width,
            'pixel_height': px_height,
            'real_width_in': round(real_width_in, 2),
            'real_height_in': round(real_height_in, 2),
            'real_width_ft': round(real_width_ft, 4),
            'real_height_ft': round(real_height_ft, 4),
            'area_sf': round(area_sf, 2),
            'perimeter_lf': round(perimeter_lf, 2),
            'is_triangle': is_triangle
        }
        supabase_request('POST', 'extraction_detection_details', detail)
        
        confidences.append(confidence)
        
        # Aggregate by class
        if class_name in ['building', 'exterior_wall']:
            counts['building' if class_name == 'building' else 'exterior_wall'] += 1
            areas['gross_facade_sf'] += area_sf
        elif class_name == 'window':
            counts['window'] += 1
            areas['window_area_sf'] += area_sf
            linear['window_perimeter_lf'] += perimeter_lf
            linear['window_head_lf'] += real_width_ft
            linear['window_jamb_lf'] += real_height_ft * 2  # Both jambs
            linear['window_sill_lf'] += real_width_ft
        elif class_name == 'door':
            counts['door'] += 1
            areas['door_area_sf'] += area_sf
            linear['door_perimeter_lf'] += perimeter_lf
            linear['door_head_lf'] += real_width_ft
            linear['door_jamb_lf'] += real_height_ft * 2
        elif class_name == 'garage':
            counts['garage'] += 1
            areas['garage_area_sf'] += area_sf
            linear['garage_head_lf'] += real_width_ft
        elif class_name == 'gable':
            counts['gable'] += 1
            areas['gable_area_sf'] += area_sf
            # Gable rake = 2 sloped edges, approximate using pythagorean
            import math
            rake_length = math.sqrt((real_width_ft/2)**2 + real_height_ft**2) * 2
            linear['gable_rake_lf'] += rake_length
        elif class_name == 'roof':
            counts['roof'] += 1
            areas['roof_area_sf'] += area_sf
            # Roof eave = horizontal edges (top and bottom width)
            linear['roof_eave_lf'] += real_width_ft
    
    # Delete existing elevation calc for this page
    supabase_request('DELETE', 'extraction_elevation_calcs', filters={'page_id': f'eq.{page_id}'})
    
    # Store elevation calculations
    elevation_calc = {
        'job_id': job_id,
        'page_id': page_id,
        'elevation_name': elevation_name,
        'building_count': counts['building'],
        'window_count': counts['window'],
        'door_count': counts['door'],
        'garage_count': counts['garage'],
        'gable_count': counts['gable'],
        'roof_count': counts['roof'],
        'exterior_wall_count': counts['exterior_wall'],
        'gross_facade_sf': round(areas['gross_facade_sf'], 2),
        'window_area_sf': round(areas['window_area_sf'], 2),
        'door_area_sf': round(areas['door_area_sf'], 2),
        'garage_area_sf': round(areas['garage_area_sf'], 2),
        'gable_area_sf': round(areas['gable_area_sf'], 2),
        'roof_area_sf': round(areas['roof_area_sf'], 2),
        'window_perimeter_lf': round(linear['window_perimeter_lf'], 2),
        'window_head_lf': round(linear['window_head_lf'], 2),
        'window_jamb_lf': round(linear['window_jamb_lf'], 2),
        'window_sill_lf': round(linear['window_sill_lf'], 2),
        'door_perimeter_lf': round(linear['door_perimeter_lf'], 2),
        'door_head_lf': round(linear['door_head_lf'], 2),
        'door_jamb_lf': round(linear['door_jamb_lf'], 2),
        'garage_head_lf': round(linear['garage_head_lf'], 2),
        'gable_rake_lf': round(linear['gable_rake_lf'], 2),
        'roof_eave_lf': round(linear['roof_eave_lf'], 2),
        'roof_rake_lf': round(linear['roof_rake_lf'], 2),
        'scale_ratio': scale_ratio,
        'dpi': dpi,
        'confidence_avg': round(sum(confidences) / len(confidences), 4) if confidences else 0
    }
    supabase_request('POST', 'extraction_elevation_calcs', elevation_calc)
    
    return {
        "success": True,
        "page_id": page_id,
        "elevation": elevation_name,
        "counts": counts,
        "areas": {k: round(v, 2) for k, v in areas.items()},
        "linear": {k: round(v, 2) for k, v in linear.items()},
        "net_siding_sf": round(areas['gross_facade_sf'] - areas['window_area_sf'] - areas['door_area_sf'] - areas['garage_area_sf'] + areas['gable_area_sf'], 2)
    }


def calculate_takeoff_for_job(job_id):
    """
    Calculate takeoff for all elevation pages in a job.
    Aggregates results into extraction_job_totals.
    """
    # Get all elevation pages
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.elevation',
        'status': 'eq.complete',
        'order': 'page_number'
    })
    
    if not pages:
        return {"error": "No elevation pages found"}
    
    results = []
    elevations_processed = []
    
    # Process each elevation
    for page in pages:
        result = calculate_takeoff_for_page(page['id'])
        results.append(result)
        if result.get('success'):
            elevations_processed.append(result.get('elevation', 'unknown'))
    
    # Aggregate totals from elevation_calcs
    calcs = supabase_request('GET', 'extraction_elevation_calcs', filters={
        'job_id': f'eq.{job_id}'
    })
    
    if not calcs:
        return {"error": "No calculations found"}
    
    totals = {
        'elevation_count': len(calcs),
        'elevations_processed': elevations_processed,
        'total_windows': sum(c.get('window_count', 0) for c in calcs),
        'total_doors': sum(c.get('door_count', 0) for c in calcs),
        'total_garages': sum(c.get('garage_count', 0) for c in calcs),
        'total_gables': sum(c.get('gable_count', 0) for c in calcs),
        'total_gross_facade_sf': round(sum(float(c.get('gross_facade_sf', 0)) for c in calcs), 2),
        'total_openings_sf': round(sum(float(c.get('window_area_sf', 0)) + float(c.get('door_area_sf', 0)) + float(c.get('garage_area_sf', 0)) for c in calcs), 2),
        'total_net_siding_sf': round(sum(float(c.get('net_siding_sf', 0)) for c in calcs), 2),
        'total_gable_sf': round(sum(float(c.get('gable_area_sf', 0)) for c in calcs), 2),
        'total_roof_sf': round(sum(float(c.get('roof_area_sf', 0)) for c in calcs), 2),
        'total_window_head_lf': round(sum(float(c.get('window_head_lf', 0)) for c in calcs), 2),
        'total_window_jamb_lf': round(sum(float(c.get('window_jamb_lf', 0)) for c in calcs), 2),
        'total_window_sill_lf': round(sum(float(c.get('window_sill_lf', 0)) for c in calcs), 2),
        'total_window_perimeter_lf': round(sum(float(c.get('window_perimeter_lf', 0)) for c in calcs), 2),
        'total_door_head_lf': round(sum(float(c.get('door_head_lf', 0)) for c in calcs), 2),
        'total_door_jamb_lf': round(sum(float(c.get('door_jamb_lf', 0)) for c in calcs), 2),
        'total_door_perimeter_lf': round(sum(float(c.get('door_perimeter_lf', 0)) for c in calcs), 2),
        'total_garage_head_lf': round(sum(float(c.get('garage_head_lf', 0)) for c in calcs), 2),
        'total_gable_rake_lf': round(sum(float(c.get('gable_rake_lf', 0)) for c in calcs), 2),
        'total_roof_eave_lf': round(sum(float(c.get('roof_eave_lf', 0)) for c in calcs), 2)
    }
    
    # Delete existing job totals
    supabase_request('DELETE', 'extraction_job_totals', filters={'job_id': f'eq.{job_id}'})
    
    # Store job totals
    totals['job_id'] = job_id
    supabase_request('POST', 'extraction_job_totals', totals)
    
    return {
        "success": True,
        "job_id": job_id,
        "elevations_processed": len(results),
        "totals": totals,
        "per_elevation": results
    }


@app.route('/calculate-takeoff', methods=['POST'])
def calculate_takeoff():
    """Calculate takeoff measurements from detections"""
    data = request.json
    
    if data.get('page_id'):
        return jsonify(calculate_takeoff_for_page(data['page_id']))
    elif data.get('job_id'):
        return jsonify(calculate_takeoff_for_job(data['job_id']))
    
    return jsonify({"error": "page_id or job_id required"}), 400


@app.route('/job-takeoff', methods=['GET'])
def get_job_takeoff():
    """Get calculated takeoff totals for a job"""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    totals = supabase_request('GET', 'extraction_job_totals', filters={'job_id': f'eq.{job_id}'})
    if not totals:
        return jsonify({"error": "No takeoff calculated. Run /calculate-takeoff first"}), 404
    
    elevations = supabase_request('GET', 'extraction_elevation_calcs', filters={
        'job_id': f'eq.{job_id}',
        'order': 'elevation_name'
    })
    
    return jsonify({
        "totals": totals[0],
        "elevations": elevations or []
    })



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
        scale_ratio = float(page.get('scale_ratio') or 48)
        
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


# ============================================================
# FLOOR PLAN CORNER ANALYSIS
# ============================================================

def analyze_floor_plan_corners(image_url):
    """
    Use Claude Vision to count outside and inside corners from a floor plan.
    """
    try:
        # Download and encode image
        response = requests.get(image_url)
        if response.status_code != 200:
            return {"error": f"Failed to download image: {response.status_code}"}
        
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        prompt = """Analyze this architectural floor plan. Focus ONLY on the EXTERIOR BUILDING PERIMETER (the outermost walls that define the building footprint - typically shown as thick black lines).

TASK: Trace the exterior perimeter and count corners.

DEFINITIONS:
- OUTSIDE CORNER (convex): Where two exterior walls meet and point OUTWARD (away from building interior). These are the "bump out" corners.
- INSIDE CORNER (concave): Where two exterior walls meet and point INWARD (into the building). These are the "cut in" corners, like where an L-shape occurs.

DO NOT COUNT:
- Interior wall corners (rooms, closets, etc.)
- Window or door openings
- Garage door openings
- Porch railings or deck edges (only count if they have siding)

If there are multiple floor plans or units on this sheet, analyze each separately.

Return ONLY valid JSON in this exact format:
{
  "floor_plans": [
    {
      "name": "Unit/Floor Plan Name",
      "outside_corners": <integer>,
      "inside_corners": <integer>,
      "confidence": "high/medium/low",
      "notes": "any observations about complexity"
    }
  ],
  "total_outside_corners": <sum of all>,
  "total_inside_corners": <sum of all>
}"""

        api_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            },
            timeout=60
        )
        
        if api_response.status_code == 200:
            result = api_response.json()
            content = result.get('content', [{}])[0].get('text', '{}')
            # Clean up response - extract JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            return json.loads(content.strip())
        else:
            return {"error": f"API error: {api_response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}


@app.route('/analyze-floor-plan', methods=['POST'])
def analyze_floor_plan():
    """
    Analyze floor plan(s) to count corners.
    
    POST body:
    - page_id: specific floor plan page
    - job_id: analyze all floor plans for a job
    """
    data = request.json
    
    if data.get('page_id'):
        # Single page analysis
        page = supabase_request('GET', 'extraction_pages', filters={'id': f"eq.{data['page_id']}"})
        if not page:
            return jsonify({"error": "Page not found"}), 404
        page = page[0]
        
        if page.get('page_type') != 'floor_plan':
            return jsonify({"error": f"Page is type '{page.get('page_type')}', not floor_plan"}), 400
        
        result = analyze_floor_plan_corners(page['image_url'])
        result['page_id'] = data['page_id']
        result['page_number'] = page.get('page_number')
        return jsonify(result)
    
    elif data.get('job_id'):
        # Analyze all floor plans for job
        pages = supabase_request('GET', 'extraction_pages', filters={
            'job_id': f"eq.{data['job_id']}",
            'page_type': 'eq.floor_plan',
            'order': 'page_number'
        })
        
        if not pages:
            return jsonify({"error": "No floor plan pages found"}), 404
        
        results = []
        total_outside = 0
        total_inside = 0
        
        for page in pages:
            print(f"Analyzing floor plan page {page.get('page_number')}...")
            result = analyze_floor_plan_corners(page['image_url'])
            result['page_id'] = page['id']
            result['page_number'] = page.get('page_number')
            results.append(result)
            
            if 'total_outside_corners' in result:
                total_outside += result.get('total_outside_corners', 0)
                total_inside += result.get('total_inside_corners', 0)
        
        # Get wall height from job totals for LF calculation
        job_totals = supabase_request('GET', 'extraction_job_totals', filters={
            'job_id': f"eq.{data['job_id']}"
        })
        
        wall_height = 9.0  # Default
        if job_totals and job_totals[0].get('total_gross_facade_sf') and job_totals[0].get('total_roof_eave_lf'):
            facade = float(job_totals[0]['total_gross_facade_sf'])
            eave = float(job_totals[0]['total_roof_eave_lf'])
            if eave > 0:
                wall_height = facade / eave
        
        # Calculate LF
        outside_lf = round(total_outside * wall_height, 2)
        inside_lf = round(total_inside * wall_height, 2)
        
        # Update job totals
        if job_totals:
            supabase_request('PATCH', 'extraction_job_totals', 
                filters={'job_id': f"eq.{data['job_id']}"},
                data={
                    'outside_corners_count': total_outside,
                    'inside_corners_count': total_inside,
                    'outside_corners_lf': outside_lf,
                    'inside_corners_lf': inside_lf,
                    'corner_source': 'floor_plan_analysis'
                }
            )
        
        return jsonify({
            "job_id": data['job_id'],
            "floor_plans_analyzed": len(results),
            "total_outside_corners": total_outside,
            "total_inside_corners": total_inside,
            "wall_height_used": round(wall_height, 2),
            "outside_corners_lf": outside_lf,
            "inside_corners_lf": inside_lf,
            "per_page_results": results
        })
    
    return jsonify({"error": "page_id or job_id required"}), 400




# ============================================================
# COMPREHENSIVE MARKUP GENERATION
# ============================================================

def generate_comprehensive_markup(page_id):
    """
    Generate a single comprehensive markup image for an elevation.
    Shows all detections with tinted fills, dimensions and a summary legend.
    """
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    # Get page data
    page = supabase_request('GET', 'extraction_pages', filters={'id': f'eq.{page_id}'})
    if not page:
        return {"error": "Page not found"}
    page = page[0]
    
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
    
    # Try to load a font, fall back to default
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
        'building': {'outline': '#22C55E', 'fill': (34, 197, 94, 60)},       # Green
        'exterior_wall': {'outline': '#22C55E', 'fill': (34, 197, 94, 60)},  # Green
        'window': {'outline': '#3B82F6', 'fill': (59, 130, 246, 80)},        # Blue
        'door': {'outline': '#F97316', 'fill': (249, 115, 22, 80)},          # Orange
        'garage': {'outline': '#A855F7', 'fill': (168, 85, 247, 80)},        # Purple
        'roof': {'outline': '#EF4444', 'fill': (239, 68, 68, 50)},           # Red
        'gable': {'outline': '#EC4899', 'fill': (236, 72, 153, 70)},         # Pink
    }
    
    # Count by class for legend
    class_counts = {}
    class_areas = {}
    
    # Sort detections: draw larger items first (buildings), then smaller (windows/doors)
    # This ensures windows/doors appear on top of buildings
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
        
        # Get pixel coordinates
        px = float(det.get('pixel_x', 0))
        py = float(det.get('pixel_y', 0))
        pw = float(det.get('pixel_width', 0))
        ph = float(det.get('pixel_height', 0))
        
        # Calculate bounding box (x,y is center)
        x1 = px - pw/2
        y1 = py - ph/2
        x2 = px + pw/2
        y2 = py + ph/2
        
        # Get real dimensions
        width_ft = float(det.get('real_width_ft', 0))
        height_ft = float(det.get('real_height_ft', 0))
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
    
    # Create another overlay for labels (so they're on top of fills)
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
        
        x1 = px - pw/2
        y1 = py - ph/2
        
        width_ft = float(det.get('real_width_ft', 0))
        height_ft = float(det.get('real_height_ft', 0))
        area_sf = float(det.get('area_sf', 0))
        
        # Draw dimension label with background for readability
        if cls in ['window', 'door', 'garage']:
            label = f"{width_ft:.1f}'x{height_ft:.1f}'"
            label_x = x1 + 3
            label_y = y1 + 2
            # Draw label background
            bbox = label_draw.textbbox((label_x, label_y), label, font=font_small)
            label_draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill=(255, 255, 255, 200))
            label_draw.text((label_x, label_y), label, fill=colors['outline'], font=font_small)
        elif cls in ['building', 'exterior_wall', 'roof', 'gable']:
            label = f"{area_sf:.0f} SF"
            label_x = x1 + 5
            label_y = y1 + 5
            bbox = label_draw.textbbox((label_x, label_y), label, font=font)
            label_draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill=(255, 255, 255, 220))
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
            
            # Color box with fill
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
    
    upload_url = f"{SUPABASE_URL}/storage/v1/object/extraction-markups/{filepath}"
    upload_response = requests.post(
        upload_url,
        headers={
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "image/png",
            "x-upsert": "true"
        },
        data=buffer.getvalue()
    )
    
    if upload_response.status_code in [200, 201]:
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/extraction-markups/{filepath}"
        return {
            "success": True,
            "page_id": page_id,
            "elevation": elevation_name,
            "markup_url": public_url,
            "summary": {
                "gross_facade_sf": gross_facade,
                "openings_sf": openings,
                "gable_sf": gable,
                "net_siding_sf": net_siding,
                "detections": class_counts
            }
        }
    else:
        return {"error": f"Upload failed: {upload_response.status_code}", "detail": upload_response.text}



@app.route('/comprehensive-markup', methods=['POST'])
def comprehensive_markup():
    """
    Generate comprehensive markup for elevation page(s).
    
    POST body:
    - page_id: specific elevation page
    - job_id: all elevations for a job
    """
    data = request.json
    
    if data.get('page_id'):
        return jsonify(generate_comprehensive_markup(data['page_id']))
    
    elif data.get('job_id'):
        # Get all elevation pages
        pages = supabase_request('GET', 'extraction_pages', filters={
            'job_id': f"eq.{data['job_id']}",
            'page_type': 'eq.elevation',
            'order': 'page_number'
        })
        
        if not pages:
            return jsonify({"error": "No elevation pages found"}), 404
        
        results = []
        for page in pages:
            result = generate_comprehensive_markup(page['id'])
            results.append(result)
        
        return jsonify({
            "job_id": data['job_id'],
            "markups_generated": len([r for r in results if r.get('success')]),
            "results": results
        })

    return jsonify({"error": "page_id or job_id required"}), 400


@app.route('/generate-facade-markup', methods=['POST'])
def generate_facade_markup():
    """
    Generate facade markup from verified detection data.
    This creates a clean visualization AFTER human review.
    """
    data = request.json
    job_id = data.get('job_id')
    page_id = data.get('page_id')

    if not job_id:
        return jsonify({"error": "job_id required"}), 400

    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json'
    }

    page_query = f"{SUPABASE_URL}/rest/v1/extraction_pages?job_id=eq.{job_id}&select=id,page_number,original_image_url,image_url"
    if page_id:
        page_query += f"&id=eq.{page_id}"

    page_response = requests.get(page_query, headers=headers)
    pages = page_response.json()

    if not pages:
        return jsonify({"error": "No pages found"}), 404

    results = []

    for page in pages:
        pg_id = page['id']
        page_number = page['page_number']
        image_url = page.get('original_image_url') or page.get('image_url')

        if not image_url:
            results.append({"page_number": page_number, "error": "No image URL"})
            continue

        det_query = f"{SUPABASE_URL}/rest/v1/extraction_detection_details?page_id=eq.{pg_id}&status=neq.deleted&select=class,pixel_x,pixel_y,pixel_width,pixel_height,area_sf,real_width_ft,real_height_ft"
        det_response = requests.get(det_query, headers=headers)
        detections = det_response.json()

        try:
            img_response = requests.get(image_url)
            img = Image.open(BytesIO(img_response.content)).convert('RGBA')
        except Exception as e:
            results.append({"page_number": page_number, "error": f"Failed to load image: {e}"})
            continue

        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        gross_facade_sf = 0
        openings_sf = 0
        window_count = 0
        door_count = 0
        garage_count = 0

        buildings = []
        openings = []

        for det in detections:
            cls = det.get('class', '').lower()
            cx = det.get('pixel_x', 0)
            cy = det.get('pixel_y', 0)
            w = det.get('pixel_width', 0)
            h = det.get('pixel_height', 0)
            area = det.get('area_sf', 0)

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            if cls == 'building':
                buildings.append((x1, y1, x2, y2))
                gross_facade_sf += area
            elif cls in ['window', 'door', 'garage']:
                openings.append((x1, y1, x2, y2, cls))
                openings_sf += area
                if cls == 'window':
                    window_count += 1
                elif cls == 'door':
                    door_count += 1
                elif cls == 'garage':
                    garage_count += 1

        net_siding_sf = gross_facade_sf - openings_sf

        for (x1, y1, x2, y2) in buildings:
            draw.rectangle([x1, y1, x2, y2], fill=(59, 130, 246, 100), outline=(59, 130, 246, 255), width=3)

        color_map = {
            'window': (249, 115, 22, 150),
            'door': (34, 197, 94, 150),
            'garage': (234, 179, 8, 150)
        }

        for (x1, y1, x2, y2, cls) in openings:
            color = color_map.get(cls, (255, 0, 0, 150))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255, 255), width=2)

        result_img = Image.alpha_composite(img, overlay)
        draw_result = ImageDraw.Draw(result_img)

        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        summary_lines = [
            f"NET FACADE: {net_siding_sf:,.0f} SF",
            f"Gross: {gross_facade_sf:,.0f} SF",
            f"Openings: {openings_sf:,.0f} SF",
            f"Win: {window_count} | Door: {door_count} | Garage: {garage_count}"
        ]

        draw_result.rectangle([10, 10, 350, 120], fill=(0, 0, 0, 200))

        y_pos = 20
        draw_result.text((20, y_pos), summary_lines[0], fill=(59, 130, 246), font=font_large)
        y_pos += 30
        for line in summary_lines[1:]:
            draw_result.text((20, y_pos), line, fill=(255, 255, 255), font=font_small)
            y_pos += 22

        legend_y = img.size[1] - 40
        legend_items = [("Facade", (59, 130, 246)), ("Window", (249, 115, 22)), ("Door", (34, 197, 94)), ("Garage", (234, 179, 8))]

        x_pos = 20
        for label, color in legend_items:
            draw_result.rectangle([x_pos, legend_y, x_pos + 20, legend_y + 20], fill=color)
            draw_result.text((x_pos + 25, legend_y), label, fill=(255, 255, 255), font=font_small)
            x_pos += 100

        buffer = BytesIO()
        result_img.convert('RGB').save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        image_bytes = buffer.getvalue()

        filename = f"{job_id}/facade_markup_{page_number:03d}.png"
        upload_url = upload_to_supabase(image_bytes, filename, 'image/png')

        results.append({
            "page_number": page_number,
            "page_id": pg_id,
            "markup_url": upload_url,
            "net_siding_sf": round(net_siding_sf, 2),
            "gross_facade_sf": round(gross_facade_sf, 2),
            "openings_sf": round(openings_sf, 2),
            "window_count": window_count,
            "door_count": door_count,
            "garage_count": garage_count
        })

    return jsonify({
        "success": True,
        "job_id": job_id,
        "pages": results
    })
