"""
Extraction API v3.1 - Cross-Reference & Derived Measurements
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
    
    simple_frac = r'(\d+)/(\d+)\s*(?:SCALE|=)'
    match = re.search(simple_frac, notation)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        return 12 / (numerator / denominator)
    
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
        'general': 'other', 'notes': 'other', 'specifications': 'other', 'spec': 'other',
    }
    return mappings.get(cleaned, 'other')


def calculate_derived_measurements(width_in, height_in, qty, element_type='window'):
    """Calculate all derived measurements for an opening"""
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


def calculate_real_measurements(predictions, scale_ratio, dpi=DEFAULT_DPI):
    if not scale_ratio or scale_ratio <= 0:
        scale_ratio = 48
        scale_warning = "Using default scale 1/4\"=1'-0\""
    else:
        scale_warning = None
    
    inches_per_pixel = 1.0 / dpi
    real_inches_per_pixel = inches_per_pixel * scale_ratio
    sqft_per_sq_pixel = (real_inches_per_pixel ** 2) / 144
    
    results = {
        'scale_used': scale_ratio,
        'scale_warning': scale_warning,
        'dpi': dpi,
        'conversion': {
            'inches_per_pixel': round(inches_per_pixel, 6),
            'real_inches_per_pixel': round(real_inches_per_pixel, 4),
            'sqft_per_sq_pixel': round(sqft_per_sq_pixel, 6)
        },
        'items': {'windows': [], 'doors': [], 'garages': [], 'buildings': [], 'roofs': [], 'gables': []},
        'counts': {'window': 0, 'door': 0, 'garage': 0, 'building': 0, 'roof': 0, 'gable': 0},
        'areas': {
            'window_sqft': 0, 'door_sqft': 0, 'garage_sqft': 0,
            'building_sqft': 0, 'roof_sqft': 0, 'gable_sqft': 0,
            'gross_wall_sqft': 0, 'net_siding_sqft': 0, 'openings_sqft': 0
        }
    }
    
    for pred in predictions:
        class_name = pred.get('class', '').lower()
        width_px = pred.get('width', 0)
        height_px = pred.get('height', 0)
        x, y = pred.get('x', 0), pred.get('y', 0)
        
        width_inches = width_px * real_inches_per_pixel
        height_inches = height_px * real_inches_per_pixel
        area_sqft = width_px * height_px * sqft_per_sq_pixel
        
        item = {
            'width_inches': round(width_inches, 1),
            'height_inches': round(height_inches, 1),
            'width_ft': round(width_inches / 12, 2),
            'height_ft': round(height_inches / 12, 2),
            'area_sqft': round(area_sqft, 1),
            'perimeter_lf': round((width_inches + height_inches) * 2 / 12, 1),
            'pixel_width': width_px,
            'pixel_height': height_px,
            'pixel_x': x,
            'pixel_y': y,
            'confidence': pred.get('confidence', 0),
            'tag': None  # Will be filled by tag detection
        }
        
        if class_name == 'window':
            results['items']['windows'].append(item)
            results['counts']['window'] += 1
            results['areas']['window_sqft'] += area_sqft
        elif class_name == 'door':
            results['items']['doors'].append(item)
            results['counts']['door'] += 1
            results['areas']['door_sqft'] += area_sqft
        elif class_name == 'garage':
            results['items']['garages'].append(item)
            results['counts']['garage'] += 1
            results['areas']['garage_sqft'] += area_sqft
        elif class_name == 'building':
            results['items']['buildings'].append(item)
            results['counts']['building'] += 1
            results['areas']['building_sqft'] += area_sqft
        elif class_name == 'roof':
            results['items']['roofs'].append(item)
            results['counts']['roof'] += 1
            results['areas']['roof_sqft'] += area_sqft
        elif class_name == 'gable':
            results['items']['gables'].append(item)
            results['counts']['gable'] += 1
            results['areas']['gable_sqft'] += area_sqft
    
    results['areas']['gross_wall_sqft'] = round(results['areas']['building_sqft'], 1)
    results['areas']['openings_sqft'] = round(
        results['areas']['window_sqft'] + results['areas']['door_sqft'] + results['areas']['garage_sqft'], 1)
    results['areas']['net_siding_sqft'] = round(
        results['areas']['gross_wall_sqft'] - results['areas']['openings_sqft'], 1)
    
    for key in results['areas']:
        results['areas'][key] = round(results['areas'][key], 1)
    
    results['validation'] = []
    for win in results['items']['windows']:
        if win['width_inches'] > 180:
            results['validation'].append(f"Window {win['width_inches']}\" wide seems too large")
    for door in results['items']['doors']:
        if door['height_inches'] > 144:
            results['validation'].append(f"Door {door['height_inches']}\" tall seems too large")
    
    return results


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
  "scale_notation": "exact scale text from drawing, e.g. 1/4\" = 1'-0\" or 1:48 or null if not found",
  "scale_location": "where scale was found",
  "contains_schedule": true | false,
  "schedule_type": "window" | "door" | "window_and_door" | null,
  "notes": "brief description"
}

page_type MUST be exactly one of: elevation, schedule, floor_plan, section, detail, cover, site_plan, other
Look carefully for scale notation in title block or near drawing title."""

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
        print(f"Claude classify error: {e}", flush=True)
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
  "windows": [
    {"tag": "W1", "width_inches": 36, "height_inches": 48, "type": "DH", "qty": 4, "notes": "tempered glass"}
  ],
  "doors": [
    {"tag": "D1", "width_inches": 36, "height_inches": 80, "type": "Entry", "qty": 1, "notes": "solid core"}
  ],
  "raw_text": "any additional schedule text"
}

Read ALL rows in the schedule table. Convert dimensions to inches (36" or 3'-0" both = 36 inches)."""

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


def build_cross_references(job_id):
    """
    Build cross-references between schedule data and elevation detections.
    Calculate derived measurements for each opening.
    """
    print(f"[{job_id}] Building cross-references...", flush=True)
    
    # Get all schedule data for this job
    schedule_pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.schedule',
        'status': 'eq.complete',
        'order': 'page_number'
    })
    
    if not schedule_pages:
        print(f"[{job_id}] No schedule pages found", flush=True)
        return {"error": "No schedule data"}
    
    # Aggregate all schedule items, de-duplicating by tag
    windows_by_tag = {}
    doors_by_tag = {}
    
    for page in schedule_pages:
        extraction_data = page.get('extraction_data', {})
        page_id = page.get('id')
        
        for window in extraction_data.get('windows', []):
            tag = window.get('tag', 'UNKNOWN')
            if tag not in windows_by_tag:
                windows_by_tag[tag] = {
                    'tag': tag,
                    'width_inches': window.get('width_inches', 0),
                    'height_inches': window.get('height_inches', 0),
                    'type': window.get('type', ''),
                    'qty': window.get('qty', 1),
                    'notes': window.get('notes', ''),
                    'schedule_page_id': page_id
                }
            else:
                # Update qty if higher (some schedules show totals)
                if window.get('qty', 1) > windows_by_tag[tag]['qty']:
                    windows_by_tag[tag]['qty'] = window.get('qty', 1)
        
        for door in extraction_data.get('doors', []):
            tag = door.get('tag', 'UNKNOWN')
            if tag not in doors_by_tag:
                doors_by_tag[tag] = {
                    'tag': tag,
                    'width_inches': door.get('width_inches', 0),
                    'height_inches': door.get('height_inches', 0),
                    'type': door.get('type', ''),
                    'qty': door.get('qty', 1),
                    'notes': door.get('notes', ''),
                    'schedule_page_id': page_id
                }
            else:
                if door.get('qty', 1) > doors_by_tag[tag]['qty']:
                    doors_by_tag[tag]['qty'] = door.get('qty', 1)
    
    print(f"[{job_id}] Found {len(windows_by_tag)} window types, {len(doors_by_tag)} door types", flush=True)
    
    # Get elevation detection counts
    elevation_pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.elevation',
        'status': 'eq.complete',
        'order': 'page_number'
    })
    
    total_detected_windows = 0
    total_detected_doors = 0
    elevation_page_ids = []
    
    for page in elevation_pages or []:
        extraction_data = page.get('extraction_data', {})
        measurements = extraction_data.get('measurements', {})
        counts = measurements.get('counts', {})
        total_detected_windows += counts.get('window', 0)
        total_detected_doors += counts.get('door', 0)
        elevation_page_ids.append(page.get('id'))
    
    print(f"[{job_id}] Detected on elevations: {total_detected_windows} windows, {total_detected_doors} doors", flush=True)
    
    # Clear existing cross-refs for this job
    supabase_request('DELETE', 'extraction_cross_refs', filters={'job_id': f'eq.{job_id}'})
    
    # Create cross-reference entries with derived measurements
    cross_refs = []
    
    # Windows
    for tag, window in windows_by_tag.items():
        derived = calculate_derived_measurements(
            window['width_inches'],
            window['height_inches'],
            window['qty'],
            'window'
        )
        
        cross_ref = {
            'job_id': job_id,
            'element_type': 'window',
            'tag': tag,
            'schedule_width': window['width_inches'],
            'schedule_height': window['height_inches'],
            'schedule_qty': window['qty'],
            'schedule_type': window['type'],
            'schedule_page_id': window['schedule_page_id'],
            'detected_qty': None,  # Would need tag detection to fill this
            'elevation_page_ids': [str(x) for x in elevation_page_ids],
            'head_trim_lf': derived['head_trim_lf'],
            'jamb_trim_lf': derived['jamb_trim_lf'],
            'sill_trim_lf': derived['sill_trim_lf'],
            'casing_lf': derived['casing_lf'],
            'rough_opening_width': derived['rough_opening_width'],
            'rough_opening_height': derived['rough_opening_height'],
            'head_flashing_lf': derived['head_flashing_lf'],
            'sill_pan_lf': derived['sill_pan_lf'],
            'needs_review': False,
            'review_notes': None
        }
        cross_refs.append(cross_ref)
    
    # Doors
    for tag, door in doors_by_tag.items():
        derived = calculate_derived_measurements(
            door['width_inches'],
            door['height_inches'],
            door['qty'],
            'door'
        )
        
        cross_ref = {
            'job_id': job_id,
            'element_type': 'door',
            'tag': tag,
            'schedule_width': door['width_inches'],
            'schedule_height': door['height_inches'],
            'schedule_qty': door['qty'],
            'schedule_type': door['type'],
            'schedule_page_id': door['schedule_page_id'],
            'detected_qty': None,
            'elevation_page_ids': [str(x) for x in elevation_page_ids],
            'head_trim_lf': derived['head_trim_lf'],
            'jamb_trim_lf': derived['jamb_trim_lf'],
            'sill_trim_lf': derived['sill_trim_lf'],
            'casing_lf': derived['casing_lf'],
            'rough_opening_width': derived['rough_opening_width'],
            'rough_opening_height': derived['rough_opening_height'],
            'head_flashing_lf': derived['head_flashing_lf'],
            'sill_pan_lf': derived['sill_pan_lf'],
            'needs_review': False,
            'review_notes': None
        }
        cross_refs.append(cross_ref)
    
    # Insert cross-refs
    for ref in cross_refs:
        supabase_request('POST', 'extraction_cross_refs', ref)
    
    print(f"[{job_id}] Created {len(cross_refs)} cross-reference entries", flush=True)
    
    # Build takeoff summary
    summary = {
        'job_id': job_id,
        'total_windows': sum(w['qty'] for w in windows_by_tag.values()),
        'total_window_sqft': sum(
            (w['width_inches'] * w['height_inches'] * w['qty']) / 144 
            for w in windows_by_tag.values()
        ),
        'total_window_head_trim_lf': sum(
            (w['width_inches'] * w['qty']) / 12 
            for w in windows_by_tag.values()
        ),
        'total_window_jamb_trim_lf': sum(
            (w['height_inches'] * 2 * w['qty']) / 12 
            for w in windows_by_tag.values()
        ),
        'total_window_sill_trim_lf': sum(
            (w['width_inches'] * w['qty']) / 12 
            for w in windows_by_tag.values()
        ),
        'total_doors': sum(d['qty'] for d in doors_by_tag.values()),
        'total_door_sqft': sum(
            (d['width_inches'] * d['height_inches'] * d['qty']) / 144 
            for d in doors_by_tag.values()
        ),
        'total_door_head_trim_lf': sum(
            (d['width_inches'] * d['qty']) / 12 
            for d in doors_by_tag.values()
        ),
        'total_door_jamb_trim_lf': sum(
            (d['height_inches'] * 2 * d['qty']) / 12 
            for d in doors_by_tag.values()
        ),
        'windows_by_tag': {tag: {**w, 'schedule_page_id': str(w['schedule_page_id'])} for tag, w in windows_by_tag.items()},
        'doors_by_tag': {tag: {**d, 'schedule_page_id': str(d['schedule_page_id'])} for tag, d in doors_by_tag.items()},
        'schedule_vs_detected_match': None,
        'discrepancies': {
            'schedule_windows': sum(w['qty'] for w in windows_by_tag.values()),
            'detected_windows': total_detected_windows,
            'schedule_doors': sum(d['qty'] for d in doors_by_tag.values()),
            'detected_doors': total_detected_doors
        }
    }
    
    # Round numeric values
    for key in ['total_window_sqft', 'total_window_head_trim_lf', 'total_window_jamb_trim_lf', 
                'total_window_sill_trim_lf', 'total_door_sqft', 'total_door_head_trim_lf', 
                'total_door_jamb_trim_lf']:
        summary[key] = round(summary[key], 2)
    
    # Upsert summary
    existing = supabase_request('GET', 'extraction_takeoff_summary', filters={'job_id': f'eq.{job_id}'})
    if existing:
        supabase_request('PATCH', 'extraction_takeoff_summary', summary, {'job_id': f'eq.{job_id}'})
    else:
        supabase_request('POST', 'extraction_takeoff_summary', summary)
    
    print(f"[{job_id}] Cross-reference complete!", flush=True)
    
    return {
        "success": True,
        "windows_count": len(windows_by_tag),
        "doors_count": len(doors_by_tag),
        "cross_refs_created": len(cross_refs),
        "summary": summary
    }


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
        print(f"[{job_id}] {total_pages} pages", flush=True)
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
                print(f"[{job_id}] {pages_converted}/{total_pages}", flush=True)
            except Exception as e:
                print(f"[{job_id}] Chunk error: {e}", flush=True)
        try:
            os.unlink(tmp_path)
        except:
            pass
        update_job(job_id, {'status': 'converted'})
        print(f"[{job_id}] Done!", flush=True)
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def classify_job_background(job_id):
    try:
        print(f"[{job_id}] Starting classification...", flush=True)
        update_job(job_id, {'status': 'classifying'})
        pages = supabase_request('GET', 'extraction_pages', filters={'job_id': f'eq.{job_id}', 'status': 'eq.pending', 'order': 'page_number'})
        if not pages:
            print(f"[{job_id}] No pending pages!", flush=True)
            return
        print(f"[{job_id}] {len(pages)} pages to classify", flush=True)
        
        elevation_count = schedule_count = floor_plan_count = other_count = 0
        scales_found = []
        
        for i, page in enumerate(pages):
            page_id = page.get('id')
            page_num = page.get('page_number', i+1)
            image_url = page.get('image_url')
            
            print(f"[{job_id}] Page {page_num}...", flush=True)
            classification = classify_page_with_claude(image_url)
            
            page_type = normalize_page_type(classification.get('page_type'))
            confidence = classification.get('confidence', 0)
            elevation_name = classification.get('elevation_name')
            scale_notation = classification.get('scale_notation')
            scale_ratio = classification.get('scale_ratio')
            
            print(f"[{job_id}] Page {page_num}: {page_type}, scale={scale_notation}", flush=True)
            
            if scale_ratio:
                scales_found.append({'page': page_num, 'notation': scale_notation, 'ratio': scale_ratio})
            
            if page_type == 'elevation': elevation_count += 1
            elif page_type == 'schedule': schedule_count += 1
            elif page_type == 'floor_plan': floor_plan_count += 1
            else: other_count += 1
            
            update_data = {
                'page_type': page_type,
                'page_type_confidence': confidence,
                'status': 'classified',
                'scale_notation': scale_notation,
                'scale_ratio': scale_ratio
            }
            if elevation_name:
                update_data['elevation_name'] = elevation_name
            
            update_page(page_id, update_data)
            update_job(job_id, {'pages_classified': i + 1})
            
            if (i + 1) % MAX_CONCURRENT_CLAUDE == 0:
                time.sleep(BATCH_DELAY_SECONDS)
        
        default_scale = None
        if scales_found:
            from collections import Counter
            ratios = [s['ratio'] for s in scales_found if s['ratio']]
            if ratios:
                default_scale = Counter(ratios).most_common(1)[0][0]
        
        update_job(job_id, {
            'status': 'classified',
            'elevation_count': elevation_count,
            'schedule_count': schedule_count,
            'floor_plan_count': floor_plan_count,
            'other_count': other_count,
            'default_scale_ratio': default_scale
        })
        print(f"[{job_id}] Done! E:{elevation_count} S:{schedule_count} F:{floor_plan_count} O:{other_count}", flush=True)
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def process_job_background(job_id, scale_override=None):
    try:
        print(f"[{job_id}] Processing...", flush=True)
        update_job(job_id, {'status': 'processing'})
        
        job = supabase_request('GET', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
        default_scale = job[0].get('default_scale_ratio') if job else None
        job_dpi = job[0].get('plan_dpi', DEFAULT_DPI) if job else DEFAULT_DPI
        
        pages = supabase_request('GET', 'extraction_pages', filters={'job_id': f'eq.{job_id}', 'status': 'eq.classified', 'order': 'page_number'})
        if not pages:
            print(f"[{job_id}] No classified pages!", flush=True)
            return
        
        elevation_pages = [p for p in pages if p.get('page_type') == 'elevation']
        schedule_pages = [p for p in pages if p.get('page_type') == 'schedule']
        print(f"[{job_id}] {len(elevation_pages)} elevations, {len(schedule_pages)} schedules", flush=True)
        
        totals = {
            'total_net_siding_sqft': 0, 'total_gross_wall_sqft': 0, 'total_openings_sqft': 0,
            'total_windows': 0, 'total_doors': 0, 'total_garages': 0,
            'scales_used': [], 'validation_warnings': []
        }
        processed = 0
        
        for page in elevation_pages:
            page_num = page.get('page_number')
            scale_ratio = scale_override or page.get('scale_ratio') or default_scale or 48
            dpi = page.get('dpi') or job_dpi
            
            print(f"[{job_id}] Elevation page {page_num} scale={scale_ratio}", flush=True)
            
            detection = detect_with_roboflow(page['image_url'])
            if 'error' not in detection:
                measurements = calculate_real_measurements(detection['predictions'], scale_ratio, dpi)
                
                totals['total_net_siding_sqft'] += measurements['areas']['net_siding_sqft']
                totals['total_gross_wall_sqft'] += measurements['areas']['gross_wall_sqft']
                totals['total_openings_sqft'] += measurements['areas']['openings_sqft']
                totals['total_windows'] += measurements['counts']['window']
                totals['total_doors'] += measurements['counts']['door']
                totals['total_garages'] += measurements['counts']['garage']
                totals['scales_used'].append({'page': page_num, 'scale': scale_ratio})
                totals['validation_warnings'].extend(measurements.get('validation', []))
                
                update_page(page['id'], {
                    'status': 'complete',
                    'extraction_data': {
                        'measurements': measurements,
                        'raw_predictions': detection['predictions']
                    }
                })
            else:
                update_page(page['id'], {'status': 'failed', 'error_message': detection.get('error')})
            
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
        
        totals['total_net_siding_sqft'] = round(totals['total_net_siding_sqft'], 1)
        totals['total_gross_wall_sqft'] = round(totals['total_gross_wall_sqft'], 1)
        totals['total_openings_sqft'] = round(totals['total_openings_sqft'], 1)
        
        update_job(job_id, {'status': 'complete', 'results_summary': totals})
        print(f"[{job_id}] Complete!", flush=True)
        
        # Auto-run cross-reference
        build_cross_references(job_id)
        
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


# === ENDPOINTS ===

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "version": "3.1", "features": ["scale_extraction", "cross_reference", "derived_measurements"]})

@app.route('/parse-scale', methods=['POST'])
def parse_scale_endpoint():
    data = request.json
    notation = data.get('notation', '')
    ratio = parse_scale_notation(notation)
    return jsonify({"notation": notation, "scale_ratio": ratio})

@app.route('/classify-page', methods=['POST'])
def classify_page_endpoint():
    data = request.json
    if not data.get('image_url'):
        return jsonify({"error": "image_url required"}), 400
    return jsonify(classify_page_with_claude(data['image_url']))

@app.route('/start-job', methods=['POST'])
def start_job():
    data = request.json
    if not data.get('pdf_url') or not data.get('project_id'):
        return jsonify({"error": "pdf_url and project_id required"}), 400
    job_result = supabase_request('POST', 'extraction_jobs', {
        'project_id': data['project_id'],
        'project_name': data.get('project_name', ''),
        'source_pdf_url': data['pdf_url'],
        'status': 'pending',
        'plan_dpi': DEFAULT_DPI
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
    scale_override = data.get('scale_ratio')
    threading.Thread(target=process_job_background, args=(data['job_id'], scale_override)).start()
    return jsonify({"success": True, "job_id": data['job_id'], "status": "processing"})

@app.route('/cross-reference', methods=['POST'])
def cross_reference():
    """Manually trigger cross-reference building"""
    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400
    result = build_cross_references(data['job_id'])
    return jsonify(result)

@app.route('/takeoff-summary', methods=['GET'])
def takeoff_summary():
    """Get takeoff summary for a job"""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    summary = supabase_request('GET', 'extraction_takeoff_summary', filters={'job_id': f'eq.{job_id}'})
    if not summary:
        return jsonify({"error": "No summary found"}), 404
    return jsonify(summary[0])

@app.route('/cross-refs', methods=['GET'])
def get_cross_refs():
    """Get cross-references for a job"""
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

@app.route('/test-update', methods=['POST'])
def test_update():
    data = request.json
    page_id = data.get('page_id')
    result = update_page(page_id, {'page_type': 'other', 'status': 'classified'})
    return jsonify({"success": bool(result), "result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5050)), debug=False)
