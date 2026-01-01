"""
Extraction API v2.3 - Stream PDF to disk for memory safety
"""

import os
import json
import base64
import requests
import time
import threading
import tempfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://okwtyttfqbfmcqtenize.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

MAX_CONCURRENT_ROBOFLOW = 5
MAX_CONCURRENT_CLAUDE = 3
BATCH_DELAY_SECONDS = 0.5
PDF_CHUNK_SIZE = 5  # Smaller chunks

ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/infer/workflows/exterior-finishes/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables"

executor = ThreadPoolExecutor(max_workers=10)


def supabase_request(method, endpoint, data=None, params=None):
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'apikey': SUPABASE_KEY,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method == 'PATCH':
            response = requests.patch(url, headers=headers, json=data, params=params)
        if response.status_code >= 400:
            print(f"Supabase error: {response.status_code} - {response.text}", flush=True)
            return None
        return response.json() if response.content else None
    except Exception as e:
        print(f"Supabase request error: {e}", flush=True)
        return None


def upload_to_supabase(image_data, filename, content_type='image/jpeg'):
    if not SUPABASE_KEY:
        return None
    try:
        upload_url = f"{SUPABASE_URL}/storage/v1/object/extraction-markups/{filename}"
        headers = {
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': content_type,
            'x-upsert': 'true'
        }
        response = requests.post(upload_url, headers=headers, data=image_data)
        if response.status_code in [200, 201]:
            return f"{SUPABASE_URL}/storage/v1/object/public/extraction-markups/{filename}"
        return None
    except Exception as e:
        print(f"Supabase upload error: {e}", flush=True)
        return None


def update_job(job_id, updates):
    supabase_request('PATCH', 'extraction_jobs', updates, {'id': f'eq.{job_id}'})


def update_page(page_id, updates):
    supabase_request('PATCH', 'extraction_pages', updates, {'id': f'eq.{page_id}'})


def detect_with_roboflow(image_url, scale_config=None):
    if scale_config is None:
        scale_config = {"sqft_per_pixel": 0.09}
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {"image": {"type": "url", "value": image_url}}
    }
    try:
        response = requests.post(ROBOFLOW_WORKFLOW_URL, json=payload, timeout=120)
        if response.status_code != 200:
            return {"error": f"Roboflow error: {response.status_code}"}
        result = response.json()
        predictions = []
        visualization = None
        if 'outputs' in result and len(result['outputs']) > 0:
            output = result['outputs'][0]
            if 'predictions' in output:
                pred_data = output['predictions']
                if isinstance(pred_data, dict) and 'predictions' in pred_data:
                    predictions = pred_data['predictions']
                elif isinstance(pred_data, list):
                    predictions = pred_data
            if 'visualization' in output:
                visualization = output['visualization']
        calculations = calculate_areas(predictions, scale_config)
        return {"predictions": predictions, "calculations": calculations, "visualization": visualization}
    except Exception as e:
        return {"error": str(e)}


def calculate_areas(predictions, scale_config):
    sqft_per_pixel = scale_config.get("sqft_per_pixel", 0.09)
    counts = {"window": 0, "door": 0, "building": 0, "roof": 0, "gable": 0, "garage": 0}
    areas = {"window_sqft": 0, "door_sqft": 0, "building_sqft": 0, "roof_sqft": 0, "gable_sqft": 0, "garage_sqft": 0}
    for pred in predictions:
        class_name = pred.get("class", "").lower()
        width = pred.get("width", 0)
        height = pred.get("height", 0)
        pixel_area = width * height
        sqft = round(pixel_area * sqft_per_pixel, 1)
        if class_name in counts:
            counts[class_name] += 1
            areas[f"{class_name}_sqft"] += sqft
    for key in areas:
        areas[key] = round(areas[key], 1)
    gross_wall_sqft = areas["building_sqft"]
    openings_sqft = areas["window_sqft"] + areas["door_sqft"] + areas["garage_sqft"]
    net_siding_sqft = round(gross_wall_sqft - openings_sqft, 1)
    opening_percentage = round((openings_sqft / gross_wall_sqft * 100), 1) if gross_wall_sqft > 0 else 0
    areas["gross_wall_sqft"] = gross_wall_sqft
    areas["net_siding_sqft"] = net_siding_sqft
    areas["opening_percentage"] = opening_percentage
    return {"counts": counts, "areas": areas}


def generate_siding_markup(image_url, predictions, calculations):
    try:
        from PIL import Image, ImageDraw
        response = requests.get(image_url, timeout=30)
        img = Image.open(BytesIO(response.content)).convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for pred in predictions:
            if pred.get('class', '').lower() == 'building':
                x, y = pred.get('x', 0), pred.get('y', 0)
                w, h = pred.get('width', 0), pred.get('height', 0)
                draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], fill=(0, 255, 0, 80))
        for pred in predictions:
            if pred.get('class', '').lower() in ['window', 'door', 'garage']:
                x, y = pred.get('x', 0), pred.get('y', 0)
                w, h = pred.get('width', 0), pred.get('height', 0)
                draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], fill=(255, 0, 0, 120))
        result = Image.alpha_composite(img, overlay)
        draw_result = ImageDraw.Draw(result)
        net_sf = calculations.get('areas', {}).get('net_siding_sqft', 0)
        opening_pct = calculations.get('areas', {}).get('opening_percentage', 0)
        text = f"Net Siding: {net_sf} SF | Openings: {opening_pct}%"
        draw_result.rectangle([10, 10, 400, 40], fill=(0, 0, 0, 200))
        draw_result.text((15, 15), text, fill=(255, 255, 255))
        buffer = BytesIO()
        result.convert('RGB').save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print(f"Markup error: {e}", flush=True)
        return None


def classify_page_with_claude(image_url):
    if not ANTHROPIC_API_KEY:
        return {"page_type": "unknown", "confidence": 0, "error": "No API key"}
    try:
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        content_type = response.headers.get('content-type', 'image/png')
        media_type = 'image/jpeg' if 'jpeg' in content_type or 'jpg' in content_type else 'image/png'
        prompt = """Classify this architectural drawing. Return ONLY JSON:
{"page_type": "elevation"|"schedule"|"floor_plan"|"section"|"detail"|"cover"|"site_plan"|"other", "confidence": 0.0-1.0, "elevation_name": "front"|"rear"|"left"|"right"|null, "contains_schedule": true|false, "schedule_type": "window"|"door"|"window_and_door"|null, "notes": "brief description"}"""
        api_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 500, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_base64}}, {"type": "text", "text": prompt}]}]},
            timeout=60
        )
        if api_response.status_code == 200:
            import re
            text = api_response.json()['content'][0]['text']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        return {"page_type": "unknown", "confidence": 0}
    except Exception as e:
        return {"page_type": "unknown", "confidence": 0, "error": str(e)}


def ocr_schedule_with_claude(image_url):
    if not ANTHROPIC_API_KEY:
        return {"error": "No API key"}
    try:
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        media_type = 'image/png'
        prompt = """Extract window/door schedule. Return ONLY JSON:
{"windows": [{"tag": "W1", "width_inches": 36, "height_inches": 48, "type": "DH", "qty": 4}], "doors": [{"tag": "D1", "width_inches": 36, "height_inches": 80, "type": "Entry", "qty": 1}]}"""
        api_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 2000, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_base64}}, {"type": "text", "text": prompt}]}]},
            timeout=90
        )
        if api_response.status_code == 200:
            import re
            text = api_response.json()['content'][0]['text']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        return {"error": f"API error: {api_response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def convert_pdf_background(job_id, pdf_url):
    """Convert PDF using temp file to save memory"""
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
        
        print(f"[{job_id}] Downloading PDF to temp file...", flush=True)
        update_job(job_id, {'status': 'converting'})
        
        # Stream PDF to temp file instead of memory
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            response = requests.get(pdf_url, timeout=300, stream=True)
            if response.status_code != 200:
                update_job(job_id, {'status': 'failed', 'error_message': f'Download failed: {response.status_code}'})
                return
            
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            
            tmp_path = tmp_file.name
        
        print(f"[{job_id}] PDF saved to {tmp_path}", flush=True)
        
        # Get page count
        try:
            info = pdfinfo_from_path(tmp_path)
            total_pages = info['Pages']
        except:
            total_pages = 81
        
        print(f"[{job_id}] PDF has {total_pages} pages", flush=True)
        update_job(job_id, {'total_pages': total_pages})
        
        pages_converted = 0
        
        # Process in small chunks using file path (more memory efficient)
        for start_page in range(1, total_pages + 1, PDF_CHUNK_SIZE):
            end_page = min(start_page + PDF_CHUNK_SIZE - 1, total_pages)
            
            print(f"[{job_id}] Converting pages {start_page}-{end_page}...", flush=True)
            
            try:
                images = convert_from_path(
                    tmp_path,
                    dpi=100,  # Lower DPI for memory
                    fmt='png',
                    first_page=start_page,
                    last_page=end_page,
                    thread_count=1  # Single thread for memory
                )
                
                for i, img in enumerate(images):
                    page_num = start_page + i
                    
                    # Resize if too large
                    max_size = 2000
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size))
                    
                    buffer = BytesIO()
                    img.save(buffer, format='PNG', optimize=True)
                    buffer.seek(0)
                    image_bytes = buffer.getvalue()
                    
                    filename = f"{job_id}/page_{page_num:03d}.png"
                    image_url = upload_to_supabase(image_bytes, filename, 'image/png')
                    
                    # Smaller thumbnail
                    thumb = img.copy()
                    thumb.thumbnail((200, 200))
                    thumb_buffer = BytesIO()
                    thumb.save(thumb_buffer, format='PNG', optimize=True)
                    thumb_buffer.seek(0)
                    thumb_filename = f"{job_id}/thumb_{page_num:03d}.png"
                    thumb_url = upload_to_supabase(thumb_buffer.getvalue(), thumb_filename, 'image/png')
                    
                    page_data = {
                        'job_id': job_id,
                        'page_number': page_num,
                        'image_url': image_url,
                        'thumbnail_url': thumb_url,
                        'status': 'pending'
                    }
                    supabase_request('POST', 'extraction_pages', page_data)
                    
                    pages_converted += 1
                    del img
                
                del images
                update_job(job_id, {'pages_converted': pages_converted})
                print(f"[{job_id}] Progress: {pages_converted}/{total_pages}", flush=True)
                
            except Exception as chunk_error:
                print(f"[{job_id}] Chunk error: {chunk_error}", flush=True)
                continue
        
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        print(f"[{job_id}] Conversion complete! {pages_converted} pages", flush=True)
        update_job(job_id, {'status': 'converted', 'pages_converted': pages_converted})
        
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def classify_job_background(job_id):
    try:
        print(f"[{job_id}] Starting classification...", flush=True)
        update_job(job_id, {'status': 'classifying'})
        pages = supabase_request('GET', 'extraction_pages', params={'job_id': f'eq.{job_id}', 'status': 'eq.pending', 'order': 'page_number'})
        if not pages:
            return
        elevation_count = schedule_count = floor_plan_count = other_count = 0
        for i, page in enumerate(pages):
            print(f"[{job_id}] Classifying page {page['page_number']}/{len(pages)}...", flush=True)
            classification = classify_page_with_claude(page['image_url'])
            page_type = classification.get('page_type', 'other')
            if page_type == 'elevation': elevation_count += 1
            elif page_type == 'schedule': schedule_count += 1
            elif page_type == 'floor_plan': floor_plan_count += 1
            else: other_count += 1
            update_page(page['id'], {'page_type': page_type, 'page_type_confidence': classification.get('confidence', 0), 'elevation_name': classification.get('elevation_name'), 'status': 'classified'})
            update_job(job_id, {'pages_classified': i + 1})
            if (i + 1) % MAX_CONCURRENT_CLAUDE == 0:
                time.sleep(BATCH_DELAY_SECONDS)
        update_job(job_id, {'status': 'classified', 'elevation_count': elevation_count, 'schedule_count': schedule_count, 'floor_plan_count': floor_plan_count, 'other_count': other_count})
    except Exception as e:
        print(f"[{job_id}] Classification error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def process_job_background(job_id, scale_config=None):
    try:
        print(f"[{job_id}] Starting processing...", flush=True)
        update_job(job_id, {'status': 'processing'})
        pages = supabase_request('GET', 'extraction_pages', params={'job_id': f'eq.{job_id}', 'status': 'eq.classified', 'order': 'page_number'})
        if not pages:
            return
        elevation_pages = [p for p in pages if p['page_type'] == 'elevation']
        schedule_pages = [p for p in pages if p['page_type'] == 'schedule']
        other_pages = [p for p in pages if p['page_type'] not in ['elevation', 'schedule']]
        totals = {'total_net_siding_sqft': 0, 'total_gross_wall_sqft': 0, 'total_windows': 0, 'total_doors': 0}
        processed_count = 0
        for page in elevation_pages:
            print(f"[{job_id}] Processing elevation {page['page_number']}...", flush=True)
            detection = detect_with_roboflow(page['image_url'], scale_config)
            if 'error' not in detection:
                timestamp = int(time.time() * 1000)
                roboflow_viz_url = None
                if detection.get('visualization'):
                    viz = detection['visualization']
                    if isinstance(viz, dict) and viz.get('value'):
                        filename = f"{job_id}/roboflow_p{page['page_number']}_{timestamp}.jpg"
                        roboflow_viz_url = upload_to_supabase(base64.b64decode(viz['value']), filename, 'image/jpeg')
                markup_url = None
                markup_bytes = generate_siding_markup(page['image_url'], detection.get('predictions', []), detection.get('calculations', {}))
                if markup_bytes:
                    filename = f"{job_id}/siding_p{page['page_number']}_{timestamp}.png"
                    markup_url = upload_to_supabase(markup_bytes, filename, 'image/png')
                calc = detection.get('calculations', {})
                totals['total_net_siding_sqft'] += calc.get('areas', {}).get('net_siding_sqft', 0)
                totals['total_gross_wall_sqft'] += calc.get('areas', {}).get('gross_wall_sqft', 0)
                totals['total_windows'] += calc.get('counts', {}).get('window', 0)
                totals['total_doors'] += calc.get('counts', {}).get('door', 0)
                update_page(page['id'], {'status': 'complete', 'extraction_data': {'calculations': calc, 'roboflow_viz_url': roboflow_viz_url, 'siding_markup_url': markup_url}})
            else:
                update_page(page['id'], {'status': 'failed', 'error_message': detection.get('error')})
            processed_count += 1
            update_job(job_id, {'pages_processed': processed_count})
        for page in schedule_pages:
            print(f"[{job_id}] Processing schedule {page['page_number']}...", flush=True)
            schedule_data = ocr_schedule_with_claude(page['image_url'])
            if 'error' not in schedule_data:
                update_page(page['id'], {'status': 'complete', 'extraction_data': schedule_data, 'schedule_data': schedule_data})
            else:
                update_page(page['id'], {'status': 'failed', 'error_message': schedule_data.get('error')})
            processed_count += 1
            update_job(job_id, {'pages_processed': processed_count})
        for page in other_pages:
            update_page(page['id'], {'status': 'skipped'})
            processed_count += 1
            update_job(job_id, {'pages_processed': processed_count})
        update_job(job_id, {'status': 'complete', 'results_summary': totals})
    except Exception as e:
        print(f"[{job_id}] Processing error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "version": "2.3", "services": {"roboflow": bool(ROBOFLOW_API_KEY), "anthropic": bool(ANTHROPIC_API_KEY), "supabase": bool(SUPABASE_KEY)}})

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if not data.get('image_url'):
        return jsonify({"error": "image_url required"}), 400
    return jsonify(detect_with_roboflow(data['image_url'], data.get('scale_config')))

@app.route('/classify-page', methods=['POST'])
def classify_page():
    data = request.json
    if not data.get('image_url'):
        return jsonify({"error": "image_url required"}), 400
    return jsonify(classify_page_with_claude(data['image_url']))

@app.route('/ocr-schedule', methods=['POST'])
def ocr_schedule():
    data = request.json
    if not data.get('image_url'):
        return jsonify({"error": "image_url required"}), 400
    return jsonify(ocr_schedule_with_claude(data['image_url']))

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    image_url = data.get('image_url')
    project_id = data.get('project_id')
    page_number = data.get('page_number', 1)
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    timestamp = int(time.time() * 1000)
    detection = detect_with_roboflow(image_url, data.get('scale_config'))
    roboflow_viz_url = markup_url = None
    if project_id and detection.get('visualization'):
        viz = detection['visualization']
        if isinstance(viz, dict) and viz.get('value'):
            roboflow_viz_url = upload_to_supabase(base64.b64decode(viz['value']), f"{project_id}/roboflow_p{page_number}_{timestamp}.jpg", 'image/jpeg')
    markup_bytes = generate_siding_markup(image_url, detection.get('predictions', []), detection.get('calculations', {}))
    if project_id and markup_bytes:
        markup_url = upload_to_supabase(markup_bytes, f"{project_id}/siding_p{page_number}_{timestamp}.png", 'image/png')
    return jsonify({"success": True, "project_id": project_id, "page_number": page_number, "calculations": detection.get('calculations', {}), "roboflow_viz_url": roboflow_viz_url, "siding_markup_url": markup_url})

@app.route('/start-job', methods=['POST'])
def start_job():
    data = request.json
    if not data.get('pdf_url') or not data.get('project_id'):
        return jsonify({"error": "pdf_url and project_id required"}), 400
    job_result = supabase_request('POST', 'extraction_jobs', {'project_id': data['project_id'], 'project_name': data.get('project_name', ''), 'source_pdf_url': data['pdf_url'], 'status': 'pending'})
    if not job_result:
        return jsonify({"error": "Failed to create job"}), 500
    job_id = job_result[0]['id']
    threading.Thread(target=convert_pdf_background, args=(job_id, data['pdf_url'])).start()
    return jsonify({"success": True, "job_id": job_id, "project_id": data['project_id'], "status": "pending"})

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
    threading.Thread(target=process_job_background, args=(data['job_id'], data.get('scale_config'))).start()
    return jsonify({"success": True, "job_id": data['job_id'], "status": "processing"})

@app.route('/job-status', methods=['GET'])
def job_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    job = supabase_request('GET', 'extraction_jobs', params={'id': f'eq.{job_id}'})
    if not job:
        return jsonify({"error": "Job not found"}), 404
    result = {"job": job[0]}
    if request.args.get('include_pages', 'false').lower() == 'true':
        result["pages"] = supabase_request('GET', 'extraction_pages', params={'job_id': f'eq.{job_id}', 'order': 'page_number'})
    return jsonify(result)

@app.route('/list-jobs', methods=['GET'])
def list_jobs():
    return jsonify({"jobs": supabase_request('GET', 'extraction_jobs', params={'order': 'created_at.desc', 'limit': '50'}) or []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5050)), debug=False)
