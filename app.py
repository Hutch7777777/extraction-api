"""
Extraction API v2.4 - Fixed page updates
"""

import os
import json
import base64
import requests
import time
import threading
import tempfile
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

ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/infer/workflows/exterior-finishes/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables"


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
        print(f"FAILED to update page {page_id}", flush=True)
    return result


def detect_with_roboflow(image_url, scale_config=None):
    if scale_config is None:
        scale_config = {"sqft_per_pixel": 0.09}
    payload = {"api_key": ROBOFLOW_API_KEY, "inputs": {"image": {"type": "url", "value": image_url}}}
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
        sqft_per_pixel = scale_config.get("sqft_per_pixel", 0.09)
        counts = {"window": 0, "door": 0, "building": 0, "roof": 0, "gable": 0, "garage": 0}
        areas = {"window_sqft": 0, "door_sqft": 0, "building_sqft": 0, "roof_sqft": 0, "gable_sqft": 0, "garage_sqft": 0}
        for pred in predictions:
            cn = pred.get("class", "").lower()
            w, h = pred.get("width", 0), pred.get("height", 0)
            sqft = round(w * h * sqft_per_pixel, 1)
            if cn in counts:
                counts[cn] += 1
                areas[f"{cn}_sqft"] += sqft
        gross = areas["building_sqft"]
        openings = areas["window_sqft"] + areas["door_sqft"] + areas["garage_sqft"]
        areas["gross_wall_sqft"] = gross
        areas["net_siding_sqft"] = round(gross - openings, 1)
        areas["opening_percentage"] = round((openings / gross * 100), 1) if gross > 0 else 0
        return {"predictions": predictions, "calculations": {"counts": counts, "areas": areas}, "visualization": visualization}
    except Exception as e:
        return {"error": str(e)}


def classify_page_with_claude(image_url):
    if not ANTHROPIC_API_KEY:
        return {"page_type": "other", "confidence": 0}
    try:
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        prompt = """Classify this architectural drawing. Return ONLY JSON:
{"page_type": "elevation"|"schedule"|"floor_plan"|"section"|"detail"|"cover"|"site_plan"|"other", "confidence": 0.0-1.0, "elevation_name": "front"|"rear"|"left"|"right"|null, "contains_schedule": true|false, "schedule_type": "window"|"door"|"window_and_door"|null, "notes": "brief description"}"""
        api_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 500, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}}, {"type": "text", "text": prompt}]}]},
            timeout=60
        )
        if api_response.status_code == 200:
            import re
            text = api_response.json()['content'][0]['text']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        return {"page_type": "other", "confidence": 0}
    except Exception as e:
        return {"page_type": "other", "confidence": 0, "error": str(e)}


def ocr_schedule_with_claude(image_url):
    if not ANTHROPIC_API_KEY:
        return {"error": "No API key"}
    try:
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        prompt = """Extract window/door schedule. Return ONLY JSON:
{"windows": [{"tag": "W1", "width_inches": 36, "height_inches": 48, "type": "DH", "qty": 4}], "doors": [{"tag": "D1", "width_inches": 36, "height_inches": 80, "type": "Entry", "qty": 1}]}"""
        api_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 2000, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}}, {"type": "text", "text": prompt}]}]},
            timeout=90
        )
        if api_response.status_code == 200:
            import re
            text = api_response.json()['content'][0]['text']
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        return {"error": "API error"}
    except Exception as e:
        return {"error": str(e)}


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
        update_job(job_id, {'total_pages': total_pages})
        pages_converted = 0
        for start_page in range(1, total_pages + 1, PDF_CHUNK_SIZE):
            end_page = min(start_page + PDF_CHUNK_SIZE - 1, total_pages)
            try:
                images = convert_from_path(tmp_path, dpi=100, fmt='png', first_page=start_page, last_page=end_page, thread_count=1)
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
                    supabase_request('POST', 'extraction_pages', {'job_id': job_id, 'page_number': page_num, 'image_url': image_url, 'thumbnail_url': thumb_url, 'status': 'pending'})
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
            print(f"[{job_id}] No pages!", flush=True)
            return
        print(f"[{job_id}] {len(pages)} pages", flush=True)
        elevation_count = schedule_count = floor_plan_count = other_count = 0
        for i, page in enumerate(pages):
            page_id = page.get('id')
            page_num = page.get('page_number', i+1)
            image_url = page.get('image_url')
            print(f"[{job_id}] Page {page_num}...", flush=True)
            classification = classify_page_with_claude(image_url)
            page_type = classification.get('page_type', 'other')
            confidence = classification.get('confidence', 0)
            elevation_name = classification.get('elevation_name')
            print(f"[{job_id}] Page {page_num}: {page_type}", flush=True)
            if page_type == 'elevation': elevation_count += 1
            elif page_type == 'schedule': schedule_count += 1
            elif page_type == 'floor_plan': floor_plan_count += 1
            else: other_count += 1
            update_page(page_id, {'page_type': page_type, 'page_type_confidence': confidence, 'elevation_name': elevation_name, 'status': 'classified'})
            update_job(job_id, {'pages_classified': i + 1})
            if (i + 1) % MAX_CONCURRENT_CLAUDE == 0:
                time.sleep(BATCH_DELAY_SECONDS)
        update_job(job_id, {'status': 'classified', 'elevation_count': elevation_count, 'schedule_count': schedule_count, 'floor_plan_count': floor_plan_count, 'other_count': other_count})
        print(f"[{job_id}] Done! E:{elevation_count} S:{schedule_count} F:{floor_plan_count} O:{other_count}", flush=True)
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def process_job_background(job_id, scale_config=None):
    try:
        print(f"[{job_id}] Processing...", flush=True)
        update_job(job_id, {'status': 'processing'})
        pages = supabase_request('GET', 'extraction_pages', filters={'job_id': f'eq.{job_id}', 'status': 'eq.classified', 'order': 'page_number'})
        if not pages:
            return
        elevation_pages = [p for p in pages if p.get('page_type') == 'elevation']
        schedule_pages = [p for p in pages if p.get('page_type') == 'schedule']
        totals = {'total_net_siding_sqft': 0, 'total_gross_wall_sqft': 0, 'total_windows': 0, 'total_doors': 0}
        processed = 0
        for page in elevation_pages:
            detection = detect_with_roboflow(page['image_url'], scale_config)
            if 'error' not in detection:
                calc = detection.get('calculations', {})
                totals['total_net_siding_sqft'] += calc.get('areas', {}).get('net_siding_sqft', 0)
                totals['total_gross_wall_sqft'] += calc.get('areas', {}).get('gross_wall_sqft', 0)
                totals['total_windows'] += calc.get('counts', {}).get('window', 0)
                totals['total_doors'] += calc.get('counts', {}).get('door', 0)
                update_page(page['id'], {'status': 'complete', 'extraction_data': {'calculations': calc}})
            processed += 1
            update_job(job_id, {'pages_processed': processed})
        for page in schedule_pages:
            schedule_data = ocr_schedule_with_claude(page['image_url'])
            update_page(page['id'], {'status': 'complete' if 'error' not in schedule_data else 'failed', 'extraction_data': schedule_data})
            processed += 1
            update_job(job_id, {'pages_processed': processed})
        for page in pages:
            if page.get('page_type') not in ['elevation', 'schedule']:
                update_page(page['id'], {'status': 'skipped'})
        update_job(job_id, {'status': 'complete', 'results_summary': totals})
        print(f"[{job_id}] Complete!", flush=True)
    except Exception as e:
        print(f"[{job_id}] Error: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "version": "2.4", "services": {"roboflow": bool(ROBOFLOW_API_KEY), "anthropic": bool(ANTHROPIC_API_KEY), "supabase": bool(SUPABASE_KEY)}})

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
    job_result = supabase_request('POST', 'extraction_jobs', {'project_id': data['project_id'], 'project_name': data.get('project_name', ''), 'source_pdf_url': data['pdf_url'], 'status': 'pending'})
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
    threading.Thread(target=process_job_background, args=(data['job_id'], data.get('scale_config'))).start()
    return jsonify({"success": True, "job_id": data['job_id'], "status": "processing"})

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
    result = update_page(page_id, {'page_type': 'test', 'status': 'classified'})
    return jsonify({"success": bool(result), "result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5050)), debug=False)
