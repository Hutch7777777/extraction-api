"""
Extraction API v2 - Full Job Processing with Parallel Execution
Deployed on Railway
"""

import os
import json
import base64
import requests
import time
import asyncio
import aiohttp
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://okwtyttfqbfmcqtenize.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Rate limiting
MAX_CONCURRENT_ROBOFLOW = 5
MAX_CONCURRENT_CLAUDE = 3
BATCH_DELAY_SECONDS = 0.5

# Roboflow endpoint
ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/infer/workflows/exterior-finishes/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables"

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=10)


# ============================================
# SUPABASE HELPERS
# ============================================

def supabase_request(method, endpoint, data=None, params=None):
    """Make authenticated request to Supabase REST API"""
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'apikey': SUPABASE_KEY,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    if method == 'GET':
        response = requests.get(url, headers=headers, params=params)
    elif method == 'POST':
        response = requests.post(url, headers=headers, json=data)
    elif method == 'PATCH':
        response = requests.patch(url, headers=headers, json=data, params=params)
    
    return response.json() if response.content else None


def upload_to_supabase(image_data, filename, content_type='image/jpeg'):
    """Upload image to Supabase Storage"""
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
        print(f"Supabase upload error: {e}")
        return None


def update_job_status(job_id, status, error_message=None):
    """Update job status in database"""
    data = {'status': status}
    if error_message:
        data['error_message'] = error_message
    if status == 'processing':
        data['started_at'] = 'now()'
    if status == 'complete':
        data['completed_at'] = 'now()'
    
    supabase_request('PATCH', 'extraction_jobs', data, {'id': f'eq.{job_id}'})


def update_page_status(page_id, status, extraction_data=None, error_message=None):
    """Update page status in database"""
    data = {'status': status}
    if extraction_data:
        data['extraction_data'] = extraction_data
        data['processed_at'] = 'now()'
    if error_message:
        data['error_message'] = error_message
    
    supabase_request('PATCH', 'extraction_pages', data, {'id': f'eq.{page_id}'})


# ============================================
# ROBOFLOW DETECTION
# ============================================

def detect_with_roboflow(image_url, scale_config=None):
    """Call Roboflow RAPID workflow for object detection"""
    if scale_config is None:
        scale_config = {"sqft_per_pixel": 0.09, "lf_per_pixel": 0.3}
    
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {"image": {"type": "url", "value": image_url}}
    }
    
    response = requests.post(ROBOFLOW_WORKFLOW_URL, json=payload, timeout=120)
    if response.status_code != 200:
        return {"error": f"Roboflow API error: {response.status_code}"}
    
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


def calculate_areas(predictions, scale_config):
    """Calculate areas from predictions"""
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


# ============================================
# MARKUP GENERATION
# ============================================

def generate_siding_markup(image_url, predictions, calculations):
    """Generate siding markup visualization"""
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
        print(f"Markup generation error: {e}")
        return None


# ============================================
# PAGE CLASSIFICATION (Claude Vision)
# ============================================

def classify_page_with_claude(image_url):
    """Use Claude Vision to classify page type"""
    if not ANTHROPIC_API_KEY:
        return {"page_type": "unknown", "confidence": 0, "error": "No Anthropic API key"}
    
    try:
        # Download image and convert to base64
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Determine media type
        content_type = response.headers.get('content-type', 'image/png')
        if 'jpeg' in content_type or 'jpg' in content_type:
            media_type = 'image/jpeg'
        else:
            media_type = 'image/png'
        
        prompt = """Analyze this architectural/construction drawing and classify it.

Return ONLY a JSON object with these fields:
{
  "page_type": "elevation" | "schedule" | "floor_plan" | "section" | "detail" | "cover" | "site_plan" | "other",
  "confidence": 0.0-1.0,
  "elevation_name": "front" | "rear" | "left" | "right" | "north" | "south" | "east" | "west" | null,
  "contains_schedule": true | false,
  "schedule_type": "window" | "door" | "window_and_door" | "finish" | null,
  "notes": "brief description"
}

Classification guide:
- elevation: Side view of building exterior showing walls, windows, doors, rooflines
- schedule: Table/chart listing windows, doors, or finishes with dimensions
- floor_plan: Top-down view showing room layout, walls, dimensions
- section: Cut-through view showing internal structure
- detail: Zoomed view of specific construction element
- cover: Title sheet with project info
- site_plan: Bird's eye view of property/lot
- other: Anything else

Return ONLY the JSON, no other text."""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['content'][0]['text']
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        return {"page_type": "unknown", "confidence": 0, "error": f"API error: {response.status_code}"}
        
    except Exception as e:
        return {"page_type": "unknown", "confidence": 0, "error": str(e)}


# ============================================
# SCHEDULE OCR (Claude Vision)
# ============================================

def ocr_schedule_with_claude(image_url):
    """Extract window/door schedule data using Claude Vision"""
    if not ANTHROPIC_API_KEY:
        return {"error": "No Anthropic API key"}
    
    try:
        response = requests.get(image_url, timeout=30)
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        content_type = response.headers.get('content-type', 'image/png')
        media_type = 'image/jpeg' if 'jpeg' in content_type or 'jpg' in content_type else 'image/png'
        
        prompt = """Extract ALL window and door schedule information from this architectural drawing.

Return ONLY a JSON object:
{
  "windows": [
    {"tag": "W1", "width_inches": 36, "height_inches": 48, "type": "DH", "qty": 4, "notes": ""},
    ...
  ],
  "doors": [
    {"tag": "D1", "width_inches": 36, "height_inches": 80, "type": "Entry", "qty": 1, "notes": ""},
    ...
  ],
  "raw_text": "any other relevant text found"
}

Notes:
- Convert all dimensions to inches (36" or 3'-0" both become 36)
- Include ALL entries from the schedule
- type: DH=Double Hung, SH=Single Hung, CAS=Casement, FX=Fixed, SLD=Sliding, etc.
- If you can't read a value, use null

Return ONLY valid JSON."""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            },
            timeout=90
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['content'][0]['text']
            
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        return {"error": f"API error: {response.status_code}"}
        
    except Exception as e:
        return {"error": str(e)}


# ============================================
# PDF CONVERSION
# ============================================

def convert_pdf_to_images(pdf_url, job_id):
    """Convert PDF to PNG images, upload to Supabase"""
    try:
        from pdf2image import convert_from_bytes
        
        # Download PDF
        response = requests.get(pdf_url, timeout=120)
        if response.status_code != 200:
            return {"error": f"Failed to download PDF: {response.status_code}"}
        
        # Convert to images
        images = convert_from_bytes(response.content, dpi=150, fmt='png')
        
        results = []
        for i, img in enumerate(images):
            page_num = i + 1
            
            # Convert to bytes
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            
            # Upload to Supabase
            filename = f"{job_id}/page_{page_num:03d}.png"
            image_url = upload_to_supabase(image_bytes, filename, 'image/png')
            
            # Create thumbnail
            thumb = img.copy()
            thumb.thumbnail((300, 300))
            thumb_buffer = BytesIO()
            thumb.save(thumb_buffer, format='PNG')
            thumb_buffer.seek(0)
            thumb_filename = f"{job_id}/thumb_{page_num:03d}.png"
            thumb_url = upload_to_supabase(thumb_buffer.getvalue(), thumb_filename, 'image/png')
            
            results.append({
                "page_number": page_num,
                "image_url": image_url,
                "thumbnail_url": thumb_url
            })
        
        return {"pages": results, "total_pages": len(results)}
        
    except Exception as e:
        return {"error": str(e)}


# ============================================
# SINGLE PAGE PROCESSOR
# ============================================

def process_single_page(page_data, job_id, scale_config=None):
    """Process a single page based on its type"""
    page_id = page_data.get('id')
    page_number = page_data.get('page_number')
    image_url = page_data.get('image_url')
    page_type = page_data.get('page_type')
    
    start_time = time.time()
    result = {"page_id": page_id, "page_number": page_number, "page_type": page_type}
    
    try:
        if page_type == 'elevation':
            # Full Roboflow detection + markup
            detection = detect_with_roboflow(image_url, scale_config)
            
            if 'error' not in detection:
                timestamp = int(time.time() * 1000)
                
                # Upload Roboflow visualization
                roboflow_viz_url = None
                if detection.get('visualization'):
                    viz = detection['visualization']
                    if isinstance(viz, dict) and viz.get('value'):
                        filename = f"{job_id}/roboflow_p{page_number}_{timestamp}.jpg"
                        image_data = base64.b64decode(viz['value'])
                        roboflow_viz_url = upload_to_supabase(image_data, filename, 'image/jpeg')
                
                # Generate and upload siding markup
                markup_url = None
                markup_bytes = generate_siding_markup(image_url, detection.get('predictions', []), detection.get('calculations', {}))
                if markup_bytes:
                    filename = f"{job_id}/siding_p{page_number}_{timestamp}.png"
                    markup_url = upload_to_supabase(markup_bytes, filename, 'image/png')
                
                result['extraction_data'] = {
                    'calculations': detection.get('calculations'),
                    'prediction_count': len(detection.get('predictions', [])),
                    'roboflow_viz_url': roboflow_viz_url,
                    'siding_markup_url': markup_url
                }
                result['status'] = 'complete'
            else:
                result['status'] = 'failed'
                result['error'] = detection.get('error')
        
        elif page_type == 'schedule':
            # OCR the schedule
            schedule_data = ocr_schedule_with_claude(image_url)
            
            if 'error' not in schedule_data:
                result['extraction_data'] = schedule_data
                result['status'] = 'complete'
            else:
                result['status'] = 'failed'
                result['error'] = schedule_data.get('error')
        
        elif page_type in ['floor_plan', 'section', 'detail']:
            # For now, just mark as processed but extract nothing
            result['extraction_data'] = {'note': f'{page_type} - no extraction implemented yet'}
            result['status'] = 'complete'
        
        else:
            # Skip other page types
            result['status'] = 'skipped'
        
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)
    
    return result


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "version": "2.0",
        "services": {
            "roboflow": bool(ROBOFLOW_API_KEY),
            "anthropic": bool(ANTHROPIC_API_KEY),
            "supabase": bool(SUPABASE_KEY)
        }
    })


@app.route('/detect', methods=['POST'])
def detect():
    """Detect objects in single image"""
    data = request.json
    image_url = data.get('image_url')
    scale_config = data.get('scale_config')
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    result = detect_with_roboflow(image_url, scale_config)
    return jsonify(result)


@app.route('/classify-page', methods=['POST'])
def classify_page():
    """Classify a single page"""
    data = request.json
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    result = classify_page_with_claude(image_url)
    return jsonify(result)


@app.route('/ocr-schedule', methods=['POST'])
def ocr_schedule():
    """OCR a schedule page"""
    data = request.json
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    result = ocr_schedule_with_claude(image_url)
    return jsonify(result)


@app.route('/extract', methods=['POST'])
def extract():
    """Full extraction for single image (existing endpoint)"""
    data = request.json
    image_url = data.get('image_url')
    project_id = data.get('project_id')
    page_number = data.get('page_number', 1)
    scale_config = data.get('scale_config')
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    timestamp = int(time.time() * 1000)
    detection = detect_with_roboflow(image_url, scale_config)
    
    roboflow_viz_url = None
    if project_id and detection.get('visualization'):
        viz = detection['visualization']
        if isinstance(viz, dict) and viz.get('value'):
            filename = f"{project_id}/roboflow_p{page_number}_{timestamp}.jpg"
            image_data = base64.b64decode(viz['value'])
            roboflow_viz_url = upload_to_supabase(image_data, filename, 'image/jpeg')
    
    markup_url = None
    markup_bytes = generate_siding_markup(image_url, detection.get('predictions', []), detection.get('calculations', {}))
    if project_id and markup_bytes:
        filename = f"{project_id}/siding_p{page_number}_{timestamp}.png"
        markup_url = upload_to_supabase(markup_bytes, filename, 'image/png')
    
    return jsonify({
        "success": True,
        "project_id": project_id,
        "page_number": page_number,
        "image_url": image_url,
        "calculations": detection.get('calculations', {}),
        "predictions": detection.get('predictions', []),
        "roboflow_viz_url": roboflow_viz_url,
        "siding_markup_url": markup_url
    })


@app.route('/convert-pdf', methods=['POST'])
def convert_pdf():
    """Convert PDF to images and create job"""
    data = request.json
    pdf_url = data.get('pdf_url')
    project_id = data.get('project_id')
    project_name = data.get('project_name', '')
    
    if not pdf_url or not project_id:
        return jsonify({"error": "pdf_url and project_id required"}), 400
    
    # Create job record
    job_data = {
        'project_id': project_id,
        'project_name': project_name,
        'source_pdf_url': pdf_url,
        'status': 'converting'
    }
    job_result = supabase_request('POST', 'extraction_jobs', job_data)
    
    if not job_result or len(job_result) == 0:
        return jsonify({"error": "Failed to create job"}), 500
    
    job_id = job_result[0]['id']
    
    # Convert PDF
    conversion_result = convert_pdf_to_images(pdf_url, job_id)
    
    if 'error' in conversion_result:
        update_job_status(job_id, 'failed', conversion_result['error'])
        return jsonify({"error": conversion_result['error']}), 500
    
    # Update job with page count
    supabase_request('PATCH', 'extraction_jobs', 
        {'total_pages': conversion_result['total_pages'], 'pages_converted': conversion_result['total_pages']},
        {'id': f'eq.{job_id}'})
    
    # Create page records
    for page in conversion_result['pages']:
        page_data = {
            'job_id': job_id,
            'page_number': page['page_number'],
            'image_url': page['image_url'],
            'thumbnail_url': page['thumbnail_url'],
            'status': 'pending'
        }
        supabase_request('POST', 'extraction_pages', page_data)
    
    update_job_status(job_id, 'classifying')
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "total_pages": conversion_result['total_pages'],
        "pages": conversion_result['pages']
    })


@app.route('/classify-job', methods=['POST'])
def classify_job():
    """Classify all pages in a job (parallel)"""
    data = request.json
    job_id = data.get('job_id')
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    # Get all pending pages
    pages = supabase_request('GET', 'extraction_pages', 
        params={'job_id': f'eq.{job_id}', 'status': 'eq.pending', 'order': 'page_number'})
    
    if not pages:
        return jsonify({"error": "No pages found"}), 404
    
    update_job_status(job_id, 'classifying')
    
    results = []
    elevation_count = 0
    schedule_count = 0
    
    # Process in batches
    for i in range(0, len(pages), MAX_CONCURRENT_CLAUDE):
        batch = pages[i:i+MAX_CONCURRENT_CLAUDE]
        
        for page in batch:
            classification = classify_page_with_claude(page['image_url'])
            
            page_type = classification.get('page_type', 'other')
            confidence = classification.get('confidence', 0)
            elevation_name = classification.get('elevation_name')
            
            if page_type == 'elevation':
                elevation_count += 1
            elif page_type == 'schedule':
                schedule_count += 1
            
            # Update page record
            update_data = {
                'page_type': page_type,
                'page_type_confidence': confidence,
                'elevation_name': elevation_name,
                'status': 'classified'
            }
            supabase_request('PATCH', 'extraction_pages', update_data, {'id': f'eq.{page["id"]}'})
            
            results.append({
                'page_number': page['page_number'],
                'page_type': page_type,
                'confidence': confidence,
                'elevation_name': elevation_name
            })
        
        # Rate limit delay
        if i + MAX_CONCURRENT_CLAUDE < len(pages):
            time.sleep(BATCH_DELAY_SECONDS)
    
    # Update job counts
    supabase_request('PATCH', 'extraction_jobs', {
        'pages_classified': len(results),
        'elevation_count': elevation_count,
        'schedule_count': schedule_count,
        'status': 'processing'
    }, {'id': f'eq.{job_id}'})
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "pages_classified": len(results),
        "elevation_count": elevation_count,
        "schedule_count": schedule_count,
        "results": results
    })


@app.route('/process-job', methods=['POST'])
def process_job():
    """Process all classified pages in a job (parallel)"""
    data = request.json
    job_id = data.get('job_id')
    scale_config = data.get('scale_config')
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    # Get classified pages
    pages = supabase_request('GET', 'extraction_pages',
        params={'job_id': f'eq.{job_id}', 'status': 'eq.classified', 'order': 'page_number'})
    
    if not pages:
        return jsonify({"error": "No classified pages found"}), 404
    
    update_job_status(job_id, 'processing')
    
    results = []
    elevation_results = []
    schedule_results = []
    
    # Process elevations first (Roboflow - can be parallelized more)
    elevation_pages = [p for p in pages if p['page_type'] == 'elevation']
    schedule_pages = [p for p in pages if p['page_type'] == 'schedule']
    other_pages = [p for p in pages if p['page_type'] not in ['elevation', 'schedule']]
    
    # Process elevations
    for i in range(0, len(elevation_pages), MAX_CONCURRENT_ROBOFLOW):
        batch = elevation_pages[i:i+MAX_CONCURRENT_ROBOFLOW]
        
        for page in batch:
            result = process_single_page(page, job_id, scale_config)
            
            # Update database
            update_data = {
                'status': result.get('status', 'failed'),
                'extraction_data': result.get('extraction_data'),
                'processing_time_ms': result.get('processing_time_ms'),
                'processed_at': 'now()'
            }
            if result.get('error'):
                update_data['error_message'] = result['error']
            
            supabase_request('PATCH', 'extraction_pages', update_data, {'id': f'eq.{page["id"]}'})
            
            results.append(result)
            if result.get('status') == 'complete':
                elevation_results.append(result)
        
        if i + MAX_CONCURRENT_ROBOFLOW < len(elevation_pages):
            time.sleep(BATCH_DELAY_SECONDS)
    
    # Process schedules
    for i in range(0, len(schedule_pages), MAX_CONCURRENT_CLAUDE):
        batch = schedule_pages[i:i+MAX_CONCURRENT_CLAUDE]
        
        for page in batch:
            result = process_single_page(page, job_id)
            
            update_data = {
                'status': result.get('status', 'failed'),
                'extraction_data': result.get('extraction_data'),
                'schedule_data': result.get('extraction_data'),
                'processing_time_ms': result.get('processing_time_ms'),
                'processed_at': 'now()'
            }
            if result.get('error'):
                update_data['error_message'] = result['error']
            
            supabase_request('PATCH', 'extraction_pages', update_data, {'id': f'eq.{page["id"]}'})
            
            results.append(result)
            if result.get('status') == 'complete':
                schedule_results.append(result)
        
        if i + MAX_CONCURRENT_CLAUDE < len(schedule_pages):
            time.sleep(BATCH_DELAY_SECONDS)
    
    # Mark other pages as skipped
    for page in other_pages:
        supabase_request('PATCH', 'extraction_pages', 
            {'status': 'skipped', 'processed_at': 'now()'}, 
            {'id': f'eq.{page["id"]}'})
        results.append({'page_number': page['page_number'], 'status': 'skipped'})
    
    # Calculate totals from elevation results
    totals = {
        'total_net_siding_sqft': 0,
        'total_gross_wall_sqft': 0,
        'total_windows': 0,
        'total_doors': 0
    }
    
    for r in elevation_results:
        if r.get('extraction_data', {}).get('calculations'):
            calc = r['extraction_data']['calculations']
            totals['total_net_siding_sqft'] += calc.get('areas', {}).get('net_siding_sqft', 0)
            totals['total_gross_wall_sqft'] += calc.get('areas', {}).get('gross_wall_sqft', 0)
            totals['total_windows'] += calc.get('counts', {}).get('window', 0)
            totals['total_doors'] += calc.get('counts', {}).get('door', 0)
    
    # Update job as complete
    supabase_request('PATCH', 'extraction_jobs', {
        'pages_processed': len(results),
        'status': 'complete',
        'completed_at': 'now()',
        'results_summary': totals
    }, {'id': f'eq.{job_id}'})
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "pages_processed": len(results),
        "elevations_processed": len(elevation_results),
        "schedules_processed": len(schedule_results),
        "totals": totals,
        "results": results
    })


@app.route('/job-status', methods=['GET'])
def job_status():
    """Get current job status"""
    job_id = request.args.get('job_id')
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    job = supabase_request('GET', 'extraction_jobs', params={'id': f'eq.{job_id}'})
    
    if not job or len(job) == 0:
        return jsonify({"error": "Job not found"}), 404
    
    pages = supabase_request('GET', 'extraction_pages', 
        params={'job_id': f'eq.{job_id}', 'order': 'page_number'})
    
    return jsonify({
        "job": job[0],
        "pages": pages
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
