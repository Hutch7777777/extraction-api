"""
Extraction API - Combined Command A + Roboflow with Supabase Upload
Deployed on Railway
"""

import os
import json
import base64
import requests
import time
from io import BytesIO
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://okwtyttfqbfmcqtenize.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/infer/workflows/exterior-finishes/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables"


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
        print(f"Supabase upload error: {e}")
        return None


def detect_with_roboflow(image_url, scale_config=None):
    if scale_config is None:
        scale_config = {"sqft_per_pixel": 0.09, "lf_per_pixel": 0.3}
    
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {"image": {"type": "url", "value": image_url}}
    }
    
    response = requests.post(ROBOFLOW_WORKFLOW_URL, json=payload)
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
    return {"predictions": predictions, "calculations": calculations, "visualization": visualization, "image": image_url}


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
        response = requests.get(image_url)
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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "services": {"cohere": bool(COHERE_API_KEY), "roboflow": bool(ROBOFLOW_API_KEY), "supabase": bool(SUPABASE_KEY)}})


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_url = data.get('image_url')
    scale_config = data.get('scale_config')
    project_id = data.get('project_id')
    page_number = data.get('page_number', 1)
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    result = detect_with_roboflow(image_url, scale_config)
    
    if project_id and result.get('visualization'):
        viz = result['visualization']
        if isinstance(viz, dict) and viz.get('value'):
            timestamp = int(time.time() * 1000)
            filename = f"{project_id}/roboflow_p{page_number}_{timestamp}.jpg"
            image_data = base64.b64decode(viz['value'])
            url = upload_to_supabase(image_data, filename, 'image/jpeg')
            if url:
                result['visualization_url'] = url
    
    return jsonify(result)


@app.route('/markup', methods=['POST'])
def markup():
    data = request.json
    image_url = data.get('image_url')
    predictions = data.get('predictions', [])
    calculations = data.get('calculations', {})
    project_id = data.get('project_id')
    page_number = data.get('page_number', 1)
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    image_bytes = generate_siding_markup(image_url, predictions, calculations)
    if not image_bytes:
        return jsonify({"error": "Failed to generate markup"}), 500
    
    result = {"success": True, "image_base64": base64.b64encode(image_bytes).decode('utf-8')}
    
    if project_id:
        timestamp = int(time.time() * 1000)
        filename = f"{project_id}/siding_p{page_number}_{timestamp}.png"
        url = upload_to_supabase(image_bytes, filename, 'image/png')
        if url:
            result['markup_url'] = url
    
    return jsonify(result)


@app.route('/extract', methods=['POST'])
def extract():
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


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
