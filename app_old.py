#!/usr/bin/env python3
"""
Construction Plan Extraction API
Combines Command A (OCR) + Roboflow (Detection) for HOVER-compatible output

Endpoints:
- POST /extract - Full extraction from PDF URL
- POST /classify - Command A page classification only
- POST /detect - Roboflow detection only
- POST /markup - Generate siding markup
- GET /health - Health check
"""

import os
import io
import json
import base64
import tempfile
import requests
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
import cohere
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration from environment
COHERE_API_KEY = os.environ.get('COHERE_API_KEY', 'dp40w3rEh3EJMuKMs4thgcnVo2MZq0CfNgvb0Eus')
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', 'K4sXaOf0Ws3Po4e65Xni')
ROBOFLOW_MODEL = os.environ.get('ROBOFLOW_MODEL', 'architectural-detail-detection/1')
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://okwtyttfqbfmcqtenize.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

# Initialize Cohere client
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

# Command A extraction prompt
CLASSIFICATION_PROMPT = """Analyze this construction plan page and extract information.

Return JSON only:
{
  "page_type": "elevation|floor_plan|roof_plan|schedule|detail|cover|section|other",
  "elevation_name": "front|rear|left|right|north|south|east|west|null",
  "scale": "scale if shown or null",
  "dimensions": {
    "wall_heights": ["list heights"],
    "building_width": "width or null",
    "building_length": "length or null"
  },
  "windows": [{"tag": "W1", "width": "3'-0\"", "height": "4'-0\"", "count": 1, "type": "double_hung|slider|fixed"}],
  "doors": [{"tag": "D1", "width": "3'-0\"", "height": "6'-8\"", "type": "entry|interior|slider|garage"}],
  "materials": ["list materials mentioned"],
  "raw_dimensions": ["all dimension text found"]
}

Extract EVERYTHING you can read."""


def image_to_base64(image):
    """Convert PIL image to base64 data URL"""
    if image.width > 1500:
        ratio = 1500 / image.width
        image = image.resize((1500, int(image.height * ratio)))
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    b64 = base64.standard_b64encode(buffer.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def extract_with_command_a(image):
    """Send image to Command A for extraction"""
    try:
        data_url = image_to_base64(image)
        
        response = cohere_client.chat(
            model="command-a-vision-07-2025",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "image": data_url},
                    {"type": "text", "text": CLASSIFICATION_PROMPT}
                ]
            }]
        )
        
        text = response.message.content[0].text
        
        # Try to parse JSON
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            pass
        
        return {"raw_text": text, "parse_error": True}
        
    except Exception as e:
        return {"error": str(e)}


def detect_with_roboflow(image_url):
    """Send image to Roboflow Workflow (RAPID) for object detection"""
    try:
        workflow_url = os.environ.get('ROBOFLOW_WORKFLOW_URL', 
            'https://serverless.roboflow.com/exterior-finishes/workflows/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables')
        
        response = requests.post(
            workflow_url,
            headers={"Content-Type": "application/json"},
            json={
                "api_key": ROBOFLOW_API_KEY,
                "inputs": {
                    "image": {"type": "url", "value": image_url}
                }
            }
        )
        result = response.json()
        
        # RAPID workflow returns nested structure - extract predictions
        if 'outputs' in result and len(result['outputs']) > 0:
            output = result['outputs'][0]
            if 'predictions' in output and 'predictions' in output['predictions']:
                return {
                    'predictions': output['predictions']['predictions'],
                    'image': output['predictions'].get('image', {}),
                    'visualization': output.get('visualization', {})
                }
        
        return result
    except Exception as e:
        return {"error": str(e)}


def calculate_areas(predictions, scale_config):
    """Calculate areas from Roboflow predictions"""
    sqft_per_pixel = scale_config.get('sqft_per_pixel', 0.09)
    lf_per_pixel = scale_config.get('lf_per_pixel', 0.3)
    
    results = {
        'building': [], 'roof': [], 'window': [], 
        'door': [], 'garage': [], 'gable': []
    }
    
    # Group predictions by class
    for pred in predictions:
        class_name = pred.get('class', '').lower()
        if class_name in results:
            width = pred.get('width', 0)
            height = pred.get('height', 0)
            area_sqft = width * height * sqft_per_pixel
            
            results[class_name].append({
                'x': pred.get('x', 0),
                'y': pred.get('y', 0),
                'width': width,
                'height': height,
                'width_ft': width * lf_per_pixel,
                'height_ft': height * lf_per_pixel,
                'area_sqft': area_sqft,
                'confidence': pred.get('confidence', 0)
            })
    
    # Calculate totals
    building_sqft = sum(d['area_sqft'] for d in results['building'])
    roof_sqft = sum(d['area_sqft'] for d in results['roof'])
    window_sqft = sum(d['area_sqft'] for d in results['window'])
    door_sqft = sum(d['area_sqft'] for d in results['door'])
    garage_sqft = sum(d['area_sqft'] for d in results['garage'])
    gable_sqft = sum(d['area_sqft'] for d in results['gable'])
    
    gross_wall_sqft = building_sqft - roof_sqft - gable_sqft
    openings_sqft = window_sqft + door_sqft + garage_sqft
    net_siding_sqft = gross_wall_sqft - openings_sqft
    
    opening_pct = (openings_sqft / gross_wall_sqft * 100) if gross_wall_sqft > 0 else 0
    
    return {
        'detections': results,
        'counts': {k: len(v) for k, v in results.items()},
        'areas': {
            'building_sqft': round(building_sqft, 1),
            'roof_sqft': round(roof_sqft, 1),
            'window_sqft': round(window_sqft, 1),
            'door_sqft': round(door_sqft, 1),
            'garage_sqft': round(garage_sqft, 1),
            'gable_sqft': round(gable_sqft, 1),
            'gross_wall_sqft': round(gross_wall_sqft, 1),
            'net_siding_sqft': round(net_siding_sqft, 1),
            'opening_percentage': round(opening_pct, 1)
        }
    }


def calculate_lineal_footage(roboflow_data, command_a_data):
    """Calculate lineal footage from combined data"""
    
    # Get counts from Roboflow
    window_count = roboflow_data['counts'].get('window', 0)
    door_count = roboflow_data['counts'].get('door', 0)
    garage_count = roboflow_data['counts'].get('garage', 0)
    
    # Try to get average dimensions from Command A schedules
    windows = command_a_data.get('windows', [])
    doors = command_a_data.get('doors', [])
    
    # Parse dimensions (convert "3'-0"" to 3.0)
    def parse_dimension(dim_str):
        if not dim_str:
            return 0
        try:
            # Handle format like "3'-0\"" or "3'-6\""
            dim_str = dim_str.replace('"', '').replace("'", ".")
            parts = dim_str.split('.')
            feet = float(parts[0]) if parts[0] else 0
            inches = float(parts[1]) / 12 if len(parts) > 1 and parts[1] else 0
            return feet + inches
        except:
            return 0
    
    # Calculate average window dimensions from schedule
    if windows:
        avg_window_width = sum(parse_dimension(w.get('width', '3')) for w in windows) / len(windows)
        avg_window_height = sum(parse_dimension(w.get('height', '4')) for w in windows) / len(windows)
    else:
        # Default to standard 3' x 4' window
        avg_window_width = 3.0
        avg_window_height = 4.0
    
    # Calculate average door dimensions from schedule
    if doors:
        avg_door_width = sum(parse_dimension(d.get('width', '3')) for d in doors) / len(doors)
        avg_door_height = sum(parse_dimension(d.get('height', '6.67')) for d in doors) / len(doors)
    else:
        # Default to standard 3' x 6'-8" door
        avg_door_width = 3.0
        avg_door_height = 6.67
    
    # Calculate LF
    return {
        'windows': {
            'count': window_count,
            'avg_width_ft': round(avg_window_width, 2),
            'avg_height_ft': round(avg_window_height, 2),
            'tops_lf': round(window_count * avg_window_width, 1),
            'sills_lf': round(window_count * avg_window_width, 1),
            'sides_lf': round(window_count * avg_window_height * 2, 1),
            'perimeter_lf': round(window_count * (avg_window_width + avg_window_height) * 2, 1)
        },
        'doors': {
            'count': door_count,
            'avg_width_ft': round(avg_door_width, 2),
            'avg_height_ft': round(avg_door_height, 2),
            'tops_lf': round(door_count * avg_door_width, 1),
            'sides_lf': round(door_count * avg_door_height * 2, 1),
            'perimeter_lf': round(door_count * (avg_door_width + avg_door_height) * 2, 1)
        },
        'garages': {
            'count': garage_count,
            'headers_lf': round(garage_count * 16, 1)  # Assume 16' garage doors
        }
    }


def generate_siding_markup(image, predictions, calculations):
    """Generate siding markup visualization"""
    draw = ImageDraw.Draw(image, 'RGBA')
    
    # Colors
    SIDING_COLOR = (66, 135, 245, 100)    # Blue
    WINDOW_COLOR = (255, 152, 0, 120)     # Orange
    DOOR_COLOR = (76, 175, 80, 120)       # Green
    ROOF_COLOR = (156, 39, 176, 100)      # Purple
    
    img_width, img_height = image.size
    
    # Draw building areas first (siding)
    for pred in predictions:
        if pred.get('class', '').lower() == 'building':
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            draw.rectangle([x1, y1, x2, y2], fill=SIDING_COLOR, outline=(66, 135, 245, 255), width=2)
    
    # Draw roofs
    for pred in predictions:
        if pred.get('class', '').lower() == 'roof':
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            draw.rectangle([x1, y1, x2, y2], fill=ROOF_COLOR, outline=(156, 39, 176, 255), width=2)
    
    # Draw windows
    for pred in predictions:
        if pred.get('class', '').lower() == 'window':
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            draw.rectangle([x1, y1, x2, y2], fill=WINDOW_COLOR, outline=(255, 152, 0, 255), width=2)
    
    # Draw doors
    for pred in predictions:
        if pred.get('class', '').lower() in ['door', 'garage']:
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            draw.rectangle([x1, y1, x2, y2], fill=DOOR_COLOR, outline=(76, 175, 80, 255), width=2)
    
    # Add legend
    legend_y = img_height - 120
    legend_x = 20
    box_size = 20
    
    draw.rectangle([legend_x, legend_y, legend_x + 150, legend_y + 100], fill=(255, 255, 255, 220))
    
    items = [
        (SIDING_COLOR, "Siding"),
        (WINDOW_COLOR, "Windows"),
        (DOOR_COLOR, "Doors"),
        (ROOF_COLOR, "Roof")
    ]
    
    for i, (color, label) in enumerate(items):
        y = legend_y + 10 + i * 22
        draw.rectangle([legend_x + 10, y, legend_x + 10 + box_size, y + box_size], fill=color)
        draw.text((legend_x + 40, y), label, fill=(0, 0, 0, 255))
    
    return image


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "services": {
            "cohere": bool(COHERE_API_KEY),
            "roboflow": bool(ROBOFLOW_API_KEY)
        }
    })


@app.route('/extract', methods=['POST'])
def extract():
    """
    Full extraction pipeline
    
    Input:
    {
        "pdf_url": "https://...",  // OR
        "image_urls": ["https://..."],
        "scale_config": {
            "scale_inches": 0.1875,
            "scale_feet": 1,
            "dpi": 144,
            "original_width": 5185,
            "original_height": 3456
        },
        "project_id": "project-123"
    }
    """
    data = request.json
    
    pdf_url = data.get('pdf_url')
    image_urls = data.get('image_urls', [])
    scale_config = data.get('scale_config', {})
    project_id = data.get('project_id', 'unknown')
    
    # Calculate scale factors
    scale_inches = scale_config.get('scale_inches', 0.1875)
    scale_feet = scale_config.get('scale_feet', 1)
    dpi = scale_config.get('dpi', 144)
    original_width = scale_config.get('original_width', 5185)
    
    feet_per_inch = scale_feet / scale_inches
    inches_per_pixel = 1 / dpi
    roboflow_scale = original_width / 640  # Roboflow resizes to 640
    
    sqft_per_pixel = (roboflow_scale * inches_per_pixel * feet_per_inch) ** 2
    lf_per_pixel = roboflow_scale * inches_per_pixel * feet_per_inch
    
    scale_config['sqft_per_pixel'] = sqft_per_pixel
    scale_config['lf_per_pixel'] = lf_per_pixel
    
    results = {
        'project_id': project_id,
        'pages': [],
        'elevation_pages': [],
        'schedule_data': {'windows': [], 'doors': []},
        'combined_measurements': None
    }
    
    # Convert PDF to images if needed
    images = []
    if pdf_url:
        try:
            response = requests.get(pdf_url)
            images = convert_from_bytes(response.content, dpi=150)
        except Exception as e:
            return jsonify({"error": f"Failed to convert PDF: {str(e)}"}), 400
    elif image_urls:
        for url in image_urls:
            try:
                response = requests.get(url)
                img = Image.open(io.BytesIO(response.content))
                images.append(img)
            except Exception as e:
                images.append(None)
    
    # Process each page
    for i, image in enumerate(images, 1):
        if image is None:
            results['pages'].append({'page': i, 'error': 'Failed to load'})
            continue
        
        # Command A extraction
        command_a_result = extract_with_command_a(image)
        
        page_result = {
            'page': i,
            'command_a': command_a_result,
            'roboflow': None,
            'calculations': None
        }
        
        # Collect schedule data
        if command_a_result.get('windows'):
            results['schedule_data']['windows'].extend(command_a_result['windows'])
        if command_a_result.get('doors'):
            results['schedule_data']['doors'].extend(command_a_result['doors'])
        
        # If elevation, also run Roboflow
        if command_a_result.get('page_type') == 'elevation':
            results['elevation_pages'].append(i)
            
            # For Roboflow, we need a URL - upload to temp storage or use provided URL
            if i <= len(image_urls):
                roboflow_result = detect_with_roboflow(image_urls[i-1])
                
                if 'predictions' in roboflow_result:
                    predictions = roboflow_result['predictions']
                    calculations = calculate_areas(predictions, scale_config)
                    lf = calculate_lineal_footage(calculations, command_a_result)
                    
                    page_result['roboflow'] = roboflow_result
                    page_result['calculations'] = {
                        **calculations,
                        'lineal_footage': lf
                    }
        
        results['pages'].append(page_result)
    
    # Combine measurements from all elevation pages
    if results['elevation_pages']:
        combined = {
            'net_siding_sqft': 0,
            'gross_wall_sqft': 0,
            'window_sqft': 0,
            'door_sqft': 0,
            'window_count': 0,
            'door_count': 0,
            'window_perimeter_lf': 0,
            'door_perimeter_lf': 0
        }
        
        for page in results['pages']:
            if page.get('calculations'):
                calc = page['calculations']
                combined['net_siding_sqft'] += calc['areas'].get('net_siding_sqft', 0)
                combined['gross_wall_sqft'] += calc['areas'].get('gross_wall_sqft', 0)
                combined['window_sqft'] += calc['areas'].get('window_sqft', 0)
                combined['door_sqft'] += calc['areas'].get('door_sqft', 0)
                combined['window_count'] += calc['counts'].get('window', 0)
                combined['door_count'] += calc['counts'].get('door', 0)
                
                lf = calc.get('lineal_footage', {})
                combined['window_perimeter_lf'] += lf.get('windows', {}).get('perimeter_lf', 0)
                combined['door_perimeter_lf'] += lf.get('doors', {}).get('perimeter_lf', 0)
        
        results['combined_measurements'] = combined
    
    return jsonify(results)


@app.route('/classify', methods=['POST'])
def classify():
    """Command A classification only"""
    data = request.json
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        result = extract_with_command_a(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect', methods=['POST'])
def detect():
    """Roboflow detection only"""
    data = request.json
    image_url = data.get('image_url')
    scale_config = data.get('scale_config', {})
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    result = detect_with_roboflow(image_url)
    
    if 'predictions' in result:
        calculations = calculate_areas(result['predictions'], scale_config)
        result['calculations'] = calculations
    
    return jsonify(result)


@app.route('/markup', methods=['POST'])
def markup():
    """Generate siding markup visualization"""
    data = request.json
    image_url = data.get('image_url')
    predictions = data.get('predictions', [])
    calculations = data.get('calculations', {})
    
    if not image_url:
        return jsonify({"error": "image_url required"}), 400
    
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGBA')
        
        # If no predictions provided, run Roboflow
        if not predictions:
            roboflow_result = detect_with_roboflow(image_url)
            predictions = roboflow_result.get('predictions', [])
        
        # Generate markup
        marked_image = generate_siding_markup(image, predictions, calculations)
        
        # Convert to base64
        buffer = io.BytesIO()
        marked_image.convert('RGB').save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        b64 = base64.standard_b64encode(buffer.read()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image_base64": b64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True)
