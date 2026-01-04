"""
Takeoff calculation service
"""

import math
from database import (
    get_page, get_elevation_pages, supabase_request
)
from config import config


def calculate_takeoff_for_page(page_id):
    """
    Calculate all takeoff measurements for a single elevation page.
    
    Stores results in:
    - extraction_detection_details (individual detections)
    - extraction_elevation_calcs (aggregated per elevation)
    
    FORMULA: Gross Facade = Building Area - Roof Area
    """
    # Get page data
    page = get_page(page_id)
    if not page:
        return {"error": "Page not found"}
    
    job_id = page.get('job_id')
    elevation_name = page.get('elevation_name', 'unknown')
    extraction_data = page.get('extraction_data', {})
    predictions = extraction_data.get('raw_predictions', [])
    scale_ratio = float(page.get('scale_ratio') or 48)
    dpi = int(page.get('dpi') or config.DEFAULT_DPI)
    
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
        'building_area_sf': 0,
        'roof_area_sf': 0,
        'window_area_sf': 0, 'door_area_sf': 0,
        'garage_area_sf': 0, 'gable_area_sf': 0
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
            area_sf = (real_width_in * real_height_in) / 144 / 2
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
            areas['building_area_sf'] += area_sf
        elif class_name == 'roof':
            counts['roof'] += 1
            areas['roof_area_sf'] += area_sf
            linear['roof_eave_lf'] += real_width_ft
        elif class_name == 'window':
            counts['window'] += 1
            areas['window_area_sf'] += area_sf
            linear['window_perimeter_lf'] += perimeter_lf
            linear['window_head_lf'] += real_width_ft
            linear['window_jamb_lf'] += real_height_ft * 2
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
            rake_length = math.sqrt((real_width_ft / 2) ** 2 + real_height_ft ** 2) * 2
            linear['gable_rake_lf'] += rake_length
    
    # CRITICAL: Gross Facade = Building - Roof
    gross_facade_sf = areas['building_area_sf'] - areas['roof_area_sf']
    
    # Net siding = Gross Facade - Openings + Gables
    net_siding_sf = gross_facade_sf - areas['window_area_sf'] - areas['door_area_sf'] - areas['garage_area_sf'] + areas['gable_area_sf']
    
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
        'gross_facade_sf': round(gross_facade_sf, 2),
        'window_area_sf': round(areas['window_area_sf'], 2),
        'door_area_sf': round(areas['door_area_sf'], 2),
        'garage_area_sf': round(areas['garage_area_sf'], 2),
        'gable_area_sf': round(areas['gable_area_sf'], 2),
        'roof_area_sf': round(areas['roof_area_sf'], 2),
        'net_siding_sf': round(net_siding_sf, 2),
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
        "areas": {
            'building_area_sf': round(areas['building_area_sf'], 2),
            'roof_area_sf': round(areas['roof_area_sf'], 2),
            'gross_facade_sf': round(gross_facade_sf, 2),
            'window_area_sf': round(areas['window_area_sf'], 2),
            'door_area_sf': round(areas['door_area_sf'], 2),
            'garage_area_sf': round(areas['garage_area_sf'], 2),
            'gable_area_sf': round(areas['gable_area_sf'], 2)
        },
        "linear": {k: round(v, 2) for k, v in linear.items()},
        "net_siding_sf": round(net_siding_sf, 2)
    }


def calculate_takeoff_for_job(job_id):
    """
    Calculate takeoff for all elevation pages in a job.
    Aggregates results into extraction_job_totals.
    """
    # Get all elevation pages
    pages = get_elevation_pages(job_id, status='complete')
    
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
        'total_openings_sf': round(sum(
            float(c.get('window_area_sf', 0)) + float(c.get('door_area_sf', 0)) + float(c.get('garage_area_sf', 0))
            for c in calcs
        ), 2),
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
