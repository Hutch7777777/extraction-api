"""
Cross-reference service - matches schedule data with detections
"""

from database import supabase_request
from geometry import calculate_derived_measurements


def build_cross_references(job_id):
    """
    Build cross-references between schedule and detections.
    
    This links window/door tags from schedules to detected openings,
    calculating derived measurements for trim, flashing, etc.
    
    Stores results in:
    - extraction_cross_refs (individual items)
    - extraction_takeoff_summary (aggregated totals)
    """
    # Get schedule pages
    schedule_pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.schedule',
        'status': 'eq.complete'
    })
    
    if not schedule_pages:
        return {"error": "No schedule data"}
    
    # Aggregate schedule data by tag
    windows_by_tag = {}
    doors_by_tag = {}
    
    for page in schedule_pages:
        extraction_data = page.get('extraction_data', {})
        page_id = page.get('id')
        
        for window in extraction_data.get('windows', []):
            tag = window.get('tag', 'UNKNOWN')
            # Keep the one with higher quantity if duplicate
            if tag not in windows_by_tag or window.get('qty', 1) > windows_by_tag[tag]['qty']:
                windows_by_tag[tag] = {**window, 'schedule_page_id': page_id}
        
        for door in extraction_data.get('doors', []):
            tag = door.get('tag', 'UNKNOWN')
            if tag not in doors_by_tag or door.get('qty', 1) > doors_by_tag[tag]['qty']:
                doors_by_tag[tag] = {**door, 'schedule_page_id': page_id}
    
    # Get elevation pages for detection counts
    elevation_pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.elevation',
        'status': 'eq.complete'
    })
    
    elevation_page_ids = [p.get('id') for p in elevation_pages or []]
    
    # Count detections from elevations
    total_detected_windows = sum(
        p.get('extraction_data', {}).get('measurements', {}).get('counts', {}).get('window', 0)
        for p in elevation_pages or []
    )
    total_detected_doors = sum(
        p.get('extraction_data', {}).get('measurements', {}).get('counts', {}).get('door', 0)
        for p in elevation_pages or []
    )
    
    # Delete existing cross-refs for this job
    supabase_request('DELETE', 'extraction_cross_refs', filters={'job_id': f'eq.{job_id}'})
    
    # Create cross-refs for windows
    for tag, window in windows_by_tag.items():
        width = window.get('width_inches', 0)
        height = window.get('height_inches', 0)
        qty = window.get('qty', 1)
        
        derived = calculate_derived_measurements(width, height, qty, 'window')
        
        supabase_request('POST', 'extraction_cross_refs', {
            'job_id': job_id,
            'element_type': 'window',
            'tag': tag,
            'schedule_width': width,
            'schedule_height': height,
            'schedule_qty': qty,
            'schedule_type': window.get('type', ''),
            'schedule_page_id': window.get('schedule_page_id'),
            'elevation_page_ids': [str(x) for x in elevation_page_ids],
            'head_trim_lf': derived['head_trim_lf'],
            'jamb_trim_lf': derived['jamb_trim_lf'],
            'sill_trim_lf': derived['sill_trim_lf'],
            'casing_lf': derived['casing_lf'],
            'rough_opening_width': derived['rough_opening_width'],
            'rough_opening_height': derived['rough_opening_height'],
            'head_flashing_lf': derived['head_flashing_lf'],
            'sill_pan_lf': derived['sill_pan_lf'],
            'needs_review': False
        })
    
    # Create cross-refs for doors
    for tag, door in doors_by_tag.items():
        width = door.get('width_inches', 0)
        height = door.get('height_inches', 0)
        qty = door.get('qty', 1)
        
        derived = calculate_derived_measurements(width, height, qty, 'door')
        
        supabase_request('POST', 'extraction_cross_refs', {
            'job_id': job_id,
            'element_type': 'door',
            'tag': tag,
            'schedule_width': width,
            'schedule_height': height,
            'schedule_qty': qty,
            'schedule_type': door.get('type', ''),
            'schedule_page_id': door.get('schedule_page_id'),
            'elevation_page_ids': [str(x) for x in elevation_page_ids],
            'head_trim_lf': derived['head_trim_lf'],
            'jamb_trim_lf': derived['jamb_trim_lf'],
            'sill_trim_lf': derived['sill_trim_lf'],
            'casing_lf': derived['casing_lf'],
            'rough_opening_width': derived['rough_opening_width'],
            'rough_opening_height': derived['rough_opening_height'],
            'head_flashing_lf': derived['head_flashing_lf'],
            'sill_pan_lf': derived['sill_pan_lf'],
            'needs_review': False
        })
    
    # Build summary
    summary = {
        'job_id': job_id,
        'total_windows': sum(w.get('qty', 1) for w in windows_by_tag.values()),
        'total_window_sqft': round(sum(
            (w.get('width_inches', 0) * w.get('height_inches', 0) * w.get('qty', 1)) / 144
            for w in windows_by_tag.values()
        ), 2),
        'total_window_head_trim_lf': round(sum(
            (w.get('width_inches', 0) * w.get('qty', 1)) / 12
            for w in windows_by_tag.values()
        ), 2),
        'total_window_jamb_trim_lf': round(sum(
            (w.get('height_inches', 0) * 2 * w.get('qty', 1)) / 12
            for w in windows_by_tag.values()
        ), 2),
        'total_window_sill_trim_lf': round(sum(
            (w.get('width_inches', 0) * w.get('qty', 1)) / 12
            for w in windows_by_tag.values()
        ), 2),
        'total_doors': sum(d.get('qty', 1) for d in doors_by_tag.values()),
        'total_door_sqft': round(sum(
            (d.get('width_inches', 0) * d.get('height_inches', 0) * d.get('qty', 1)) / 144
            for d in doors_by_tag.values()
        ), 2),
        'total_door_head_trim_lf': round(sum(
            (d.get('width_inches', 0) * d.get('qty', 1)) / 12
            for d in doors_by_tag.values()
        ), 2),
        'total_door_jamb_trim_lf': round(sum(
            (d.get('height_inches', 0) * 2 * d.get('qty', 1)) / 12
            for d in doors_by_tag.values()
        ), 2),
        'windows_by_tag': {
            t: {**w, 'schedule_page_id': str(w['schedule_page_id'])}
            for t, w in windows_by_tag.items()
        },
        'doors_by_tag': {
            t: {**d, 'schedule_page_id': str(d['schedule_page_id'])}
            for t, d in doors_by_tag.items()
        },
        'discrepancies': {
            'schedule_windows': sum(w.get('qty', 1) for w in windows_by_tag.values()),
            'detected_windows': total_detected_windows,
            'schedule_doors': sum(d.get('qty', 1) for d in doors_by_tag.values()),
            'detected_doors': total_detected_doors
        }
    }
    
    # Upsert summary
    existing = supabase_request('GET', 'extraction_takeoff_summary', filters={'job_id': f'eq.{job_id}'})
    if existing:
        supabase_request('PATCH', 'extraction_takeoff_summary', summary, {'job_id': f'eq.{job_id}'})
    else:
        supabase_request('POST', 'extraction_takeoff_summary', summary)
    
    return {
        "success": True,
        "windows_count": len(windows_by_tag),
        "doors_count": len(doors_by_tag),
        "discrepancies": summary['discrepancies']
    }
