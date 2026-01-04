"""
Fusion Service - Combines OCR, Roboflow detections, and Schedule data

This service implements data fusion to improve measurement accuracy by:
1. Matching OCR callouts (W-1, D-1) to Roboflow detections by proximity
2. Looking up schedule dimensions for matched callouts
3. Applying dimension priority: schedule > OCR > calculated
4. Tracking dimension sources for transparency
5. Flagging discrepancies where sources disagree
"""

import math
from database import supabase_request, get_page


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum pixel distance to match a callout to a detection
MAX_CALLOUT_DISTANCE_PX = 150

# Maximum pixel distance to match a dimension text to a detection
MAX_DIMENSION_DISTANCE_PX = 100

# Dimension source priorities (lower = higher priority)
SOURCE_PRIORITY = {
    'manual': 0,    # User override - highest priority
    'schedule': 1,  # From schedule - very reliable
    'ocr': 2,       # From OCR text on drawing
    'calculated': 3 # From pixel math - fallback
}

# Discrepancy threshold (inches) - flag if sources differ by more than this
DISCREPANCY_THRESHOLD_IN = 3

# Maximum reasonable dimensions by element type (in inches)
# These filter out building-level dimensions being matched to individual elements
MAX_REASONABLE_DIMENSIONS = {
    'window': {'width': 120, 'height': 96},   # 10' x 8' max window
    'door': {'width': 96, 'height': 120},      # 8' x 10' max door
    'garage': {'width': 216, 'height': 120},   # 18' x 10' max garage
    'default': {'width': 240, 'height': 240}   # 20' x 20' fallback
}

# Minimum reasonable dimensions (filter out noise)
MIN_REASONABLE_DIMENSIONS = {
    'window': {'width': 12, 'height': 12},    # 1' x 1' min window
    'door': {'width': 24, 'height': 72},       # 2' x 6' min door
    'garage': {'width': 84, 'height': 72},     # 7' x 6' min garage
    'default': {'width': 6, 'height': 6}       # 6" x 6" fallback
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_reasonable_dimension(width_in, height_in, element_class):
    """
    Check if dimensions are reasonable for the element type.
    
    Filters out:
    - Building-level dimensions matched to windows/doors
    - Noise/artifacts that are too small
    
    Args:
        width_in: Width in inches
        height_in: Height in inches  
        element_class: 'window', 'door', 'garage', etc.
    
    Returns:
        bool: True if dimensions are reasonable
    """
    max_dims = MAX_REASONABLE_DIMENSIONS.get(element_class, MAX_REASONABLE_DIMENSIONS['default'])
    min_dims = MIN_REASONABLE_DIMENSIONS.get(element_class, MIN_REASONABLE_DIMENSIONS['default'])
    
    # Check width if provided
    if width_in is not None:
        if width_in > max_dims['width']:
            return False
        if width_in < min_dims['width']:
            return False
    
    # Check height if provided
    if height_in is not None:
        if height_in > max_dims['height']:
            return False
        if height_in < min_dims['height']:
            return False
    
    return True


# =============================================================================
# MAIN FUSION FUNCTION
# =============================================================================

def fuse_page_data(page_id):
    """
    Main fusion function for a single page.
    
    Combines:
    - Roboflow detections from extraction_detection_details
    - OCR data from extraction_ocr_data
    - Schedule data from extraction_pages (schedule pages in same job)
    
    Updates:
    - extraction_detection_details with matched_tag, dimension_source, final dimensions
    - extraction_dimension_sources with all source records
    
    Returns:
        Dict with fusion results and statistics
    """
    # Get page info
    page = get_page(page_id)
    if not page:
        return {"error": "Page not found"}
    
    job_id = page.get('job_id')
    
    # Get detections for this page
    detections = supabase_request('GET', 'extraction_detection_details', filters={
        'page_id': f'eq.{page_id}',
        'status': 'neq.deleted'
    })
    
    if not detections:
        return {"error": "No detections found for page", "page_id": page_id}
    
    # Get OCR data for this page
    ocr_data = supabase_request('GET', 'extraction_ocr_data', filters={
        'page_id': f'eq.{page_id}'
    })
    ocr = ocr_data[0] if ocr_data else {}
    
    # Get schedule data for this job
    schedule_data = get_schedule_data_for_job(job_id)
    
    # Track results
    results = {
        'page_id': page_id,
        'job_id': job_id,
        'detections_processed': 0,
        'callouts_matched': 0,
        'schedule_matches': 0,
        'ocr_dimension_matches': 0,
        'rejected_ocr_matches': [],
        'discrepancies': [],
        'updated_detections': []
    }
    
    # Get callouts and dimensions from OCR
    callouts = ocr.get('element_callouts', [])
    dimension_texts = ocr.get('dimension_text', [])
    
    # Process each detection
    for det in detections:
        det_id = det['id']
        det_class = det.get('class', '').lower()
        det_cx = det.get('pixel_x', 0)
        det_cy = det.get('pixel_y', 0)
        
        # Only process windows and doors for callout matching
        if det_class not in ['window', 'door']:
            results['detections_processed'] += 1
            continue
        
        # Current calculated dimensions (fallback)
        calc_width_in = det.get('real_width_in')
        calc_height_in = det.get('real_height_in')
        
        # Try to match a callout to this detection
        matched_callout = match_callout_to_detection(callouts, det_cx, det_cy, det_class)
        
        # Initialize dimension tracking
        final_width_in = calc_width_in
        final_height_in = calc_height_in
        dimension_source = 'calculated'
        matched_tag = None
        has_discrepancy = False
        discrepancy_notes = None
        sources_found = []
        
        if matched_callout:
            matched_tag = matched_callout['tag']
            results['callouts_matched'] += 1
            
            # Look up schedule dimensions
            schedule_dims = lookup_schedule_dimensions(matched_tag, schedule_data, det_class)
            
            if schedule_dims:
                results['schedule_matches'] += 1
                sources_found.append({
                    'source': 'schedule',
                    'width_in': schedule_dims['width_inches'],
                    'height_in': schedule_dims['height_inches'],
                    'tag': matched_tag
                })
                
                # Schedule is highest priority
                final_width_in = schedule_dims['width_inches']
                final_height_in = schedule_dims['height_inches']
                dimension_source = 'schedule'
        
        # Try to match dimension text to this detection
        dim_match = match_dimension_to_detection(dimension_texts, det_cx, det_cy, det)
        
        if dim_match:
            ocr_width = dim_match.get('width_inches')
            ocr_height = dim_match.get('height_inches')
            
            # Validate OCR dimensions are reasonable for this element type
            if is_reasonable_dimension(ocr_width, ocr_height, det_class):
                results['ocr_dimension_matches'] += 1
                sources_found.append({
                    'source': 'ocr',
                    'width_in': ocr_width,
                    'height_in': ocr_height,
                    'raw_text': dim_match.get('raw_text')
                })
                
                # Use OCR if no schedule match
                if dimension_source == 'calculated':
                    if ocr_width:
                        final_width_in = ocr_width
                    if ocr_height:
                        final_height_in = ocr_height
                    dimension_source = 'ocr'
            else:
                # Log rejected dimension for debugging
                results['rejected_ocr_matches'].append({
                    'detection_id': det_id,
                    'class': det_class,
                    'ocr_width': ocr_width,
                    'ocr_height': ocr_height,
                    'reason': 'dimensions_out_of_range'
                })
        
        # Add calculated as a source for comparison
        sources_found.append({
            'source': 'calculated',
            'width_in': calc_width_in,
            'height_in': calc_height_in
        })
        
        # Check for discrepancies between sources
        discrepancy = check_discrepancies(sources_found)
        if discrepancy:
            has_discrepancy = True
            discrepancy_notes = discrepancy
            results['discrepancies'].append({
                'detection_id': det_id,
                'class': det_class,
                'tag': matched_tag,
                'notes': discrepancy,
                'sources': sources_found
            })
        
        # Calculate final area
        final_area_sf = None
        if final_width_in and final_height_in:
            final_area_sf = (final_width_in * final_height_in) / 144.0
        
        # Update detection record
        update_data = {
            'matched_tag': matched_tag,
            'dimension_source': dimension_source,
            'final_width_in': final_width_in,
            'final_height_in': final_height_in,
            'final_area_sf': final_area_sf,
            'has_discrepancy': has_discrepancy,
            'discrepancy_notes': discrepancy_notes
        }
        
        supabase_request('PATCH', 'extraction_detection_details',
                        data=update_data,
                        filters={'id': f'eq.{det_id}'})
        
        # Store dimension sources
        store_dimension_sources(det_id, sources_found, dimension_source)
        
        results['updated_detections'].append({
            'detection_id': det_id,
            'class': det_class,
            'matched_tag': matched_tag,
            'dimension_source': dimension_source,
            'final_width_in': final_width_in,
            'final_height_in': final_height_in
        })
        
        results['detections_processed'] += 1
    
    # Update page with wall height from OCR if available
    if ocr.get('average_wall_height_ft'):
        supabase_request('PATCH', 'extraction_elevation_calcs',
                        data={
                            'wall_height_ft': ocr['average_wall_height_ft'],
                            'wall_height_source': 'ocr'
                        },
                        filters={'page_id': f'eq.{page_id}'})
    
    return results


def fuse_job_data(job_id):
    """
    Run fusion on all elevation pages in a job.
    
    Returns:
        Dict with aggregated results for all pages
    """
    # Get all elevation pages
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.elevation'
    })
    
    if not pages:
        return {"error": "No elevation pages found", "job_id": job_id}
    
    results = {
        'job_id': job_id,
        'pages_processed': 0,
        'total_callouts_matched': 0,
        'total_schedule_matches': 0,
        'total_discrepancies': 0,
        'page_results': []
    }
    
    for page in pages:
        page_result = fuse_page_data(page['id'])
        
        if not page_result.get('error'):
            results['pages_processed'] += 1
            results['total_callouts_matched'] += page_result.get('callouts_matched', 0)
            results['total_schedule_matches'] += page_result.get('schedule_matches', 0)
            results['total_discrepancies'] += len(page_result.get('discrepancies', []))
        
        results['page_results'].append({
            'page_id': page['id'],
            'page_number': page.get('page_number'),
            'elevation_name': page.get('elevation_name'),
            **page_result
        })
    
    return results


# =============================================================================
# MATCHING FUNCTIONS
# =============================================================================

def match_callout_to_detection(callouts, det_x, det_y, det_class):
    """
    Match an OCR callout to a detection by proximity.
    
    Args:
        callouts: List of OCR callouts with x_pixel, y_pixel, element_type
        det_x: Detection center X pixel
        det_y: Detection center Y pixel
        det_class: Detection class (window, door)
    
    Returns:
        Best matching callout or None
    """
    if not callouts:
        return None
    
    best_match = None
    best_distance = float('inf')
    
    for callout in callouts:
        # Check element type matches
        callout_type = callout.get('element_type', '').lower()
        if callout_type and callout_type != det_class:
            continue
        
        # Calculate distance
        cx = callout.get('x_pixel', 0)
        cy = callout.get('y_pixel', 0)
        distance = math.sqrt((cx - det_x)**2 + (cy - det_y)**2)
        
        if distance < MAX_CALLOUT_DISTANCE_PX and distance < best_distance:
            best_distance = distance
            best_match = callout
    
    return best_match


def match_dimension_to_detection(dimensions, det_x, det_y, detection):
    """
    Match dimension text to a detection.
    
    Looks for dimension strings near the detection that could describe its size.
    
    Args:
        dimensions: List of OCR dimension texts
        det_x: Detection center X pixel
        det_y: Detection center Y pixel
        detection: Full detection dict
    
    Returns:
        Dict with matched dimensions or None
    """
    if not dimensions:
        return None
    
    det_width = detection.get('pixel_width', 0)
    det_height = detection.get('pixel_height', 0)
    
    # Look for horizontal dimension (width) near top or bottom of detection
    # Look for vertical dimension (height) near left or right of detection
    
    width_match = None
    height_match = None
    
    for dim in dimensions:
        dx = dim.get('x_pixel', 0)
        dy = dim.get('y_pixel', 0)
        orientation = dim.get('orientation', '').lower()
        value = dim.get('value_inches')
        
        if not value:
            continue
        
        # Check if dimension is near this detection
        distance = math.sqrt((dx - det_x)**2 + (dy - det_y)**2)
        
        if distance > MAX_DIMENSION_DISTANCE_PX + max(det_width, det_height) / 2:
            continue
        
        # Horizontal dimensions are likely widths
        if orientation == 'horizontal' and not width_match:
            width_match = value
        
        # Vertical dimensions are likely heights
        elif orientation == 'vertical' and not height_match:
            height_match = value
    
    if width_match or height_match:
        return {
            'width_inches': width_match,
            'height_inches': height_match
        }
    
    return None


# =============================================================================
# SCHEDULE LOOKUP
# =============================================================================

def get_schedule_data_for_job(job_id):
    """
    Get all schedule data (windows and doors) for a job.
    
    Looks for schedule pages and extracts window/door specifications.
    
    Returns:
        Dict with 'windows' and 'doors' lists
    """
    # Get schedule pages
    schedule_pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.schedule'
    })
    
    result = {
        'windows': [],
        'doors': []
    }
    
    if not schedule_pages:
        return result
    
    for page in schedule_pages:
        extraction_data = page.get('extraction_data', {})
        
        # Add windows
        windows = extraction_data.get('windows', [])
        for w in windows:
            w['source_page_id'] = page['id']
        result['windows'].extend(windows)
        
        # Add doors
        doors = extraction_data.get('doors', [])
        for d in doors:
            d['source_page_id'] = page['id']
        result['doors'].extend(doors)
    
    return result


def lookup_schedule_dimensions(tag, schedule_data, element_type):
    """
    Look up dimensions from schedule by tag.
    
    Args:
        tag: Element tag like "W-1", "D-2"
        schedule_data: Schedule data dict with 'windows' and 'doors'
        element_type: 'window' or 'door'
    
    Returns:
        Dict with width_inches, height_inches or None
    """
    if not tag or not schedule_data:
        return None
    
    # Normalize tag for comparison
    tag_normalized = tag.upper().replace(' ', '').replace('-', '')
    
    # Choose list based on element type
    if element_type == 'window':
        items = schedule_data.get('windows', [])
    elif element_type == 'door':
        items = schedule_data.get('doors', [])
    else:
        items = schedule_data.get('windows', []) + schedule_data.get('doors', [])
    
    for item in items:
        item_tag = item.get('tag', '').upper().replace(' ', '').replace('-', '')
        
        if item_tag == tag_normalized:
            width = item.get('width_inches')
            height = item.get('height_inches')
            
            if width and height:
                return {
                    'width_inches': width,
                    'height_inches': height,
                    'tag': item.get('tag'),
                    'type': item.get('type'),
                    'source_page_id': item.get('source_page_id')
                }
    
    return None


# =============================================================================
# DISCREPANCY DETECTION
# =============================================================================

def check_discrepancies(sources):
    """
    Check if dimension sources have significant discrepancies.
    
    Args:
        sources: List of source dicts with width_in, height_in
    
    Returns:
        String describing discrepancy or None
    """
    if len(sources) < 2:
        return None
    
    # Get all non-None width and height values
    widths = [(s['source'], s.get('width_in')) for s in sources if s.get('width_in')]
    heights = [(s['source'], s.get('height_in')) for s in sources if s.get('height_in')]
    
    discrepancies = []
    
    # Check width discrepancies
    if len(widths) >= 2:
        width_values = [w[1] for w in widths]
        max_diff = max(width_values) - min(width_values)
        if max_diff > DISCREPANCY_THRESHOLD_IN:
            discrepancies.append(f"Width varies by {max_diff:.1f}in")
    
    # Check height discrepancies
    if len(heights) >= 2:
        height_values = [h[1] for h in heights]
        max_diff = max(height_values) - min(height_values)
        if max_diff > DISCREPANCY_THRESHOLD_IN:
            discrepancies.append(f"Height varies by {max_diff:.1f}in")
    
    if discrepancies:
        return "; ".join(discrepancies)
    
    return None


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def store_dimension_sources(detection_id, sources, primary_source):
    """
    Store all dimension sources for a detection.
    
    Args:
        detection_id: UUID of detection
        sources: List of source dicts
        primary_source: Which source was used ('schedule', 'ocr', 'calculated')
    """
    for source in sources:
        source_type = source.get('source')
        is_primary = (source_type == primary_source)
        
        record = {
            'detection_id': detection_id,
            'source_type': source_type,
            'source_priority': SOURCE_PRIORITY.get(source_type, 99),
            'width_inches': source.get('width_in'),
            'height_inches': source.get('height_in'),
            'is_primary': is_primary
        }
        
        # Add OCR-specific fields
        if source_type == 'ocr':
            record['raw_text'] = source.get('raw_text')
        
        # Add schedule-specific fields
        if source_type == 'schedule':
            record['schedule_tag'] = source.get('tag')
        
        supabase_request('POST', 'extraction_dimension_sources', record)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_fusion_summary(job_id):
    """
    Get a summary of fusion results for a job.
    
    Returns breakdown of dimension sources and discrepancies.
    """
    # Get all detections for job
    detections = supabase_request('GET', 'extraction_detection_details', filters={
        'job_id': f'eq.{job_id}',
        'select': 'class,dimension_source,has_discrepancy,matched_tag'
    })
    
    if not detections:
        return {"error": "No detections found"}
    
    summary = {
        'job_id': job_id,
        'total_detections': len(detections),
        'by_source': {
            'schedule': 0,
            'ocr': 0,
            'calculated': 0,
            'manual': 0
        },
        'by_class': {},
        'matched_to_schedule': 0,
        'with_discrepancies': 0
    }
    
    for det in detections:
        source = det.get('dimension_source', 'calculated')
        cls = det.get('class', 'unknown')
        
        summary['by_source'][source] = summary['by_source'].get(source, 0) + 1
        summary['by_class'][cls] = summary['by_class'].get(cls, 0) + 1
        
        if det.get('matched_tag'):
            summary['matched_to_schedule'] += 1
        
        if det.get('has_discrepancy'):
            summary['with_discrepancies'] += 1
    
    return summary
