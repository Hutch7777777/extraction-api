"""
Linear Service - Corner and linear element calculations

Phase 4: Enhanced corner counting and linear measurements

Calculates:
- Outside corner LF (corner trim)
- Inside corner LF (J-channel)
- Building perimeter LF (starter strip, water table)
- Band board LF (between floors)
- Frieze board LF (at soffit)
"""

from database import supabase_request, get_page


# =============================================================================
# CONSTANTS
# =============================================================================

# Default values (feet)
DEFAULT_WALL_HEIGHT = 9.0
DEFAULT_STORIES = 2

# Material lengths for calculating quantities
CORNER_POST_LENGTH = 10.0  # 10' corner posts
J_CHANNEL_LENGTH = 12.5    # 12.5' J-channel
STARTER_STRIP_LENGTH = 12.0  # 12' starter strip


# =============================================================================
# WALL HEIGHT CALCULATIONS
# =============================================================================

def get_wall_heights_from_ocr(job_id):
    """
    Get wall heights from OCR data.
    
    Returns:
        Dict with floor-by-floor heights and totals
    """
    # Get OCR data for this job's elevations
    ocr_data = supabase_request('GET', 'extraction_ocr_data', filters={
        'select': 'page_id,wall_heights,extraction_confidence',
        'order': 'created_at.asc'
    })
    
    if not ocr_data:
        return {
            'source': 'default',
            'first_floor_ft': DEFAULT_WALL_HEIGHT,
            'second_floor_ft': DEFAULT_WALL_HEIGHT,
            'total_wall_height_ft': DEFAULT_WALL_HEIGHT * DEFAULT_STORIES,
            'story_count': DEFAULT_STORIES
        }
    
    # Filter to this job
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.elevation',
        'select': 'id'
    })
    page_ids = [p['id'] for p in (pages or [])]
    
    job_ocr = [o for o in ocr_data if o.get('page_id') in page_ids]
    
    if not job_ocr:
        return {
            'source': 'default',
            'first_floor_ft': DEFAULT_WALL_HEIGHT,
            'second_floor_ft': DEFAULT_WALL_HEIGHT,
            'total_wall_height_ft': DEFAULT_WALL_HEIGHT * DEFAULT_STORIES,
            'story_count': DEFAULT_STORIES
        }
    
    # Extract wall heights from all elevations
    first_floor_heights = []
    second_floor_heights = []
    
    for ocr in job_ocr:
        for wh in ocr.get('wall_heights', []):
            label = (wh.get('label') or '').upper()
            inches = wh.get('value_inches')
            
            if inches and inches > 0:
                height_ft = inches / 12.0
                
                if '1ST' in label or 'FIRST' in label or '1' in label:
                    first_floor_heights.append(height_ft)
                elif '2ND' in label or 'SECOND' in label or '2' in label:
                    second_floor_heights.append(height_ft)
    
    # Calculate averages
    first_floor_avg = sum(first_floor_heights) / len(first_floor_heights) if first_floor_heights else DEFAULT_WALL_HEIGHT
    second_floor_avg = sum(second_floor_heights) / len(second_floor_heights) if second_floor_heights else DEFAULT_WALL_HEIGHT
    
    # Determine story count
    story_count = 1
    if second_floor_heights:
        story_count = 2
    
    total_height = first_floor_avg
    if story_count == 2:
        total_height += second_floor_avg
    
    return {
        'source': 'ocr',
        'first_floor_ft': round(first_floor_avg, 2),
        'second_floor_ft': round(second_floor_avg, 2) if story_count > 1 else None,
        'total_wall_height_ft': round(total_height, 2),
        'story_count': story_count,
        'samples': {
            'first_floor': first_floor_heights,
            'second_floor': second_floor_heights
        }
    }


# =============================================================================
# CORNER CALCULATIONS
# =============================================================================

def calculate_corner_lf(outside_corners, inside_corners, wall_height_ft):
    """
    Calculate linear feet for corner trim.
    
    Args:
        outside_corners: Count of outside (90°) corners
        inside_corners: Count of inside corners
        wall_height_ft: Total wall height (all stories)
    
    Returns:
        Dict with corner LF calculations
    """
    outside_lf = (outside_corners or 0) * wall_height_ft
    inside_lf = (inside_corners or 0) * wall_height_ft
    
    return {
        'outside_corners_count': outside_corners or 0,
        'inside_corners_count': inside_corners or 0,
        'wall_height_used_ft': wall_height_ft,
        'outside_corners_lf': round(outside_lf, 2),
        'inside_corners_lf': round(inside_lf, 2),
        'total_corner_lf': round(outside_lf + inside_lf, 2),
        # Material quantities
        'corner_posts_needed': calculate_pieces_needed(outside_lf, CORNER_POST_LENGTH),
        'j_channel_for_corners_lf': round(inside_lf, 2),
        'j_channel_pieces_needed': calculate_pieces_needed(inside_lf, J_CHANNEL_LENGTH)
    }


def calculate_pieces_needed(total_lf, piece_length):
    """
    Calculate number of pieces needed with 10% waste.
    """
    if total_lf <= 0:
        return 0
    
    pieces = total_lf / piece_length
    with_waste = pieces * 1.10  # 10% waste
    return int(round(with_waste + 0.5))  # Round up


# =============================================================================
# PERIMETER CALCULATIONS
# =============================================================================

def calculate_perimeter_from_facade(gross_facade_sf, wall_height_ft):
    """
    Estimate building perimeter from facade area.
    
    Perimeter ≈ Facade SF / Wall Height
    
    This is an approximation - actual perimeter from floor plan is more accurate.
    """
    if not gross_facade_sf or not wall_height_ft:
        return None
    
    perimeter_lf = gross_facade_sf / wall_height_ft
    return round(perimeter_lf, 2)


def calculate_perimeter_elements(perimeter_lf, story_count=2):
    """
    Calculate linear elements based on building perimeter.
    
    Args:
        perimeter_lf: Building perimeter in linear feet
        story_count: Number of stories
    
    Returns:
        Dict with linear element quantities
    """
    if not perimeter_lf:
        return {}
    
    return {
        'building_perimeter_lf': perimeter_lf,
        # Starter strip at base
        'starter_strip_lf': perimeter_lf,
        'starter_strip_pieces': calculate_pieces_needed(perimeter_lf, STARTER_STRIP_LENGTH),
        # Water table (if used)
        'water_table_lf': perimeter_lf,
        # Band board between floors (for 2+ story)
        'band_board_lf': perimeter_lf * (story_count - 1) if story_count > 1 else 0,
        # Frieze board at soffit
        'frieze_board_lf': perimeter_lf
    }


# =============================================================================
# MAIN CALCULATION FUNCTIONS
# =============================================================================

def calculate_linear_elements_for_job(job_id):
    """
    Calculate all linear elements for a job.
    
    Combines:
    - Corner data from floor plan analysis
    - Wall heights from elevation OCR
    - Perimeter calculations
    
    Returns:
        Dict with all linear element calculations
    """
    # Get wall heights from OCR
    wall_data = get_wall_heights_from_ocr(job_id)
    
    # Get job totals (has corner counts and facade areas)
    job_totals = supabase_request('GET', 'extraction_job_totals', filters={
        'job_id': f'eq.{job_id}'
    })
    
    if not job_totals:
        return {"error": "No job totals found - run extraction first"}
    
    totals = job_totals[0]
    
    # Get corner counts
    outside_corners = totals.get('outside_corners_count') or 0
    inside_corners = totals.get('inside_corners_count') or 0
    
    # Calculate corner LF with OCR wall height
    corner_calcs = calculate_corner_lf(
        outside_corners,
        inside_corners,
        wall_data['total_wall_height_ft']
    )
    
    # Calculate perimeter from facade
    perimeter_lf = calculate_perimeter_from_facade(
        totals.get('total_gross_facade_sf'),
        wall_data['total_wall_height_ft']
    )
    
    # Calculate perimeter elements
    perimeter_calcs = calculate_perimeter_elements(
        perimeter_lf,
        wall_data['story_count']
    )
    
    # Get existing trim calculations from elevations
    trim_calcs = {
        'window_perimeter_lf': totals.get('total_window_perimeter_lf'),
        'window_head_lf': totals.get('total_window_head_lf'),
        'window_jamb_lf': totals.get('total_window_jamb_lf'),
        'window_sill_lf': totals.get('total_window_sill_lf'),
        'door_perimeter_lf': totals.get('total_door_perimeter_lf'),
        'door_head_lf': totals.get('total_door_head_lf'),
        'door_jamb_lf': totals.get('total_door_jamb_lf'),
        'garage_head_lf': totals.get('total_garage_head_lf'),
        'gable_rake_lf': totals.get('total_gable_rake_lf'),
    }
    
    # Compile results
    result = {
        'job_id': job_id,
        'wall_heights': wall_data,
        'corners': corner_calcs,
        'perimeter': perimeter_calcs,
        'trim': trim_calcs,
        'summary': {
            'total_corner_lf': corner_calcs['total_corner_lf'],
            'total_perimeter_lf': perimeter_lf,
            'total_window_trim_lf': trim_calcs['window_perimeter_lf'],
            'total_door_trim_lf': (trim_calcs['door_perimeter_lf'] or 0) + (trim_calcs['garage_head_lf'] or 0) * 2,
            'total_rake_lf': trim_calcs['gable_rake_lf']
        }
    }
    
    # Store in linear_elements table
    store_linear_elements(job_id, result)
    
    # Update job_totals with recalculated corner LF
    supabase_request('PATCH', 'extraction_job_totals',
        {
            'outside_corners_lf': corner_calcs['outside_corners_lf'],
            'inside_corners_lf': corner_calcs['inside_corners_lf'],
            'corner_source': f"ocr_enhanced_{wall_data['source']}"
        },
        filters={'job_id': f'eq.{job_id}'}
    )
    
    return result


def store_linear_elements(job_id, calculations):
    """
    Store linear element calculations in database.
    """
    elements_to_store = []
    
    # Corner elements
    if calculations.get('corners'):
        corners = calculations['corners']
        elements_to_store.append({
            'job_id': job_id,
            'element_type': 'outside_corner',
            'length_lf': corners['outside_corners_lf'],
            'source': 'calculated',
            'notes': f"{corners['outside_corners_count']} corners × {corners['wall_height_used_ft']:.1f}' height"
        })
        elements_to_store.append({
            'job_id': job_id,
            'element_type': 'inside_corner',
            'length_lf': corners['inside_corners_lf'],
            'source': 'calculated',
            'notes': f"{corners['inside_corners_count']} corners × {corners['wall_height_used_ft']:.1f}' height"
        })
    
    # Perimeter elements
    if calculations.get('perimeter'):
        perimeter = calculations['perimeter']
        for elem_type in ['starter_strip', 'water_table', 'band_board', 'frieze_board']:
            key = f'{elem_type}_lf'
            if perimeter.get(key):
                elements_to_store.append({
                    'job_id': job_id,
                    'element_type': elem_type,
                    'length_lf': perimeter[key],
                    'source': 'calculated',
                    'notes': f"From perimeter: {perimeter['building_perimeter_lf']:.1f}'"
                })
    
    # Store each element
    for elem in elements_to_store:
        # Check if exists
        existing = supabase_request('GET', 'extraction_linear_elements', filters={
            'job_id': f'eq.{job_id}',
            'element_type': f"eq.{elem['element_type']}"
        })
        
        if existing:
            supabase_request('PATCH', 'extraction_linear_elements',
                elem,
                filters={
                    'job_id': f'eq.{job_id}',
                    'element_type': f"eq.{elem['element_type']}"
                }
            )
        else:
            supabase_request('POST', 'extraction_linear_elements', elem)


def get_linear_summary(job_id):
    """
    Get summary of all linear elements for a job.
    """
    elements = supabase_request('GET', 'extraction_linear_elements', filters={
        'job_id': f'eq.{job_id}',
        'order': 'element_type'
    })
    
    # Get job totals for trim data
    totals = supabase_request('GET', 'extraction_job_totals', filters={
        'job_id': f'eq.{job_id}'
    })
    
    return {
        'job_id': job_id,
        'linear_elements': elements or [],
        'trim_totals': {
            'window_perimeter_lf': totals[0].get('total_window_perimeter_lf') if totals else None,
            'door_perimeter_lf': totals[0].get('total_door_perimeter_lf') if totals else None,
            'gable_rake_lf': totals[0].get('total_gable_rake_lf') if totals else None
        } if totals else {}
    }


# =============================================================================
# MANUAL OVERRIDE FUNCTIONS
# =============================================================================

def set_corner_counts(job_id, outside_corners, inside_corners, wall_height_ft=None):
    """
    Manually set corner counts and recalculate LF.
    
    Use when floor plan analysis is inaccurate.
    """
    # Get wall height
    if wall_height_ft is None:
        wall_data = get_wall_heights_from_ocr(job_id)
        wall_height_ft = wall_data['total_wall_height_ft']
    
    # Calculate
    corner_calcs = calculate_corner_lf(outside_corners, inside_corners, wall_height_ft)
    
    # Update job totals
    supabase_request('PATCH', 'extraction_job_totals',
        {
            'outside_corners_count': outside_corners,
            'inside_corners_count': inside_corners,
            'outside_corners_lf': corner_calcs['outside_corners_lf'],
            'inside_corners_lf': corner_calcs['inside_corners_lf'],
            'corner_source': 'manual'
        },
        filters={'job_id': f'eq.{job_id}'}
    )
    
    # Store in linear_elements
    store_linear_elements(job_id, {'corners': corner_calcs})
    
    return corner_calcs


def set_wall_height(job_id, first_floor_ft, second_floor_ft=None):
    """
    Manually set wall heights and recalculate all linear elements.
    """
    total_height = first_floor_ft + (second_floor_ft or 0)
    
    # Get current corner counts
    totals = supabase_request('GET', 'extraction_job_totals', filters={
        'job_id': f'eq.{job_id}'
    })
    
    if totals:
        return calculate_corner_lf(
            totals[0].get('outside_corners_count'),
            totals[0].get('inside_corners_count'),
            total_height
        )
    
    return {"error": "No job totals found"}
