"""
Roof Service - Roof intelligence and calculations

Handles:
- Roof plan OCR processing
- Pitch factor calculations
- Soffit/fascia calculations
- True area calculations from projected area
- Linear element aggregation
"""

import math
from database import supabase_request, get_page, update_page


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard pitch factors (pre-calculated for performance)
PITCH_FACTORS = {
    '1:12': 1.003, '2:12': 1.014, '3:12': 1.031, '4:12': 1.054,
    '5:12': 1.083, '6:12': 1.118, '7:12': 1.158, '8:12': 1.202,
    '9:12': 1.250, '10:12': 1.302, '11:12': 1.357, '12:12': 1.414,
    '14:12': 1.537, '16:12': 1.667
}

# Waste factors by pitch (steeper = more waste)
PITCH_WASTE_FACTORS = {
    '1:12': 0.08, '2:12': 0.08, '3:12': 0.08, '4:12': 0.10,
    '5:12': 0.10, '6:12': 0.10, '7:12': 0.12, '8:12': 0.12,
    '9:12': 0.15, '10:12': 0.15, '11:12': 0.18, '12:12': 0.18,
    '14:12': 0.20, '16:12': 0.22
}

# Default overhang depth for soffit calculation (inches)
DEFAULT_OVERHANG_IN = 12


# =============================================================================
# PITCH CALCULATIONS
# =============================================================================

def calculate_pitch_factor(pitch_notation):
    """
    Calculate pitch factor from notation.
    
    pitch_factor = sqrt(1 + (rise/run)^2)
    
    Args:
        pitch_notation: String like '6:12', '6/12'
        
    Returns:
        float pitch factor
    """
    if not pitch_notation:
        return 1.0
    
    # Check lookup table first
    normalized = normalize_pitch_notation(pitch_notation)
    if normalized in PITCH_FACTORS:
        return PITCH_FACTORS[normalized]
    
    # Parse and calculate
    import re
    match = re.match(r'(\d+)\s*[:/\-]\s*(\d+)', str(pitch_notation))
    if match:
        rise = int(match.group(1))
        run = int(match.group(2))
        if run > 0:
            return round(math.sqrt(1 + (rise / run) ** 2), 3)
    
    return 1.0


def normalize_pitch_notation(pitch):
    """
    Normalize pitch notation to standard format.
    
    '6/12' -> '6:12'
    '6-12' -> '6:12'
    """
    if not pitch:
        return None
    
    import re
    match = re.match(r'(\d+)\s*[:/\-]\s*(\d+)', str(pitch))
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return pitch


def get_pitch_waste_factor(pitch_notation):
    """
    Get additional waste factor for pitch.
    
    Steeper roofs require more waste allowance.
    """
    normalized = normalize_pitch_notation(pitch_notation)
    return PITCH_WASTE_FACTORS.get(normalized, 0.10)


def pitch_to_degrees(pitch_notation):
    """
    Convert pitch notation to degrees.
    
    Args:
        pitch_notation: '6:12' etc.
        
    Returns:
        Angle in degrees
    """
    import re
    match = re.match(r'(\d+)\s*[:/\-]\s*(\d+)', str(pitch_notation))
    if match:
        rise = int(match.group(1))
        run = int(match.group(2))
        if run > 0:
            return round(math.degrees(math.atan(rise / run)), 2)
    return 0


# =============================================================================
# ROOF AREA CALCULATIONS
# =============================================================================

def calculate_true_area(projected_area_sf, pitch_notation):
    """
    Convert projected (plan view) area to true surface area.
    
    True Area = Projected Area × Pitch Factor
    
    Args:
        projected_area_sf: Flat area from roof plan
        pitch_notation: '6:12' etc.
        
    Returns:
        True surface area in SF
    """
    factor = calculate_pitch_factor(pitch_notation)
    return round(projected_area_sf * factor, 1)


def calculate_roofing_squares(true_area_sf, waste_factor=None, pitch=None):
    """
    Calculate roofing squares needed.
    
    1 square = 100 SF
    Adds waste factor for cuts and pitch.
    
    Args:
        true_area_sf: True surface area
        waste_factor: Override waste factor (0.10 = 10%)
        pitch: Pitch notation to auto-calculate waste
        
    Returns:
        Number of squares needed
    """
    if waste_factor is None:
        waste_factor = get_pitch_waste_factor(pitch) if pitch else 0.10
    
    total_sf = true_area_sf * (1 + waste_factor)
    return round(total_sf / 100, 1)


# =============================================================================
# LINEAR ELEMENT CALCULATIONS
# =============================================================================

def calculate_soffit_area(eave_lf, overhang_in=DEFAULT_OVERHANG_IN):
    """
    Calculate soffit area from eave length.
    
    Soffit SF = Eave LF × Overhang Depth
    
    Args:
        eave_lf: Total eave length in feet
        overhang_in: Overhang depth in inches
        
    Returns:
        Soffit area in SF
    """
    if not eave_lf:
        return 0
    
    overhang_ft = overhang_in / 12
    return round(eave_lf * overhang_ft, 1)


def calculate_fascia_lf(eave_lf, rake_lf=None):
    """
    Calculate total fascia length.
    
    Fascia runs along eaves and rakes.
    
    Args:
        eave_lf: Eave length
        rake_lf: Rake length (optional)
        
    Returns:
        Total fascia LF
    """
    total = eave_lf or 0
    if rake_lf:
        total += rake_lf
    return round(total, 1)


def calculate_drip_edge_lf(eave_lf, rake_lf=None):
    """
    Calculate drip edge needed.
    
    Drip edge goes on eaves and rakes.
    """
    return calculate_fascia_lf(eave_lf, rake_lf)


def calculate_starter_lf(eave_lf):
    """
    Calculate starter strip length.
    
    Starter goes along eaves only.
    """
    return eave_lf or 0


def calculate_ridge_cap_lf(ridge_lf, hip_lf=None):
    """
    Calculate ridge cap length needed.
    
    Ridge cap covers ridges and hips.
    
    Args:
        ridge_lf: Ridge length
        hip_lf: Hip length (optional)
        
    Returns:
        Total ridge cap LF
    """
    total = ridge_lf or 0
    if hip_lf:
        total += hip_lf
    return round(total, 1)


# =============================================================================
# ROOF PLAN PROCESSING
# =============================================================================

def process_roof_plan(page_id):
    """
    Process a roof plan page with OCR and store results.
    
    Args:
        page_id: UUID of roof plan page
        
    Returns:
        Dict with processing results
    """
    from core import extract_roof_plan_data
    
    # Get page
    page = get_page(page_id)
    if not page:
        return {"error": "Page not found"}
    
    job_id = page.get('job_id')
    image_url = page.get('image_url')
    
    if not image_url:
        return {"error": "No image URL"}
    
    # Run OCR
    print(f"[{job_id}] Running roof plan OCR on page {page_id}...", flush=True)
    ocr_result = extract_roof_plan_data(image_url)
    
    if ocr_result.get('error'):
        update_page(page_id, {'roof_ocr_status': 'failed'})
        return {"error": ocr_result['error']}
    
    # Store roof sections
    sections = ocr_result.get('roof_sections', [])
    for i, section in enumerate(sections):
        store_roof_section(job_id, page_id, section, i + 1)
    
    # Store linear elements if present
    linear = ocr_result.get('linear_elements', {})
    if any(linear.values()):
        store_linear_elements(job_id, page_id, linear, ocr_result.get('dominant_pitch'))
    
    # Update page status
    update_page(page_id, {
        'roof_ocr_status': 'complete',
        'roof_ocr_processed_at': 'now()'
    })
    
    # Recalculate job summary
    recalculate_roof_summary(job_id)
    
    return {
        "success": True,
        "page_id": page_id,
        "sections_found": len(sections),
        "total_area_sf": ocr_result.get('total_roof_area_sf'),
        "dominant_pitch": ocr_result.get('dominant_pitch'),
        "linear_elements": linear,
        "confidence": ocr_result.get('extraction_confidence')
    }


def store_roof_section(job_id, page_id, section_data, index):
    """
    Store a roof section in the database.
    """
    pitch = section_data.get('pitch')
    pitch_factor = calculate_pitch_factor(pitch)
    projected_sf = section_data.get('area_sf')
    true_sf = calculate_true_area(projected_sf, pitch) if projected_sf else None
    
    record = {
        'job_id': job_id,
        'page_id': page_id,
        'section_name': section_data.get('section_name', f'Section {index}'),
        'section_index': index,
        'projected_area_sf': projected_sf,
        'true_area_sf': true_sf,
        'pitch_notation': normalize_pitch_notation(pitch),
        'pitch_factor': pitch_factor,
        'pitch_degrees': pitch_to_degrees(pitch) if pitch else None,
        'source': 'ocr',
        'confidence': section_data.get('confidence', 0.8)
    }
    
    supabase_request('POST', 'extraction_roof_sections', record)


def store_linear_elements(job_id, page_id, linear_data, dominant_pitch=None):
    """
    Store linear elements as a roof section (for simplicity).
    
    In a more complex system, these might go in a separate table.
    """
    record = {
        'job_id': job_id,
        'page_id': page_id,
        'section_name': 'Linear Elements',
        'section_index': 0,
        'eave_lf': linear_data.get('eave_lf'),
        'ridge_lf': linear_data.get('ridge_lf'),
        'valley_lf': linear_data.get('valley_lf'),
        'hip_lf': linear_data.get('hip_lf'),
        'rake_lf': linear_data.get('rake_lf'),
        'pitch_notation': normalize_pitch_notation(dominant_pitch),
        'source': 'ocr'
    }
    
    supabase_request('POST', 'extraction_roof_sections', record)


def recalculate_roof_summary(job_id):
    """
    Recalculate roof summary for a job from all sections.
    """
    # Get all sections
    sections = supabase_request('GET', 'extraction_roof_sections', filters={
        'job_id': f'eq.{job_id}'
    })
    
    if not sections:
        return
    
    # Aggregate
    total_projected = sum(s.get('projected_area_sf') or 0 for s in sections)
    total_true = sum(s.get('true_area_sf') or 0 for s in sections)
    total_eave = sum(s.get('eave_lf') or 0 for s in sections)
    total_ridge = sum(s.get('ridge_lf') or 0 for s in sections)
    total_valley = sum(s.get('valley_lf') or 0 for s in sections)
    total_hip = sum(s.get('hip_lf') or 0 for s in sections)
    total_rake = sum(s.get('rake_lf') or 0 for s in sections)
    
    # Calculate derived values
    soffit_lf = total_eave
    soffit_sf = calculate_soffit_area(total_eave)
    fascia_lf = calculate_fascia_lf(total_eave, total_rake)
    drip_edge_lf = calculate_drip_edge_lf(total_eave, total_rake)
    starter_lf = calculate_starter_lf(total_eave)
    ridge_cap_lf = calculate_ridge_cap_lf(total_ridge, total_hip)
    
    # Average pitch factor
    pitch_factors = [s.get('pitch_factor') for s in sections if s.get('pitch_factor')]
    avg_pitch_factor = sum(pitch_factors) / len(pitch_factors) if pitch_factors else 1.0
    
    summary = {
        'job_id': job_id,
        'section_count': len([s for s in sections if s.get('projected_area_sf')]),
        'total_projected_sf': total_projected,
        'total_true_sf': total_true,
        'average_pitch_factor': round(avg_pitch_factor, 3),
        'total_eave_lf': total_eave,
        'total_ridge_lf': total_ridge,
        'total_valley_lf': total_valley,
        'total_hip_lf': total_hip,
        'total_rake_lf': total_rake,
        'soffit_lf': soffit_lf,
        'soffit_sf': soffit_sf,
        'fascia_lf': fascia_lf,
        'drip_edge_lf': drip_edge_lf,
        'starter_lf': starter_lf,
        'ridge_cap_lf': ridge_cap_lf
    }
    
    # Upsert
    existing = supabase_request('GET', 'extraction_roof_summary', filters={
        'job_id': f'eq.{job_id}'
    })
    
    if existing:
        supabase_request('PATCH', 'extraction_roof_summary',
                        data=summary,
                        filters={'job_id': f'eq.{job_id}'})
    else:
        supabase_request('POST', 'extraction_roof_summary', summary)


# =============================================================================
# JOB-LEVEL PROCESSING
# =============================================================================

def process_roof_plans_for_job(job_id):
    """
    Process all roof plan pages in a job.
    
    Args:
        job_id: UUID of job
        
    Returns:
        Dict with processing results
    """
    # Get roof plan pages
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'page_type': 'eq.roof_plan'
    })
    
    if not pages:
        return {"message": "No roof plan pages found", "job_id": job_id}
    
    results = []
    for page in pages:
        result = process_roof_plan(page['id'])
        results.append(result)
    
    # Get summary
    summary = supabase_request('GET', 'extraction_roof_summary', filters={
        'job_id': f'eq.{job_id}'
    })
    
    return {
        "job_id": job_id,
        "pages_processed": len(results),
        "results": results,
        "summary": summary[0] if summary else None
    }


def get_roof_summary(job_id):
    """
    Get roof summary for a job.
    """
    summary = supabase_request('GET', 'extraction_roof_summary', filters={
        'job_id': f'eq.{job_id}'
    })
    
    sections = supabase_request('GET', 'extraction_roof_sections', filters={
        'job_id': f'eq.{job_id}',
        'order': 'section_index'
    })
    
    return {
        "job_id": job_id,
        "summary": summary[0] if summary else None,
        "sections": sections or []
    }
