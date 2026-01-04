"""
Floor plan analysis service - corner counting and analysis
"""

from database import supabase_request, get_floor_plan_pages
from core import analyze_floor_plan_corners


def analyze_floor_plan_for_job(job_id):
    """
    Analyze all floor plans for a job to count corners.
    
    Corners are important for siding estimation:
    - Outside corners need corner trim
    - Inside corners need J-channel or corner trim
    
    Returns:
        Dict with corner counts and LF calculations
    """
    # Get floor plan pages
    pages = get_floor_plan_pages(job_id)
    
    if not pages:
        return {"error": "No floor plan pages found"}
    
    results = []
    total_outside = 0
    total_inside = 0
    
    for page in pages:
        print(f"Analyzing floor plan page {page.get('page_number')}...", flush=True)
        
        image_url = page.get('image_url')
        result = analyze_floor_plan_corners(image_url)
        result['page_id'] = page['id']
        result['page_number'] = page.get('page_number')
        results.append(result)
        
        if 'total_outside_corners' in result:
            total_outside += result.get('total_outside_corners', 0)
            total_inside += result.get('total_inside_corners', 0)
    
    # Get wall height from job totals for LF calculation
    job_totals = supabase_request('GET', 'extraction_job_totals', filters={
        'job_id': f'eq.{job_id}'
    })
    
    # Calculate average wall height from facade/eave ratio
    wall_height = 9.0  # Default 9' walls
    if job_totals and job_totals[0].get('total_gross_facade_sf') and job_totals[0].get('total_roof_eave_lf'):
        facade = float(job_totals[0]['total_gross_facade_sf'])
        eave = float(job_totals[0]['total_roof_eave_lf'])
        if eave > 0:
            wall_height = facade / eave
    
    # Calculate linear feet
    outside_lf = round(total_outside * wall_height, 2)
    inside_lf = round(total_inside * wall_height, 2)
    
    # Update job totals with corner data
    if job_totals:
        supabase_request('PATCH', 'extraction_job_totals',
            {
                'outside_corners_count': total_outside,
                'inside_corners_count': total_inside,
                'outside_corners_lf': outside_lf,
                'inside_corners_lf': inside_lf,
                'corner_source': 'floor_plan_analysis'
            },
            filters={'job_id': f'eq.{job_id}'}
        )
    
    return {
        "job_id": job_id,
        "floor_plans_analyzed": len(results),
        "total_outside_corners": total_outside,
        "total_inside_corners": total_inside,
        "wall_height_used": round(wall_height, 2),
        "outside_corners_lf": outside_lf,
        "inside_corners_lf": inside_lf,
        "per_page_results": results
    }


def analyze_single_floor_plan(page_id):
    """
    Analyze a single floor plan page for corners.
    
    Args:
        page_id: UUID of the floor plan page
    
    Returns:
        Dict with corner analysis results
    """
    from database import get_page
    
    page = get_page(page_id)
    if not page:
        return {"error": "Page not found"}
    
    if page.get('page_type') != 'floor_plan':
        return {"error": f"Page is type '{page.get('page_type')}', not floor_plan"}
    
    image_url = page.get('image_url')
    result = analyze_floor_plan_corners(image_url)
    result['page_id'] = page_id
    result['page_number'] = page.get('page_number')
    
    return result
