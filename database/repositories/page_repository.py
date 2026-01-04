"""
Page repository - CRUD operations for extraction_pages
"""

from database.client import supabase_request


def get_page(page_id):
    """Get page by ID"""
    result = supabase_request('GET', 'extraction_pages', filters={'id': f'eq.{page_id}'})
    return result[0] if result else None


def create_page(data):
    """Create new page"""
    return supabase_request('POST', 'extraction_pages', data)


def update_page(page_id, updates):
    """Update page by ID"""
    result = supabase_request('PATCH', 'extraction_pages', updates, {'id': f'eq.{page_id}'})
    if not result:
        print(f"FAILED update page {page_id}", flush=True)
    return result


def delete_page(page_id):
    """Delete page by ID"""
    return supabase_request('DELETE', 'extraction_pages', filters={'id': f'eq.{page_id}'})


def get_pages_by_job(job_id, page_type=None, status=None):
    """Get all pages for a job with optional filters"""
    filters = {'job_id': f'eq.{job_id}', 'order': 'page_number'}
    
    if page_type:
        filters['page_type'] = f'eq.{page_type}'
    if status:
        filters['status'] = f'eq.{status}'
    
    return supabase_request('GET', 'extraction_pages', filters=filters) or []


def get_elevation_pages(job_id, status='complete'):
    """Get elevation pages for a job"""
    return get_pages_by_job(job_id, page_type='elevation', status=status)


def get_schedule_pages(job_id, status='complete'):
    """Get schedule pages for a job"""
    return get_pages_by_job(job_id, page_type='schedule', status=status)


def get_floor_plan_pages(job_id):
    """Get floor plan pages for a job"""
    return get_pages_by_job(job_id, page_type='floor_plan')


def get_pending_pages(job_id):
    """Get pages pending classification"""
    return get_pages_by_job(job_id, status='pending')


def get_classified_pages(job_id):
    """Get classified pages ready for processing"""
    return get_pages_by_job(job_id, status='classified')
