"""
Detection repository - CRUD operations for extraction_detection_details
"""

from database.client import supabase_request


def get_detection(detection_id):
    """Get detection by ID"""
    result = supabase_request('GET', 'extraction_detection_details', filters={'id': f'eq.{detection_id}'})
    return result[0] if result else None


def create_detection(data):
    """Create new detection"""
    return supabase_request('POST', 'extraction_detection_details', data)


def update_detection(detection_id, updates):
    """Update detection by ID"""
    return supabase_request('PATCH', 'extraction_detection_details', updates, {'id': f'eq.{detection_id}'})


def delete_detection(detection_id):
    """Delete detection by ID"""
    return supabase_request('DELETE', 'extraction_detection_details', filters={'id': f'eq.{detection_id}'})


def get_detections_by_page(page_id, status=None):
    """Get all detections for a page"""
    filters = {'page_id': f'eq.{page_id}', 'order': 'class,detection_index'}
    
    if status:
        filters['status'] = f'eq.{status}'
    
    return supabase_request('GET', 'extraction_detection_details', filters=filters) or []


def get_detections_by_job(job_id):
    """Get all detections for a job"""
    return supabase_request('GET', 'extraction_detection_details', filters={
        'job_id': f'eq.{job_id}',
        'order': 'page_id,class,detection_index'
    }) or []


def delete_detections_by_page(page_id):
    """Delete all detections for a page"""
    return supabase_request('DELETE', 'extraction_detection_details', filters={'page_id': f'eq.{page_id}'})


def get_active_detections_by_page(page_id):
    """Get non-deleted detections for a page"""
    return supabase_request('GET', 'extraction_detection_details', filters={
        'page_id': f'eq.{page_id}',
        'status': 'neq.deleted',
        'order': 'class,detection_index'
    }) or []


def batch_create_detections(detections_list):
    """
    Create multiple detections in a single request.

    Args:
        detections_list: List of detection dicts, each with:
            - job_id, page_id, class, detection_index
            - x, y, width, height (pixel coordinates)
            - confidence, source, status

    Returns:
        List of created detection records
    """
    if not detections_list:
        return []

    return supabase_request('POST', 'extraction_detection_details', detections_list) or []
