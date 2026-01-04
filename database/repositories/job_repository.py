"""
Job repository - CRUD operations for extraction_jobs
"""

from database.client import supabase_request


def get_job(job_id):
    """Get job by ID"""
    result = supabase_request('GET', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
    return result[0] if result else None


def create_job(data):
    """Create new job"""
    result = supabase_request('POST', 'extraction_jobs', data)
    return result[0] if result else None


def update_job(job_id, updates):
    """Update job by ID"""
    return supabase_request('PATCH', 'extraction_jobs', updates, {'id': f'eq.{job_id}'})


def list_jobs(limit=50):
    """List recent jobs"""
    return supabase_request('GET', 'extraction_jobs', filters={
        'order': 'created_at.desc',
        'limit': str(limit)
    }) or []


def delete_job(job_id):
    """Delete job by ID"""
    return supabase_request('DELETE', 'extraction_jobs', filters={'id': f'eq.{job_id}'})
