"""
Database module exports
"""

from database.client import supabase_request
from database.storage import upload_to_storage
from database.repositories.job_repository import (
    get_job, create_job, update_job, list_jobs, delete_job
)
from database.repositories.page_repository import (
    get_page, create_page, update_page, delete_page,
    get_pages_by_job, get_elevation_pages, get_schedule_pages,
    get_floor_plan_pages, get_pending_pages, get_classified_pages
)
from database.repositories.detection_repository import (
    get_detection, create_detection, update_detection, delete_detection,
    get_detections_by_page, get_detections_by_job, delete_detections_by_page,
    get_active_detections_by_page
)

__all__ = [
    # Client
    'supabase_request',
    
    # Storage
    'upload_to_storage',
    
    # Jobs
    'get_job', 'create_job', 'update_job', 'list_jobs', 'delete_job',
    
    # Pages
    'get_page', 'create_page', 'update_page', 'delete_page',
    'get_pages_by_job', 'get_elevation_pages', 'get_schedule_pages',
    'get_floor_plan_pages', 'get_pending_pages', 'get_classified_pages',
    
    # Detections
    'get_detection', 'create_detection', 'update_detection', 'delete_detection',
    'get_detections_by_page', 'get_detections_by_job', 'delete_detections_by_page',
    'get_active_detections_by_page'
]
