"""
Page classification service
"""

import time
from collections import Counter

from config import config
from database import update_job, update_page, get_pending_pages
from core import classify_page_with_claude
from utils import normalize_page_type


def classify_job_background(job_id):
    """
    Background task to classify all pages in a job.
    
    Steps:
    1. Get all pending pages
    2. Classify each page with Claude Vision
    3. Update page records
    4. Update job summary
    """
    try:
        update_job(job_id, {'status': 'classifying'})
        
        pages = get_pending_pages(job_id)
        if not pages:
            print(f"[{job_id}] No pending pages to classify", flush=True)
            return
        
        print(f"[{job_id}] Classifying {len(pages)} pages...", flush=True)
        
        elevation_count = 0
        schedule_count = 0
        floor_plan_count = 0
        other_count = 0
        scales_found = []
        
        for i, page in enumerate(pages):
            page_id = page.get('id')
            image_url = page.get('image_url')
            
            # Classify with Claude
            classification = classify_page_with_claude(image_url)
            page_type = normalize_page_type(classification.get('page_type'))
            
            # Count by type
            if page_type == 'elevation':
                elevation_count += 1
            elif page_type == 'schedule':
                schedule_count += 1
            elif page_type == 'floor_plan':
                floor_plan_count += 1
            else:
                other_count += 1
            
            # Track scales
            scale_ratio = classification.get('scale_ratio')
            if scale_ratio:
                scales_found.append(scale_ratio)
            
            # Update page record
            update_page(page_id, {
                'page_type': page_type,
                'page_type_confidence': classification.get('confidence', 0),
                'status': 'classified',
                'scale_notation': classification.get('scale_notation'),
                'scale_ratio': scale_ratio,
                'elevation_name': classification.get('elevation_name')
            })
            
            # Update job progress
            update_job(job_id, {'pages_classified': i + 1})
            
            # Rate limiting
            if (i + 1) % config.MAX_CONCURRENT_CLAUDE == 0:
                time.sleep(config.BATCH_DELAY_SECONDS)
            
            print(f"[{job_id}] Classified page {i+1}/{len(pages)}: {page_type}", flush=True)
        
        # Find most common scale
        default_scale = Counter(scales_found).most_common(1)[0][0] if scales_found else None
        
        # Update job summary
        update_job(job_id, {
            'status': 'classified',
            'elevation_count': elevation_count,
            'schedule_count': schedule_count,
            'floor_plan_count': floor_plan_count,
            'other_count': other_count,
            'default_scale_ratio': default_scale
        })
        
        print(f"[{job_id}] Classification complete: {elevation_count} elevations, {schedule_count} schedules", flush=True)
    
    except Exception as e:
        print(f"[{job_id}] Classification failed: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})
