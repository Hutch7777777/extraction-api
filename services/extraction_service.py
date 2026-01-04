"""
Extraction service - main processing orchestration
"""

from database import (
    get_job, update_job, update_page,
    get_classified_pages, get_elevation_pages, get_schedule_pages
)
from core import detect_with_roboflow, ocr_schedule_with_claude
from geometry import calculate_real_measurements
from config import config


def process_job_background(job_id, scale_override=None, generate_markups=True):
    """
    Background task to process all pages in a job.
    
    Steps:
    1. Run Roboflow detection on elevation pages
    2. Calculate real-world measurements
    3. Run OCR on schedule pages
    4. Update page records
    5. Optionally generate markups
    6. Build cross-references
    """
    try:
        update_job(job_id, {'status': 'processing'})
        
        # Get job info
        job = get_job(job_id)
        if not job:
            print(f"[{job_id}] Job not found", flush=True)
            return
        
        default_scale = job.get('default_scale_ratio')
        job_dpi = job.get('plan_dpi', config.DEFAULT_DPI)
        
        # Get classified pages
        pages = get_classified_pages(job_id)
        if not pages:
            print(f"[{job_id}] No classified pages to process", flush=True)
            return
        
        print(f"[{job_id}] Processing {len(pages)} pages...", flush=True)
        
        elevation_pages = [p for p in pages if p.get('page_type') == 'elevation']
        schedule_pages = [p for p in pages if p.get('page_type') == 'schedule']
        
        totals = {
            'total_net_siding_sqft': 0,
            'total_gross_wall_sqft': 0,
            'total_windows': 0,
            'total_doors': 0
        }
        processed = 0
        
        # Process elevation pages
        for page in elevation_pages:
            page_id = page.get('id')
            image_url = page.get('image_url')
            
            # Use scale priority: override > page > job default > fallback
            scale_ratio = scale_override or page.get('scale_ratio') or default_scale or 48
            dpi = page.get('dpi') or job_dpi
            
            # Run Roboflow detection
            detection = detect_with_roboflow(image_url)
            
            if 'error' not in detection:
                predictions = detection.get('predictions', [])
                
                # Calculate real measurements
                measurements = calculate_real_measurements(predictions, scale_ratio, dpi)
                
                # Accumulate totals
                totals['total_net_siding_sqft'] += measurements['areas'].get('net_siding_sqft', 0)
                totals['total_gross_wall_sqft'] += measurements['areas'].get('gross_wall_sqft', 0)
                totals['total_windows'] += measurements['counts'].get('window', 0)
                totals['total_doors'] += measurements['counts'].get('door', 0)
                
                # Update page
                update_page(page_id, {
                    'status': 'complete',
                    'extraction_data': {
                        'measurements': measurements,
                        'raw_predictions': predictions
                    }
                })
                
                print(f"[{job_id}] Processed elevation: {measurements['counts'].get('window', 0)} windows", flush=True)
            else:
                print(f"[{job_id}] Detection failed for page {page_id}: {detection.get('error')}", flush=True)
                update_page(page_id, {'status': 'failed'})
            
            processed += 1
            update_job(job_id, {'pages_processed': processed})
        
        # Process schedule pages
        for page in schedule_pages:
            page_id = page.get('id')
            image_url = page.get('image_url')
            
            # Extract schedule data with Claude
            schedule_data = ocr_schedule_with_claude(image_url)
            
            if 'error' not in schedule_data:
                update_page(page_id, {
                    'status': 'complete',
                    'extraction_data': schedule_data
                })
                print(f"[{job_id}] Extracted schedule: {len(schedule_data.get('windows', []))} windows, {len(schedule_data.get('doors', []))} doors", flush=True)
            else:
                print(f"[{job_id}] Schedule extraction failed: {schedule_data.get('error')}", flush=True)
                update_page(page_id, {'status': 'failed'})
            
            processed += 1
            update_job(job_id, {'pages_processed': processed})
        
        # Skip other page types
        for page in pages:
            if page.get('page_type') not in ['elevation', 'schedule']:
                update_page(page.get('id'), {'status': 'skipped'})
        
        # Update job with totals
        update_job(job_id, {'status': 'complete', 'results_summary': totals})
        
        # Auto-generate markups
        if generate_markups:
            from services.markup_service import generate_markups_for_job
            print(f"[{job_id}] Generating markups...", flush=True)
            generate_markups_for_job(job_id, trades=['all', 'siding', 'roofing'])
        
        # Auto-run cross-reference
        from services.cross_ref_service import build_cross_references
        print(f"[{job_id}] Building cross-references...", flush=True)
        build_cross_references(job_id)
        
        print(f"[{job_id}] Processing complete!", flush=True)
    
    except Exception as e:
        print(f"[{job_id}] Processing failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})
