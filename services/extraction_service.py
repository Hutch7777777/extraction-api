"""
Extraction service - main processing orchestration
"""

from database import (
    get_job, update_job, update_page,
    get_classified_pages, get_elevation_pages, get_schedule_pages,
    supabase_request
)
from core import detect_with_roboflow, ocr_schedule_with_claude, extract_elevation_dimensions
from geometry import calculate_real_measurements
from config import config
import datetime


def _insert_detections(job_id, page_id, predictions, scale_ratio, dpi):
    """
    Insert Roboflow predictions into extraction_detection_details table.
    
    Args:
        job_id: Job UUID
        page_id: Page UUID
        predictions: List of Roboflow predictions
        scale_ratio: Scale ratio (e.g., 48 for 1/4" = 1')
        dpi: DPI of the image
    """
    if not predictions:
        return 0
    
    # Calculate pixels to real units conversion
    # At 200 DPI and 1/4"=1' (scale_ratio=48): 1 pixel = 0.24 inches real
    inches_per_pixel = scale_ratio / dpi
    
    inserted = 0
    for idx, pred in enumerate(predictions):
        detection_class = (pred.get('class') or '').lower().strip() or 'unknown'
        
        # Roboflow returns center x,y - convert to top-left corner
        pixel_width = pred.get('width', 0)
        pixel_height = pred.get('height', 0)
        pixel_x = pred.get('x', 0) - (pixel_width / 2)
        pixel_y = pred.get('y', 0) - (pixel_height / 2)
        confidence = pred.get('confidence', 1.0)
        
        # Calculate real dimensions
        real_width_in = pixel_width * inches_per_pixel
        real_height_in = pixel_height * inches_per_pixel
        real_width_ft = real_width_in / 12
        real_height_ft = real_height_in / 12
        
        # Calculate area - triangles for gables
        is_triangle = detection_class == 'gable'
        if is_triangle:
            area_sf = (real_width_ft * real_height_ft) / 2
        else:
            area_sf = real_width_ft * real_height_ft
        
        perimeter_lf = 2 * (real_width_ft + real_height_ft)
        
        detection_record = {
            'job_id': job_id,
            'page_id': page_id,
            'class': detection_class,
            'detection_index': idx + 1,
            'confidence': round(confidence, 4),
            'pixel_x': round(pixel_x, 2),
            'pixel_y': round(pixel_y, 2),
            'pixel_width': round(pixel_width, 2),
            'pixel_height': round(pixel_height, 2),
            'real_width_in': round(real_width_in, 2),
            'real_height_in': round(real_height_in, 2),
            'real_width_ft': round(real_width_ft, 2),
            'real_height_ft': round(real_height_ft, 2),
            'area_sf': round(area_sf, 2),
            'perimeter_lf': round(perimeter_lf, 2),
            'is_triangle': is_triangle,
            'status': 'auto',
            'original_bbox': {
                'x': pixel_x,
                'y': pixel_y,
                'width': pixel_width,
                'height': pixel_height
            }
        }
        
        print(f"[DEBUG] Inserting detection {idx+1}: {detection_class}", flush=True)
        result = supabase_request('POST', 'extraction_detection_details', detection_record)
        if result:
            inserted += 1
    
    return inserted


def process_job_background(job_id, scale_override=None, generate_markups=True):
    """
    Background task to process all pages in a job.
    
    Steps:
    1. Run Roboflow detection on elevation pages
    2. Insert detections into database
    3. Calculate real-world measurements
    4. Run OCR on schedule pages
    5. Update page records
    6. Optionally generate markups
    7. Build cross-references
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
            'total_doors': 0,
            'total_detections': 0
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
                
                # INSERT DETECTIONS INTO DATABASE
                inserted_count = _insert_detections(job_id, page_id, predictions, scale_ratio, dpi)
                totals['total_detections'] += inserted_count
                print(f"[{job_id}] Inserted {inserted_count} detections for page {page_id}", flush=True)
                
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
                
                # Run OCR on elevation to extract wall heights, callouts, dimensions
                try:
                    print(f"[{job_id}] Running OCR on elevation page {page_id}...", flush=True)
                    ocr_result = extract_elevation_dimensions(image_url)
                    
                    if ocr_result and 'error' not in ocr_result:
                        # Store OCR results
                        _store_ocr_results(job_id, page_id, ocr_result)
                        print(f"[{job_id}] OCR complete: {len(ocr_result.get('wall_heights', []))} wall heights, {len(ocr_result.get('element_callouts', []))} callouts", flush=True)
                    else:
                        print(f"[{job_id}] OCR returned no data for page {page_id}", flush=True)
                except Exception as ocr_err:
                    print(f"[{job_id}] OCR failed for page {page_id}: {ocr_err}", flush=True)
                    # Don't fail the whole extraction if OCR fails
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
        update_job(job_id, {
            'status': 'complete',
            'results_summary': totals,
            'total_detections': totals['total_detections']
        })
        
        # Auto-generate markups
        if generate_markups:
            from services.markup_service import generate_markups_for_job
            print(f"[{job_id}] Generating markups...", flush=True)
            generate_markups_for_job(job_id, trades=['all', 'siding', 'roofing'])
        
        # Auto-run cross-reference
        from services.cross_ref_service import build_cross_references
        print(f"[{job_id}] Building cross-references...", flush=True)
        build_cross_references(job_id)
        
        # Auto-run data fusion to combine OCR + detections + schedules
        try:
            from services.fusion_service import fuse_job_data
            print(f"[{job_id}] Running data fusion...", flush=True)
            fusion_results = fuse_job_data(job_id)
            if fusion_results and 'error' not in fusion_results:
                print(f"[{job_id}] Fusion complete: {fusion_results.get('total_callouts_matched', 0)} callouts matched, {fusion_results.get('total_schedule_matches', 0)} schedule matches", flush=True)
        except Exception as fusion_err:
            print(f"[{job_id}] Fusion failed: {fusion_err}", flush=True)
            # Don't fail the job if fusion fails
        
        print(f"[{job_id}] Processing complete!", flush=True)
    
    except Exception as e:
        print(f"[{job_id}] Processing failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})


def _store_ocr_results(job_id, page_id, ocr_result):
    """
    Store OCR extraction results in the database.
    
    Args:
        job_id: Job UUID
        page_id: Page UUID
        ocr_result: Dict from extract_elevation_dimensions()
    """
    ocr_record = {
        'job_id': job_id,
        'page_id': page_id,
        'wall_heights': ocr_result.get('wall_heights', []),
        'dimension_text': ocr_result.get('dimension_text', []),
        'element_callouts': ocr_result.get('element_callouts', []),
        'level_markers': ocr_result.get('level_markers', []),
        'eave_height_ft': ocr_result.get('eave_height_ft'),
        'ridge_height_ft': ocr_result.get('ridge_height_ft'),
        'average_wall_height_ft': ocr_result.get('average_wall_height_ft'),
        'total_building_height_ft': ocr_result.get('total_building_height_ft'),
        'extraction_confidence': ocr_result.get('extraction_confidence'),
        'processing_time_ms': ocr_result.get('processing_time_ms'),
        'claude_model': ocr_result.get('raw_response', {}).get('model', 'unknown')
    }
    
    # Check if record exists
    existing = supabase_request('GET', 'extraction_ocr_data', filters={
        'page_id': f'eq.{page_id}'
    })
    
    if existing:
        # Update existing
        supabase_request('PATCH', 'extraction_ocr_data',
                        data=ocr_record,
                        filters={'page_id': f'eq.{page_id}'})
    else:
        # Insert new
        supabase_request('POST', 'extraction_ocr_data', ocr_record)
    
    # Update page OCR status
    update_page(page_id, {
        'ocr_status': 'complete',
        'ocr_processed_at': datetime.datetime.utcnow().isoformat()
    })
    
    # Update elevation calcs with wall height if found
    if ocr_result.get('average_wall_height_ft'):
        supabase_request('PATCH', 'extraction_elevation_calcs',
                        data={
                            'wall_height_ft': ocr_result['average_wall_height_ft'],
                            'wall_height_source': 'ocr',
                            'ocr_eave_height_ft': ocr_result.get('eave_height_ft'),
                            'ocr_ridge_height_ft': ocr_result.get('ridge_height_ft')
                        },
                        filters={'page_id': f'eq.{page_id}'})
