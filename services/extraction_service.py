"""
Extraction service - main processing orchestration
"""

import logging
import time
import uuid

from database import (
    get_job, update_job, update_page,
    get_classified_pages, get_elevation_pages, get_schedule_pages,
    supabase_request
)
from core import detect_with_roboflow, ocr_schedule_with_claude, extract_elevation_dimensions
from geometry import calculate_real_measurements
from geometry.area import compute_detection_area_sf, compute_detection_perimeter_lf
from services.detection_postprocess import postprocess_detections
from services.detection_normalization import normalize_detection_class
from config import config
from utils.scale import get_safe_scale_ratio, get_safe_dpi
import datetime

logger = logging.getLogger(__name__)


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
        return 0, []
    
    # Calculate pixels to real units conversion
    # At 200 DPI and 1/4"=1' (scale_ratio=48): 1 pixel = 0.24 inches real
    safe_dpi = get_safe_dpi(dpi, context="insert_detections")
    inches_per_pixel = scale_ratio / safe_dpi
    
    inserted = 0
    inserted_records = []
    for idx, pred in enumerate(predictions):
        detection_class = normalize_detection_class(pred.get('class')) or 'unknown'
        
        # Roboflow returns center x,y - store as center (frontend converts)
        pixel_x = pred.get('x', 0)
        pixel_y = pred.get('y', 0)
        pixel_width = pred.get('width', 0)
        pixel_height = pred.get('height', 0)
        confidence = pred.get('confidence', 1.0)

        if not pixel_width or not pixel_height:
            logger.warning(
                f"Detection missing dimensions: class={detection_class}, "
                f"width={pixel_width}, height={pixel_height}, "
                f"confidence={confidence}"
            )

        # Calculate real dimensions
        real_width_in = pixel_width * inches_per_pixel
        real_height_in = pixel_height * inches_per_pixel
        real_width_ft = real_width_in / 12
        real_height_ft = real_height_in / 12

        # Build detection dict for shared area function
        detection_dict = {
            'pixel_width': pixel_width,
            'pixel_height': pixel_height,
            'polygon_points': pred.get('polygon_points'),  # Usually None for raw predictions
            'class': detection_class
        }

        # Calculate area using shared function (prefers Shoelace on polygon_points, falls back to bbox)
        is_triangle = detection_class == 'gable'
        area_sf = compute_detection_area_sf(detection_dict, scale_ratio)
        if is_triangle and not pred.get('polygon_points'):
            # Apply triangle factor for bbox-based gable calculation
            area_sf = area_sf / 2

        # Calculate perimeter using shared function
        perimeter_lf = compute_detection_perimeter_lf(detection_dict, scale_ratio)
        
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
        
        # # print(f"[DEBUG] Inserting detection {idx+1}: {detection_class}", flush=True)
        result = supabase_request('POST', 'extraction_detection_details', detection_record)
        if result:
            inserted += 1
            if isinstance(result, list):
                inserted_records.extend(result)
            else:
                inserted_records.append(result)
    
    return inserted, inserted_records


def _seed_draft_detections_for_page(job_id, page_id, detail_rows=None):
    """
    Copy AI detections into the editor draft table when a page has no drafts yet.

    The Detection Editor saves all user changes to extraction_detections_draft.
    Roboflow's first-pass detections land in extraction_detection_details, so this
    initializer gives the editor a real editable layer without overwriting manual
    edits on subsequent retries.
    """
    existing_drafts = supabase_request('GET', 'extraction_detections_draft', filters={
        'page_id': f'eq.{page_id}',
        'select': 'id',
        'limit': '1'
    }) or []
    if existing_drafts:
        return 0

    detail_rows = detail_rows or supabase_request('GET', 'extraction_detection_details', filters={
        'page_id': f'eq.{page_id}',
        'status': 'neq.deleted',
        'order': 'detection_index.asc'
    }) or []
    if not detail_rows:
        return 0

    draft_rows = []
    for detail in detail_rows:
        draft_rows.append({
            'id': str(uuid.uuid4()),
            'job_id': job_id,
            'page_id': page_id,
            'source_detection_id': detail.get('id'),
            'class': detail.get('class'),
            'pixel_x': detail.get('pixel_x'),
            'pixel_y': detail.get('pixel_y'),
            'pixel_width': detail.get('pixel_width'),
            'pixel_height': detail.get('pixel_height'),
            'confidence': detail.get('confidence'),
            'detection_index': detail.get('detection_index'),
            'matched_tag': detail.get('matched_tag'),
            'is_triangle': detail.get('is_triangle') or False,
            'assigned_material_id': None,
            'material_notes': None,
            'is_deleted': False,
            'is_user_created': False,
            'polygon_points': detail.get('polygon_points'),
            'markup_type': detail.get('markup_type') or 'polygon',
            'status': detail.get('status') or 'auto',
            'area_sf': detail.get('area_sf'),
            'perimeter_lf': detail.get('perimeter_lf'),
            'item_count': detail.get('item_count') or 1,
        })

    inserted = 0
    batch_size = 50
    for i in range(0, len(draft_rows), batch_size):
        batch = draft_rows[i:i + batch_size]
        result = None
        for attempt in range(3):
            result = supabase_request('POST', 'extraction_detections_draft', batch)
            if result:
                break
            if attempt < 2:
                time.sleep(1)
        if result:
            inserted += len(result) if isinstance(result, list) else len(batch)
        else:
            print(
                f"[{job_id}] Failed to seed draft detections for page {page_id} "
                f"batch {i // batch_size + 1}",
                flush=True
            )

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
        update_job(job_id, {'status': 'processing', 'error_message': None})
        
        # Get job info
        job = get_job(job_id)
        if not job:
            print(f"[{job_id}] Job not found", flush=True)
            return
        
        default_scale = job.get('default_scale_ratio')
        job_dpi = get_safe_dpi(job.get('plan_dpi'), context=f"job {job_id}")
        
        # Get classified pages
        pages = get_classified_pages(job_id)
        if not pages:
            print(f"[{job_id}] No classified pages to process", flush=True)
            update_job(job_id, {
                'status': 'failed',
                'error_message': 'No classified pages were available for detection. Review and approve page types first.'
            })
            return
        
        print(f"[{job_id}] Processing {len(pages)} pages...", flush=True)
        
        detection_page_types = set(config.ROBOFLOW_PAGE_TYPES or ['elevation'])
        detection_pages = [
            p for p in pages
            if (p.get('page_type') or '').lower() in detection_page_types
        ]
        schedule_pages = [p for p in pages if p.get('page_type') == 'schedule']

        if not detection_pages:
            page_type_list = ', '.join(sorted(detection_page_types))
            print(f"[{job_id}] No pages selected for Roboflow detection", flush=True)
            update_job(job_id, {
                'status': 'failed',
                'error_message': (
                    'No pages were selected for Roboflow detection. '
                    f'Review and approve at least one page with type: {page_type_list}.'
                )
            })
            return
        
        totals = {
            'total_net_siding_sqft': 0,
            'total_gross_wall_sqft': 0,
            'total_windows': 0,
            'total_doors': 0,
            'total_detections': 0
        }
        processed = 0
        detection_failures = []
        
        # Process pages configured for Roboflow object detection.
        for page in detection_pages:
            page_id = page.get('id')
            image_url = page.get('image_url')
            
            # Use scale priority: override > page > job default > fallback
            scale_ratio = get_safe_scale_ratio(
                scale_override or page.get('scale_ratio') or default_scale,
                context=f"extraction page {page_id}"
            )
            dpi = page.get('dpi') or job_dpi
            
            # Run Roboflow detection
            detection = detect_with_roboflow(image_url)
            
            if 'error' not in detection:
                raw_predictions = detection.get('predictions', [])

                # Extract and save image dimensions from Roboflow response
                if raw_predictions and raw_predictions[0].get('rle_mask', {}).get('size'):
                    mask_size = raw_predictions[0]['rle_mask']['size']
                    img_height, img_width = mask_size[0], mask_size[1]
                    supabase_request('PATCH', f'extraction_pages?id=eq.{page_id}', {
                        'original_width': img_width,
                        'original_height': img_height
                    })
                    print(f"[{job_id}] Saved dimensions: {img_width}x{img_height}", flush=True)

                # POST-PROCESS: Filter duplicates, merge garages, drop low-confidence
                print(f"[{job_id}] Raw Roboflow predictions: {len(raw_predictions)}", flush=True)
                postprocess_result = postprocess_detections(raw_predictions)
                predictions = postprocess_result['predictions']
                pp_stats = postprocess_result['stats']

                # Log post-processing stats
                if pp_stats['confidence_filtered'] or pp_stats['iou_suppressed'] or pp_stats['doors_merged_to_garages']:
                    print(f"[{job_id}] Post-process: conf={pp_stats['confidence_filtered']}, "
                          f"iou={pp_stats['iou_suppressed']}, size={pp_stats['size_filtered']}, "
                          f"contained={pp_stats['containment_filtered']}, garage_merge={pp_stats['doors_merged_to_garages']}", flush=True)

                # INSERT DETECTIONS INTO DATABASE (post-processed)
                inserted_count, inserted_records = _insert_detections(job_id, page_id, predictions, scale_ratio, dpi)
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
                    'error_message': None,
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
                detection_error = detection.get('error') or 'Roboflow detection failed.'
                detection_failures.append({
                    'page_id': page_id,
                    'page_number': page.get('page_number'),
                    'error': detection_error
                })
                print(f"[{job_id}] Detection failed for page {page_id}: {detection_error}", flush=True)
                update_page(page_id, {
                    'status': 'failed',
                    'error_message': detection_error
                })
            
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
                    'error_message': None,
                    'extraction_data': schedule_data
                })
                print(f"[{job_id}] Extracted schedule: {len(schedule_data.get('windows', []))} windows, {len(schedule_data.get('doors', []))} doors", flush=True)
            else:
                schedule_error = schedule_data.get('error') or 'Schedule extraction failed.'
                print(f"[{job_id}] Schedule extraction failed: {schedule_error}", flush=True)
                update_page(page_id, {
                    'status': 'failed',
                    'error_message': schedule_error
                })
            
            processed += 1
            update_job(job_id, {'pages_processed': processed})
        
        # Skip other page types
        for page in pages:
            if page.get('page_type') != 'schedule' and (page.get('page_type') or '').lower() not in detection_page_types:
                update_page(page.get('id'), {
                    'status': 'skipped',
                    'error_message': None
                })

        if detection_failures:
            totals['detection_errors'] = detection_failures

        if detection_failures and len(detection_failures) == len(detection_pages) and totals['total_detections'] == 0:
            first_error = detection_failures[0].get('error') or 'Roboflow detection failed.'
            error_message = (
                f"Roboflow detection failed for all {len(detection_pages)} detection pages: "
                f"{first_error}"
            )
            print(f"[{job_id}] {error_message}", flush=True)
            update_job(job_id, {
                'status': 'failed',
                'error_message': error_message,
                'results_summary': totals,
                'total_detections': totals['total_detections']
            })
            return

        partial_error_message = None
        if detection_failures:
            partial_error_message = (
                f"Roboflow detection failed for {len(detection_failures)} of "
                f"{len(detection_pages)} detection pages."
            )

        if config.REFINEMENT_AUTO_ENABLED:
            try:
                from services.job_refinement_service import refine_job_detections
                print(f"[{job_id}] Refining detections before editor handoff...", flush=True)
                update_job(job_id, {'status': 'refining'})
                refinement_summary = refine_job_detections(
                    job_id,
                    mode='auto',
                    page_ids=[p.get('id') for p in detection_pages if p.get('id')],
                    set_job_status=False,
                )
                totals['refinement'] = {
                    'pages_processed': refinement_summary.get('pages_processed'),
                    'pages_refined': refinement_summary.get('pages_refined'),
                    'pages_seeded_from_refined': refinement_summary.get('pages_seeded_from_refined'),
                    'pages_seeded_from_raw': refinement_summary.get('pages_seeded_from_raw'),
                    'applied_actions': refinement_summary.get('applied_actions'),
                    'blocked_actions': refinement_summary.get('blocked_actions'),
                    'review_flags': refinement_summary.get('review_flags'),
                    'storage_available': refinement_summary.get('storage_available'),
                    'openai_available': refinement_summary.get('openai_available'),
                }
                print(
                    f"[{job_id}] Refinement complete: "
                    f"{refinement_summary.get('pages_refined', 0)} page(s) refined, "
                    f"{refinement_summary.get('pages_seeded_from_refined', 0)} seeded from refined, "
                    f"{refinement_summary.get('pages_seeded_from_raw', 0)} seeded from raw",
                    flush=True,
                )
            except Exception as refinement_err:
                print(f"[{job_id}] Refinement failed; seeding raw detections: {refinement_err}", flush=True)
                totals['refinement_error'] = str(refinement_err)
                raw_seeded = 0
                for page in detection_pages:
                    raw_seeded += _seed_draft_detections_for_page(job_id, page.get('id'))
                totals['raw_draft_seeded_after_refinement_error'] = raw_seeded
        else:
            raw_seeded = 0
            for page in detection_pages:
                raw_seeded += _seed_draft_detections_for_page(job_id, page.get('id'))
            totals['raw_draft_seeded'] = raw_seeded
        
        # Update job with totals
        update_job(job_id, {
            'status': 'complete',
            'error_message': partial_error_message,
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
