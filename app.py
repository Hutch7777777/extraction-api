"""
Extraction API v4.0 - Modular Architecture

This is the main Flask entry point. All business logic is in modules:
- config.py: Configuration
- core/: External API clients (Roboflow, Claude)
- services/: Business logic (extraction, markup, takeoff)
- database/: Supabase client and repositories
- geometry/: Scale parsing and calculations
- utils/: Validation and utilities
"""

import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import config
from database import (
    supabase_request,
    get_job, create_job, update_job, list_jobs,
    get_page, update_page, get_pages_by_job, get_elevation_pages
)
from core import claude_client
from geometry import parse_scale_notation


# ============================================================
# FLASK APP INITIALIZATION
# ============================================================

app = Flask(__name__)
CORS(app, origins=config.CORS_ORIGINS)


# ============================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "4.8",
        "architecture": "modular",
        "features": [
            "markups",
            "scale_extraction",
            "cross_reference",
            "floor_plan_analysis",
            "elevation_ocr",
            "dimension_sources",
            "roof_intelligence",
            "linear_elements",
            "intelligent_analysis",
            "bluebeam_export",
            "bluebeam_import"  # NEW: Import Bluebeam-edited PDFs for round-trip workflow
        ]
    })


@app.route('/job-status', methods=['GET'])
def job_status():
    """Get job status with optional page details"""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    result = {"job": job}
    
    if request.args.get('include_pages', 'false').lower() == 'true':
        result["pages"] = get_pages_by_job(job_id)
    
    return jsonify(result)


@app.route('/list-jobs', methods=['GET'])
def list_jobs_endpoint():
    """List recent jobs"""
    return jsonify({"jobs": list_jobs()})


# ============================================================
# JOB MANAGEMENT ENDPOINTS
# ============================================================

@app.route('/start-job', methods=['POST'])
def start_job():
    """Start a new extraction job"""
    from services.pdf_service import convert_pdf_background
    
    data = request.json
    if not data.get('pdf_url') or not data.get('project_id'):
        return jsonify({"error": "pdf_url and project_id required"}), 400
    
    job = create_job({
        'project_id': data['project_id'],
        'project_name': data.get('project_name', ''),
        'source_pdf_url': data['pdf_url'],
        'status': 'pending',
        'plan_dpi': config.DEFAULT_DPI
    })
    
    if not job:
        return jsonify({"error": "Failed to create job"}), 500
    
    job_id = job['id']
    threading.Thread(target=convert_pdf_background, args=(job_id, data['pdf_url'])).start()
    
    return jsonify({"success": True, "job_id": job_id})


@app.route('/classify-job', methods=['POST'])
def classify_job():
    """Classify all pages in a job (LEGACY - use /analyze-job instead)"""
    from services.classification_service import classify_job_background

    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400

    threading.Thread(target=classify_job_background, args=(data['job_id'],)).start()

    return jsonify({"success": True, "job_id": data['job_id'], "status": "classifying"})


# ============================================================
# INTELLIGENT ANALYSIS ENDPOINTS (v4.5)
# ============================================================

@app.route('/analyze-job', methods=['POST'])
def analyze_job_endpoint():
    """
    Run intelligent analysis on all pages in a job.

    This is the next-generation replacement for /classify-job.
    Uses parallel processing (10 concurrent) and extracts comprehensive data
    in a single API call per page.

    Request body:
        - job_id: UUID of job to analyze
        - sync: bool (default: false) - if true, wait for completion and return results

    Returns:
        Async mode: {"success": true, "job_id": "...", "status": "analyzing"}
        Sync mode: Full analysis results with all extracted data
    """
    from services.intelligent_analysis_service import analyze_job_background, analyze_job

    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400

    job_id = data['job_id']
    sync_mode = data.get('sync', False)

    # Verify job exists
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if sync_mode:
        # Run synchronously and return full results
        result = analyze_job(job_id)
        return jsonify(result)
    else:
        # Run in background thread
        threading.Thread(target=analyze_job_background, args=(job_id,)).start()
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "analyzing",
            "message": "Intelligent analysis started. Poll /job-status for progress."
        })


@app.route('/page-analysis/<job_id>', methods=['GET'])
def get_page_analysis(job_id):
    """
    Get extracted analysis data for all pages in a job.

    Returns pages with their extracted_data JSONB parsed, including:
    - classification (page_type, confidence, elevation_name)
    - scale (notation, ratio)
    - element_counts (windows, doors, garages, corners)
    - dimensions (wall heights, plate heights)
    - materials (siding, trim, fascia)
    - spatial_context (stories, roof style)
    - quality_indicators

    Query params:
        - page_type: Filter by page type (elevation, floor_plan, etc.)
        - include_raw: Include raw extraction data (default: true)

    Returns:
        List of pages with extracted data
    """
    # Get job to verify it exists
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Get all pages
    page_type_filter = request.args.get('page_type')
    include_raw = request.args.get('include_raw', 'true').lower() == 'true'

    if page_type_filter:
        pages = get_pages_by_job(job_id, page_type=page_type_filter)
    else:
        pages = get_pages_by_job(job_id)

    # Process pages to extract summary data
    results = []
    for page in pages:
        page_data = {
            "page_id": page.get('id'),
            "page_number": page.get('page_number'),
            "page_type": page.get('page_type'),
            "page_type_confidence": page.get('page_type_confidence'),
            "elevation_name": page.get('elevation_name'),
            "scale_notation": page.get('scale_notation'),
            "scale_ratio": page.get('scale_ratio'),
            "status": page.get('status'),
            "image_url": page.get('image_url'),
            "thumbnail_url": page.get('thumbnail_url')
        }

        # Include extracted_data if available and requested
        extracted = page.get('extracted_data')
        if extracted and include_raw:
            page_data['extracted_data'] = extracted

        # Always include a summary if extracted_data exists
        if extracted:
            from services.intelligent_analysis_service import get_extraction_summary
            page_data['summary'] = get_extraction_summary(extracted)

        results.append(page_data)

    return jsonify({
        "job_id": job_id,
        "job_status": job.get('status'),
        "total_pages": len(results),
        "pages": results
    })


@app.route('/analyze-page', methods=['POST'])
def analyze_single_page_endpoint():
    """
    Analyze a single page with intelligent extraction.

    Useful for:
    - Testing extraction on specific pages
    - Re-running analysis after corrections
    - Debugging extraction issues

    Request body:
        - page_id: UUID of page to analyze

    Returns:
        Full extraction result including all extracted data
    """
    from services.intelligent_analysis_service import analyze_single_page_sync

    data = request.json
    if not data.get('page_id'):
        return jsonify({"error": "page_id required"}), 400

    page_id = data['page_id']

    # Verify page exists
    page = get_page(page_id)
    if not page:
        return jsonify({"error": "Page not found"}), 404

    # Run analysis
    result = analyze_single_page_sync(page_id)

    return jsonify({
        "page_id": page_id,
        "page_number": page.get('page_number'),
        **result
    })


@app.route('/analysis-estimate/<job_id>', methods=['GET'])
def get_analysis_estimate(job_id):
    """
    Get cost and time estimate for analyzing a job.

    Call this before /analyze-job to preview:
    - Estimated API token usage
    - Estimated cost in USD
    - Estimated processing time

    Returns:
        Estimate data including tokens, cost, and time
    """
    from services.intelligent_analysis_service import estimate_job_cost

    # Get job
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    page_count = job.get('total_pages') or 0

    if page_count == 0:
        # Count pages directly
        pages = get_pages_by_job(job_id)
        page_count = len(pages)

    estimate = estimate_job_cost(page_count)

    return jsonify({
        "job_id": job_id,
        "job_status": job.get('status'),
        **estimate
    })


@app.route('/process-job', methods=['POST'])
def process_job():
    """Process all pages in a job (detection + measurements)"""
    from services.extraction_service import process_job_background

    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400

    threading.Thread(
        target=process_job_background,
        args=(data['job_id'], data.get('scale_ratio'), data.get('generate_markups', True))
    ).start()

    return jsonify({"success": True, "job_id": data['job_id'], "status": "processing"})


# ============================================================
# AGGREGATION ENDPOINTS (v4.5)
# ============================================================

@app.route('/aggregate-job/<job_id>', methods=['POST'])
def aggregate_job_data(job_id):
    """
    Aggregate extracted data from all pages into unified measurements.

    Combines floor plan corners with elevation heights to calculate
    corner linear feet. Merges window/door counts from schedules and
    elevations. Tracks sources and confidence for all values.

    Returns:
        Aggregated data with:
        - corners: outside/inside counts with LF calculations
        - heights: story heights and total wall height
        - elements: windows, doors, gables, garages with source preference
        - materials: siding type and profile
        - spatial: stories, roof style, foundation type
        - quality: data completeness and warnings
    """
    from services.aggregation_service import aggregate_job

    result = aggregate_job(job_id)

    if not result.get('success'):
        return jsonify(result), 404 if result.get('error') == 'Job not found' else 400

    return jsonify(result)


@app.route('/job-summary/<job_id>', methods=['GET'])
def get_job_summary(job_id):
    """
    Get the aggregated summary for a job.

    Must run POST /aggregate-job/<job_id> first to generate the summary.

    Returns:
        Previously aggregated data with corners, heights, elements,
        materials, spatial context, and quality metrics.
    """
    from services.aggregation_service import get_aggregated_summary

    result = get_aggregated_summary(job_id)

    if not result:
        return jsonify({
            "error": "No aggregated data found. Run POST /aggregate-job/<job_id> first.",
            "job_id": job_id
        }), 404

    return jsonify(result)


@app.route('/recalculate-corners/<job_id>', methods=['POST'])
def recalculate_corner_lf(job_id):
    """
    Recalculate corner linear feet with a manually specified wall height.

    Useful when the user corrects the wall height after reviewing.

    Request body:
        - wall_height_ft: Total wall height in feet (e.g., 18.0 for 2-story)

    Returns:
        Updated corner LF calculations
    """
    from services.aggregation_service import recalculate_corner_lf as recalc

    data = request.json
    if not data.get('wall_height_ft'):
        return jsonify({"error": "wall_height_ft required"}), 400

    result = recalc(job_id, data['wall_height_ft'])

    if not result.get('success'):
        return jsonify(result), 404

    return jsonify(result)


# ============================================================
# MARKUP ENDPOINTS
# ============================================================

@app.route('/generate-markups', methods=['POST'])
def generate_markups():
    """Generate markup images for a job or single page"""
    from services.markup_service import generate_markups_for_page, generate_markups_for_job
    
    data = request.json
    trades = data.get('trades', ['all', 'siding', 'roofing'])
    
    if data.get('page_id'):
        result = generate_markups_for_page(data['page_id'], trades)
    elif data.get('job_id'):
        result = generate_markups_for_job(data['job_id'], trades)
    else:
        return jsonify({"error": "job_id or page_id required"}), 400
    
    return jsonify(result)


@app.route('/comprehensive-markup', methods=['POST'])
def comprehensive_markup():
    """Generate comprehensive markup with legend"""
    from services.markup_service import generate_comprehensive_markup
    
    data = request.json
    
    if data.get('page_id'):
        return jsonify(generate_comprehensive_markup(data['page_id']))
    
    elif data.get('job_id'):
        pages = get_elevation_pages(data['job_id'])
        if not pages:
            return jsonify({"error": "No elevation pages found"}), 404
        
        results = [generate_comprehensive_markup(p['id']) for p in pages]
        return jsonify({
            "job_id": data['job_id'],
            "markups_generated": len([r for r in results if r.get('success')]),
            "results": results
        })
    
    return jsonify({"error": "page_id or job_id required"}), 400


# ============================================================
# BLUEBEAM EXPORT ENDPOINT
# ============================================================

@app.route('/export-bluebeam', methods=['POST'])
def export_bluebeam():
    """Export detections to Bluebeam-compatible annotated PDF"""
    from services.bluebeam_service import export_bluebeam_pdf

    data = request.json
    job_id = data.get('job_id')

    if not job_id:
        return jsonify({'success': False, 'error': 'job_id required'}), 400

    include_materials = data.get('include_materials', True)

    try:
        result = export_bluebeam_pdf(job_id, include_materials)
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
    except Exception as e:
        import traceback
        print(f"[export-bluebeam] Error: {traceback.format_exc()}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/import-bluebeam', methods=['POST'])
def import_bluebeam():
    """
    Import a Bluebeam-edited PDF and generate a diff against current detections.

    Accepts multipart form data with:
    - file: The PDF file to import
    - job_id: The extraction job UUID to compare against
    - modification_threshold: (optional) IoU threshold for modifications (default 0.8)

    Returns a diff summary showing MATCHED, MODIFIED, DELETED, and ADDED annotations.
    Does NOT apply changes - use /apply-bluebeam-changes for that (future endpoint).
    """
    from services.bluebeam_import_service import import_bluebeam_pdf

    # Check for file in request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded. Use multipart form with "file" field.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    # Check file extension
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'File must be a PDF'}), 400

    # Get job_id from form data
    job_id = request.form.get('job_id')
    if not job_id:
        return jsonify({'success': False, 'error': 'job_id required in form data'}), 400

    # Get optional modification threshold
    try:
        modification_threshold = float(request.form.get('modification_threshold', 0.8))
    except ValueError:
        modification_threshold = 0.8

    try:
        # Read PDF bytes
        pdf_bytes = file.read()
        print(f"[import-bluebeam] Received {len(pdf_bytes)} bytes for job {job_id}", flush=True)

        # Process import
        result = import_bluebeam_pdf(pdf_bytes, job_id, modification_threshold)

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        import traceback
        print(f"[import-bluebeam] Error: {traceback.format_exc()}", flush=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# TAKEOFF ENDPOINTS
# ============================================================

@app.route('/calculate-takeoff', methods=['POST'])
def calculate_takeoff():
    """Calculate takeoff measurements from detections"""
    from services.takeoff_service import calculate_takeoff_for_page, calculate_takeoff_for_job
    
    data = request.json
    
    if data.get('page_id'):
        return jsonify(calculate_takeoff_for_page(data['page_id']))
    elif data.get('job_id'):
        return jsonify(calculate_takeoff_for_job(data['job_id']))
    
    return jsonify({"error": "page_id or job_id required"}), 400


@app.route('/job-takeoff', methods=['GET'])
def get_job_takeoff():
    """Get calculated takeoff totals for a job"""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    totals = supabase_request('GET', 'extraction_job_totals', filters={'job_id': f'eq.{job_id}'})
    if not totals:
        return jsonify({"error": "No takeoff calculated. Run /calculate-takeoff first"}), 404
    
    elevations = supabase_request('GET', 'extraction_elevation_calcs', filters={
        'job_id': f'eq.{job_id}',
        'order': 'elevation_name'
    })
    
    return jsonify({"totals": totals[0], "elevations": elevations or []})


# ============================================================
# CROSS-REFERENCE ENDPOINTS
# ============================================================

@app.route('/cross-reference', methods=['POST'])
def cross_reference():
    """Build cross-references between schedules and detections"""
    from services.cross_ref_service import build_cross_references
    
    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400
    
    return jsonify(build_cross_references(data['job_id']))


@app.route('/takeoff-summary', methods=['GET'])
def takeoff_summary():
    """Get takeoff summary with schedule data"""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    summary = supabase_request('GET', 'extraction_takeoff_summary', filters={'job_id': f'eq.{job_id}'})
    if not summary:
        return jsonify({"error": "No summary found"}), 404
    
    return jsonify(summary[0])


@app.route('/cross-refs', methods=['GET'])
def get_cross_refs():
    """Get cross-reference data for a job"""
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    refs = supabase_request('GET', 'extraction_cross_refs', filters={
        'job_id': f'eq.{job_id}',
        'order': 'element_type,tag'
    })
    
    return jsonify({"cross_refs": refs or []})


# ============================================================
# FLOOR PLAN ANALYSIS ENDPOINTS
# ============================================================

@app.route('/analyze-floor-plan', methods=['POST'])
def analyze_floor_plan():
    """Analyze floor plan(s) for corner counts"""
    from services.floor_plan_service import analyze_floor_plan_for_job, analyze_single_floor_plan
    
    data = request.json
    
    if data.get('page_id'):
        return jsonify(analyze_single_floor_plan(data['page_id']))
    elif data.get('job_id'):
        return jsonify(analyze_floor_plan_for_job(data['job_id']))
    
    return jsonify({"error": "page_id or job_id required"}), 400


# ============================================================
# OCR EXTRACTION ENDPOINTS (Phase 2)
# ============================================================

@app.route('/extract-dimensions', methods=['POST'])
def extract_dimensions():
    """
    Extract dimension text from an elevation page using Claude Vision OCR.
    
    This extracts:
    - Wall heights
    - Window/door callouts
    - Dimension strings with locations
    - Level markers
    - Eave/ridge heights
    
    Request body:
        - page_id: UUID of elevation page
        OR
        - image_url: Direct URL to image
    
    Returns:
        OCR extraction results with confidence scores
    """
    from core import extract_elevation_dimensions
    
    data = request.json
    
    # Get image URL
    if data.get('page_id'):
        page = get_page(data['page_id'])
        if not page:
            return jsonify({"error": "Page not found"}), 404
        image_url = page.get('image_url')
        if not image_url:
            return jsonify({"error": "Page has no image URL"}), 400
    elif data.get('image_url'):
        image_url = data['image_url']
    else:
        return jsonify({"error": "page_id or image_url required"}), 400
    
    # Run OCR extraction
    result = extract_elevation_dimensions(image_url)
    
    # If page_id provided, store results in database
    if data.get('page_id') and not result.get('error'):
        _store_ocr_results(data['page_id'], result)
    
    return jsonify(result)


@app.route('/ocr-job', methods=['POST'])
def ocr_job():
    """
    Run OCR extraction on all elevation pages in a job.
    
    Request body:
        - job_id: UUID of job
        - reprocess: bool - if true, reprocess pages with existing OCR (default: false)
    
    Returns:
        Summary of OCR extraction for all pages
    """
    from core import extract_elevation_dimensions
    
    data = request.json
    if not data.get('job_id'):
        return jsonify({"error": "job_id required"}), 400
    
    job_id = data['job_id']
    reprocess = data.get('reprocess', False)
    
    # Get elevation pages
    pages = get_elevation_pages(job_id)
    if not pages:
        return jsonify({"error": "No elevation pages found"}), 404
    
    results = []
    
    for page in pages:
        page_id = page['id']
        ocr_status = page.get('ocr_status')
        
        # Skip if already processed (unless reprocess=true)
        if ocr_status == 'complete' and not reprocess:
            results.append({
                "page_id": page_id,
                "status": "skipped",
                "reason": "already processed"
            })
            continue
        
        image_url = page.get('image_url')
        if not image_url:
            results.append({
                "page_id": page_id,
                "status": "error",
                "error": "no image URL"
            })
            continue
        
        # Update status to processing
        update_page(page_id, {'ocr_status': 'processing'})
        
        # Run OCR
        ocr_result = extract_elevation_dimensions(image_url)
        
        if ocr_result.get('error'):
            update_page(page_id, {'ocr_status': 'failed'})
            results.append({
                "page_id": page_id,
                "status": "error",
                "error": ocr_result['error']
            })
        else:
            # Store results
            _store_ocr_results(page_id, ocr_result)
            results.append({
                "page_id": page_id,
                "status": "success",
                "wall_heights_found": len(ocr_result.get('wall_heights', [])),
                "callouts_found": len(ocr_result.get('element_callouts', [])),
                "dimensions_found": len(ocr_result.get('dimension_text', [])),
                "confidence": ocr_result.get('extraction_confidence')
            })
    
    successful = len([r for r in results if r.get('status') == 'success'])
    
    return jsonify({
        "job_id": job_id,
        "pages_processed": len(results),
        "successful": successful,
        "results": results
    })


@app.route('/ocr-data', methods=['GET'])
def get_ocr_data():
    """
    Get OCR extraction data for a page or job.
    
    Query params:
        - page_id: UUID of page
        OR
        - job_id: UUID of job (returns all pages)
    
    Returns:
        OCR data from extraction_ocr_data table
    """
    page_id = request.args.get('page_id')
    job_id = request.args.get('job_id')
    
    if page_id:
        data = supabase_request('GET', 'extraction_ocr_data', filters={
            'page_id': f'eq.{page_id}'
        })
        return jsonify(data[0] if data else {"error": "No OCR data found"})
    
    elif job_id:
        data = supabase_request('GET', 'v_page_ocr_summary', filters={
            'job_id': f'eq.{job_id}',
            'order': 'page_number'
        })
        return jsonify({"job_id": job_id, "pages": data or []})
    
    return jsonify({"error": "page_id or job_id required"}), 400


def _store_ocr_results(page_id, ocr_result):
    """
    Store OCR extraction results in database.
    
    - Inserts/updates extraction_ocr_data
    - Updates extraction_pages.ocr_status
    - Updates extraction_elevation_calcs with wall height
    """
    import datetime
    
    page = get_page(page_id)
    if not page:
        return
    
    job_id = page.get('job_id')
    
    # Prepare OCR data record
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
        'claude_model': 'claude-sonnet-4-20250514',
        'processing_time_ms': ocr_result.get('processing_time_ms')
    }
    
    # Check if OCR data already exists for this page
    existing = supabase_request('GET', 'extraction_ocr_data', filters={
        'page_id': f'eq.{page_id}'
    })
    
    if existing:
        # Update existing record
        supabase_request('PATCH', 'extraction_ocr_data', 
                        data=ocr_record,
                        filters={'page_id': f'eq.{page_id}'})
        ocr_data_id = existing[0]['id']
    else:
        # Insert new record
        result = supabase_request('POST', 'extraction_ocr_data', ocr_record)
        ocr_data_id = result[0]['id'] if result else None
    
    # Update page status
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
                            'ocr_data_id': ocr_data_id,
                            'ocr_eave_height_ft': ocr_result.get('eave_height_ft'),
                            'ocr_ridge_height_ft': ocr_result.get('ridge_height_ft')
                        },
                        filters={'page_id': f'eq.{page_id}'})


# ============================================================
# FUSION ENDPOINTS (Phase 2)
# ============================================================

@app.route('/fuse-data', methods=['POST'])
def fuse_data():
    """
    Run data fusion on a page or job.
    
    Combines OCR callouts with Roboflow detections and schedule data
    to improve dimension accuracy.
    
    Request body:
        - page_id: UUID of page to fuse
        OR
        - job_id: UUID of job (fuses all elevation pages)
    
    Returns:
        Fusion results with matched callouts, dimension sources, discrepancies
    """
    from services.fusion_service import fuse_page_data, fuse_job_data
    
    data = request.json
    
    if data.get('page_id'):
        return jsonify(fuse_page_data(data['page_id']))
    elif data.get('job_id'):
        return jsonify(fuse_job_data(data['job_id']))
    
    return jsonify({"error": "page_id or job_id required"}), 400


@app.route('/fusion-summary', methods=['GET'])
def fusion_summary():
    """
    Get fusion summary for a job.
    
    Shows breakdown of dimension sources and discrepancies.
    
    Query params:
        - job_id: UUID of job
    
    Returns:
        Summary statistics
    """
    from services.fusion_service import get_fusion_summary
    
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    return jsonify(get_fusion_summary(job_id))


@app.route('/dimension-sources', methods=['GET'])
def get_dimension_sources():
    """
    Get all dimension sources for a detection.
    
    Query params:
        - detection_id: UUID of detection
    
    Returns:
        List of dimension sources with priorities
    """
    detection_id = request.args.get('detection_id')
    if not detection_id:
        return jsonify({"error": "detection_id required"}), 400
    
    sources = supabase_request('GET', 'extraction_dimension_sources', filters={
        'detection_id': f'eq.{detection_id}',
        'order': 'source_priority'
    })
    
    return jsonify({"detection_id": detection_id, "sources": sources or []})


@app.route('/detections-with-sources', methods=['GET'])
def get_detections_with_sources():
    """
    Get detections with their dimension source info.
    
    Query params:
        - page_id: UUID of page
        OR
        - job_id: UUID of job
    
    Returns:
        Detections with dimension_source, matched_tag, discrepancy info
    """
    page_id = request.args.get('page_id')
    job_id = request.args.get('job_id')
    
    if page_id:
        data = supabase_request('GET', 'v_detection_with_sources', filters={
            'page_id': f'eq.{page_id}'
        })
    elif job_id:
        data = supabase_request('GET', 'v_detection_with_sources', filters={
            'job_id': f'eq.{job_id}'
        })
    else:
        return jsonify({"error": "page_id or job_id required"}), 400
    
    return jsonify({"detections": data or []})


# ============================================================
# ROOF INTELLIGENCE ENDPOINTS (Phase 3)
# ============================================================

@app.route('/extract-roof-data', methods=['POST'])
def extract_roof_data():
    """
    Extract roof data from a roof plan page using Claude Vision.
    
    Request body:
        - page_id: UUID of roof plan page
        OR
        - image_url: Direct URL to roof plan image
    
    Returns:
        Extracted roof sections, pitches, and linear elements
    """
    from core import extract_roof_plan_data
    from services.roof_service import process_roof_plan
    
    data = request.json
    
    if data.get('page_id'):
        # Process and store in database
        return jsonify(process_roof_plan(data['page_id']))
    
    elif data.get('image_url'):
        # Just run OCR, don't store
        result = extract_roof_plan_data(data['image_url'])
        return jsonify(result)
    
    return jsonify({"error": "page_id or image_url required"}), 400


@app.route('/roof-job', methods=['POST'])
def process_roof_job():
    """
    Process all roof plan pages in a job.
    
    Request body:
        - job_id: UUID of job
        - reprocess: Boolean, reprocess already-processed pages (optional)
    
    Returns:
        Summary of pages processed with totals
    """
    from services.roof_service import process_roof_plans_for_job
    
    data = request.json
    job_id = data.get('job_id')
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    return jsonify(process_roof_plans_for_job(job_id))


@app.route('/roof-summary', methods=['GET'])
def get_roof_summary_endpoint():
    """
    Get roof summary for a job.
    
    Query params:
        - job_id: UUID of job
    
    Returns:
        Roof summary with sections, areas, linear elements, and derived calculations
    """
    from services.roof_service import get_roof_summary
    
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    return jsonify(get_roof_summary(job_id))


@app.route('/calculate-pitch', methods=['POST'])
def calculate_pitch():
    """
    Calculate pitch factor and related values.
    
    Request body:
        - pitch: Pitch notation like '6:12'
        - projected_area_sf: Optional - calculate true area
    
    Returns:
        Pitch factor, true area, waste factor, angle in degrees
    """
    from services.roof_service import (
        calculate_pitch_factor, calculate_true_area,
        get_pitch_waste_factor, pitch_to_degrees
    )
    
    data = request.json
    pitch = data.get('pitch')
    
    if not pitch:
        return jsonify({"error": "pitch required"}), 400
    
    factor = calculate_pitch_factor(pitch)
    result = {
        "pitch": pitch,
        "pitch_factor": factor,
        "waste_factor": get_pitch_waste_factor(pitch),
        "degrees": pitch_to_degrees(pitch)
    }
    
    if data.get('projected_area_sf'):
        result['projected_area_sf'] = data['projected_area_sf']
        result['true_area_sf'] = calculate_true_area(data['projected_area_sf'], pitch)
    
    return jsonify(result)


@app.route('/calculate-roofing', methods=['POST'])
def calculate_roofing():
    """
    Calculate roofing materials needed.
    
    Request body:
        - true_area_sf: True roof surface area
        - pitch: Optional pitch for auto waste factor
        - waste_factor: Optional override waste factor (0.10 = 10%)
        - eave_lf: Eave length for soffit/starter
        - ridge_lf: Ridge length
        - rake_lf: Rake length
        - hip_lf: Hip length
        - overhang_in: Overhang depth in inches (default 12)
    
    Returns:
        Calculated quantities for all roofing components
    """
    from services.roof_service import (
        calculate_roofing_squares, calculate_soffit_area,
        calculate_fascia_lf, calculate_drip_edge_lf,
        calculate_starter_lf, calculate_ridge_cap_lf
    )
    
    data = request.json
    true_area = data.get('true_area_sf', 0)
    pitch = data.get('pitch')
    waste = data.get('waste_factor')
    eave_lf = data.get('eave_lf', 0)
    ridge_lf = data.get('ridge_lf', 0)
    rake_lf = data.get('rake_lf', 0)
    hip_lf = data.get('hip_lf', 0)
    overhang_in = data.get('overhang_in', 12)
    
    return jsonify({
        "input": {
            "true_area_sf": true_area,
            "eave_lf": eave_lf,
            "ridge_lf": ridge_lf,
            "rake_lf": rake_lf,
            "hip_lf": hip_lf
        },
        "calculations": {
            "roofing_squares": calculate_roofing_squares(true_area, waste, pitch),
            "soffit_sf": calculate_soffit_area(eave_lf, overhang_in),
            "soffit_lf": eave_lf,
            "fascia_lf": calculate_fascia_lf(eave_lf, rake_lf),
            "drip_edge_lf": calculate_drip_edge_lf(eave_lf, rake_lf),
            "starter_lf": calculate_starter_lf(eave_lf),
            "ridge_cap_lf": calculate_ridge_cap_lf(ridge_lf, hip_lf)
        }
    })


# ============================================================
# LINEAR ELEMENTS ENDPOINTS (Phase 4)
# ============================================================

@app.route('/calculate-linear', methods=['POST'])
def calculate_linear():
    """
    Calculate all linear elements for a job.
    
    Uses OCR wall heights + corner counts to calculate:
    - Corner LF (outside and inside)
    - Perimeter elements (starter strip, water table, etc.)
    
    Request body:
        - job_id: UUID of job
    
    Returns:
        Complete linear element calculations
    """
    from services.linear_service import calculate_linear_elements_for_job
    
    data = request.json
    job_id = data.get('job_id')
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    return jsonify(calculate_linear_elements_for_job(job_id))


@app.route('/linear-summary', methods=['GET'])
def get_linear_summary_endpoint():
    """
    Get linear element summary for a job.
    
    Query params:
        - job_id: UUID of job
    
    Returns:
        All stored linear elements and trim totals
    """
    from services.linear_service import get_linear_summary
    
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    return jsonify(get_linear_summary(job_id))


@app.route('/wall-heights', methods=['GET'])
def get_wall_heights():
    """
    Get wall heights from OCR data for a job.
    
    Query params:
        - job_id: UUID of job
    
    Returns:
        Floor-by-floor wall heights from elevation OCR
    """
    from services.linear_service import get_wall_heights_from_ocr
    
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    return jsonify(get_wall_heights_from_ocr(job_id))


@app.route('/set-corners', methods=['POST'])
def set_corners():
    """
    Manually set corner counts and recalculate LF.
    
    Use when floor plan analysis is inaccurate.
    
    Request body:
        - job_id: UUID of job
        - outside_corners: Count of outside corners
        - inside_corners: Count of inside corners
        - wall_height_ft: Optional override wall height
    
    Returns:
        Updated corner calculations
    """
    from services.linear_service import set_corner_counts
    
    data = request.json
    job_id = data.get('job_id')
    outside = data.get('outside_corners')
    inside = data.get('inside_corners')
    wall_height = data.get('wall_height_ft')
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    if outside is None or inside is None:
        return jsonify({"error": "outside_corners and inside_corners required"}), 400
    
    return jsonify(set_corner_counts(job_id, outside, inside, wall_height))


@app.route('/calculate-corners', methods=['POST'])
def calculate_corners():
    """
    Calculate corner LF from counts and wall height.
    
    Standalone calculator - doesn't require job data.
    
    Request body:
        - outside_corners: Count
        - inside_corners: Count
        - wall_height_ft: Total wall height (all stories)
        OR
        - first_floor_ft: First floor height
        - second_floor_ft: Optional second floor height
    
    Returns:
        Corner LF calculations and material quantities
    """
    from services.linear_service import calculate_corner_lf
    
    data = request.json
    outside = data.get('outside_corners', 0)
    inside = data.get('inside_corners', 0)
    
    # Get wall height
    if data.get('wall_height_ft'):
        wall_height = data['wall_height_ft']
    else:
        first = data.get('first_floor_ft', 9.0)
        second = data.get('second_floor_ft', 0)
        wall_height = first + second
    
    return jsonify(calculate_corner_lf(outside, inside, wall_height))


# ============================================================
# UTILITY ENDPOINTS
# ============================================================

@app.route('/parse-scale', methods=['POST'])
def parse_scale_endpoint():
    """Parse scale notation to ratio"""
    data = request.json
    notation = data.get('notation', '')
    return jsonify({
        "notation": notation,
        "scale_ratio": parse_scale_notation(notation)
    })


@app.route('/debug-markup', methods=['POST'])
def debug_markup():
    """Debug markup with full error return"""
    import requests
    from io import BytesIO
    from PIL import Image, ImageDraw
    from database import upload_to_storage
    
    data = request.json
    page_id = data.get('page_id')
    
    try:
        page = get_page(page_id)
        if not page:
            return jsonify({"step": "get_page", "error": "Page not found"})
        
        image_url = page.get('image_url')
        extraction_data = page.get('extraction_data', {})
        predictions = extraction_data.get('raw_predictions', [])
        scale_ratio = float(page.get('scale_ratio') or 48)
        job_id = page.get('job_id')
        page_num = page.get('page_number')
        
        if not predictions:
            return jsonify({"step": "check_predictions", "error": "No predictions"})
        
        # Download image
        response = requests.get(image_url, timeout=30)
        image_data = response.content
        
        # Simple markup - just draw boxes
        img = Image.open(BytesIO(image_data)).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        count = 0
        for pred in predictions[:10]:
            x = pred.get('x', 0)
            y = pred.get('y', 0)
            w = pred.get('width', 0)
            h = pred.get('height', 0)
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            count += 1
        
        # Save and upload
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        filename = f"{job_id}/debug_markup_{page_num:03d}.png"
        markup_url = upload_to_storage(buffer.getvalue(), filename, 'image/png')
        
        return jsonify({
            "success": True,
            "boxes_drawn": count,
            "url": markup_url
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})


@app.route('/test-markup', methods=['POST'])
def test_markup():
    """Test markup data availability"""
    import requests
    from io import BytesIO
    from PIL import Image
    
    data = request.json
    page_id = data.get('page_id')
    
    try:
        page = get_page(page_id)
        if not page:
            return jsonify({"error": "Page not found"})
        
        image_url = page.get('image_url')
        extraction_data = page.get('extraction_data', {})
        predictions = extraction_data.get('raw_predictions', [])
        scale_ratio = float(page.get('scale_ratio') or 48)
        
        # Download and check image
        response = requests.get(image_url, timeout=30)
        img = Image.open(BytesIO(response.content))
        
        return jsonify({
            "image_url": image_url,
            "image_size": img.size,
            "predictions_count": len(predictions),
            "scale_ratio": scale_ratio,
            "first_prediction": predictions[0] if predictions else None
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})


# ============================================================
# FACADE MARKUP ENDPOINT
# ============================================================

@app.route('/generate-facade-markup', methods=['POST'])
def generate_facade_markup():
    """
    Generate facade markup from verified detection data.
    Shows Building - Roof = Gross Facade calculation.
    """
    import requests
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    from database import upload_to_storage
    
    data = request.json
    job_id = data.get('job_id')
    page_id = data.get('page_id')
    
    if not job_id:
        return jsonify({"error": "job_id required"}), 400
    
    # Build page query
    filters = {'job_id': f'eq.{job_id}', 'select': 'id,page_number,original_image_url,image_url'}
    if page_id:
        filters['id'] = f'eq.{page_id}'
    
    pages = supabase_request('GET', 'extraction_pages', filters=filters)
    if not pages:
        return jsonify({"error": "No pages found"}), 404
    
    results = []
    
    for page in pages:
        pg_id = page['id']
        page_number = page['page_number']
        image_url = page.get('original_image_url') or page.get('image_url')
        
        if not image_url:
            results.append({"page_number": page_number, "error": "No image URL"})
            continue
        
        # Get detections
        detections = supabase_request('GET', 'extraction_detection_details', filters={
            'page_id': f'eq.{pg_id}',
            'status': 'neq.deleted',
            'select': 'class,pixel_x,pixel_y,pixel_width,pixel_height,area_sf,real_width_ft,real_height_ft'
        })
        
        try:
            img_response = requests.get(image_url)
            img = Image.open(BytesIO(img_response.content)).convert('RGBA')
        except Exception as e:
            results.append({"page_number": page_number, "error": f"Failed to load image: {e}"})
            continue
        
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Track areas
        building_area_sf = 0
        roof_area_sf = 0
        openings_sf = 0
        window_count = door_count = garage_count = 0
        
        buildings = []
        roofs = []
        openings = []
        
        for det in (detections or []):
            cls = det.get('class', '').lower()
            cx = det.get('pixel_x', 0)
            cy = det.get('pixel_y', 0)
            w = det.get('pixel_width', 0)
            h = det.get('pixel_height', 0)
            area = det.get('area_sf', 0)
            
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            if cls == 'building':
                buildings.append((x1, y1, x2, y2))
                building_area_sf += area
            elif cls == 'roof':
                roofs.append((x1, y1, x2, y2))
                roof_area_sf += area
            elif cls in ['window', 'door', 'garage']:
                openings.append((x1, y1, x2, y2, cls))
                openings_sf += area
                if cls == 'window': window_count += 1
                elif cls == 'door': door_count += 1
                elif cls == 'garage': garage_count += 1
        
        gross_facade_sf = building_area_sf - roof_area_sf
        net_siding_sf = gross_facade_sf - openings_sf
        
        # Draw building (blue)
        for (x1, y1, x2, y2) in buildings:
            draw.rectangle([x1, y1, x2, y2], fill=(59, 130, 246, 100), outline=(59, 130, 246, 255), width=3)
        
        # Draw roof with crosshatch (red, excluded)
        for (x1, y1, x2, y2) in roofs:
            draw.rectangle([x1, y1, x2, y2], fill=(220, 38, 38, 120), outline=(220, 38, 38, 255), width=2)
            spacing = 15
            for i in range(int(x1), int(x2), spacing):
                draw.line([(i, y1), (min(i + (y2-y1), x2), y2)], fill=(220, 38, 38, 180), width=1)
                draw.line([(i, y2), (min(i + (y2-y1), x2), y1)], fill=(220, 38, 38, 180), width=1)
        
        # Draw openings
        color_map = {
            'window': (249, 115, 22, 150),
            'door': (34, 197, 94, 150),
            'garage': (234, 179, 8, 150)
        }
        for (x1, y1, x2, y2, cls) in openings:
            color = color_map.get(cls, (255, 0, 0, 150))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(255, 255, 255, 255), width=2)
        
        result_img = Image.alpha_composite(img, overlay)
        draw_result = ImageDraw.Draw(result_img)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw summary box
        summary_lines = [
            f"NET SIDING: {net_siding_sf:,.0f} SF",
            f"Building: {building_area_sf:,.0f} SF - Roof: {roof_area_sf:,.0f} SF",
            f"Gross Facade: {gross_facade_sf:,.0f} SF",
            f"Openings: {openings_sf:,.0f} SF",
            f"Win: {window_count} | Door: {door_count} | Garage: {garage_count}"
        ]
        
        draw_result.rectangle([10, 10, 400, 145], fill=(0, 0, 0, 200))
        y_pos = 20
        draw_result.text((20, y_pos), summary_lines[0], fill=(59, 130, 246), font=font_large)
        y_pos += 30
        for line in summary_lines[1:]:
            draw_result.text((20, y_pos), line, fill=(255, 255, 255), font=font_small)
            y_pos += 22
        
        # Draw legend
        legend_y = img.size[1] - 40
        legend_items = [
            ("Facade", (59, 130, 246)),
            ("Roof (Excl)", (220, 38, 38)),
            ("Window", (249, 115, 22)),
            ("Door", (34, 197, 94)),
            ("Garage", (234, 179, 8))
        ]
        
        x_pos = 20
        for label, color in legend_items:
            draw_result.rectangle([x_pos, legend_y, x_pos + 20, legend_y + 20], fill=color)
            draw_result.text((x_pos + 25, legend_y), label, fill=(255, 255, 255), font=font_small)
            x_pos += 110
        
        # Save and upload
        buffer = BytesIO()
        result_img.convert('RGB').save(buffer, format='PNG', quality=95)
        buffer.seek(0)
        
        filename = f"{job_id}/facade_markup_{page_number:03d}.png"
        markup_url = upload_to_storage(buffer.getvalue(), filename, 'image/png')
        
        results.append({
            "page_number": page_number,
            "page_id": pg_id,
            "markup_url": markup_url,
            "net_siding_sf": round(net_siding_sf, 2),
            "building_area_sf": round(building_area_sf, 2),
            "roof_area_sf": round(roof_area_sf, 2),
            "gross_facade_sf": round(gross_facade_sf, 2),
            "openings_sf": round(openings_sf, 2),
            "window_count": window_count,
            "door_count": door_count,
            "garage_count": garage_count
        })
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "pages": results
    })


# ============================================================
# SIDING POLYGON ENDPOINTS
# ============================================================

@app.route('/siding-polygons', methods=['POST'])
def get_siding_polygons():
    """
    Return polygon coordinates for net siding area (building - roof - openings).
    Used by frontend to render siding overlay in DetectionCanvas.
    """
    data = request.json
    page_id = data.get('page_id')
    job_id = data.get('job_id')

    if not page_id and not job_id:
        return jsonify({"error": "page_id or job_id required"}), 400

    # If job_id provided, get all pages
    if job_id and not page_id:
        pages = supabase_request('GET', 'extraction_pages', filters={
            'job_id': f'eq.{job_id}',
            'page_type': 'eq.elevation'
        })
        results = []
        for page in (pages or []):
            result = _calculate_siding_polygon(page['id'])
            if result:
                results.append(result)
        return jsonify({"success": True, "job_id": job_id, "pages": results})

    # Single page
    result = _calculate_siding_polygon(page_id)
    if not result:
        return jsonify({"error": "No detections found"}), 404

    return jsonify(result)


def _calculate_siding_polygon(page_id):
    """
    Calculate siding polygons for a single page.
    Handles multiple buildings per page - returns one polygon per building.
    """
    # Get all detections for this page
    detections = supabase_request('GET', 'extraction_detection_details', filters={
        'page_id': f'eq.{page_id}',
        'status': 'neq.deleted'
    })

    if not detections:
        return None

    # Collect all buildings, roofs, and openings
    buildings = []
    roofs = []
    openings = []

    for det in detections:
        cls = (det.get('class') or '').lower()
        cx = det.get('pixel_x', 0)
        cy = det.get('pixel_y', 0)
        w = det.get('pixel_width', 0)
        h = det.get('pixel_height', 0)
        area = det.get('area_sf') or 0
        det_id = det.get('id', '')

        # Convert center coords to corner coords
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        bbox = {'id': det_id, 'cx': cx, 'cy': cy, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'area_sf': area}

        if cls == 'building':
            buildings.append(bbox)
        elif cls == 'roof':
            roofs.append(bbox)
        elif cls in ['window', 'door', 'garage']:
            openings.append({'class': cls, **bbox})

    if not buildings:
        return None

    # Calculate siding polygon for each building
    siding_polygons = []
    total_net_siding_sf = 0

    for building in buildings:
        # Find roof that overlaps this building
        matching_roof = None
        for roof in roofs:
            if _detection_overlaps_building(roof, building):
                matching_roof = roof
                break

        # Find openings inside this building
        building_openings = [op for op in openings if _detection_overlaps_building(op, building)]

        # Calculate areas
        building_sf = building['area_sf']
        roof_sf = matching_roof['area_sf'] if matching_roof else 0
        openings_sf = sum(op['area_sf'] for op in building_openings)

        # Calculate exterior polygon (building minus roof from top)
        exterior = _calculate_exterior_bounds(building, matching_roof)

        # Convert openings to hole polygons (simple rectangles)
        holes = []
        for op in building_openings:
            holes.append({
                'class': op['class'],
                'points': [
                    [op['x1'], op['y1']],  # top-left
                    [op['x2'], op['y1']],  # top-right
                    [op['x2'], op['y2']],  # bottom-right
                    [op['x1'], op['y2']]   # bottom-left
                ],
                'area_sf': op['area_sf']
            })

        gross_facade_sf = building_sf - roof_sf
        net_siding_sf = gross_facade_sf - openings_sf
        total_net_siding_sf += net_siding_sf

        siding_polygons.append({
            "building_id": building['id'],
            "exterior": {
                "points": exterior,
                "gross_facade_sf": round(gross_facade_sf, 2)
            },
            "holes": holes,
            "summary": {
                "building_sf": round(building_sf, 2),
                "roof_sf": round(roof_sf, 2),
                "gross_facade_sf": round(gross_facade_sf, 2),
                "openings_sf": round(openings_sf, 2),
                "net_siding_sf": round(net_siding_sf, 2),
                "opening_count": len(building_openings)
            }
        })

    # For backwards compatibility, also include legacy format for single building
    # (first building's data at top level)
    first_polygon = siding_polygons[0] if siding_polygons else None

    return {
        "success": True,
        "page_id": page_id,
        # Legacy format (for backwards compatibility with existing frontend)
        "exterior": first_polygon["exterior"] if first_polygon else {"points": [], "gross_facade_sf": 0},
        "holes": first_polygon["holes"] if first_polygon else [],
        "summary": first_polygon["summary"] if first_polygon else {
            "building_sf": 0, "roof_sf": 0, "gross_facade_sf": 0,
            "openings_sf": 0, "net_siding_sf": 0, "opening_count": 0
        },
        # New multi-building format
        "siding_polygons": siding_polygons,
        "page_summary": {
            "total_buildings": len(buildings),
            "total_net_siding_sf": round(total_net_siding_sf, 2)
        }
    }


def _detection_overlaps_building(detection, building):
    """
    Check if detection center is within building bounds.
    Used to associate roofs/openings with their parent building.
    """
    det_cx = detection['cx']
    det_cy = detection['cy']
    return (building['x1'] <= det_cx <= building['x2'] and
            building['y1'] <= det_cy <= building['y2'])


def _calculate_exterior_bounds(building, roof):
    """
    Calculate exterior polygon points.
    If roof overlaps building, clip building at roof's bottom edge.
    """
    x1, y1, x2, y2 = building['x1'], building['y1'], building['x2'], building['y2']

    if roof:
        # If roof's bottom edge is below building's top edge,
        # start the facade from roof's bottom
        roof_bottom = roof['y2']
        if roof_bottom > y1 and roof_bottom < y2:
            y1 = roof_bottom

    # Return rectangle as polygon points (clockwise)
    return [
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2]   # bottom-left
    ]


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Extraction API v4.0 - Modular Architecture")
    print("=" * 60)
    print(f"Port: {config.PORT}")
    print(f"Debug: {config.DEBUG}")
    print(f"CORS Origins: {config.CORS_ORIGINS}")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=config.DEBUG
    )
