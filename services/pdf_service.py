"""
PDF conversion service
"""

import os
import tempfile
import requests
from io import BytesIO

from config import config
from database import update_job, create_page, upload_to_storage


def convert_pdf_background(job_id, pdf_url):
    """
    Background task to convert PDF to images.
    
    Steps:
    1. Download PDF
    2. Convert each page to PNG
    3. Upload to storage
    4. Create extraction_pages records
    5. Auto-trigger classification
    """
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
        
        print(f"[{job_id}] Downloading PDF...", flush=True)
        update_job(job_id, {'status': 'converting'})
        
        # Download PDF to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            response = requests.get(pdf_url, timeout=300, stream=True)
            if response.status_code != 200:
                update_job(job_id, {'status': 'failed', 'error_message': 'Download failed'})
                return
            
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Get page count
        try:
            info = pdfinfo_from_path(tmp_path)
            total_pages = info['Pages']
        except:
            total_pages = 100
        
        update_job(job_id, {'total_pages': total_pages, 'plan_dpi': config.DEFAULT_DPI})
        
        # Convert in chunks
        pages_converted = 0
        for start_page in range(1, total_pages + 1, config.PDF_CHUNK_SIZE):
            end_page = min(start_page + config.PDF_CHUNK_SIZE - 1, total_pages)
            
            try:
                images = convert_from_path(
                    tmp_path,
                    dpi=config.DEFAULT_DPI,
                    fmt='png',
                    first_page=start_page,
                    last_page=end_page,
                    thread_count=1
                )
                
                for i, img in enumerate(images):
                    page_num = start_page + i
                    
                    # Resize if too large
                    if img.width > 2000 or img.height > 2000:
                        img.thumbnail((2000, 2000))
                    
                    # Save full image
                    buffer = BytesIO()
                    img.save(buffer, format='PNG', optimize=True)
                    buffer.seek(0)
                    
                    filename = f"{job_id}/page_{page_num:03d}.png"
                    image_url = upload_to_storage(buffer.getvalue(), filename, 'image/png')
                    
                    # Create thumbnail
                    thumb = img.copy()
                    thumb.thumbnail((200, 200))
                    thumb_buffer = BytesIO()
                    thumb.save(thumb_buffer, format='PNG')
                    
                    thumb_filename = f"{job_id}/thumb_{page_num:03d}.png"
                    thumb_url = upload_to_storage(thumb_buffer.getvalue(), thumb_filename, 'image/png')
                    
                    # Create page record
                    create_page({
                        'job_id': job_id,
                        'page_number': page_num,
                        'image_url': image_url,
                        'thumbnail_url': thumb_url,
                        'status': 'pending',
                        'dpi': config.DEFAULT_DPI
                    })
                    
                    pages_converted += 1
                    del img
                
                del images
                update_job(job_id, {'pages_converted': pages_converted})
            
            except Exception as e:
                print(f"[{job_id}] Chunk error: {e}", flush=True)
        
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        update_job(job_id, {'status': 'classifying'})
        print(f"[{job_id}] Conversion complete: {pages_converted} pages", flush=True)

        # Auto-trigger intelligent analysis (parallel, 50+ fields extracted)
        # Replaces legacy sequential classification
        from services.intelligent_analysis_service import analyze_job_background
        analyze_job_background(job_id)

        # After analysis completes, run aggregation to calculate corner LF
        from services.aggregation_service import aggregate_job
        try:
            aggregate_job(job_id)
            print(f"[{job_id}] ✅ Aggregation complete", flush=True)
        except Exception as e:
            print(f"[{job_id}] ⚠️ Aggregation failed (non-fatal): {e}", flush=True)
            # Don't fail the job - aggregation is optional enhancement
    
    except Exception as e:
        print(f"[{job_id}] Conversion failed: {e}", flush=True)
        update_job(job_id, {'status': 'failed', 'error_message': str(e)})
