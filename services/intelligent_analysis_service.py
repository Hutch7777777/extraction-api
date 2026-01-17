"""
Intelligent Page Analysis Service

High-performance parallel processing of architectural drawings using Claude Vision.
Extracts comprehensive data in a single API call per page:
- Page classification
- Scale detection
- Element counts (windows, doors, etc.)
- Dimension OCR
- Material callouts
- Spatial context
- Quality indicators

Uses asyncio + aiohttp for 10x speedup over sequential processing.
"""

import asyncio
import aiohttp
import base64
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter

from config import config
from database import update_page, update_job, get_pages_by_job
from database.repositories.detection_repository import batch_create_detections
from utils.validation import normalize_page_type


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CONCURRENT_PAGES = 10  # Parallel API calls
MAX_RETRIES = 2
API_TIMEOUT_SECONDS = 180
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Token pricing (as of Jan 2025)
INPUT_TOKEN_COST_PER_1K = 0.003  # $3 per 1M input tokens
OUTPUT_TOKEN_COST_PER_1K = 0.015  # $15 per 1M output tokens


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class AnalysisResult:
    """Result of analyzing a single page"""
    page_id: str
    page_number: int
    success: bool
    error: Optional[str] = None
    processing_time_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    extracted_data: Optional[Dict] = None


@dataclass
class JobAnalysisResult:
    """Result of analyzing an entire job"""
    job_id: str
    total_pages: int
    successful_pages: int
    failed_pages: int
    total_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    page_type_counts: Dict[str, int]
    results: List[AnalysisResult]


# ============================================================
# COMPREHENSIVE EXTRACTION PROMPT
# ============================================================

EXTRACTION_PROMPT = '''Analyze this architectural drawing and extract ALL available information.

Return ONLY a valid JSON object with this exact structure:

{
  "classification": {
    "page_type": "elevation" | "schedule" | "floor_plan" | "section" | "detail" | "cover" | "site_plan" | "roof_plan" | "other",
    "confidence": 0.0-1.0,
    "elevation_name": "front" | "rear" | "left" | "right" | "north" | "south" | "east" | "west" | null,
    "contains_schedule": true | false,
    "schedule_type": "window" | "door" | "window_and_door" | "finish" | null,
    "is_mirror_reversed": true | false
  },

  "scale": {
    "notation": "exact text like '1/4\" = 1'-0\"' or null",
    "ratio": 48 | 24 | 96 | null,
    "location": "where scale was found on drawing",
    "confidence": 0.0-1.0
  },

  "element_counts": {
    "windows": {"count": 0, "confidence": 0.0-1.0, "types_seen": ["DH", "casement", "picture"]},
    "doors": {"count": 0, "confidence": 0.0-1.0, "types_seen": ["entry", "patio", "french"]},
    "garages": {"count": 0, "confidence": 0.0-1.0, "widths_seen": ["16'", "9'"]},
    "gables": {"count": 0, "confidence": 0.0-1.0},
    "outside_corners": {"count": 0, "confidence": 0.0-1.0, "notes": ""},
    "inside_corners": {"count": 0, "confidence": 0.0-1.0, "notes": ""}
  },

  "dimensions": {
    "wall_heights": [
      {"label": "1ST FLOOR", "text": "9'-0\"", "inches": 108, "confidence": 0.9}
    ],
    "plate_heights": [
      {"label": "T.O. PLATE", "text": "19'-4 1/4\"", "inches": 232.25}
    ],
    "building_width_ft": null,
    "building_length_ft": null,
    "eave_height_ft": null,
    "ridge_height_ft": null,
    "overhang_inches": null,
    "raw_dimensions": [
      {"text": "12'-6\"", "value_inches": 150, "orientation": "horizontal", "context": "window width"}
    ]
  },

  "materials": {
    "siding": {
      "type": "lap" | "shake" | "panel" | "board_and_batten" | null,
      "profile": "HardiePlank" | "Artisan" | null,
      "exposure_inches": 7 | 8 | null,
      "notes": ""
    },
    "trim": {
      "type": "smooth" | "woodgrain" | null,
      "window_width_inches": 3.5 | 4 | 5 | null,
      "door_width_inches": 3.5 | 4 | 5 | null,
      "corner_width_inches": 3 | 4 | 5 | null
    },
    "fascia": {
      "width_inches": 6 | 8 | null,
      "type": null
    },
    "soffit": {
      "type": "vented" | "solid" | null,
      "width_inches": null
    },
    "other_callouts": []
  },

  "spatial_context": {
    "building_orientation": "front_facing" | "side_facing" | "corner" | "unknown",
    "roof_style": "gable" | "hip" | "shed" | "flat" | "complex" | "unknown",
    "roof_pitch": "6:12" | "8:12" | null,
    "stories": 1 | 1.5 | 2 | 2.5 | 3 | null,
    "has_garage": true | false,
    "garage_position": "attached_front" | "attached_side" | "detached" | null,
    "has_porch": true | false,
    "porch_type": "covered" | "open" | null,
    "foundation_type": "slab" | "crawl" | "basement" | null
  },

  "trim_details": {
    "rake_width_inches": null,
    "frieze_height_inches": null,
    "water_table": {"present": false, "height_inches": null},
    "belly_band": {"present": false, "height_inches": null, "location": null},
    "window_sills": {"present": false, "projection_inches": null},
    "decorative_elements": []
  },

  "quality_indicators": {
    "drawing_clarity": "high" | "medium" | "low",
    "dimension_completeness": "complete" | "partial" | "minimal" | "none",
    "has_notes_or_callouts": true | false,
    "estimator_notes": [],
    "missing_information": [],
    "confidence_overall": 0.0-1.0
  },

  "corner_positions": {
    "outside_corners": [
      {"x": 150, "y": 200, "confidence": 0.9, "description": "NW corner of main structure"}
    ],
    "inside_corners": [
      {"x": 250, "y": 350, "confidence": 0.8, "description": "garage setback corner"}
    ],
    "coordinate_reference": "pixel coordinates from top-left origin"
  }
}

EXTRACTION GUIDELINES:
1. Only include data you can actually see - use null for missing values
2. For counts, only count elements you can clearly identify
3. Parse dimensions like "9'-6"" to inches (114)
4. Scale ratio: 1/4"=1'-0" → 48, 1/8"=1'-0" → 96, 1/2"=1'-0" → 24
5. Confidence: 0.9+ for clear data, 0.7-0.9 for inferred, <0.7 for uncertain
6. For floor plans: count corners on the EXTERIOR perimeter only
7. Look for material notes, specifications, and legends on the drawing

CORNER POSITION EXTRACTION (FOR FLOOR PLAN PAGES ONLY):
When analyzing floor plans, identify corner positions along the EXTERIOR building perimeter:
- OUTSIDE CORNERS: Where two exterior walls meet forming an outward-facing angle (convex, building sticks out)
- INSIDE CORNERS: Where two exterior walls meet forming an inward-facing angle (concave, building goes in)
- Return x,y as pixel coordinates from the top-left corner of the image
- Mark the approximate CENTER of each corner intersection
- Include attached garage corners
- Do NOT include interior room corners - only exterior perimeter
- If not a floor plan or no corners identifiable, return empty arrays for corner_positions

Return ONLY the JSON object, no markdown formatting or explanation.'''


# ============================================================
# ASYNC CLAUDE CLIENT
# ============================================================

class AsyncClaudeClient:
    """Async Claude Vision API client using aiohttp"""

    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = CLAUDE_MODEL

    async def _download_image(self, session: aiohttp.ClientSession, image_url: str) -> tuple[str, str]:
        """Download image and return base64 + media type"""
        async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            content = await response.read()

            # Determine media type from content-type header or URL
            content_type = response.headers.get('content-type', '')
            if 'png' in content_type or image_url.lower().endswith('.png'):
                media_type = 'image/png'
            elif 'jpeg' in content_type or 'jpg' in content_type or image_url.lower().endswith(('.jpg', '.jpeg')):
                media_type = 'image/jpeg'
            else:
                media_type = 'image/png'  # Default

            return base64.b64encode(content).decode('utf-8'), media_type

    async def analyze_page(
        self,
        session: aiohttp.ClientSession,
        image_url: str,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Analyze a single page with Claude Vision"""

        async with semaphore:  # Limit concurrent requests
            start_time = time.time()

            try:
                # Download and encode image
                image_base64, media_type = await self._download_image(session, image_url)

                # Prepare request
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }

                payload = {
                    "model": self.model,
                    "max_tokens": 4000,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {"type": "text", "text": EXTRACTION_PROMPT}
                        ]
                    }]
                }

                # Make API request
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
                ) as response:

                    processing_time = int((time.time() - start_time) * 1000)

                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"API error {response.status}: {error_text[:200]}",
                            "processing_time_ms": processing_time
                        }

                    result = await response.json()

                    # Extract usage stats
                    usage = result.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)

                    # Parse response
                    text = result.get('content', [{}])[0].get('text', '{}')
                    extracted_data = self._parse_json_response(text)

                    if extracted_data is None:
                        return {
                            "success": False,
                            "error": "Failed to parse JSON response",
                            "raw_text": text[:500],
                            "processing_time_ms": processing_time,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens
                        }

                    return {
                        "success": True,
                        "extracted_data": extracted_data,
                        "processing_time_ms": processing_time,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }

            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": "Request timeout",
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Parse JSON from Claude's response, handling markdown code blocks"""

        # Remove markdown code blocks if present
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]

        # Try to find JSON object
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object within text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return None


# ============================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================

async def analyze_pages_async(
    pages: List[Dict],
    job_id: str
) -> List[AnalysisResult]:
    """Analyze multiple pages in parallel using asyncio"""

    client = AsyncClaudeClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)
    results = []

    # Create shared session for connection pooling
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_PAGES)
    async with aiohttp.ClientSession(connector=connector) as session:

        # Create tasks for all pages
        tasks = []
        for page in pages:
            task = analyze_single_page(
                session=session,
                client=client,
                semaphore=semaphore,
                page=page,
                job_id=job_id
            )
            tasks.append(task)

        # Execute all tasks in parallel with progress tracking
        completed = 0
        total = len(tasks)

        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            results.append(result)

            # Log progress
            if completed % 5 == 0 or completed == total:
                print(f"[{job_id[:8]}] Analyzed {completed}/{total} pages", flush=True)

    return results


async def analyze_single_page(
    session: aiohttp.ClientSession,
    client: AsyncClaudeClient,
    semaphore: asyncio.Semaphore,
    page: Dict,
    job_id: str
) -> AnalysisResult:
    """Analyze a single page with retry logic"""

    page_id = page.get('id')
    page_number = page.get('page_number', 0)
    image_url = page.get('image_url')

    if not image_url:
        return AnalysisResult(
            page_id=page_id,
            page_number=page_number,
            success=False,
            error="No image URL"
        )

    # Retry loop
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        result = await client.analyze_page(session, image_url, semaphore)

        if result.get('success'):
            extracted_data = result['extracted_data']

            # Extract backward-compatible fields
            classification = extracted_data.get('classification', {})
            page_type = normalize_page_type(classification.get('page_type'))

            scale = extracted_data.get('scale', {})

            # Update database
            # Note: status must be 'classified' (not 'analyzed') due to DB check constraint
            update_data = {
                'page_type': page_type,
                'page_type_confidence': classification.get('confidence', 0),
                'elevation_name': classification.get('elevation_name'),
                'scale_notation': scale.get('notation'),
                'scale_ratio': scale.get('ratio'),
                'extracted_data': extracted_data,  # Full JSONB data
                'status': 'classified',  # Use 'classified' for DB constraint compatibility
                'processing_time_ms': result.get('processing_time_ms', 0)
            }
            update_page(page_id, update_data)

            # Store corner positions as detections for floor plan pages
            if page_type == 'floor_plan':
                corner_positions = extracted_data.get('corner_positions', {})
                if corner_positions.get('outside_corners') or corner_positions.get('inside_corners'):
                    store_corner_detections(job_id, page_id, extracted_data)

            return AnalysisResult(
                page_id=page_id,
                page_number=page_number,
                success=True,
                processing_time_ms=result.get('processing_time_ms', 0),
                input_tokens=result.get('input_tokens', 0),
                output_tokens=result.get('output_tokens', 0),
                extracted_data=extracted_data
            )

        last_error = result.get('error', 'Unknown error')

        if attempt < MAX_RETRIES:
            # Wait before retry (exponential backoff)
            await asyncio.sleep(2 ** attempt)

    # All retries failed
    update_page(page_id, {
        'status': 'failed',
        'error_message': last_error
    })

    return AnalysisResult(
        page_id=page_id,
        page_number=page_number,
        success=False,
        error=last_error,
        processing_time_ms=result.get('processing_time_ms', 0)
    )


def analyze_job_background(job_id: str) -> None:
    """
    Background task to analyze all pages in a job.

    This is the main entry point for async analysis.
    Runs the async event loop and updates job status.
    """
    print(f"[{job_id[:8]}] Starting intelligent analysis...", flush=True)
    start_time = time.time()

    try:
        # Update job status
        update_job(job_id, {'status': 'analyzing', 'stage': 'intelligent_analysis'})

        # Get pending pages
        pages = get_pages_by_job(job_id, status='pending')
        if not pages:
            # Try getting all pages if none are pending
            pages = get_pages_by_job(job_id)

        if not pages:
            print(f"[{job_id[:8]}] No pages to analyze", flush=True)
            update_job(job_id, {'status': 'failed', 'error_message': 'No pages found'})
            return

        print(f"[{job_id[:8]}] Analyzing {len(pages)} pages in parallel (max {MAX_CONCURRENT_PAGES} concurrent)", flush=True)

        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(analyze_pages_async(pages, job_id))
        finally:
            loop.close()

        # Calculate statistics
        total_time = time.time() - start_time
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)
        estimated_cost = (
            (total_input_tokens / 1000) * INPUT_TOKEN_COST_PER_1K +
            (total_output_tokens / 1000) * OUTPUT_TOKEN_COST_PER_1K
        )

        # Count page types
        page_type_counts = Counter()
        element_totals = {
            'windows': 0,
            'doors': 0,
            'garages': 0,
            'gables': 0,
            'outside_corners': 0,
            'inside_corners': 0
        }

        for result in successful:
            if result.extracted_data:
                classification = result.extracted_data.get('classification', {})
                page_type = classification.get('page_type', 'other')
                page_type_counts[page_type] += 1

                # Sum element counts
                elements = result.extracted_data.get('element_counts', {})
                for key in element_totals:
                    if key in elements:
                        count = elements[key].get('count', 0)
                        if isinstance(count, (int, float)):
                            element_totals[key] += count

        # Update job with results
        # Note: Use 'classified' status for DB constraint compatibility
        update_job(job_id, {
            'status': 'classified',
            'pages_classified': len(successful),
            'elevation_count': page_type_counts.get('elevation', 0),
            'schedule_count': page_type_counts.get('schedule', 0),
            'floor_plan_count': page_type_counts.get('floor_plan', 0),
            'other_count': sum(v for k, v in page_type_counts.items()
                             if k not in ('elevation', 'schedule', 'floor_plan')),
            'results_summary': {
                'total_pages_analyzed': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'total_time_seconds': round(total_time, 2),
                'avg_time_per_page_ms': round((total_time * 1000) / len(pages), 0) if pages else 0,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'estimated_cost_usd': round(estimated_cost, 4),
                'page_type_counts': dict(page_type_counts),
                'element_totals': element_totals
            }
        })

        # Log summary
        print(f"[{job_id[:8]}] Analysis complete:", flush=True)
        print(f"  - Pages: {len(successful)}/{len(pages)} successful", flush=True)
        print(f"  - Time: {total_time:.1f}s ({total_time/len(pages):.2f}s/page avg)", flush=True)
        print(f"  - Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out", flush=True)
        print(f"  - Cost: ${estimated_cost:.4f}", flush=True)
        print(f"  - Page types: {dict(page_type_counts)}", flush=True)

    except Exception as e:
        print(f"[{job_id[:8]}] Analysis failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        update_job(job_id, {
            'status': 'failed',
            'error_message': str(e)
        })


def analyze_job(job_id: str) -> Dict:
    """
    Synchronous wrapper for job analysis.

    Returns a dict with analysis results.
    """
    start_time = time.time()

    try:
        # Get pages
        pages = get_pages_by_job(job_id, status='pending')
        if not pages:
            pages = get_pages_by_job(job_id)

        if not pages:
            return {
                "success": False,
                "error": "No pages found",
                "job_id": job_id
            }

        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(analyze_pages_async(pages, job_id))
        finally:
            loop.close()

        # Calculate statistics
        total_time = time.time() - start_time
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)
        estimated_cost = (
            (total_input_tokens / 1000) * INPUT_TOKEN_COST_PER_1K +
            (total_output_tokens / 1000) * OUTPUT_TOKEN_COST_PER_1K
        )

        # Count page types
        page_type_counts = Counter()
        for result in successful:
            if result.extracted_data:
                classification = result.extracted_data.get('classification', {})
                page_type = classification.get('page_type', 'other')
                page_type_counts[page_type] += 1

        return {
            "success": True,
            "job_id": job_id,
            "total_pages": len(pages),
            "successful_pages": len(successful),
            "failed_pages": len(failed),
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_page_seconds": round(total_time / len(pages), 2) if pages else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "page_type_counts": dict(page_type_counts),
            "results": [asdict(r) for r in results]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "job_id": job_id
        }


# ============================================================
# SINGLE PAGE ANALYSIS (for testing/individual use)
# ============================================================

def analyze_single_page_sync(page_id: str) -> Dict:
    """
    Analyze a single page synchronously.

    Useful for testing or analyzing individual pages.
    """
    from database import get_page

    page = get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    image_url = page.get('image_url')
    if not image_url:
        return {"success": False, "error": "No image URL"}

    # Create a simple sync wrapper
    async def _analyze():
        connector = aiohttp.TCPConnector(limit=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            client = AsyncClaudeClient()
            semaphore = asyncio.Semaphore(1)
            return await client.analyze_page(session, image_url, semaphore)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_analyze())
    finally:
        loop.close()

    if result.get('success'):
        extracted_data = result['extracted_data']

        # Update database
        classification = extracted_data.get('classification', {})
        scale = extracted_data.get('scale', {})

        update_page(page_id, {
            'page_type': normalize_page_type(classification.get('page_type')),
            'page_type_confidence': classification.get('confidence', 0),
            'elevation_name': classification.get('elevation_name'),
            'scale_notation': scale.get('notation'),
            'scale_ratio': scale.get('ratio'),
            'extracted_data': extracted_data,
            'status': 'classified',  # Use 'classified' for DB constraint compatibility
            'processing_time_ms': result.get('processing_time_ms', 0)
        })

    return result


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def estimate_job_cost(page_count: int) -> Dict:
    """
    Estimate the cost of analyzing a job before running.

    Based on typical token usage patterns.
    """
    # Typical token usage per page (based on observations)
    avg_input_tokens = 1500  # Image + prompt
    avg_output_tokens = 800   # JSON response

    total_input = page_count * avg_input_tokens
    total_output = page_count * avg_output_tokens

    cost = (
        (total_input / 1000) * INPUT_TOKEN_COST_PER_1K +
        (total_output / 1000) * OUTPUT_TOKEN_COST_PER_1K
    )

    # Estimate time with parallel processing
    # ~3 seconds per API call, but 10 concurrent
    batches = (page_count + MAX_CONCURRENT_PAGES - 1) // MAX_CONCURRENT_PAGES
    estimated_seconds = batches * 3.5

    return {
        "page_count": page_count,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": round(cost, 4),
        "estimated_time_seconds": estimated_seconds,
        "estimated_time_formatted": f"{int(estimated_seconds // 60)}m {int(estimated_seconds % 60)}s"
    }


def store_corner_detections(
    job_id: str,
    page_id: str,
    extracted_data: Dict
) -> int:
    """
    Store extracted corner positions as detection records.

    Converts corner_positions from Claude Vision analysis into
    extraction_detection_details records that can be displayed
    in the Detection Editor.

    Args:
        job_id: The extraction job ID
        page_id: The page ID where corners were detected
        extracted_data: The full extracted_data from Claude Vision

    Returns:
        Number of corner detections created
    """
    corner_positions = extracted_data.get('corner_positions', {})
    outside_corners = corner_positions.get('outside_corners', [])
    inside_corners = corner_positions.get('inside_corners', [])

    if not outside_corners and not inside_corners:
        return 0

    detections = []

    # Size for corner point markers (small square)
    CORNER_SIZE = 20  # pixels

    # Process outside corners
    for idx, corner in enumerate(outside_corners):
        x = corner.get('x', 0)
        y = corner.get('y', 0)
        confidence = corner.get('confidence', 0.8)
        description = corner.get('description', '')

        detections.append({
            'job_id': job_id,
            'page_id': page_id,
            'class': 'outside_corner',
            'detection_index': idx,
            'pixel_x': x - CORNER_SIZE // 2,  # Center the marker
            'pixel_y': y - CORNER_SIZE // 2,
            'pixel_width': CORNER_SIZE,
            'pixel_height': CORNER_SIZE,
            'confidence': confidence,
            'source': 'intelligent_analysis',
            'status': 'pending',
            'notes': description
        })

    # Process inside corners
    for idx, corner in enumerate(inside_corners):
        x = corner.get('x', 0)
        y = corner.get('y', 0)
        confidence = corner.get('confidence', 0.8)
        description = corner.get('description', '')

        detections.append({
            'job_id': job_id,
            'page_id': page_id,
            'class': 'inside_corner',
            'detection_index': idx,
            'pixel_x': x - CORNER_SIZE // 2,
            'pixel_y': y - CORNER_SIZE // 2,
            'pixel_width': CORNER_SIZE,
            'pixel_height': CORNER_SIZE,
            'confidence': confidence,
            'source': 'intelligent_analysis',
            'status': 'pending',
            'notes': description
        })

    # Batch insert all corner detections
    if detections:
        try:
            batch_create_detections(detections)
            print(f"  Created {len(detections)} corner detections ({len(outside_corners)} outside, {len(inside_corners)} inside)", flush=True)
        except Exception as e:
            print(f"  Warning: Failed to store corner detections: {e}", flush=True)
            return 0

    return len(detections)


def get_extraction_summary(extracted_data: Dict) -> Dict:
    """
    Get a human-readable summary from extracted data.

    Useful for displaying in UI.
    """
    if not extracted_data:
        return {"error": "No data"}

    classification = extracted_data.get('classification', {})
    scale = extracted_data.get('scale', {})
    elements = extracted_data.get('element_counts', {})
    dimensions = extracted_data.get('dimensions', {})
    materials = extracted_data.get('materials', {})
    spatial = extracted_data.get('spatial_context', {})
    quality = extracted_data.get('quality_indicators', {})

    return {
        "page_type": classification.get('page_type'),
        "elevation_name": classification.get('elevation_name'),
        "scale": scale.get('notation'),
        "stories": spatial.get('stories'),
        "roof_style": spatial.get('roof_style'),
        "window_count": elements.get('windows', {}).get('count', 0),
        "door_count": elements.get('doors', {}).get('count', 0),
        "garage_count": elements.get('garages', {}).get('count', 0),
        "wall_heights": [h.get('text') for h in dimensions.get('wall_heights', [])],
        "siding_type": materials.get('siding', {}).get('type'),
        "drawing_quality": quality.get('drawing_clarity'),
        "overall_confidence": quality.get('confidence_overall')
    }
