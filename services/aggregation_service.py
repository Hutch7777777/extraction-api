"""
Aggregation Service

Combines extracted data from multiple pages to calculate derived measurements.
Intelligently merges floor plan corners, elevation heights, schedule counts,
and other data to produce unified job-level measurements with confidence
scores and full source tracking.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

from database import get_pages_by_job, get_job, update_job, supabase_request


# ============================================================
# CONFIGURATION
# ============================================================

# Confidence thresholds
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.5

# Similarity threshold for averaging (20%)
SIMILARITY_THRESHOLD = 0.2


# ============================================================
# DATA STRUCTURES
# ============================================================

class AggregatedData:
    """Container for aggregated job data with source tracking"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.aggregated_at = datetime.utcnow().isoformat()

        # Core aggregated values
        self.corners = {}
        self.heights = {}
        self.calculated = {}
        self.elements = {}
        self.materials = {}
        self.spatial = {}
        self.quality = {
            "data_completeness": 0.0,
            "missing_data": [],
            "warnings": []
        }

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "aggregated_at": self.aggregated_at,
            "corners": self.corners,
            "heights": self.heights,
            "calculated": self.calculated,
            "elements": self.elements,
            "materials": self.materials,
            "spatial": self.spatial,
            "quality": self.quality
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_value_with_confidence(items: List[Dict], key: str) -> Tuple[Optional[Any], float, str]:
    """
    Extract best value from multiple sources based on confidence.

    Returns: (value, confidence, source)
    """
    if not items:
        return None, 0.0, ""

    # Filter out None/zero values and sort by confidence
    valid_items = [
        (item.get(key), item.get('confidence', 0.5), item.get('source', ''))
        for item in items
        if item.get(key) is not None and item.get(key) != 0
    ]

    if not valid_items:
        return None, 0.0, ""

    # Sort by confidence descending
    valid_items.sort(key=lambda x: x[1], reverse=True)

    return valid_items[0]


def average_similar_counts(counts: List[Dict]) -> Tuple[float, float, str]:
    """
    Average counts that are similar (within threshold), otherwise use highest confidence.

    Returns: (count, confidence, source_description)
    """
    if not counts:
        return 0, 0.0, ""

    # Extract values with confidence
    valid = [(c.get('count', 0), c.get('confidence', 0.5), c.get('source', ''))
             for c in counts if c.get('count', 0) > 0]

    if not valid:
        return 0, 0.0, ""

    if len(valid) == 1:
        return valid[0]

    # Check if values are similar (within threshold)
    values = [v[0] for v in valid]
    max_val = max(values)
    min_val = min(values)

    if max_val > 0 and (max_val - min_val) / max_val <= SIMILARITY_THRESHOLD:
        # Values are similar - average them
        avg_count = statistics.mean(values)
        avg_confidence = statistics.mean([v[1] for v in valid])
        sources = [v[2] for v in valid if v[2]]
        return avg_count, avg_confidence, f"average of {len(valid)} sources: {', '.join(sources)}"
    else:
        # Values differ significantly - use highest confidence
        valid.sort(key=lambda x: x[1], reverse=True)
        best = valid[0]
        return best[0], best[1], best[2]


def parse_height_to_feet(height_data: Dict) -> Optional[float]:
    """Convert height data to feet"""
    inches = height_data.get('inches')
    if inches:
        return inches / 12.0

    text = height_data.get('text', '')
    # Try to parse "9'-0"" format
    if "'" in text:
        try:
            parts = text.replace('"', '').split("'")
            feet = float(parts[0].strip())
            inches_part = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
            return feet + inches_part / 12.0
        except (ValueError, IndexError):
            pass

    # Try to parse "9.0'" or "9.0" format
    try:
        clean = text.replace("'", "").replace('"', '').strip()
        if clean:
            return float(clean)
    except ValueError:
        pass

    return None


# ============================================================
# PAGE TYPE EXTRACTORS
# ============================================================

def extract_from_floor_plans(pages: List[Dict]) -> Dict:
    """Extract corner counts and building footprint from floor plans"""

    results = {
        "outside_corners": [],
        "inside_corners": [],
        "building_dimensions": [],
        "garage_info": []
    }

    for page in pages:
        extracted = page.get('extracted_data') or {}
        page_num = page.get('page_number', 0)
        source = f"floor_plan_page_{page_num}"

        # Extract element counts
        elements = extracted.get('element_counts', {})

        # Outside corners
        outside = elements.get('outside_corners', {})
        if outside.get('count', 0) > 0:
            results["outside_corners"].append({
                "count": outside.get('count', 0),
                "confidence": outside.get('confidence', 0.5),
                "source": source,
                "notes": outside.get('notes', '')
            })

        # Inside corners
        inside = elements.get('inside_corners', {})
        if inside.get('count', 0) > 0:
            results["inside_corners"].append({
                "count": inside.get('count', 0),
                "confidence": inside.get('confidence', 0.5),
                "source": source,
                "notes": inside.get('notes', '')
            })

        # Building dimensions
        dimensions = extracted.get('dimensions', {})
        if dimensions.get('building_width_ft') or dimensions.get('building_length_ft'):
            results["building_dimensions"].append({
                "width_ft": dimensions.get('building_width_ft'),
                "length_ft": dimensions.get('building_length_ft'),
                "source": source
            })

        # Garage info
        spatial = extracted.get('spatial_context', {})
        garages = elements.get('garages', {})
        if spatial.get('has_garage') or garages.get('count', 0) > 0:
            results["garage_info"].append({
                "count": garages.get('count', 0),
                "position": spatial.get('garage_position'),
                "widths": garages.get('widths_seen', []),
                "source": source
            })

    return results


def extract_from_elevations(pages: List[Dict]) -> Dict:
    """Extract heights, element counts, and materials from elevations"""

    results = {
        "wall_heights": [],
        "story_heights": {},
        "windows_by_elevation": [],
        "doors_by_elevation": [],
        "gables": [],
        "siding": [],
        "stories": [],
        "total_windows": 0,
        "total_doors": 0
    }

    for page in pages:
        extracted = page.get('extracted_data') or {}
        page_num = page.get('page_number', 0)
        elevation_name = page.get('elevation_name') or extracted.get('classification', {}).get('elevation_name')
        source = f"elevation_page_{page_num}"
        if elevation_name:
            source = f"{elevation_name}_elevation_page_{page_num}"

        # Wall heights
        dimensions = extracted.get('dimensions', {})
        for height in dimensions.get('wall_heights', []):
            height_ft = parse_height_to_feet(height)
            if height_ft:
                label = height.get('label', 'unknown')
                results["wall_heights"].append({
                    "label": label,
                    "height_ft": height_ft,
                    "confidence": height.get('confidence', 0.7),
                    "source": source
                })

                # Track unique story heights
                normalized_label = label.upper().replace(' ', '_')
                if normalized_label not in results["story_heights"]:
                    results["story_heights"][normalized_label] = {
                        "label": label,
                        "height_ft": height_ft,
                        "source": source
                    }

        # Element counts
        elements = extracted.get('element_counts', {})

        # Windows
        windows = elements.get('windows', {})
        window_count = windows.get('count', 0)
        if window_count > 0:
            results["windows_by_elevation"].append({
                "elevation": elevation_name,
                "count": window_count,
                "confidence": windows.get('confidence', 0.5),
                "types": windows.get('types_seen', []),
                "source": source
            })
            results["total_windows"] += window_count

        # Doors
        doors = elements.get('doors', {})
        door_count = doors.get('count', 0)
        if door_count > 0:
            results["doors_by_elevation"].append({
                "elevation": elevation_name,
                "count": door_count,
                "confidence": doors.get('confidence', 0.5),
                "types": doors.get('types_seen', []),
                "source": source
            })
            results["total_doors"] += door_count

        # Gables
        gables = elements.get('gables', {})
        if gables.get('count', 0) > 0:
            results["gables"].append({
                "count": gables.get('count', 0),
                "confidence": gables.get('confidence', 0.5),
                "source": source
            })

        # Materials
        materials = extracted.get('materials', {})
        siding = materials.get('siding', {})
        if siding.get('type'):
            results["siding"].append({
                "type": siding.get('type'),
                "profile": siding.get('profile'),
                "exposure": siding.get('exposure_inches'),
                "source": source
            })

        # Stories
        spatial = extracted.get('spatial_context', {})
        if spatial.get('stories'):
            results["stories"].append({
                "count": spatial.get('stories'),
                "source": source
            })

    return results


def extract_from_schedules(pages: List[Dict]) -> Dict:
    """Extract window/door counts and dimensions from schedules"""

    results = {
        "window_count": 0,
        "door_count": 0,
        "window_details": [],
        "door_details": [],
        "sources": []
    }

    for page in pages:
        extracted = page.get('extracted_data') or {}
        page_num = page.get('page_number', 0)
        source = f"schedule_page_{page_num}"

        classification = extracted.get('classification', {})
        schedule_type = classification.get('schedule_type')

        # Check for schedule data in extraction_data field (from previous processing)
        existing_data = page.get('extraction_data') or {}

        if existing_data.get('windows'):
            results["window_count"] = len(existing_data['windows'])
            results["window_details"] = existing_data['windows']
            results["sources"].append(source)

        if existing_data.get('doors'):
            results["door_count"] = len(existing_data['doors'])
            results["door_details"] = existing_data['doors']
            results["sources"].append(source)

        # Also check element_counts from intelligent analysis
        elements = extracted.get('element_counts', {})
        if schedule_type in ['window', 'window_and_door']:
            window_data = elements.get('windows', {})
            if window_data.get('count', 0) > results["window_count"]:
                results["window_count"] = window_data.get('count', 0)

        if schedule_type in ['door', 'window_and_door']:
            door_data = elements.get('doors', {})
            if door_data.get('count', 0) > results["door_count"]:
                results["door_count"] = door_data.get('count', 0)

    return results


def extract_from_roof_plans(pages: List[Dict]) -> Dict:
    """Extract roof-related data from roof plans"""

    results = {
        "gables": [],
        "roof_style": [],
        "roof_pitch": []
    }

    for page in pages:
        extracted = page.get('extracted_data') or {}
        page_num = page.get('page_number', 0)
        source = f"roof_plan_page_{page_num}"

        elements = extracted.get('element_counts', {})
        spatial = extracted.get('spatial_context', {})

        # Gables
        gables = elements.get('gables', {})
        if gables.get('count', 0) > 0:
            results["gables"].append({
                "count": gables.get('count', 0),
                "confidence": gables.get('confidence', 0.5),
                "source": source
            })

        # Roof style
        if spatial.get('roof_style'):
            results["roof_style"].append({
                "style": spatial.get('roof_style'),
                "source": source
            })

        # Roof pitch
        if spatial.get('roof_pitch'):
            results["roof_pitch"].append({
                "pitch": spatial.get('roof_pitch'),
                "source": source
            })

    return results


# ============================================================
# MAIN AGGREGATION LOGIC
# ============================================================

def aggregate_job(job_id: str) -> Dict:
    """
    Main aggregation function.

    Fetches all pages, extracts data by page type, and combines
    into unified measurements with confidence scores.
    """
    print(f"[{job_id[:8]}] Starting aggregation...", flush=True)

    # Verify job exists
    job = get_job(job_id)
    if not job:
        return {"success": False, "error": "Job not found"}

    # Get all pages with extracted_data
    pages = get_pages_by_job(job_id)
    if not pages:
        return {"success": False, "error": "No pages found"}

    # Filter pages by type
    floor_plans = [p for p in pages if p.get('page_type') == 'floor_plan']
    elevations = [p for p in pages if p.get('page_type') == 'elevation']
    schedules = [p for p in pages if p.get('page_type') == 'schedule']
    roof_plans = [p for p in pages if p.get('page_type') == 'roof_plan']

    print(f"[{job_id[:8]}] Found: {len(floor_plans)} floor plans, {len(elevations)} elevations, "
          f"{len(schedules)} schedules, {len(roof_plans)} roof plans", flush=True)

    # Extract data from each page type
    floor_plan_data = extract_from_floor_plans(floor_plans)
    elevation_data = extract_from_elevations(elevations)
    schedule_data = extract_from_schedules(schedules)
    roof_plan_data = extract_from_roof_plans(roof_plans)

    # Build aggregated result
    result = AggregatedData(job_id)
    warnings = []
    missing = []

    # ========== CORNERS ==========
    outside_count, outside_conf, outside_source = average_similar_counts(
        floor_plan_data["outside_corners"]
    )
    inside_count, inside_conf, inside_source = average_similar_counts(
        floor_plan_data["inside_corners"]
    )

    if outside_count > 0:
        result.corners["outside_count"] = round(outside_count)
        result.corners["outside_count_confidence"] = round(outside_conf, 2)
        result.corners["outside_count_source"] = outside_source
    else:
        missing.append("outside_corners")

    if inside_count > 0:
        result.corners["inside_count"] = round(inside_count)
        result.corners["inside_count_confidence"] = round(inside_conf, 2)
        result.corners["inside_count_source"] = inside_source
    else:
        missing.append("inside_corners")

    # Check for corner count variance
    if len(floor_plan_data["outside_corners"]) > 1:
        counts = [c["count"] for c in floor_plan_data["outside_corners"]]
        if max(counts) - min(counts) > 2:
            warnings.append(f"Outside corner counts vary between floor plans: {counts}")

    # ========== HEIGHTS ==========
    story_heights = list(elevation_data["story_heights"].values())
    if story_heights:
        result.heights["story_heights"] = story_heights
        total_height = sum(h["height_ft"] for h in story_heights)
        result.heights["total_wall_height_ft"] = round(total_height, 2)
    else:
        missing.append("wall_heights")
        result.heights["total_wall_height_ft"] = 0

    # Stories count
    if elevation_data["stories"]:
        # Use most common story count
        story_counts = [s["count"] for s in elevation_data["stories"]]
        result.heights["stories"] = max(set(story_counts), key=story_counts.count)
    else:
        result.heights["stories"] = None
        missing.append("stories")

    # ========== CALCULATED MEASUREMENTS ==========
    total_height_ft = result.heights.get("total_wall_height_ft", 0)

    # Corner linear feet
    outside_corner_count = result.corners.get("outside_count", 0)
    inside_corner_count = result.corners.get("inside_count", 0)

    if outside_corner_count and total_height_ft:
        result.calculated["outside_corner_lf"] = round(outside_corner_count * total_height_ft, 2)
    else:
        result.calculated["outside_corner_lf"] = 0
        if outside_corner_count and not total_height_ft:
            warnings.append("Cannot calculate outside corner LF - missing wall heights")

    if inside_corner_count and total_height_ft:
        result.calculated["inside_corner_lf"] = round(inside_corner_count * total_height_ft, 2)
    else:
        result.calculated["inside_corner_lf"] = 0

    result.calculated["total_corner_lf"] = (
        result.calculated.get("outside_corner_lf", 0) +
        result.calculated.get("inside_corner_lf", 0)
    )

    # ========== ELEMENTS ==========
    # Windows - prefer schedule count, fall back to elevation sum
    window_from_schedule = schedule_data.get("window_count", 0)
    window_from_elevations = elevation_data.get("total_windows", 0)

    result.elements["windows"] = {
        "count_from_schedule": window_from_schedule,
        "count_from_elevations": window_from_elevations,
        "recommended_count": window_from_schedule if window_from_schedule > 0 else window_from_elevations,
        "source": "schedule" if window_from_schedule > 0 else "elevations"
    }

    # Doors - prefer schedule count
    door_from_schedule = schedule_data.get("door_count", 0)
    door_from_elevations = elevation_data.get("total_doors", 0)

    result.elements["doors"] = {
        "count_from_schedule": door_from_schedule,
        "count_from_elevations": door_from_elevations,
        "recommended_count": door_from_schedule if door_from_schedule > 0 else door_from_elevations,
        "source": "schedule" if door_from_schedule > 0 else "elevations"
    }

    # Gables - combine from elevations and roof plans
    all_gables = elevation_data.get("gables", []) + roof_plan_data.get("gables", [])
    if all_gables:
        gable_count, gable_conf, gable_source = average_similar_counts(all_gables)
        result.elements["gables"] = {
            "count": round(gable_count),
            "confidence": round(gable_conf, 2),
            "source": gable_source
        }

    # Garages
    if floor_plan_data["garage_info"]:
        # Use first floor plan with garage info
        garage = floor_plan_data["garage_info"][0]
        result.elements["garages"] = {
            "count": garage.get("count", 0),
            "position": garage.get("position"),
            "widths": garage.get("widths", []),
            "source": garage.get("source", "")
        }

    # ========== MATERIALS ==========
    if elevation_data["siding"]:
        # Use first siding info found
        siding = elevation_data["siding"][0]
        result.materials = {
            "siding_type": siding.get("type"),
            "siding_profile": siding.get("profile"),
            "siding_exposure_inches": siding.get("exposure"),
            "source": siding.get("source", "")
        }
    else:
        missing.append("siding_type")

    # ========== SPATIAL ==========
    result.spatial["stories"] = result.heights.get("stories")

    # Roof style - prefer roof plan, fall back to elevations
    if roof_plan_data["roof_style"]:
        result.spatial["roof_style"] = roof_plan_data["roof_style"][0]["style"]
    elif elevation_data.get("roof_style"):
        result.spatial["roof_style"] = elevation_data["roof_style"][0]["style"]

    # Roof pitch
    if roof_plan_data["roof_pitch"]:
        result.spatial["roof_pitch"] = roof_plan_data["roof_pitch"][0]["pitch"]

    # Foundation, porch (from any page with spatial_context)
    for page in pages:
        extracted = page.get('extracted_data') or {}
        spatial = extracted.get('spatial_context', {})

        if spatial.get('foundation_type') and 'foundation_type' not in result.spatial:
            result.spatial["foundation_type"] = spatial['foundation_type']

        if spatial.get('has_porch') and 'has_porch' not in result.spatial:
            result.spatial["has_porch"] = spatial['has_porch']
            result.spatial["porch_type"] = spatial.get('porch_type')

    # ========== QUALITY METRICS ==========
    # Calculate completeness
    total_fields = 10
    filled_fields = sum([
        bool(result.corners.get("outside_count")),
        bool(result.corners.get("inside_count")),
        bool(result.heights.get("total_wall_height_ft")),
        bool(result.elements.get("windows", {}).get("recommended_count")),
        bool(result.elements.get("doors", {}).get("recommended_count")),
        bool(result.elements.get("gables")),
        bool(result.materials.get("siding_type")),
        bool(result.spatial.get("stories")),
        bool(result.spatial.get("roof_style")),
        bool(result.spatial.get("foundation_type"))
    ])

    result.quality["data_completeness"] = round(filled_fields / total_fields, 2)
    result.quality["missing_data"] = missing
    result.quality["warnings"] = warnings

    # ========== STORE RESULTS ==========
    aggregated_dict = result.to_dict()

    # Update extraction_job_totals if exists (without aggregated_data column which may not exist)
    totals = supabase_request('GET', 'extraction_job_totals',
                              filters={'job_id': f'eq.{job_id}'})

    # Core totals update (columns that definitely exist)
    totals_update = {
        'outside_corners_count': result.corners.get("outside_count"),
        'inside_corners_count': result.corners.get("inside_count"),
        'outside_corners_lf': result.calculated.get("outside_corner_lf"),
        'inside_corners_lf': result.calculated.get("inside_corner_lf"),
        'corner_source': result.corners.get("outside_count_source"),
        'updated_at': datetime.utcnow().isoformat()
    }

    if totals:
        # Update existing totals record
        supabase_request('PATCH', 'extraction_job_totals', totals_update,
                         filters={'job_id': f'eq.{job_id}'})
        print(f"[{job_id[:8]}] Updated extraction_job_totals", flush=True)
    else:
        # Create new totals record
        totals_update['job_id'] = job_id
        del totals_update['updated_at']  # Not needed for insert
        supabase_request('POST', 'extraction_job_totals', totals_update)
        print(f"[{job_id[:8]}] Created extraction_job_totals record", flush=True)

    # Store full aggregated data in job's results_summary (always works)
    existing_summary = job.get('results_summary') or {}
    existing_summary['aggregation'] = aggregated_dict
    update_job(job_id, {'results_summary': existing_summary})

    print(f"[{job_id[:8]}] Aggregation complete. Completeness: {result.quality['data_completeness']:.0%}", flush=True)

    return {
        "success": True,
        "job_id": job_id,
        **aggregated_dict
    }


def get_aggregated_summary(job_id: str) -> Optional[Dict]:
    """
    Retrieve previously aggregated data for a job.
    """
    # Check extraction_job_totals first
    totals = supabase_request('GET', 'extraction_job_totals',
                              filters={'job_id': f'eq.{job_id}'})

    if totals and totals[0].get('aggregated_data'):
        return totals[0]['aggregated_data']

    # Fall back to job results_summary
    job = get_job(job_id)
    if job and job.get('results_summary', {}).get('aggregation'):
        return job['results_summary']['aggregation']

    return None


def recalculate_corner_lf(job_id: str, wall_height_ft: float) -> Dict:
    """
    Recalculate corner linear feet with a new wall height.

    Useful when user manually corrects wall height.
    """
    # Get existing aggregated data
    existing = get_aggregated_summary(job_id)
    if not existing:
        return {"success": False, "error": "No aggregated data found"}

    outside_count = existing.get('corners', {}).get('outside_count', 0)
    inside_count = existing.get('corners', {}).get('inside_count', 0)

    outside_lf = round(outside_count * wall_height_ft, 2) if outside_count else 0
    inside_lf = round(inside_count * wall_height_ft, 2) if inside_count else 0
    total_lf = outside_lf + inside_lf

    # Update database
    supabase_request('PATCH', 'extraction_job_totals', {
        'outside_corners_lf': outside_lf,
        'inside_corners_lf': inside_lf,
        'updated_at': datetime.utcnow().isoformat()
    }, filters={'job_id': f'eq.{job_id}'})

    return {
        "success": True,
        "job_id": job_id,
        "wall_height_ft": wall_height_ft,
        "outside_corners": outside_count,
        "inside_corners": inside_count,
        "outside_corner_lf": outside_lf,
        "inside_corner_lf": inside_lf,
        "total_corner_lf": total_lf
    }
