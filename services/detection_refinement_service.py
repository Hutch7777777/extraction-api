"""
Second-pass vision refinement for draft detections.

Roboflow produces first-pass candidates. This service asks a vision model
for structured edit suggestions, validates them deterministically, and can
optionally apply the safe subset to extraction_detections_draft.
"""

import json
import uuid
import base64
from io import BytesIO

import requests
from PIL import Image, ImageDraw

from config import config
from database import supabase_request
from database.repositories.page_repository import get_page, get_pages_by_job
from geometry.area import compute_detection_area_sf, compute_detection_perimeter_lf
from utils.detection_classes import normalize_detection_class
from utils.scale import get_safe_scale_ratio


ACTION_TYPES = {
    'add',
    'keep',
    'resize',
    'reclassify',
    'delete',
    'convert_to_polygon',
    'merge',
    'flag_review',
}

POLYGON_CLASSES = {'exterior_wall', 'building', 'roof', 'gable'}
FACADE_OVERLAP_CLASSES = {'exterior_wall', 'building', 'gable'}
ALLOWED_CLASSES = {
    'window',
    'door',
    'garage',
    'building',
    'exterior_wall',
    'roof',
    'gable',
}


REFINEMENT_PROMPT_VERSION = "detection-refinement-v2"


REFINEMENT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "actions": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": sorted(ACTION_TYPES),
                    },
                    "detection_id": {"type": ["string", "null"]},
                    "detection_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20,
                    },
                    "target_class": {"type": ["string", "null"]},
                    "new_class": {"type": ["string", "null"]},
                    "new_box": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                    "polygon_points": {
                        "type": "array",
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "type",
                    "detection_id",
                    "detection_ids",
                    "target_class",
                    "new_class",
                    "new_box",
                    "polygon_points",
                    "reason",
                    "confidence",
                ],
            },
        },
        "page_observations": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string"},
        },
        "model_confidence": {"type": "number"},
    },
    "required": ["actions", "page_observations", "model_confidence"],
}


def refine_page_detections(page_id, apply=False, requested_classes=None, detection_ids=None, proposed_actions=None):
    """
    Run a second-pass vision refinement for one page.

    Args:
        page_id: extraction_pages.id
        apply: when False, return validated actions only; when True, mutate draft rows
        requested_classes: optional class allow-list for candidate actions
        detection_ids: optional detection id allow-list
    """
    page = get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}, 404

    detections = _get_active_draft_detections(page_id)
    if not detections:
        return {"success": False, "error": "No active draft detections found for page"}, 404

    result, status_code = preview_refinement_for_detections(
        page=page,
        detections=detections,
        requested_classes=requested_classes,
        detection_ids=detection_ids,
        proposed_actions=proposed_actions,
    )
    if status_code != 200:
        return result, status_code

    if apply:
        scale_ratio = get_safe_scale_ratio(page.get('scale_ratio'), context=f"refinement page {page_id}")
        apply_summary = _apply_validated_actions(
            result.get('actions', []),
            detections,
            page_id,
            page.get('job_id'),
            scale_ratio,
        )
        apply_summary['facade_overlap_cleanup'] = cleanup_facade_overlaps(page_id, apply=True)
        result['applied'] = True
        result['apply_summary'] = apply_summary

    return result, 200


def preview_refinement_for_detections(page, detections, requested_classes=None, detection_ids=None, proposed_actions=None):
    """
    Run the model/validator against a provided detection collection without
    assuming those detections already live in the human-editable draft layer.

    This is used both by the editor button (draft detections) and the automatic
    pre-editor job refinement pass (raw Roboflow detections).
    """
    page_id = page.get('id')
    target_classes = _normalize_requested_classes(requested_classes)
    detection_id_set = set(detection_ids or [])
    candidate_detections = [
        d for d in detections
        if (not target_classes or normalize_detection_class(d.get('class')) in target_classes)
        and (not detection_id_set or d.get('id') in detection_id_set)
    ]
    if not candidate_detections:
        return {"success": False, "error": "No detections matched the refinement filters"}, 400

    context_pages = _get_refinement_context_pages(page)

    if not config.OPENAI_API_KEY and proposed_actions is None:
        return {
            "success": False,
            "setup_required": True,
            "error": "OPENAI_API_KEY is not configured",
            "required_env": ["OPENAI_API_KEY"],
            "optional_env": [
                "OPENAI_REFINEMENT_MODEL",
                "OPENAI_REFINEMENT_MAX_OUTPUT_TOKENS",
                "REFINEMENT_CONTEXT_PAGE_TYPES",
                "REFINEMENT_MAX_CONTEXT_PAGES",
            ],
        }, 400

    image_url = page.get('original_image_url') or page.get('image_url')
    if not image_url:
        return {"success": False, "error": "Page has no image_url"}, 400

    width, height = _get_page_dimensions(page, image_url)
    scale_ratio = get_safe_scale_ratio(page.get('scale_ratio'), context=f"refinement page {page_id}")

    if proposed_actions is None:
        model_result = _request_openai_refinement(
            page=page,
            image_url=image_url,
            detections=detections,
            candidate_detections=candidate_detections,
            context_pages=context_pages,
            width=width,
            height=height,
        )
    else:
        model_result = {
            "actions": proposed_actions,
            "page_observations": [],
            "model_confidence": None,
        }
    if model_result.get('error'):
        return {"success": False, **model_result}, 502

    validated = _validate_actions(
        model_result.get('actions', []),
        detections=detections,
        candidate_detections=candidate_detections,
        page_width=width,
        page_height=height,
        scale_ratio=scale_ratio,
    )

    return {
        "success": True,
        "page_id": page_id,
        "applied": False,
        "model": config.OPENAI_REFINEMENT_MODEL,
        "prompt_version": REFINEMENT_PROMPT_VERSION,
        "image_size": {"width": width, "height": height},
        "input_detection_count": len(detections),
        "candidate_detection_count": len(candidate_detections),
        "context_pages": [_compact_page_context(p) for p in context_pages],
        "wall_planes": _compact_wall_plane_context(detections),
        "actions": validated,
        "apply_summary": None,
        "page_observations": model_result.get('page_observations', []),
        "model_confidence": model_result.get('model_confidence'),
    }, 200


def cleanup_facade_overlaps(page_id, apply=False):
    """
    Retire clearly redundant facade detections that already overlap in draft data.

    This is intentionally conservative: it auto-deletes stale model-created boxes
    when a better polygon covers the same facade, and only flags ambiguous pairs.
    """
    page = get_page(page_id)
    scale_ratio = get_safe_scale_ratio((page or {}).get('scale_ratio'), context=f"overlap cleanup page {page_id}")
    detections = _get_active_draft_detections(page_id)
    cleanup_plan = _plan_facade_overlap_cleanup(detections, scale_ratio)
    deleted = []
    trimmed = []
    flagged = []
    errors = []

    if apply:
        for item in cleanup_plan:
            try:
                if item['action'] == 'delete':
                    det = item['detection']
                    note = _append_material_note(
                        det.get('material_notes'),
                        f"AI overlap cleanup: {item['reason']}"
                    )
                    result = supabase_request('PATCH', 'extraction_detections_draft', {
                        'is_deleted': True,
                        'status': 'deleted',
                        'material_notes': note,
                    }, {'id': f"eq.{det.get('id')}"})
                    if not result:
                        raise ValueError(f"Supabase update failed for {det.get('id')}")
                    deleted.append({
                        'id': det.get('id'),
                        'detection_index': det.get('detection_index'),
                        'reason': item['reason'],
                    })
                elif item['action'] == 'trim':
                    det = item['detection']
                    updates = dict(item.get('updates') or {})
                    note = _append_material_note(
                        det.get('material_notes'),
                        f"AI overlap cleanup: {item['reason']}"
                    )
                    updates.update({
                        'is_deleted': False,
                        'status': 'auto',
                        'markup_type': 'polygon',
                        'material_notes': note,
                    })
                    result = supabase_request('PATCH', 'extraction_detections_draft', updates, {'id': f"eq.{det.get('id')}"})
                    if not result:
                        raise ValueError(f"Supabase update failed for {det.get('id')}")
                    trimmed.append({
                        'id': det.get('id'),
                        'detection_index': det.get('detection_index'),
                        'reason': item['reason'],
                    })
                elif item['action'] == 'flag_review':
                    for det in item['detections']:
                        note = _append_material_note(
                            det.get('material_notes'),
                            f"AI overlap review: {item['reason']}"
                        )
                        supabase_request('PATCH', 'extraction_detections_draft', {
                            'material_notes': note,
                        }, {'id': f"eq.{det.get('id')}"})
                        flagged.append({
                            'id': det.get('id'),
                            'detection_index': det.get('detection_index'),
                            'reason': item['reason'],
                        })
            except Exception as exc:
                errors.append({
                    'id': (item.get('detection') or {}).get('id'),
                    'action': item.get('action'),
                    'error': str(exc),
                })

    return {
        'applied': bool(apply),
        'planned': len(cleanup_plan),
        'delete_count': len([item for item in cleanup_plan if item['action'] == 'delete']),
        'trim_count': len([item for item in cleanup_plan if item['action'] == 'trim']),
        'flag_review_count': len([item for item in cleanup_plan if item['action'] == 'flag_review']),
        'deleted': deleted,
        'trimmed': trimmed,
        'flagged': flagged,
        'errors': errors,
    }


def _get_active_draft_detections(page_id):
    return supabase_request('GET', 'extraction_detections_draft', filters={
        'page_id': f'eq.{page_id}',
        'is_deleted': 'eq.false',
        'order': 'detection_index.asc',
    }) or []


def _get_refinement_context_pages(page):
    job_id = page.get('job_id')
    if not job_id or not config.REFINEMENT_CONTEXT_PAGE_TYPES or config.REFINEMENT_MAX_CONTEXT_PAGES <= 0:
        return []

    allowed_types = set(config.REFINEMENT_CONTEXT_PAGE_TYPES)
    pages = get_pages_by_job(job_id, status=None)
    candidates = []
    for candidate in pages:
        if candidate.get('id') == page.get('id'):
            continue
        page_type = (candidate.get('page_type') or '').strip().lower()
        if page_type not in allowed_types:
            continue
        if not (candidate.get('image_url') or candidate.get('original_image_url')):
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda p: (
        _context_page_rank(p),
        abs(int(p.get('page_number') or 0) - int(page.get('page_number') or 0)),
        int(p.get('page_number') or 0),
    ))
    return candidates[:config.REFINEMENT_MAX_CONTEXT_PAGES]


def _context_page_rank(page):
    page_type = (page.get('page_type') or '').strip().lower()
    rank = {
        'roof_plan': 0,
        'floor_plan': 1,
        'site_plan': 2,
        'section': 3,
    }
    return rank.get(page_type, 99)


def _compact_page_context(page):
    return {
        'page_id': page.get('id'),
        'page_number': page.get('page_number'),
        'page_type': page.get('page_type'),
        'elevation_name': page.get('elevation_name'),
        'status': page.get('status'),
        'image_url': page.get('image_url'),
        'floor_plan_data': _compact_json(page.get('floor_plan_data'), max_chars=2500),
        'roof_plan_data': _compact_json(page.get('roof_plan_data'), max_chars=2500),
        'ocr_data': _compact_json(page.get('ocr_data'), max_chars=1500),
    }


def _get_elevation_ocr_context(page_id):
    rows = supabase_request('GET', 'extraction_ocr_data', filters={
        'page_id': f'eq.{page_id}',
        'limit': '1',
    }) or []
    return rows[0] if rows else None


def _compact_json(value, max_chars=2000):
    if value in (None, "", [], {}):
        return None
    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "summary": text[:max_chars],
    }


def _normalize_requested_classes(classes):
    if not classes:
        return set()
    return {
        normalize_detection_class(cls)
        for cls in classes
        if normalize_detection_class(cls)
    }


def _get_page_dimensions(page, image_url):
    width = page.get('original_width') or page.get('width')
    height = page.get('original_height') or page.get('height')
    if width and height:
        return int(width), int(height)

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img.size


def _build_annotated_target_image_data_url(image_url, detections, candidate_detections):
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGBA')
    except Exception:
        return None

    candidate_ids = {d.get('id') for d in candidate_detections}
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    context_detections = [
        det for det in detections
        if det.get('id') not in candidate_ids
        and normalize_detection_class(det.get('class')) in FACADE_OVERLAP_CLASSES
    ]
    for det in context_detections:
        _draw_detection_overlay(draw, det, outline=(16, 185, 129, 120), fill=(16, 185, 129, 26), width=3)

    for idx, det in enumerate(candidate_detections, start=1):
        color = (37, 99, 235, 245) if idx % 2 else (219, 39, 119, 245)
        fill = (37, 99, 235, 46) if idx % 2 else (219, 39, 119, 46)
        _draw_detection_overlay(draw, det, outline=color, fill=fill, width=8)
        center = (_to_float(det.get('pixel_x'), 0), _to_float(det.get('pixel_y'), 0))
        _draw_candidate_badge(draw, center, str(idx), color)

    annotated = Image.alpha_composite(image, overlay).convert('RGB')
    buffer = BytesIO()
    annotated.save(buffer, format='JPEG', quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f"data:image/jpeg;base64,{encoded}"


def _draw_detection_overlay(draw, det, outline, fill, width):
    points = _points_from_detection(det)
    if len(points) >= 3:
        xy = [(p['x'], p['y']) for p in points]
        draw.polygon(xy, fill=fill)
        draw.line(xy + [xy[0]], fill=outline, width=width, joint='curve')
        return

    box_points = _points_from_box_values(
        det.get('pixel_x'),
        det.get('pixel_y'),
        det.get('pixel_width'),
        det.get('pixel_height'),
    )
    if len(box_points) >= 4:
        left, top, right, bottom = _points_bbox(box_points)
        draw.rectangle((left, top, right, bottom), outline=outline, width=width, fill=fill)


def _draw_candidate_badge(draw, center, label, color):
    x, y = center
    radius = 20
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bbox, fill=color, outline=(255, 255, 255, 255), width=3)
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text(
        (x - text_width / 2, y - text_height / 2 - 1),
        label,
        fill=(255, 255, 255, 255),
    )


def _request_openai_refinement(page, image_url, detections, candidate_detections, context_pages, width, height):
    prompt = _build_refinement_prompt(page, detections, candidate_detections, context_pages, width, height)
    annotated_image_url = _build_annotated_target_image_data_url(
        image_url,
        detections=detections,
        candidate_detections=candidate_detections,
    )
    user_content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": "TARGET ELEVATION PAGE IMAGE. Return coordinates only in this image's pixel coordinate system."},
        {"type": "input_image", "image_url": image_url},
    ]
    if annotated_image_url:
        user_content.extend([
            {
                "type": "input_text",
                "text": (
                    "ANNOTATED TARGET ELEVATION IMAGE. Bright blue/pink markups are the candidate detections you may alter. "
                    "Muted outlines are other current siding/gable context detections. Use this overlay to identify the selected markup; "
                    "still return coordinates in the original target elevation pixel system."
                ),
            },
            {"type": "input_image", "image_url": annotated_image_url},
        ])
    for context_page in context_pages:
        context_url = context_page.get('image_url') or context_page.get('original_image_url')
        if not context_url:
            continue
        user_content.extend([
            {
                "type": "input_text",
                "text": (
                    f"REFERENCE CONTEXT PAGE {context_page.get('page_number')} "
                    f"({context_page.get('page_type')}). Use for footprint/roof/depth reasoning only; "
                    "do not return coordinates from this page."
                ),
            },
            {"type": "input_image", "image_url": context_url},
        ])

    payload = {
        "model": config.OPENAI_REFINEMENT_MODEL,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a construction plan markup QA assistant. "
                            "You inspect architectural elevation screenshots and propose precise, conservative "
                            "edits to existing detection markups. Return only schema-valid JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "detection_refinement_actions",
                "schema": REFINEMENT_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": config.OPENAI_REFINEMENT_MAX_OUTPUT_TOKENS,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
    except requests.RequestException as exc:
        return {
            "error": "OpenAI refinement request failed",
            "detail": str(exc),
        }
    if response.status_code >= 400:
        return {
            "error": "OpenAI refinement request failed",
            "status_code": response.status_code,
            "detail": _safe_error_text(response.text),
        }

    data = response.json()
    text = _extract_response_text(data)
    if not text:
        return {"error": "OpenAI response did not include output text"}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "error": "OpenAI response was not valid JSON",
            "detail": str(exc),
            "raw": text[:1000],
        }

    return {
        "actions": parsed.get('actions', []),
        "page_observations": parsed.get('page_observations', []),
        "model_confidence": parsed.get('model_confidence'),
    }


def _build_refinement_prompt(page, detections, candidate_detections, context_pages, width, height):
    page_info = {
        "page_id": page.get('id'),
        "page_number": page.get('page_number'),
        "page_type": page.get('page_type'),
        "elevation_name": page.get('elevation_name'),
        "image_width": width,
        "image_height": height,
    }
    compact_detections = [_compact_detection(d) for d in detections]
    candidate_ids = [d.get('id') for d in candidate_detections]
    compact_context_pages = [_compact_page_context(p) for p in context_pages]
    wall_plane_context = _compact_wall_plane_context(detections)
    target_extracted_context = {
        "page_ocr_data": _compact_json(page.get('ocr_data'), max_chars=1800),
        "elevation_ocr_data": _compact_json(_get_elevation_ocr_context(page.get('id')), max_chars=3000),
    }

    return (
        "Refine the existing markups for this architectural elevation page.\n\n"
        "Coordinates are image pixels. Existing boxes use center coordinates: "
        "pixel_x, pixel_y, pixel_width, pixel_height. If you return new_box, use "
        "the same center-coordinate convention. polygon_points must be absolute "
        "image pixels in clockwise or counterclockwise order. Only return coordinates "
        "for the TARGET ELEVATION PAGE image, never for a reference context page.\n\n"
        "Cross-sheet context:\n"
        "- Reference roof/floor plan pages may be attached after the target elevation image.\n"
        "- Use roof plans to infer projections, bump-outs, return walls, ridges, valleys, hips, eaves, and depth changes.\n"
        "- Use floor plans to infer footprint edges and whether a visible siding region is a front plane, side plane, or recessed return.\n"
        "- Do not directly overlay reference-page coordinates onto the elevation; use them only for construction reasoning.\n\n"
        "Inferred wall-plane context:\n"
        "- view_group separates different elevation drawings on the same sheet.\n"
        "- plane_type is a heuristic hint, not a command. Use the image if the hint is wrong.\n"
        "- Duplicate overlaps are only unsafe within the same physical wall plane; return_plane and projecting_gable may be separate depth planes.\n\n"
        "Primary goals:\n"
        "- Convert gable rectangles into triangular polygon_points when the triangle is visually clear.\n"
        "- Convert or merge exterior_wall rectangles into more accurate facade/siding polygons when clear.\n"
        "- Add a missing exterior_wall or gable polygon only when the full visible surface boundary is clear.\n"
        "- Expand partial exterior_wall slivers into the full visible wall plane when the roof rake/eave, base line, outside corner, and projection boundary are clear.\n"
        "- Separate obvious return walls, side planes, recessed walls, and projecting gables from the main facade instead of merging across perspective/depth changes.\n"
        "- Do not create duplicate overlaps inside the same wall plane. Adjacent same-plane polygons should meet at edges or stay separate.\n"
        "- Count visible takeoff surfaces only. Use roof/floor plans to choose boundaries, but do not infer or count hidden wall area behind another visible plane.\n"
        "- A different depth plane, such as a return wall beside a projecting gable, may need its own polygon, but its counted polygon must still be visible in the target elevation.\n"
        "- If a broad detection should become multiple surfaces, represent the split as add actions for the replacement visible polygons plus delete or flag_review for the original.\n"
        "- For siding/wall polygons, follow construction boundaries: roof rake/eave at top, grade/base/stone band at bottom, outside corners or trim boards at sides, and projection edges where planes change.\n"
        "- Keep windows, doors, and garages rectangular unless a box is plainly wrong.\n"
        "- Delete only obvious false positives such as labels, notes, dimension marks, or title block artifacts.\n"
        "- Prefer flag_review over risky edits. Be conservative.\n\n"
        "Action rules:\n"
        "- Use keep when no edit is needed.\n"
        "- Use add for a missed visible siding/gable surface.\n"
        "- Use convert_to_polygon for one detection changing from a box to polygon_points.\n"
        "- Use merge only when multiple exterior_wall/gable fragments clearly describe one physical surface.\n"
        "- For merge, include detection_ids and either polygon_points or new_box.\n"
        "- For every action, include a concise construction-plan reason and confidence from 0 to 1.\n\n"
        f"Page info:\n{json.dumps(page_info, indent=2)}\n\n"
        f"Target page extracted context:\n{json.dumps(target_extracted_context, indent=2)}\n\n"
        f"Reference context pages included:\n{json.dumps(compact_context_pages, indent=2)}\n\n"
        f"Inferred wall planes:\n{json.dumps(wall_plane_context, indent=2)}\n\n"
        f"Candidate detection ids you may alter:\n{json.dumps(candidate_ids, indent=2)}\n\n"
        f"All current detections for context:\n{json.dumps(compact_detections, indent=2)}"
    )


def _compact_detection(det):
    return {
        "id": det.get('id'),
        "detection_index": det.get('detection_index'),
        "class": normalize_detection_class(det.get('class')) or det.get('class'),
        "confidence": det.get('confidence'),
        "pixel_x": det.get('pixel_x'),
        "pixel_y": det.get('pixel_y'),
        "pixel_width": det.get('pixel_width'),
        "pixel_height": det.get('pixel_height'),
        "area_sf": det.get('area_sf'),
        "polygon_points": det.get('polygon_points'),
    }


def _extract_response_text(response_data):
    if response_data.get('output_text'):
        return response_data['output_text']

    parts = []
    for item in response_data.get('output', []):
        for content in item.get('content', []):
            if isinstance(content, dict) and content.get('text'):
                parts.append(content['text'])
    return ''.join(parts)


def _safe_error_text(text):
    if not text:
        return ''
    try:
        parsed = json.loads(text)
        message = parsed.get('error', {}).get('message')
        if message:
            return message
    except Exception:
        pass
    return text[:500]


def _validate_actions(actions, detections, candidate_detections, page_width, page_height, scale_ratio):
    by_id = {d.get('id'): d for d in detections}
    candidate_ids = {d.get('id') for d in candidate_detections}
    validated = []
    delete_count = 0
    max_delete_count = max(5, int(len(candidate_ids) * 0.35))

    for raw in actions:
        action = _normalize_action(raw)
        action_type = action.get('type')
        reasons = []
        status = 'valid'
        updates = {}

        if action_type not in ACTION_TYPES:
            reasons.append(f"Unsupported action type: {action_type}")

        ids = _action_detection_ids(action)
        missing = [det_id for det_id in ids if det_id not in by_id]
        outside_scope = [det_id for det_id in ids if det_id not in candidate_ids]
        if missing:
            reasons.append(f"Unknown detection ids: {missing}")
        if outside_scope:
            reasons.append(f"Detection ids outside refinement scope: {outside_scope}")

        target_class = _target_class_for_action(action, by_id)
        if action_type in {'resize', 'reclassify', 'delete', 'convert_to_polygon', 'flag_review'} and not action.get('detection_id'):
            reasons.append(f"{action_type} action needs detection_id")
        if action_type == 'merge' and len(ids) < 2:
            reasons.append("merge action needs at least two detection_ids")
        if action_type == 'add' and not target_class:
            reasons.append("add action needs target_class")
        if target_class and target_class not in ALLOWED_CLASSES:
            reasons.append(f"Unsupported target class: {target_class}")

        if action_type in {'add', 'convert_to_polygon', 'merge'}:
            if target_class not in POLYGON_CLASSES:
                reasons.append(f"Polygons are not allowed for class: {target_class}")
            points = _clean_points(action.get('polygon_points'), page_width, page_height)
            if len(points) < 3 and not action.get('new_box'):
                reasons.append("Polygon action needs at least 3 polygon_points or a new_box")
            if points and _polygon_area_px(points) < 25:
                reasons.append("Polygon area is too small")
            if points and _polygon_self_intersects(points):
                reasons.append("Polygon self-intersects")
            if points:
                updates.update(_updates_from_polygon(points, scale_ratio))
                updates['polygon_points'] = points
                updates['markup_type'] = 'polygon'
            elif action.get('new_box'):
                box = _clean_box(action.get('new_box'), page_width, page_height)
                if box:
                    updates.update(_updates_from_box(box, scale_ratio))
                else:
                    reasons.append("new_box is outside page bounds or invalid")

        if action_type == 'resize':
            box = _clean_box(action.get('new_box'), page_width, page_height)
            if not box:
                reasons.append("resize action needs a valid new_box")
            else:
                updates.update(_updates_from_box(box, scale_ratio))

        if action_type == 'reclassify':
            new_class = normalize_detection_class(action.get('new_class'))
            if not new_class or new_class not in ALLOWED_CLASSES:
                reasons.append(f"Invalid new_class: {action.get('new_class')}")
            else:
                updates['class'] = new_class

        if action_type == 'delete':
            delete_count += 1
            if delete_count > max_delete_count:
                reasons.append("Delete safety limit exceeded")
            updates.update({'is_deleted': True, 'status': 'deleted'})

        if action_type == 'flag_review':
            note = (action.get('reason') or 'AI flagged for review').strip()[:500]
            updates['material_notes'] = f"AI review: {note}"

        if reasons:
            status = 'blocked'

        validated.append({
            **action,
            "status": status,
            "validation_errors": reasons,
            "updates": updates,
        })

    if config.REFINEMENT_ENABLE_RETURN_PLANE_EXPANSION:
        _promote_return_plane_expansions(validated, detections, candidate_detections, scale_ratio)
    _validate_final_facade_overlaps(validated, detections)
    return validated


def _normalize_action(raw):
    action = dict(raw or {})
    action['type'] = str(action.get('type') or '').strip()
    action['detection_id'] = action.get('detection_id') or None
    action['detection_ids'] = [str(x) for x in (action.get('detection_ids') or []) if x]
    action['target_class'] = normalize_detection_class(action.get('target_class')) or None
    action['new_class'] = normalize_detection_class(action.get('new_class')) or None
    action['polygon_points'] = action.get('polygon_points') or []
    action['confidence'] = _to_float(action.get('confidence'), default=0.0)
    action['reason'] = str(action.get('reason') or '').strip()
    return action


def _action_detection_ids(action):
    ids = []
    if action.get('detection_id'):
        ids.append(action['detection_id'])
    ids.extend(action.get('detection_ids') or [])
    return list(dict.fromkeys(ids))


def _target_class_for_action(action, by_id):
    if action.get('target_class'):
        return action['target_class']
    if action.get('new_class'):
        return action['new_class']
    ids = _action_detection_ids(action)
    if ids and ids[0] in by_id:
        return normalize_detection_class(by_id[ids[0]].get('class'))
    return None


def _clean_points(points, page_width, page_height):
    clean = []
    for point in points or []:
        x = _to_float((point or {}).get('x'), default=None)
        y = _to_float((point or {}).get('y'), default=None)
        if x is None or y is None:
            continue
        if x < 0 or y < 0 or x > page_width or y > page_height:
            continue
        clean.append({"x": round(x, 2), "y": round(y, 2)})
    return clean


def _clean_box(box, page_width, page_height):
    if not isinstance(box, dict):
        return None
    x = _to_float(box.get('x'), default=None)
    y = _to_float(box.get('y'), default=None)
    width = _to_float(box.get('width'), default=None)
    height = _to_float(box.get('height'), default=None)
    if None in (x, y, width, height) or width <= 1 or height <= 1:
        return None
    if x - width / 2 < 0 or y - height / 2 < 0:
        return None
    if x + width / 2 > page_width or y + height / 2 > page_height:
        return None
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
    }


def _updates_from_polygon(points, scale_ratio):
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    detection = {
        'pixel_width': max_x - min_x,
        'pixel_height': max_y - min_y,
        'polygon_points': points,
    }
    return {
        'pixel_x': round((min_x + max_x) / 2, 2),
        'pixel_y': round((min_y + max_y) / 2, 2),
        'pixel_width': round(max_x - min_x, 2),
        'pixel_height': round(max_y - min_y, 2),
        'area_sf': round(compute_detection_area_sf(detection, scale_ratio), 2),
        'perimeter_lf': round(compute_detection_perimeter_lf(detection, scale_ratio), 2),
    }


def _updates_from_box(box, scale_ratio):
    detection = {
        'pixel_width': box['width'],
        'pixel_height': box['height'],
        'polygon_points': None,
    }
    return {
        'pixel_x': box['x'],
        'pixel_y': box['y'],
        'pixel_width': box['width'],
        'pixel_height': box['height'],
        'polygon_points': None,
        'area_sf': round(compute_detection_area_sf(detection, scale_ratio), 2),
        'perimeter_lf': round(compute_detection_perimeter_lf(detection, scale_ratio), 2),
    }


def _polygon_area_px(points):
    area = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area += point['x'] * nxt['y']
        area -= nxt['x'] * point['y']
    return abs(area) / 2.0


def _validate_final_facade_overlaps(validated_actions, detections):
    max_ratio = max(0.0, min(1.0, config.REFINEMENT_MAX_FACADE_OVERLAP_RATIO))
    records_by_key = {}
    by_id = {d.get('id'): d for d in detections}

    for det in detections:
        record = _facade_record_from_detection(det)
        if record:
            records_by_key[det.get('id')] = record

    for index, action in enumerate(validated_actions):
        if action.get('status') != 'valid':
            continue

        action_type = action.get('type')
        ids = _action_detection_ids(action)

        if action_type == 'delete':
            for det_id in ids:
                records_by_key.pop(det_id, None)
            continue

        if action_type == 'merge':
            for det_id in ids:
                records_by_key.pop(det_id, None)
            record = _facade_record_from_action(action, by_id, index)
            if record:
                records_by_key[f"action:{index}"] = record
            continue

        if action_type in {'add', 'convert_to_polygon', 'resize'}:
            det_id = action.get('detection_id')
            if det_id:
                records_by_key.pop(det_id, None)
            record = _facade_record_from_action(action, by_id, index)
            if record:
                records_by_key[det_id or f"action:{index}"] = record
            continue

        if action_type == 'reclassify':
            det_id = action.get('detection_id')
            det = by_id.get(det_id)
            if det_id and det:
                records_by_key.pop(det_id, None)
                record = _facade_record_from_detection(
                    {**det, 'class': action.get('new_class')},
                    action_index=index,
                )
                if record:
                    records_by_key[det_id] = record

    records = _assign_wall_plane_info(list(records_by_key.values()))
    for left_idx, left in enumerate(records):
        for right in records[left_idx + 1:]:
            if not _records_same_wall_plane(left, right):
                continue
            if not _bboxes_overlap(left['bbox'], right['bbox']):
                continue
            overlap_area = _polygon_overlap_area(left['points'], right['points'])
            if overlap_area <= 0:
                continue

            smaller_area = min(left['area'], right['area'])
            if smaller_area <= 0:
                continue

            overlap_ratio = overlap_area / smaller_area
            if overlap_ratio <= max_ratio:
                continue

            message = (
                f"Facade overlap exceeds {max_ratio:.0%}: "
                f"{left['label']} overlaps {right['label']} by {overlap_ratio:.0%} "
                "of the smaller polygon"
            )
            for action_index in {left.get('action_index'), right.get('action_index')}:
                if action_index is not None:
                    _block_action(validated_actions[action_index], message)


def _plan_facade_overlap_cleanup(detections, scale_ratio):
    max_ratio = max(0.0, min(1.0, config.REFINEMENT_MAX_FACADE_OVERLAP_RATIO))
    by_id = {det.get('id'): det for det in detections}
    records = []

    for det in detections:
        record = _facade_record_from_detection(det)
        if record:
            record['detection'] = det
            records.append(record)

    records = _assign_wall_plane_info(records)
    overlaps = []
    for left_idx, left in enumerate(records):
        for right in records[left_idx + 1:]:
            if not _records_same_wall_plane(left, right):
                continue
            if not _bboxes_overlap(left['bbox'], right['bbox']):
                continue
            overlap_area = _polygon_overlap_area(left['points'], right['points'])
            if overlap_area <= 0:
                continue

            smaller_area = min(left['area'], right['area'])
            if smaller_area <= 0:
                continue

            overlap_ratio = overlap_area / smaller_area
            if overlap_ratio > max_ratio:
                overlaps.append({
                    'left': left,
                    'right': right,
                    'overlap_ratio': overlap_ratio,
                    'overlap_area': overlap_area,
                })

    overlaps.sort(key=lambda item: item['overlap_ratio'], reverse=True)
    planned = []
    planned_ids = set()

    for overlap in overlaps:
        left = overlap['left']
        right = overlap['right']
        left_id = left['detection'].get('id')
        right_id = right['detection'].get('id')

        if left_id in planned_ids or right_id in planned_ids:
            continue

        target, reason = _choose_facade_cleanup_target(
            left,
            right,
            overlap['overlap_ratio'],
            scale_ratio,
        )
        if target:
            target_id = target['detection'].get('id')
            if target_id and target_id in by_id:
                planned_ids.add(target_id)
                planned.append({
                    'action': target.get('cleanup_action', 'delete'),
                    'detection': by_id[target_id],
                    'updates': target.get('updates'),
                    'reason': reason,
                    'overlap_ratio': round(overlap['overlap_ratio'], 4),
                })
            continue

        reason = (
            f"Ambiguous facade overlap between detection "
            f"{left['detection'].get('detection_index')} and {right['detection'].get('detection_index')} "
            f"({overlap['overlap_ratio']:.0%} of smaller polygon)."
        )
        planned.append({
            'action': 'flag_review',
            'detections': [left['detection'], right['detection']],
            'reason': reason,
            'overlap_ratio': round(overlap['overlap_ratio'], 4),
        })

    return planned


def _choose_facade_cleanup_target(left, right, overlap_ratio, scale_ratio):
    left_det = left['detection']
    right_det = right['detection']
    left_can_delete = _can_auto_delete_detection(left_det)
    right_can_delete = _can_auto_delete_detection(right_det)

    left_has_polygon = bool(_points_from_polygon_data(left_det.get('polygon_points')))
    right_has_polygon = bool(_points_from_polygon_data(right_det.get('polygon_points')))

    if left_has_polygon and not right_has_polygon and right_can_delete:
        trimmed = _trim_box_record_against_polygon_record(right, left, scale_ratio)
        if trimmed:
            return trimmed, _cleanup_reason(trimmed, left, overlap_ratio, "trimmed rectangle into non-overlapping polygon")
        if overlap_ratio >= 0.80:
            return right, _cleanup_reason(right, left, overlap_ratio, "mostly contained rectangle overlaps a polygon")
        return None, None
    if right_has_polygon and not left_has_polygon and left_can_delete:
        trimmed = _trim_box_record_against_polygon_record(left, right, scale_ratio)
        if trimmed:
            return trimmed, _cleanup_reason(trimmed, right, overlap_ratio, "trimmed rectangle into non-overlapping polygon")
        if overlap_ratio >= 0.80:
            return left, _cleanup_reason(left, right, overlap_ratio, "mostly contained rectangle overlaps a polygon")
        return None, None

    left_flagged = _has_ai_review_note(left_det)
    right_flagged = _has_ai_review_note(right_det)
    if left_flagged and not right_flagged and left_can_delete and overlap_ratio >= 0.80:
        return left, _cleanup_reason(left, right, overlap_ratio, "review-flagged duplicate facade")
    if right_flagged and not left_flagged and right_can_delete and overlap_ratio >= 0.80:
        return right, _cleanup_reason(right, left, overlap_ratio, "review-flagged duplicate facade")

    if overlap_ratio >= 0.80:
        if left['area'] <= right['area'] and left_can_delete:
            return left, _cleanup_reason(left, right, overlap_ratio, "mostly contained duplicate facade")
        if right_can_delete:
            return right, _cleanup_reason(right, left, overlap_ratio, "mostly contained duplicate facade")

    if overlap_ratio >= 0.80:
        left_conf = _to_float(left_det.get('confidence'), default=0.0)
        right_conf = _to_float(right_det.get('confidence'), default=0.0)
        if left_conf < right_conf and left_can_delete:
            return left, _cleanup_reason(left, right, overlap_ratio, "lower-confidence overlapping facade")
        if right_conf < left_conf and right_can_delete:
            return right, _cleanup_reason(right, left, overlap_ratio, "lower-confidence overlapping facade")

    return None, None


def _cleanup_reason(target, keeper, overlap_ratio, reason):
    target_idx = target['detection'].get('detection_index')
    keeper_idx = keeper['detection'].get('detection_index')
    return (
        f"Adjusted detection {target_idx}; {reason} with detection {keeper_idx} "
        f"({overlap_ratio:.0%} overlap of the smaller facade)."
    )


def _trim_box_record_against_polygon_record(box_record, polygon_record, scale_ratio):
    box = box_record['bbox']
    poly_box = polygon_record['bbox']
    strips = _non_overlapping_strips(box, poly_box)
    if not strips:
        return None

    strips.sort(key=lambda item: item['area'], reverse=True)
    strip = strips[0]
    original_area = box_record['area']
    if original_area <= 0 or strip['area'] / original_area < 0.18:
        return None

    points = _strip_to_polygon(strip, box_record, polygon_record)
    if len(points) < 3 or _polygon_area_px(points) / original_area < 0.12:
        return None

    updates = _updates_from_polygon(points, scale_ratio)
    updates['polygon_points'] = points
    updates['markup_type'] = 'polygon'

    trimmed = dict(box_record)
    trimmed.update({
        'cleanup_action': 'trim',
        'updates': updates,
        'points': points,
        'area': _polygon_area_px(points),
        'bbox': _points_bbox(points),
    })
    return trimmed


def _non_overlapping_strips(box, blocker):
    left, top, right, bottom = box
    block_left, block_top, block_right, block_bottom = blocker
    strips = []

    candidates = [
        ('left', (left, top, min(right, block_left), bottom)),
        ('right', (max(left, block_right), top, right, bottom)),
        ('top', (left, top, right, min(bottom, block_top))),
        ('bottom', (left, max(top, block_bottom), right, bottom)),
    ]
    for side, candidate in candidates:
        c_left, c_top, c_right, c_bottom = candidate
        width = c_right - c_left
        height = c_bottom - c_top
        if width <= 8 or height <= 8:
            continue
        strips.append({
            'side': side,
            'bbox': candidate,
            'area': width * height,
        })
    return strips


def _strip_to_polygon(strip, box_record, polygon_record):
    left, top, right, bottom = strip['bbox']
    side = strip['side']
    poly_points = polygon_record['points']
    polygon_bbox = polygon_record['bbox']
    fallback = [
        {'x': left, 'y': top},
        {'x': right, 'y': top},
        {'x': right, 'y': bottom},
        {'x': left, 'y': bottom},
    ]

    if side == 'right':
        sloped = _right_strip_with_sloped_top(left, top, right, bottom, poly_points, polygon_bbox)
        if _usable_trim_polygon(sloped, fallback):
            return sloped
    if side == 'left':
        sloped = _left_strip_with_sloped_top(left, top, right, bottom, poly_points, polygon_bbox)
        if _usable_trim_polygon(sloped, fallback):
            return sloped

    return fallback


def _usable_trim_polygon(candidate, fallback):
    if not candidate or len(candidate) < 3:
        return False
    if _polygon_self_intersects(candidate):
        return False
    fallback_area = _polygon_area_px(fallback)
    if fallback_area <= 0:
        return False
    return _polygon_area_px(candidate) >= fallback_area * 0.20


def _right_strip_with_sloped_top(left, top, right, bottom, poly_points, polygon_bbox):
    shared_x = polygon_bbox[2]
    if abs(left - shared_x) > 2:
        return None

    boundary_points = [p for p in poly_points if abs(p['x'] - shared_x) <= 2]
    if len(boundary_points) < 2:
        return None

    top_shared = min(boundary_points, key=lambda p: p['y'])
    bottom_shared = max(boundary_points, key=lambda p: p['y'])
    neighbor = _non_vertical_neighbor(poly_points, top_shared)
    if not neighbor:
        return None

    slope = (top_shared['y'] - neighbor['y']) / ((top_shared['x'] - neighbor['x']) or 1e-9)
    right_top = top_shared['y'] + slope * (right - shared_x)
    right_top = max(top, min(bottom, right_top))
    left_bottom = min(bottom, bottom_shared['y'])

    return [
        {'x': shared_x, 'y': top_shared['y']},
        {'x': right, 'y': right_top},
        {'x': right, 'y': bottom},
        {'x': shared_x, 'y': left_bottom},
    ]


def _left_strip_with_sloped_top(left, top, right, bottom, poly_points, polygon_bbox):
    shared_x = polygon_bbox[0]
    if abs(right - shared_x) > 2:
        return None

    boundary_points = [p for p in poly_points if abs(p['x'] - shared_x) <= 2]
    if len(boundary_points) < 2:
        return None

    top_shared = min(boundary_points, key=lambda p: p['y'])
    bottom_shared = max(boundary_points, key=lambda p: p['y'])
    neighbor = _non_vertical_neighbor(poly_points, top_shared)
    if not neighbor:
        return None

    slope = (top_shared['y'] - neighbor['y']) / ((top_shared['x'] - neighbor['x']) or 1e-9)
    left_top = top_shared['y'] + slope * (left - shared_x)
    left_top = max(top, min(bottom, left_top))
    right_bottom = min(bottom, bottom_shared['y'])

    return [
        {'x': left, 'y': left_top},
        {'x': shared_x, 'y': top_shared['y']},
        {'x': shared_x, 'y': right_bottom},
        {'x': left, 'y': bottom},
    ]


def _non_vertical_neighbor(points, point):
    for idx, candidate in enumerate(points):
        if not _same_point(candidate, point):
            continue
        neighbors = [
            points[(idx - 1) % len(points)],
            points[(idx + 1) % len(points)],
        ]
        for neighbor in neighbors:
            if abs(neighbor['x'] - point['x']) > 2:
                return neighbor
    return None


def _can_auto_delete_detection(det):
    return not det.get('is_user_created') and not det.get('assigned_material_id')


def _has_ai_review_note(det):
    notes = (det.get('material_notes') or '').lower()
    return 'ai review:' in notes or 'ai overlap review:' in notes


def _append_material_note(existing, note):
    existing = (existing or '').strip()
    if not existing:
        return note[:1000]
    if note in existing:
        return existing[:1000]
    return f"{existing}\n{note}"[:1000]


def _block_action(action, message):
    action['status'] = 'blocked'
    errors = action.setdefault('validation_errors', [])
    if message not in errors:
        errors.append(message)


def _facade_record_from_action(action, by_id, action_index):
    target_class = _target_class_for_action(action, by_id)
    if target_class not in FACADE_OVERLAP_CLASSES:
        return None

    points = _points_from_geometry(action.get('updates') or {})
    if not points and action.get('detection_id') in by_id:
        points = _points_from_detection(by_id[action['detection_id']])
    if len(points) < 3:
        return None

    area = _polygon_area_px(points)
    if area <= 0:
        return None

    return {
        'action': action,
        'action_index': action_index,
        'area': area,
        'bbox': _points_bbox(points),
        'class': target_class,
        'label': f"{target_class} action {action_index + 1}",
        'points': points,
    }


def _facade_record_from_detection(det, action_index=None):
    if det.get('is_deleted'):
        return None

    det_class = normalize_detection_class(det.get('class'))
    if det_class not in FACADE_OVERLAP_CLASSES:
        return None

    points = _points_from_detection(det)
    if len(points) < 3:
        return None

    area = _polygon_area_px(points)
    if area <= 0:
        return None

    det_id = det.get('id') or 'existing'
    return {
        'action_index': action_index,
        'area': area,
        'bbox': _points_bbox(points),
        'class': det_class,
        'detection': det,
        'label': f"{det_class} {str(det_id)[:8]}",
        'points': points,
    }


def _assign_wall_plane_info(records):
    if not records:
        return records

    sorted_records = sorted(records, key=lambda rec: _bbox_center(rec['bbox'])[1])
    heights = [max(1, rec['bbox'][3] - rec['bbox'][1]) for rec in sorted_records]
    threshold = max(180, _median(heights) * 1.5)
    groups = []

    for record in sorted_records:
        center_y = _bbox_center(record['bbox'])[1]
        if not groups or abs(center_y - groups[-1]['center_y']) > threshold:
            groups.append({'records': [record], 'center_y': center_y})
        else:
            group = groups[-1]
            group['records'].append(record)
            group['center_y'] = sum(_bbox_center(r['bbox'])[1] for r in group['records']) / len(group['records'])

    for group_index, group in enumerate(groups):
        group_records = group['records']
        group_min_x = min(rec['bbox'][0] for rec in group_records)
        group_max_x = max(rec['bbox'][2] for rec in group_records)
        group_width = max(1, group_max_x - group_min_x)
        view_group = f"view_{group_index + 1}"

        for record in group_records:
            plane_type, confidence = _classify_wall_plane_record(
                record,
                group_index=group_index,
                group_min_x=group_min_x,
                group_max_x=group_max_x,
                group_width=group_width,
            )
            side = _record_lateral_side(record, group_min_x, group_max_x, group_width)
            record['view_group'] = view_group
            record['plane_type'] = plane_type
            record['plane_side'] = side
            record['plane_confidence'] = confidence
            record['plane_id'] = _wall_plane_id(record)

    return records


def _classify_wall_plane_record(record, group_index, group_min_x, group_max_x, group_width):
    left, top, right, bottom = record['bbox']
    width = max(1, right - left)
    height = max(1, bottom - top)
    det_class = record.get('class')
    side = _record_lateral_side(record, group_min_x, group_max_x, group_width)
    has_sloped_top = _record_has_sloped_top(record)
    has_peak = _record_has_center_peak(record)
    is_narrow = width <= max(80, height * 0.65)
    near_edge = side in {'left', 'right'}

    if group_index > 0 and width >= height * 2:
        return 'side_elevation', 0.72
    if det_class == 'gable' or (has_peak and not is_narrow):
        return 'projecting_gable', 0.78
    if has_sloped_top and (is_narrow or near_edge):
        return 'return_plane', 0.70
    if is_narrow and near_edge and height > 70:
        return 'return_plane', 0.58
    return 'main_plane', 0.62


def _records_same_wall_plane(left, right):
    if left.get('view_group') != right.get('view_group'):
        return False

    left_type = left.get('plane_type')
    right_type = right.get('plane_type')

    if 'return_plane' in {left_type, right_type}:
        return (
            left_type == right_type
            and left.get('plane_side') == right.get('plane_side')
            and _horizontal_overlap_ratio(left['bbox'], right['bbox']) > 0.45
        )

    if 'side_elevation' in {left_type, right_type}:
        return left_type == right_type

    return True


def _promote_return_plane_expansions(validated_actions, detections, candidate_detections, scale_ratio):
    candidate_ids = {d.get('id') for d in candidate_detections}
    by_id = {d.get('id'): d for d in detections}
    for action in validated_actions:
        if action.get('status') != 'valid' or action.get('type') != 'keep':
            continue
        det_id = action.get('detection_id')
        if not det_id or det_id not in candidate_ids:
            continue

        det = by_id.get(det_id)
        if not det:
            continue

        expansion = _return_plane_expansion_updates(det, detections, scale_ratio)
        if not expansion:
            continue

        action['type'] = 'convert_to_polygon'
        action['target_class'] = normalize_detection_class(det.get('class')) or 'exterior_wall'
        action['polygon_points'] = expansion['polygon_points']
        action['updates'] = expansion
        action['confidence'] = max(_to_float(action.get('confidence'), 0.0), 0.74)
        action['reason'] = (
            "Wall-plane solver expanded the selected return-plane sliver into the wider visible "
            "return wall using the adjacent projecting gable shoulder, sloped eave, outside corner, and base line."
        )


def _return_plane_expansion_updates(det, detections, scale_ratio):
    records = []
    candidate_id = det.get('id')
    candidate_record = None

    for current in detections:
        record = _facade_record_from_detection(current)
        if not record:
            continue
        records.append(record)

    records = _assign_wall_plane_info(records)
    for record in records:
        if (record.get('detection') or {}).get('id') == candidate_id:
            candidate_record = record
            break

    if not candidate_record or candidate_record.get('plane_type') != 'return_plane':
        return None

    cand_left, cand_top, cand_right, cand_bottom = candidate_record['bbox']
    cand_width = cand_right - cand_left
    cand_area = candidate_record['area']
    if cand_width <= 0 or cand_area <= 0:
        return None

    direction = candidate_record.get('plane_side')
    if direction not in {'left', 'right'}:
        return None

    projecting = _adjacent_projecting_gable(candidate_record, records, direction)
    if not projecting:
        return None

    proj_left, proj_top, proj_right, proj_bottom = projecting['bbox']
    proj_width = proj_right - proj_left
    if proj_width <= 0 or cand_width > proj_width * 0.65:
        return None

    if direction == 'right':
        if abs(proj_right - cand_left) > 14:
            return None
        points = _right_return_expansion_points(candidate_record, projecting)
    else:
        if abs(proj_left - cand_right) > 14:
            return None
        points = _left_return_expansion_points(candidate_record, projecting)

    if len(points) < 3 or _polygon_self_intersects(points):
        return None
    if _polygon_area_px(points) <= cand_area * 1.35:
        return None

    updates = _updates_from_polygon(points, scale_ratio)
    updates['polygon_points'] = points
    updates['markup_type'] = 'polygon'
    return updates


def _adjacent_projecting_gable(candidate_record, records, direction):
    candidates = []
    cand_bbox = candidate_record['bbox']
    for record in records:
        if record is candidate_record:
            continue
        if record.get('view_group') != candidate_record.get('view_group'):
            continue
        if record.get('plane_type') != 'projecting_gable':
            continue
        if not _vertical_ranges_overlap(cand_bbox, record['bbox'], min_ratio=0.35):
            continue
        if direction == 'right':
            gap = abs(record['bbox'][2] - cand_bbox[0])
        else:
            gap = abs(record['bbox'][0] - cand_bbox[2])
        if gap <= 22:
            candidates.append((gap, record))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _right_return_expansion_points(candidate, projecting):
    cand_left, _, cand_right, cand_bottom = candidate['bbox']
    proj_left, _, proj_right, proj_bottom = projecting['bbox']
    shoulders = _gable_shoulder_points(projecting)
    if not shoulders:
        return []

    left_shoulder = min(shoulders, key=lambda p: p['x'])
    right_shoulder = max(shoulders, key=lambda p: p['x'])
    right_top = _vertical_edge_point(candidate['points'], cand_right, prefer='top')
    right_bottom = _vertical_edge_point(candidate['points'], cand_right, prefer='bottom')
    if not right_top:
        right_top = {'x': cand_right, 'y': right_shoulder['y'] + max(0, cand_right - proj_right)}
    if not right_bottom:
        right_bottom = {'x': cand_right, 'y': min(cand_bottom, proj_bottom)}

    bottom_y = min(cand_bottom, proj_bottom)
    return _dedupe_polygon_points([
        {'x': left_shoulder['x'], 'y': left_shoulder['y']},
        {'x': right_shoulder['x'], 'y': right_shoulder['y']},
        {'x': right_top['x'], 'y': right_top['y']},
        {'x': right_bottom['x'], 'y': right_bottom['y']},
        {'x': proj_left, 'y': bottom_y},
    ])


def _left_return_expansion_points(candidate, projecting):
    cand_left, _, cand_right, cand_bottom = candidate['bbox']
    proj_left, _, proj_right, proj_bottom = projecting['bbox']
    shoulders = _gable_shoulder_points(projecting)
    if not shoulders:
        return []

    left_shoulder = min(shoulders, key=lambda p: p['x'])
    right_shoulder = max(shoulders, key=lambda p: p['x'])
    left_top = _vertical_edge_point(candidate['points'], cand_left, prefer='top')
    left_bottom = _vertical_edge_point(candidate['points'], cand_left, prefer='bottom')
    if not left_top:
        left_top = {'x': cand_left, 'y': left_shoulder['y'] + max(0, proj_left - cand_left)}
    if not left_bottom:
        left_bottom = {'x': cand_left, 'y': min(cand_bottom, proj_bottom)}

    bottom_y = min(cand_bottom, proj_bottom)
    return _dedupe_polygon_points([
        {'x': cand_left, 'y': left_top['y']},
        {'x': left_shoulder['x'], 'y': left_shoulder['y']},
        {'x': right_shoulder['x'], 'y': right_shoulder['y']},
        {'x': proj_right, 'y': bottom_y},
        {'x': cand_left, 'y': left_bottom['y']},
    ])


def _gable_shoulder_points(record):
    points = record.get('points') or []
    if len(points) < 4:
        return []
    min_y = min(p['y'] for p in points)
    bottom_y = record['bbox'][3]
    candidates = [
        p for p in points
        if p['y'] > min_y + 20 and p['y'] < bottom_y - 20
    ]
    if len(candidates) < 2:
        return []
    candidates.sort(key=lambda p: p['x'])
    return [candidates[0], candidates[-1]]


def _vertical_edge_point(points, x, prefer):
    edge_points = [p for p in points if abs(p['x'] - x) <= 3]
    if not edge_points:
        return None
    return min(edge_points, key=lambda p: p['y']) if prefer == 'top' else max(edge_points, key=lambda p: p['y'])


def _vertical_ranges_overlap(left, right, min_ratio=0.0):
    overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    smaller = min(max(1, left[3] - left[1]), max(1, right[3] - right[1]))
    return overlap / smaller >= min_ratio


def _wall_plane_id(record):
    center_x, _ = _bbox_center(record['bbox'])
    bucket = int(center_x // 180)
    return f"{record.get('view_group')}:{record.get('plane_type')}:{record.get('plane_side')}:{bucket}"


def _record_lateral_side(record, group_min_x, group_max_x, group_width):
    center_x, _ = _bbox_center(record['bbox'])
    if center_x <= group_min_x + group_width * 0.22:
        return 'left'
    if center_x >= group_max_x - group_width * 0.22:
        return 'right'
    return 'middle'


def _record_has_sloped_top(record):
    points = record.get('points') or []
    if len(points) < 3:
        return False
    ys = sorted(p['y'] for p in points)
    if len(ys) >= 2 and abs(ys[1] - ys[0]) > 20:
        return True

    top = record['bbox'][1]
    height = max(1, record['bbox'][3] - record['bbox'][1])
    upper_limit = top + height * 0.55
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        dx = abs(nxt['x'] - point['x'])
        dy = abs(nxt['y'] - point['y'])
        if dx > 20 and dy > 20 and min(point['y'], nxt['y']) <= upper_limit:
            return True
    return False


def _record_has_center_peak(record):
    points = record.get('points') or []
    if len(points) < 4:
        return False
    left, _, right, _ = record['bbox']
    width = max(1, right - left)
    min_y = min(p['y'] for p in points)
    top_points = [p for p in points if abs(p['y'] - min_y) <= 4]
    if len(top_points) != 1:
        return False
    peak_x = top_points[0]['x']
    return left + width * 0.25 <= peak_x <= right - width * 0.25


def _bbox_center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _horizontal_overlap_ratio(left, right):
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    smaller = min(max(1, left[2] - left[0]), max(1, right[2] - right[0]))
    return overlap / smaller


def _median(values):
    if not values:
        return 0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _compact_wall_plane_context(detections):
    records = []
    for det in detections:
        record = _facade_record_from_detection(det)
        if record:
            records.append(record)
    records = _assign_wall_plane_info(records)
    return [
        {
            'detection_id': (record.get('detection') or {}).get('id'),
            'detection_index': (record.get('detection') or {}).get('detection_index'),
            'class': record.get('class'),
            'view_group': record.get('view_group'),
            'plane_type': record.get('plane_type'),
            'plane_side': record.get('plane_side'),
            'plane_id': record.get('plane_id'),
            'plane_confidence': record.get('plane_confidence'),
            'bbox': [round(v, 2) for v in record.get('bbox', [])],
        }
        for record in records
    ]


def _points_from_detection(det):
    points = _points_from_polygon_data(det.get('polygon_points'))
    if len(points) >= 3 and _polygon_area_px(points) > 0:
        return points
    return _points_from_box_values(
        det.get('pixel_x'),
        det.get('pixel_y'),
        det.get('pixel_width'),
        det.get('pixel_height'),
    )


def _points_from_geometry(geometry):
    points = _points_from_polygon_data(geometry.get('polygon_points'))
    if len(points) >= 3 and _polygon_area_px(points) > 0:
        return points
    return _points_from_box_values(
        geometry.get('pixel_x'),
        geometry.get('pixel_y'),
        geometry.get('pixel_width'),
        geometry.get('pixel_height'),
    )


def _points_from_polygon_data(polygon_points):
    if isinstance(polygon_points, dict):
        polygon_points = polygon_points.get('outer')
    if not isinstance(polygon_points, list):
        return []

    points = []
    for point in polygon_points:
        if isinstance(point, dict):
            x = _to_float(point.get('x'), default=None)
            y = _to_float(point.get('y'), default=None)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x = _to_float(point[0], default=None)
            y = _to_float(point[1], default=None)
        else:
            continue
        if x is None or y is None:
            continue
        points.append({'x': x, 'y': y})

    if len(points) > 1 and _same_point(points[0], points[-1]):
        points.pop()
    return points


def _points_from_box_values(x, y, width, height):
    x = _to_float(x, default=None)
    y = _to_float(y, default=None)
    width = _to_float(width, default=None)
    height = _to_float(height, default=None)
    if None in (x, y, width, height) or width <= 0 or height <= 0:
        return []

    left = x - width / 2
    right = x + width / 2
    top = y - height / 2
    bottom = y + height / 2
    return [
        {'x': left, 'y': top},
        {'x': right, 'y': top},
        {'x': right, 'y': bottom},
        {'x': left, 'y': bottom},
    ]


def _points_bbox(points):
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bboxes_overlap(left, right):
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _polygon_overlap_area(subject, clip):
    if not subject or not clip:
        return 0.0
    if not _bboxes_overlap(_points_bbox(subject), _points_bbox(clip)):
        return 0.0

    if _polygon_is_convex(clip):
        clipped = _clip_polygon(subject, clip)
        return _polygon_area_px(clipped) if len(clipped) >= 3 else 0.0
    if _polygon_is_convex(subject):
        clipped = _clip_polygon(clip, subject)
        return _polygon_area_px(clipped) if len(clipped) >= 3 else 0.0

    return _sample_polygon_overlap_area(subject, clip)


def _clip_polygon(subject, clip):
    output = list(subject)
    if len(output) < 3 or len(clip) < 3:
        return []

    clip_orientation = 1 if _signed_polygon_area(clip) >= 0 else -1
    for idx, clip_start in enumerate(clip):
        clip_end = clip[(idx + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break

        prev = input_points[-1]
        for current in input_points:
            current_inside = _inside_clip_edge(current, clip_start, clip_end, clip_orientation)
            prev_inside = _inside_clip_edge(prev, clip_start, clip_end, clip_orientation)

            if current_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, current, clip_start, clip_end))
                output.append(current)
            elif prev_inside:
                output.append(_line_intersection(prev, current, clip_start, clip_end))
            prev = current

    return _dedupe_polygon_points(output)


def _inside_clip_edge(point, edge_start, edge_end, clip_orientation):
    cross = (
        (edge_end['x'] - edge_start['x']) * (point['y'] - edge_start['y'])
        - (edge_end['y'] - edge_start['y']) * (point['x'] - edge_start['x'])
    )
    return cross >= -1e-7 if clip_orientation >= 0 else cross <= 1e-7


def _line_intersection(line_a_start, line_a_end, line_b_start, line_b_end):
    x1, y1 = line_a_start['x'], line_a_start['y']
    x2, y2 = line_a_end['x'], line_a_end['y']
    x3, y3 = line_b_start['x'], line_b_start['y']
    x4, y4 = line_b_end['x'], line_b_end['y']
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denominator) < 1e-9:
        return {'x': x2, 'y': y2}

    det_a = x1 * y2 - y1 * x2
    det_b = x3 * y4 - y3 * x4
    return {
        'x': (det_a * (x3 - x4) - (x1 - x2) * det_b) / denominator,
        'y': (det_a * (y3 - y4) - (y1 - y2) * det_b) / denominator,
    }


def _signed_polygon_area(points):
    area = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area += point['x'] * nxt['y']
        area -= nxt['x'] * point['y']
    return area / 2.0


def _polygon_is_convex(points):
    if len(points) < 4:
        return True

    sign = 0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        after = points[(idx + 2) % len(points)]
        cross = (
            (nxt['x'] - point['x']) * (after['y'] - nxt['y'])
            - (nxt['y'] - point['y']) * (after['x'] - nxt['x'])
        )
        if abs(cross) < 1e-7:
            continue
        current_sign = 1 if cross > 0 else -1
        if sign and current_sign != sign:
            return False
        sign = current_sign
    return True


def _sample_polygon_overlap_area(left, right):
    left_bbox = _points_bbox(left)
    right_bbox = _points_bbox(right)
    min_x = max(left_bbox[0], right_bbox[0])
    min_y = max(left_bbox[1], right_bbox[1])
    max_x = min(left_bbox[2], right_bbox[2])
    max_y = min(left_bbox[3], right_bbox[3])
    if max_x <= min_x or max_y <= min_y:
        return 0.0

    bbox_area = (max_x - min_x) * (max_y - min_y)
    step = max(4.0, (bbox_area / 40000) ** 0.5)
    x = min_x + step / 2
    hits = 0
    while x < max_x:
        y = min_y + step / 2
        while y < max_y:
            point = {'x': x, 'y': y}
            if _point_in_polygon(point, left) and _point_in_polygon(point, right):
                hits += 1
            y += step
        x += step
    return hits * step * step


def _point_in_polygon(point, polygon):
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        intersects = (
            (current['y'] > point['y']) != (previous['y'] > point['y'])
            and point['x'] < (
                (previous['x'] - current['x'])
                * (point['y'] - current['y'])
                / ((previous['y'] - current['y']) or 1e-9)
                + current['x']
            )
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _dedupe_polygon_points(points):
    clean = []
    for point in points:
        if clean and _same_point(clean[-1], point):
            continue
        clean.append({'x': point['x'], 'y': point['y']})
    if len(clean) > 1 and _same_point(clean[0], clean[-1]):
        clean.pop()
    return clean


def _polygon_self_intersects(points):
    if len(points) < 4:
        return False

    for left_idx, left_start in enumerate(points):
        left_end = points[(left_idx + 1) % len(points)]
        for right_idx in range(left_idx + 1, len(points)):
            if abs(left_idx - right_idx) <= 1:
                continue
            if left_idx == 0 and right_idx == len(points) - 1:
                continue

            right_start = points[right_idx]
            right_end = points[(right_idx + 1) % len(points)]
            if _segments_intersect(left_start, left_end, right_start, right_end):
                return True
    return False


def _segments_intersect(a_start, a_end, b_start, b_end):
    o1 = _orientation(a_start, a_end, b_start)
    o2 = _orientation(a_start, a_end, b_end)
    o3 = _orientation(b_start, b_end, a_start)
    o4 = _orientation(b_start, b_end, a_end)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _point_on_segment(a_start, b_start, a_end):
        return True
    if o2 == 0 and _point_on_segment(a_start, b_end, a_end):
        return True
    if o3 == 0 and _point_on_segment(b_start, a_start, b_end):
        return True
    if o4 == 0 and _point_on_segment(b_start, a_end, b_end):
        return True
    return False


def _orientation(a, b, c):
    value = (b['y'] - a['y']) * (c['x'] - b['x']) - (b['x'] - a['x']) * (c['y'] - b['y'])
    if abs(value) < 1e-7:
        return 0
    return 1 if value > 0 else 2


def _point_on_segment(a, b, c):
    return (
        min(a['x'], c['x']) - 1e-7 <= b['x'] <= max(a['x'], c['x']) + 1e-7
        and min(a['y'], c['y']) - 1e-7 <= b['y'] <= max(a['y'], c['y']) + 1e-7
    )


def _same_point(left, right):
    return abs(left['x'] - right['x']) < 1e-7 and abs(left['y'] - right['y']) < 1e-7


def _apply_validated_actions(validated_actions, detections, page_id, job_id, scale_ratio):
    by_id = {d.get('id'): d for d in detections}
    applied = 0
    blocked = 0
    skipped = 0
    errors = []

    for action in validated_actions:
        action_type = action.get('type')
        if action.get('status') != 'valid':
            blocked += 1
            continue
        if action_type == 'keep':
            skipped += 1
            continue

        try:
            if action_type == 'merge':
                _apply_merge_action(action, by_id, page_id, job_id, scale_ratio)
            elif action_type == 'add':
                _apply_add_action(action, page_id, job_id)
            else:
                det_id = action.get('detection_id')
                if not det_id:
                    raise ValueError("Action is missing detection_id")
                updates = dict(action.get('updates') or {})
                if not updates:
                    skipped += 1
                    continue
                result = supabase_request('PATCH', 'extraction_detections_draft', updates, {'id': f'eq.{det_id}'})
                if not result:
                    raise ValueError(f"Supabase update failed for {det_id}")
            action['status'] = 'applied'
            applied += 1
        except Exception as exc:
            action['status'] = 'error'
            action['apply_error'] = str(exc)
            errors.append({"action": action_type, "error": str(exc)})

    return {
        "applied": applied,
        "blocked": blocked,
        "skipped": skipped,
        "errors": errors,
    }


def _apply_add_action(action, page_id, job_id):
    target_class = action.get('target_class') or action.get('new_class')
    updates = dict(action.get('updates') or {})
    if not target_class:
        raise ValueError("Add action is missing target_class")
    if not updates:
        raise ValueError("Add action has no geometry updates")

    next_index = _next_detection_index(page_id)
    new_row = {
        'id': str(uuid.uuid4()),
        'job_id': job_id,
        'page_id': page_id,
        'source_detection_id': None,
        'class': target_class,
        'confidence': action.get('confidence') or 0.5,
        'detection_index': next_index,
        'matched_tag': None,
        'is_triangle': target_class == 'gable',
        'assigned_material_id': None,
        'material_notes': f"AI added: {(action.get('reason') or '')[:450]}",
        'is_deleted': False,
        'is_user_created': False,
        'markup_type': 'polygon',
        'status': 'auto',
        'item_count': 1,
        **updates,
    }
    result = supabase_request('POST', 'extraction_detections_draft', new_row)
    if not result:
        raise ValueError("Supabase insert failed for added detection")


def _apply_merge_action(action, by_id, page_id, job_id, scale_ratio):
    source_ids = [det_id for det_id in action.get('detection_ids', []) if det_id in by_id]
    if len(source_ids) < 2:
        raise ValueError("Merge action needs at least two valid source detections")

    target_class = _target_class_for_action(action, by_id) or normalize_detection_class(by_id[source_ids[0]].get('class'))
    updates = dict(action.get('updates') or {})
    if not updates:
        raise ValueError("Merge action has no geometry updates")

    next_index = _next_detection_index(page_id)
    new_row = {
        'id': str(uuid.uuid4()),
        'job_id': job_id,
        'page_id': page_id,
        'source_detection_id': None,
        'class': target_class,
        'confidence': action.get('confidence') or 0.5,
        'detection_index': next_index,
        'matched_tag': None,
        'is_triangle': target_class == 'gable',
        'assigned_material_id': None,
        'material_notes': f"AI merged: {(action.get('reason') or '')[:450]}",
        'is_deleted': False,
        'is_user_created': False,
        'markup_type': 'polygon' if updates.get('polygon_points') else 'polygon',
        'status': 'auto',
        'item_count': 1,
        **updates,
    }
    result = supabase_request('POST', 'extraction_detections_draft', new_row)
    if not result:
        raise ValueError("Supabase insert failed for merged detection")

    for det_id in source_ids:
        supabase_request('PATCH', 'extraction_detections_draft', {
            'is_deleted': True,
            'status': 'deleted',
        }, {'id': f'eq.{det_id}'})


def _next_detection_index(page_id):
    existing = supabase_request('GET', 'extraction_detections_draft', filters={
        'page_id': f'eq.{page_id}',
        'order': 'detection_index.desc',
        'limit': '1',
        'select': 'detection_index',
    }) or []
    if not existing:
        return 1
    return int(existing[0].get('detection_index') or 0) + 1


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
