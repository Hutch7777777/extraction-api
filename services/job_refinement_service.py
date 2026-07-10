"""
Job-level pre-editor detection refinement.

This service promotes raw Roboflow detections through the second-pass validator
into a separate refined layer, then seeds the editor draft layer only when the
user has not already created/edited draft markups.
"""

import datetime
import uuid

from config import config
from database import get_job, supabase_request, update_job
from database.repositories.page_repository import get_pages_by_job
from services import detection_refinement_service as refinement
from utils.detection_classes import normalize_detection_class
from utils.scale import get_safe_scale_ratio


REFINED_TABLE = 'extraction_detections_refined'
RUNS_TABLE = 'extraction_refinement_runs'


def refine_job_detections(job_id, mode='auto', page_ids=None, set_job_status=True):
    """
    Refine all eligible detection pages for a job before the editor opens.

    Returns a serializable summary. It never raises for model/storage failures;
    failed pages fall back to raw Roboflow draft seeding.
    """
    job = get_job(job_id)
    if not job:
        return {"success": False, "error": "Job not found", "job_id": job_id}

    previous_status = job.get('status')
    if set_job_status:
        update_job(job_id, {'status': 'refining', 'error_message': None})

    requested_page_ids = set(page_ids or [])
    detection_page_types = set(config.ROBOFLOW_PAGE_TYPES or ['elevation'])
    pages = get_pages_by_job(job_id, status=None)
    target_pages = [
        p for p in pages
        if (p.get('page_type') or '').lower() in detection_page_types
        and (not requested_page_ids or p.get('id') in requested_page_ids)
    ]

    summary = {
        "success": True,
        "job_id": job_id,
        "mode": mode,
        "page_count": len(target_pages),
        "pages_processed": 0,
        "pages_refined": 0,
        "pages_seeded_from_refined": 0,
        "pages_seeded_from_raw": 0,
        "pages_skipped_for_existing_draft": 0,
        "pages_failed": 0,
        "applied_actions": 0,
        "blocked_actions": 0,
        "review_flags": 0,
        "storage_available": True,
        "openai_available": bool(config.OPENAI_API_KEY),
        "page_results": [],
    }

    for page in target_pages:
        result = refine_page_to_refined_layer(page, mode=mode)
        summary["page_results"].append(result)
        summary["pages_processed"] += 1
        if result.get('status') == 'complete':
            summary["pages_refined"] += 1
        if result.get('draft_seed_source') == 'refined':
            summary["pages_seeded_from_refined"] += 1
        if result.get('draft_seed_source') == 'raw':
            summary["pages_seeded_from_raw"] += 1
        if result.get('skipped_for_existing_draft'):
            summary["pages_skipped_for_existing_draft"] += 1
        if result.get('status') in {'failed', 'storage_failed'}:
            summary["pages_failed"] += 1
        if result.get('storage_available') is False:
            summary["storage_available"] = False
        summary["applied_actions"] += result.get('applied_actions', 0)
        summary["blocked_actions"] += result.get('blocked_actions', 0)
        summary["review_flags"] += result.get('review_flags', 0)

    if set_job_status:
        final_status = 'complete' if previous_status != 'failed' else previous_status
        update_job(job_id, {
            'status': final_status,
            'error_message': None if summary["pages_failed"] == 0 else 'Some AI refinement pages fell back to raw detections.',
        })

    return summary


def refine_page_to_refined_layer(page, mode='auto'):
    page_id = page.get('id')
    job_id = page.get('job_id')
    raw_rows = _get_raw_detection_rows(page_id)
    draft_exists = _draft_exists(page_id)
    run_id, run_persisted = _create_run(job_id, page_id, mode, raw_rows)

    base = {
        "page_id": page_id,
        "page_number": page.get('page_number'),
        "status": "pending",
        "run_id": run_id,
        "run_persisted": run_persisted,
        "raw_detection_count": len(raw_rows),
        "refined_detection_count": 0,
        "draft_seeded": 0,
        "draft_seed_source": None,
        "skipped_for_existing_draft": False,
        "storage_available": run_persisted,
        "applied_actions": 0,
        "blocked_actions": 0,
        "review_flags": 0,
    }

    if not raw_rows:
        _finish_run(run_id, run_persisted, 'skipped', error_message='No raw Roboflow detections found')
        return {**base, "status": "skipped", "reason": "No raw Roboflow detections found"}

    if draft_exists and mode == 'auto':
        _finish_run(run_id, run_persisted, 'skipped', error_message='Draft detections already exist')
        return {
            **base,
            "status": "skipped",
            "skipped_for_existing_draft": True,
            "reason": "Draft detections already exist; auto refinement will not overwrite user work",
        }

    if not config.OPENAI_API_KEY:
        seeded = 0 if draft_exists else _seed_draft_from_rows(job_id, page_id, raw_rows, source='raw')
        _finish_run(
            run_id,
            run_persisted,
            'skipped',
            actions_summary={"draft_seeded": seeded, "seed_source": "raw"},
            error_message='OPENAI_API_KEY is not configured; seeded raw Roboflow detections',
        )
        return {
            **base,
            "status": "skipped",
            "draft_seeded": seeded,
            "draft_seed_source": "raw" if seeded else None,
            "reason": "OPENAI_API_KEY is not configured",
        }

    preview, status_code = refinement.preview_refinement_for_detections(
        page=page,
        detections=raw_rows,
        requested_classes=config.REFINEMENT_AUTO_CLASSES,
    )

    if status_code != 200 or not preview.get('success'):
        seeded = 0 if draft_exists else _seed_draft_from_rows(job_id, page_id, raw_rows, source='raw')
        error_message = preview.get('error') or f"Refinement preview failed with status {status_code}"
        _finish_run(
            run_id,
            run_persisted,
            'failed',
            input_summary={"raw_detection_count": len(raw_rows)},
            actions_summary={"draft_seeded": seeded, "seed_source": "raw"},
            error_message=error_message,
        )
        return {
            **base,
            "status": "failed",
            "draft_seeded": seeded,
            "draft_seed_source": "raw" if seeded else None,
            "error": error_message,
        }

    actions = preview.get('actions', [])
    refined_rows = build_refined_rows_from_actions(
        page,
        raw_rows,
        actions,
        run_id if run_persisted else None,
    )
    wrote_refined = _replace_refined_rows(page_id, refined_rows)
    storage_available = bool(wrote_refined)

    safe_actions = [a for a in actions if a.get('status') == 'valid' and a.get('type') != 'keep']
    blocked_actions = [a for a in actions if a.get('status') == 'blocked']
    review_flags = [a for a in actions if a.get('status') == 'valid' and a.get('type') == 'flag_review']

    if wrote_refined:
        seeded = 0 if draft_exists else _seed_draft_from_rows(job_id, page_id, refined_rows, source='refined')
        _finish_run(
            run_id,
            run_persisted,
            'complete',
            input_summary={
                "raw_detection_count": len(raw_rows),
                "candidate_detection_count": preview.get('candidate_detection_count'),
                "context_pages": preview.get('context_pages'),
            },
            actions_summary={
                "applied": len(safe_actions),
                "blocked": len(blocked_actions),
                "review_flags": len(review_flags),
                "draft_seeded": seeded,
                "seed_source": "refined" if seeded else None,
            },
            blocked_summary={"actions": blocked_actions[:20]},
        )
        return {
            **base,
            "status": "complete",
            "storage_available": True,
            "refined_detection_count": len([r for r in refined_rows if not r.get('is_deleted')]),
            "draft_seeded": seeded,
            "draft_seed_source": "refined" if seeded else None,
            "applied_actions": len(safe_actions),
            "blocked_actions": len(blocked_actions),
            "review_flags": len(review_flags),
            "model_confidence": preview.get('model_confidence'),
        }

    seeded = 0 if draft_exists else _seed_draft_from_rows(job_id, page_id, raw_rows, source='raw')
    _finish_run(
        run_id,
        run_persisted,
        'failed',
        actions_summary={"draft_seeded": seeded, "seed_source": "raw"},
        error_message='Could not write refined detection rows; seeded raw detections',
    )
    return {
        **base,
        "status": "storage_failed",
        "storage_available": storage_available,
        "draft_seeded": seeded,
        "draft_seed_source": "raw" if seeded else None,
        "error": "Could not write refined detection rows",
    }


def build_refined_rows_from_actions(page, source_rows, actions, refinement_run_id):
    scale_ratio = get_safe_scale_ratio(page.get('scale_ratio'), context=f"refined rows page {page.get('id')}")
    rows_by_source = {}
    rows = []
    for source in source_rows:
        row = _base_refined_row_from_source(page, source, refinement_run_id)
        rows_by_source[source.get('id')] = row
        rows.append(row)

    next_index = max([int(r.get('detection_index') or 0) for r in rows] or [0]) + 1

    for action in actions:
        if action.get('status') != 'valid':
            continue
        action_type = action.get('type')
        source_ids = _action_detection_ids(action)
        reason = (action.get('reason') or '').strip()

        if action_type == 'keep':
            continue

        if action_type == 'flag_review':
            for source_id in source_ids:
                row = rows_by_source.get(source_id)
                if row:
                    row['needs_review'] = True
                    row['reason'] = reason
                    row['material_notes'] = _append_note(row.get('material_notes'), f"AI review: {reason}")
            continue

        if action_type == 'delete':
            for source_id in source_ids:
                row = rows_by_source.get(source_id)
                if row:
                    row.update({
                        'is_deleted': True,
                        'status': 'deleted',
                        'reason': reason,
                        'provenance': _action_provenance(action),
                    })
            continue

        if action_type in {'resize', 'reclassify', 'convert_to_polygon'}:
            source_id = action.get('detection_id')
            row = rows_by_source.get(source_id)
            if not row:
                continue
            updates = dict(action.get('updates') or {})
            if action_type == 'reclassify' and action.get('new_class'):
                updates['class'] = action.get('new_class')
            row.update(updates)
            row.update({
                'confidence': action.get('confidence') or row.get('confidence'),
                'is_triangle': (updates.get('class') or row.get('class')) == 'gable',
                'reason': reason,
                'provenance': _action_provenance(action),
                'needs_review': False,
            })
            continue

        if action_type == 'merge':
            for source_id in source_ids:
                row = rows_by_source.get(source_id)
                if row:
                    row.update({
                        'is_deleted': True,
                        'status': 'deleted',
                        'reason': f"Merged into refined detection by AI: {reason}",
                        'provenance': _action_provenance(action),
                    })
            merged = _new_refined_row_from_action(
                page,
                action,
                refinement_run_id,
                next_index,
                source_detection_ids=source_ids,
            )
            next_index += 1
            rows.append(merged)
            continue

        if action_type == 'add':
            added = _new_refined_row_from_action(
                page,
                action,
                refinement_run_id,
                next_index,
                source_detection_ids=[],
            )
            next_index += 1
            rows.append(added)

    _attach_plane_metadata(rows)
    return rows


def _get_raw_detection_rows(page_id):
    return supabase_request('GET', 'extraction_detection_details', filters={
        'page_id': f'eq.{page_id}',
        'status': 'neq.deleted',
        'order': 'detection_index.asc',
    }) or []


def _draft_exists(page_id):
    rows = supabase_request('GET', 'extraction_detections_draft', filters={
        'page_id': f'eq.{page_id}',
        'select': 'id',
        'limit': '1',
    }) or []
    return bool(rows)


def _create_run(job_id, page_id, mode, raw_rows):
    local_id = str(uuid.uuid4())
    record = {
        'id': local_id,
        'job_id': job_id,
        'page_id': page_id,
        'status': 'running',
        'mode': mode,
        'model': config.OPENAI_REFINEMENT_MODEL,
        'prompt_version': refinement.REFINEMENT_PROMPT_VERSION,
        'input_summary': {
            'raw_detection_count': len(raw_rows),
            'auto_classes': config.REFINEMENT_AUTO_CLASSES,
        },
        'started_at': _now_iso(),
    }
    result = supabase_request('POST', RUNS_TABLE, record)
    if not result:
        return local_id, False
    return (result[0] if isinstance(result, list) else result).get('id', local_id), True


def _finish_run(run_id, persisted, status, input_summary=None, actions_summary=None, blocked_summary=None, error_message=None):
    if not persisted:
        return
    updates = {
        'status': status,
        'completed_at': _now_iso(),
    }
    if input_summary is not None:
        updates['input_summary'] = input_summary
    if actions_summary is not None:
        updates['actions_summary'] = actions_summary
    if blocked_summary is not None:
        updates['blocked_summary'] = blocked_summary
    if error_message:
        updates['error_message'] = error_message
    supabase_request('PATCH', RUNS_TABLE, updates, {'id': f'eq.{run_id}'})


def _replace_refined_rows(page_id, rows):
    supabase_request('DELETE', REFINED_TABLE, filters={'page_id': f'eq.{page_id}'})
    if not rows:
        return True

    for i in range(0, len(rows), 50):
        batch = rows[i:i + 50]
        result = supabase_request('POST', REFINED_TABLE, batch)
        if not result:
            return False
    return True


def _seed_draft_from_rows(job_id, page_id, rows, source):
    if _draft_exists(page_id):
        return 0

    draft_rows = []
    for row in rows:
        if row.get('is_deleted') or row.get('status') == 'deleted':
            continue
        draft_rows.append(_draft_row_from_source(job_id, page_id, row, source))

    inserted = 0
    for i in range(0, len(draft_rows), 50):
        batch = draft_rows[i:i + 50]
        result = supabase_request('POST', 'extraction_detections_draft', batch)
        if result:
            inserted += len(result) if isinstance(result, list) else len(batch)
    return inserted


def _draft_row_from_source(job_id, page_id, row, source):
    reason = row.get('reason')
    notes = row.get('material_notes')
    if source == 'refined':
        notes = _append_note(notes, f"AI refined: {reason}" if reason else "AI refined before editor review")
    return {
        'id': str(uuid.uuid4()),
        'job_id': job_id,
        'page_id': page_id,
        'source_detection_id': row.get('source_detection_id') or row.get('id'),
        'class': row.get('class'),
        'pixel_x': row.get('pixel_x'),
        'pixel_y': row.get('pixel_y'),
        'pixel_width': row.get('pixel_width'),
        'pixel_height': row.get('pixel_height'),
        'confidence': row.get('confidence'),
        'detection_index': row.get('detection_index'),
        'matched_tag': row.get('matched_tag'),
        'is_triangle': row.get('is_triangle') or False,
        'assigned_material_id': None,
        'material_notes': notes,
        'is_deleted': False,
        'is_user_created': False,
        'polygon_points': row.get('polygon_points'),
        'markup_type': row.get('markup_type') or 'polygon',
        'status': 'auto',
        'area_sf': row.get('area_sf'),
        'perimeter_lf': row.get('perimeter_lf'),
        'item_count': row.get('item_count') or 1,
    }


def _base_refined_row_from_source(page, source, refinement_run_id):
    source_id = source.get('id')
    source_class = normalize_detection_class(source.get('class')) or source.get('class')
    return {
        'id': str(uuid.uuid4()),
        'job_id': page.get('job_id'),
        'page_id': page.get('id'),
        'source_detection_id': source_id,
        'source_detection_ids': [source_id] if source_id else [],
        'refinement_run_id': refinement_run_id,
        'class': source_class,
        'detection_index': source.get('detection_index'),
        'confidence': source.get('confidence'),
        'pixel_x': source.get('pixel_x'),
        'pixel_y': source.get('pixel_y'),
        'pixel_width': source.get('pixel_width'),
        'pixel_height': source.get('pixel_height'),
        'real_width_in': source.get('real_width_in'),
        'real_height_in': source.get('real_height_in'),
        'real_width_ft': source.get('real_width_ft'),
        'real_height_ft': source.get('real_height_ft'),
        'area_sf': source.get('area_sf'),
        'perimeter_lf': source.get('perimeter_lf'),
        'is_triangle': source.get('is_triangle') or source_class == 'gable',
        'matched_tag': source.get('matched_tag'),
        'assigned_material_id': None,
        'material_notes': source.get('material_notes'),
        'is_deleted': False,
        'is_user_created': False,
        'polygon_points': source.get('polygon_points'),
        'has_hole': source.get('has_hole') or False,
        'markup_type': source.get('markup_type') or 'polygon',
        'status': source.get('status') or 'auto',
        'item_count': source.get('item_count') or 1,
        'plane_type': None,
        'plane_id': None,
        'provenance': {'source': 'roboflow', 'stage': 'raw_passthrough'},
        'reason': None,
        'needs_review': False,
    }


def _new_refined_row_from_action(page, action, refinement_run_id, detection_index, source_detection_ids):
    target_class = action.get('target_class') or action.get('new_class')
    updates = dict(action.get('updates') or {})
    row = {
        'id': str(uuid.uuid4()),
        'job_id': page.get('job_id'),
        'page_id': page.get('id'),
        'source_detection_id': source_detection_ids[0] if len(source_detection_ids) == 1 else None,
        'source_detection_ids': source_detection_ids,
        'refinement_run_id': refinement_run_id,
        'class': target_class,
        'detection_index': detection_index,
        'confidence': action.get('confidence') or 0.5,
        'matched_tag': None,
        'assigned_material_id': None,
        'material_notes': None,
        'is_deleted': False,
        'is_user_created': False,
        'has_hole': False,
        'markup_type': 'polygon',
        'status': 'auto',
        'item_count': 1,
        'plane_type': None,
        'plane_id': None,
        'provenance': _action_provenance(action),
        'reason': (action.get('reason') or '').strip(),
        'needs_review': False,
        **updates,
    }
    row['is_triangle'] = row.get('class') == 'gable'
    return row


def _attach_plane_metadata(rows):
    records = []
    by_id = {row.get('id'): row for row in rows}
    for row in rows:
        if row.get('is_deleted'):
            continue
        record = refinement._facade_record_from_detection(row)
        if record:
            records.append(record)
    for record in refinement._assign_wall_plane_info(records):
        row = by_id.get((record.get('detection') or {}).get('id'))
        if row:
            row['plane_type'] = record.get('plane_type')
            row['plane_id'] = record.get('plane_id')


def _action_detection_ids(action):
    ids = []
    if action.get('detection_id'):
        ids.append(action['detection_id'])
    ids.extend(action.get('detection_ids') or [])
    return list(dict.fromkeys(ids))


def _action_provenance(action):
    return {
        'source': 'openai_refinement',
        'action_type': action.get('type'),
        'confidence': action.get('confidence'),
    }


def _append_note(existing, note):
    existing = (existing or '').strip()
    note = (note or '').strip()
    if not note:
        return existing or None
    if not existing:
        return note[:1000]
    if note in existing:
        return existing[:1000]
    return f"{existing}\n{note}"[:1000]


def _now_iso():
    return datetime.datetime.utcnow().isoformat()
