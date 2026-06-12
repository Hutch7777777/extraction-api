"""
Compare a job's freshly aggregated recalc payload against the known-good
approve-path measurements stored in cad_hover_measurements.

Read-only: runs aggregate_detections_for_recalc() directly (GET-only,
no webhook call, no DB writes) and diffs it against the row n8n stored
the last time Approve & Calculate ran.

Usage:
    python3 scripts/compare_recalc_vs_known_good.py <job_id>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import supabase_request  # noqa: E402
from services.bluebeam_import_service import aggregate_detections_for_recalc  # noqa: E402

# (label, payload getter, cad_hover column, note)
ROWS = [
    ('gross facade SF',     lambda p: p['facade']['gross_area_sf'],   'facade_total_sqft', ''),
    ('net siding SF',       lambda p: p['facade']['net_siding_sf'],   'net_siding_sqft', ''),
    ('level starter LF',    lambda p: p['facade']['level_starter_lf'], 'level_starter_lf',
     'DEVIATION #2: known-good is n8n-derived (net/wall height); payload sends geometric sum'),
    ('openings area SF',    lambda p: p['windows']['area_sf'] + p['doors']['area_sf'] + p['garages']['area_sf'],
     'openings_area_sqft', ''),
    ('openings tops LF',    lambda p: p['windows']['head_lf'] + p['doors']['head_lf'] + p['garages']['head_lf'],
     'openings_tops_lf', ''),
    ('openings sills LF',   lambda p: p['windows']['sill_lf'], 'openings_sills_lf', ''),
    ('openings sides LF',   lambda p: p['windows']['jamb_lf'] + p['doors']['jamb_lf'] + p['garages']['jamb_lf'],
     'openings_sides_lf', ''),
    ('openings perim LF',   lambda p: p['windows']['perimeter_lf'] + p['doors']['perimeter_lf'] + p['garages']['perimeter_lf'],
     'openings_total_perimeter_lf', ''),
    ('window count',        lambda p: p['windows']['count'], 'openings_windows_count', ''),
    ('door count',          lambda p: p['doors']['count'], 'openings_doors_count', ''),
    ('outside corners',     lambda p: p['corners']['outside_count'], 'outside_corners_count',
     'DEVIATION #1: payload sources corners from extraction_job_totals; approve path sent 0'),
    ('outside corner LF',   lambda p: p['corners']['outside_lf'], 'outside_corners_lf', 'DEVIATION #1'),
    ('inside corners',      lambda p: p['corners']['inside_count'], 'inside_corners_count', 'DEVIATION #1'),
    ('inside corner LF',    lambda p: p['corners']['inside_lf'], 'inside_corners_lf', 'DEVIATION #1'),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    job_id = sys.argv[1]

    payload = aggregate_detections_for_recalc(job_id)

    known_rows = supabase_request('GET', 'cad_hover_measurements', filters={
        'extraction_id': f'eq.{job_id}', 'order': 'created_at.desc', 'limit': '1'
    }) or []
    if not known_rows:
        print(f"No cad_hover_measurements row for extraction_id={job_id} — "
              f"no known-good payload to compare against.")
        sys.exit(2)
    known = known_rows[0]

    print(f"\nJob {job_id}")
    print(f"Known-good row created {known.get('created_at')} "
          f"(updated {known.get('updated_at')}, source {known.get('source_type')})\n")
    print(f"{'measurement':<20} {'recalc':>10} {'known-good':>11} {'delta':>9}  status")
    print('-' * 78)

    worst = 0.0
    for label, getter, column, note in ROWS:
        ours = float(getter(payload) or 0)
        theirs = float(known.get(column) or 0)
        delta = ours - theirs
        if note.startswith('DEVIATION'):
            status = 'DEVIATION (intentional)'
        elif abs(delta) <= 0.5:
            status = 'OK'
        else:
            status = 'DIFF'
            worst = max(worst, abs(delta))
        print(f"{label:<20} {ours:>10.2f} {theirs:>11.2f} {delta:>+9.2f}  {status}")
        if note:
            print(f"{'':<20}   note: {note}")

    print('-' * 78)
    print("\npayload-only fields (no cad_hover counterpart):")
    for cls, entry in sorted((payload.get('detection_counts') or {}).items()):
        print(f"  {cls}: count={entry.get('count')}, "
              f"total_sf={entry.get('total_sf')}, total_lf={entry.get('total_lf')}")
    print(f"  selected_trades: {payload.get('selected_trades')}")
    print(f"  client_name: {payload.get('client_name')!r}, address: {payload.get('address')!r}")

    sys.exit(0 if worst == 0 else 3)


if __name__ == '__main__':
    main()
