"""
Acceptance test: dry-run recalc payload == import-time payload for an
untouched job (MN568, job 240e222e).

The fixture is the job's real draft geometry (93 rows, no PII) plus the
known-good March 5 approve payload values recovered from
cad_hover_measurements. One intentional deviation: corners come from
extraction_job_totals (8/8 @ 397.36 LF) because the draft has zero
corner rows and the approve path lost them (sent 0/0).

Tolerances: ±0.5 SF/LF on derived measurements.

Run with:  python3 -m unittest discover tests -v
"""

import json
import os
import unittest
from unittest.mock import patch

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'mn568_draft.json')


class TestMN568RecalcParity(unittest.TestCase):
    """aggregate_detections_for_recalc over real MN568 draft geometry."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH) as f:
            cls.fx = json.load(f)
        cls.expected = cls.fx['expected']

    def fake_supabase(self, method, endpoint, data=None, filters=None):
        if endpoint == 'extraction_detections_draft':
            return list(self.fx['detections'])
        if endpoint == 'extraction_pages':
            return list(self.fx['pages'])
        if endpoint == 'extraction_jobs':
            return [dict(self.fx['job'])]
        if endpoint == 'projects':
            return [dict(self.fx['project'])]
        if endpoint == 'extraction_job_totals':
            return [dict(self.fx['job_totals'])]
        return []

    def build_payload(self):
        from services import bluebeam_import_service as svc
        with patch.object(svc, 'supabase_request', side_effect=self.fake_supabase), \
             patch.object(svc, '_get_product_selections', return_value={}):
            return svc.aggregate_detections_for_recalc(self.fx['job_id'])

    def assertClose(self, actual, expected, label, tol=0.5):
        self.assertAlmostEqual(
            actual, expected, delta=tol,
            msg=f"{label}: got {actual}, known-good {expected} (tol ±{tol})")

    def test_facade_matches_known_good(self):
        payload = self.build_payload()
        # Known-good gross = exterior wall class ONLY (building rows are
        # overlapping page-level outlines and must be excluded)
        self.assertClose(payload['facade']['gross_area_sf'],
                         self.expected['facade_gross_area_sf'], 'gross_area_sf')
        self.assertClose(payload['facade']['net_siding_sf'],
                         self.expected['facade_net_siding_sf'], 'net_siding_sf')
        # Documented deviation #2: known-good 260.03 was n8n-derived
        # (net_siding / 9.588 ft wall height), not a sum of markup widths.
        # The recalc sends the geometric sum of wall-section widths; n8n
        # recomputes starter downstream. See fixture starter_note.
        self.assertClose(payload['facade']['level_starter_lf'],
                         self.expected['facade_level_starter_lf'], 'level_starter_lf')

    def test_openings_match_known_good(self):
        payload = self.build_payload()
        w, d, g = payload['windows'], payload['doors'], payload['garages']
        self.assertEqual(w['count'], self.expected['windows_count'])
        self.assertEqual(d['count'], self.expected['doors_count'])
        self.assertEqual(g['count'], self.expected['garages_count'])
        self.assertClose(w['area_sf'] + d['area_sf'] + g['area_sf'],
                         self.expected['openings_area_sf'], 'openings_area_sf')
        # cad_hover mapping: tops = all head LF; sills = window sills;
        # sides = all jamb LF
        self.assertClose(w['head_lf'] + d['head_lf'] + g['head_lf'],
                         self.expected['openings_tops_lf'], 'openings_tops_lf')
        self.assertClose(w['sill_lf'],
                         self.expected['openings_sills_lf'], 'openings_sills_lf')
        self.assertClose(w['jamb_lf'] + d['jamb_lf'] + g['jamb_lf'],
                         self.expected['openings_sides_lf'], 'openings_sides_lf')

    def test_corners_sourced_from_job_totals(self):
        # Intentional deviation from the March payload (which sent 0/0):
        # draft has zero corner rows, so counts AND LF come from
        # extraction_job_totals
        payload = self.build_payload()
        c = payload['corners']
        self.assertEqual(c['outside_count'], self.expected['corners_outside_count'])
        self.assertEqual(c['inside_count'], self.expected['corners_inside_count'])
        self.assertClose(c['outside_lf'], self.expected['corners_outside_lf'], 'outside_lf')
        self.assertClose(c['inside_lf'], self.expected['corners_inside_lf'], 'inside_lf')

    def test_belly_band_carries_total_lf(self):
        # Contract for the n8n/TS side: per-class measurement totals ride
        # along with the counts
        payload = self.build_payload()
        bb = payload['detection_counts'].get('belly_band')
        self.assertIsNotNone(bb, 'belly_band missing from detection_counts')
        self.assertEqual(bb['count'], self.expected['belly_band_count'])
        self.assertClose(bb.get('total_lf', 0.0),
                         self.expected['belly_band_total_lf'], 'belly_band total_lf')

    def test_point_markers_present(self):
        # gable_topout is markup_type='point' but not in the hardcoded
        # POINT_MARKER_CLASSES set — must still appear
        payload = self.build_payload()
        counts = payload['detection_counts']
        self.assertEqual(counts.get('gable_topout', {}).get('count'),
                         self.expected['gable_topout_count'])
        self.assertEqual(counts.get('corbel', {}).get('count'),
                         self.expected['corbel_count'])

    def test_project_fields_populated(self):
        payload = self.build_payload()
        self.assertEqual(payload['client_name'], 'Fixture Client')
        self.assertEqual(payload['address'], '123 Fixture St')
        self.assertEqual(payload['selected_trades'], ['siding'])


if __name__ == '__main__':
    unittest.main()
