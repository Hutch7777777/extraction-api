"""
Tests for the shared net-siding formula (audit finding D-1) and for
aggregate_detections_for_recalc() picking up both corner spellings and
deriving linear footage (audit findings T-1 / T-2).

Run with:  python3 -m unittest discover tests -v
"""

import unittest
from unittest.mock import patch

from geometry import calculate_net_siding_sf


class TestCalculateNetSidingSf(unittest.TestCase):
    """Canonical formula: (building - roof) - openings + gables."""

    def test_full_formula(self):
        self.assertEqual(calculate_net_siding_sf(2000, 800, 100, 150), 1250)

    def test_no_gables(self):
        self.assertEqual(calculate_net_siding_sf(2000, 800, 100), 1100)

    def test_none_inputs_treated_as_zero(self):
        self.assertEqual(calculate_net_siding_sf(2000, None, None, None), 2000)


class TestAggregateDetectionsForRecalc(unittest.TestCase):
    """
    End-to-end through the aggregation: mixed corner spellings count,
    Count-markup group totals (item_count) are honored, and LF derives
    from pixel geometry when real_* columns are absent.
    """

    JOB_ID = 'job-1'
    # Page with a known scale: 48 / 200dpi → 0.24 in/px → 0.02 ft/px
    PAGES = [{'id': 'page-1', 'scale_ratio': 48, 'dpi': 200}]
    DETECTIONS = [
        # Fresh-import spelling, Count markup carrying a group total of 6
        {'id': 'd1', 'class': 'corner_outside', 'item_count': 6,
         'markup_type': 'point', 'pixel_width': 16, 'pixel_height': 16,
         'page_id': 'page-1'},
        # Roundtrip-import spelling, single corner
        {'id': 'd2', 'class': 'inside corner', 'page_id': 'page-1',
         'markup_type': 'point', 'pixel_width': 16, 'pixel_height': 16},
        # Bare 'corner' (old import maps Bluebeam 'Corner' subject to this)
        {'id': 'd3', 'class': 'corner', 'page_id': 'page-1',
         'markup_type': 'point', 'pixel_width': 16, 'pixel_height': 16},
        # Window with pixel geometry, no real_* columns:
        # 500px → 10 ft wide, 250px → 5 ft tall
        {'id': 'd4', 'class': 'window', 'area_sf': 50, 'page_id': 'page-1',
         'pixel_width': 500, 'pixel_height': 250, 'markup_type': 'rect'},
        # Building: 1000px → 20 ft of starter
        {'id': 'd5', 'class': 'building', 'area_sf': 2000, 'page_id': 'page-1',
         'pixel_width': 1000, 'pixel_height': 600, 'markup_type': 'rect'},
        {'id': 'd6', 'class': 'roof', 'area_sf': 800, 'page_id': 'page-1'},
        {'id': 'd7', 'class': 'gable', 'area_sf': 150, 'page_id': 'page-1',
         'perimeter_lf': 30},
        # Point-marker Count markup: 10 corbels in one deduped row
        {'id': 'd8', 'class': 'corbel', 'item_count': 10, 'page_id': 'page-1',
         'markup_type': 'point', 'pixel_width': 16, 'pixel_height': 16},
    ]

    def fake_supabase(self, method, endpoint, data=None, filters=None):
        if endpoint == 'extraction_detections_draft':
            return list(self.DETECTIONS)
        if endpoint == 'extraction_pages':
            return list(self.PAGES)
        if endpoint == 'extraction_jobs':
            return [{'id': self.JOB_ID, 'project_id': 'proj-1'}]
        if endpoint == 'projects':
            return [{'id': 'proj-1', 'name': 'Test Project',
                     'organization_id': 'org-1'}]
        return []

    def build_payload(self):
        from services import bluebeam_import_service as svc
        with patch.object(svc, 'supabase_request', side_effect=self.fake_supabase), \
             patch.object(svc, '_get_product_selections', return_value={}):
            return svc.aggregate_detections_for_recalc(self.JOB_ID)

    def test_both_corner_spellings_are_counted(self):
        payload = self.build_payload()
        # d1 (corner_outside, group of 6) + d3 (bare corner) = 7 outside
        self.assertEqual(payload['corners']['outside_count'], 7)
        # d2 ('inside corner') = 1 inside
        self.assertEqual(payload['corners']['inside_count'], 1)

    def test_window_lf_derived_from_pixel_geometry(self):
        payload = self.build_payload()
        self.assertEqual(payload['windows']['count'], 1)
        self.assertEqual(payload['windows']['head_lf'], 10.0)   # width
        self.assertEqual(payload['windows']['sill_lf'], 10.0)   # width
        self.assertEqual(payload['windows']['jamb_lf'], 10.0)   # height * 2

    def test_starter_lf_derived_for_building(self):
        payload = self.build_payload()
        self.assertEqual(payload['facade']['level_starter_lf'], 20.0)

    def test_point_marker_group_total_honored(self):
        payload = self.build_payload()
        self.assertEqual(payload['detection_counts']['corbel']['count'], 10)
        self.assertEqual(payload['total_point_count'], 10)

    def test_net_siding_uses_canonical_formula(self):
        payload = self.build_payload()
        # (2000 building - 800 roof) - 50 window + 150 gable = 1300
        self.assertEqual(payload['facade']['net_siding_sf'], 1300.0)

    def test_corner_lf_warns_instead_of_silent_zero(self):
        # Corner point markers have no derivable height: LF stays 0 but a
        # WARNING naming the detection must be logged (not silence)
        with self.assertLogs('services.detection_normalization', level='WARNING') as captured:
            payload = self.build_payload()
        self.assertEqual(payload['corners']['outside_lf'], 0.0)
        output = '\n'.join(captured.output)
        self.assertIn('[DimensionDerivation] WARNING', output)
        self.assertIn('d1', output)


if __name__ == '__main__':
    unittest.main()
