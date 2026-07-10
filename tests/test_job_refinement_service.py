"""
Unit tests for the pre-editor refined detection layer.

Run with: python3 -m unittest tests.test_job_refinement_service -v
"""

import unittest
from unittest.mock import patch

from services import detection_refinement_service as refinement
from services import job_refinement_service
from services.job_refinement_service import build_refined_rows_from_actions


class TestBuildRefinedRowsFromActions(unittest.TestCase):
    def test_valid_actions_build_refined_rows_without_touching_draft(self):
        page = {'id': 'page-1', 'job_id': 'job-1', 'scale_ratio': 10}
        source_rows = [
            {
                'id': '11111111-1111-1111-1111-111111111111',
                'job_id': 'job-1',
                'page_id': 'page-1',
                'class': 'gable',
                'detection_index': 1,
                'confidence': 0.7,
                'pixel_x': 50,
                'pixel_y': 50,
                'pixel_width': 40,
                'pixel_height': 30,
                'area_sf': 6,
                'perimeter_lf': 14,
                'status': 'auto',
            },
            {
                'id': '22222222-2222-2222-2222-222222222222',
                'job_id': 'job-1',
                'page_id': 'page-1',
                'class': 'exterior_wall',
                'detection_index': 2,
                'confidence': 0.6,
                'pixel_x': 130,
                'pixel_y': 60,
                'pixel_width': 40,
                'pixel_height': 40,
                'area_sf': 16,
                'perimeter_lf': 16,
                'status': 'auto',
            },
        ]

        gable_points = [
            {'x': 30, 'y': 70},
            {'x': 50, 'y': 30},
            {'x': 70, 'y': 70},
        ]
        wall_points = [
            {'x': 90, 'y': 40},
            {'x': 125, 'y': 40},
            {'x': 125, 'y': 85},
            {'x': 90, 'y': 85},
        ]
        actions = [
            {
                'type': 'convert_to_polygon',
                'status': 'valid',
                'detection_id': source_rows[0]['id'],
                'detection_ids': [],
                'target_class': 'gable',
                'new_class': None,
                'polygon_points': gable_points,
                'confidence': 0.88,
                'reason': 'Visible triangular gable.',
                'updates': {
                    **refinement._updates_from_polygon(gable_points, 10),
                    'polygon_points': gable_points,
                    'markup_type': 'polygon',
                },
            },
            {
                'type': 'delete',
                'status': 'valid',
                'detection_id': source_rows[1]['id'],
                'detection_ids': [],
                'target_class': 'exterior_wall',
                'new_class': None,
                'polygon_points': [],
                'confidence': 0.8,
                'reason': 'Duplicate wall candidate.',
                'updates': {'is_deleted': True, 'status': 'deleted'},
            },
            {
                'type': 'add',
                'status': 'valid',
                'detection_id': None,
                'detection_ids': [],
                'target_class': 'exterior_wall',
                'new_class': None,
                'polygon_points': wall_points,
                'confidence': 0.81,
                'reason': 'Missed visible siding plane.',
                'updates': {
                    **refinement._updates_from_polygon(wall_points, 10),
                    'polygon_points': wall_points,
                    'markup_type': 'polygon',
                },
            },
        ]

        rows = build_refined_rows_from_actions(page, source_rows, actions, 'run-1')

        converted = next(row for row in rows if row['source_detection_id'] == source_rows[0]['id'])
        deleted = next(row for row in rows if row['source_detection_id'] == source_rows[1]['id'])
        added = next(row for row in rows if row['source_detection_ids'] == [])

        self.assertEqual(converted['polygon_points'], gable_points)
        self.assertEqual(converted['class'], 'gable')
        self.assertEqual(converted['reason'], 'Visible triangular gable.')
        self.assertFalse(converted['is_deleted'])
        self.assertTrue(deleted['is_deleted'])
        self.assertEqual(deleted['status'], 'deleted')
        self.assertEqual(added['class'], 'exterior_wall')
        self.assertEqual(added['polygon_points'], wall_points)
        self.assertEqual(added['reason'], 'Missed visible siding plane.')
        self.assertEqual(added['refinement_run_id'], 'run-1')


class TestRefinePageToRefinedLayer(unittest.TestCase):
    def test_auto_mode_does_not_overwrite_existing_draft(self):
        page = {'id': 'page-1', 'job_id': 'job-1', 'page_number': 8}
        raw_rows = [{'id': '11111111-1111-1111-1111-111111111111', 'class': 'exterior_wall'}]

        with patch.object(job_refinement_service, '_get_raw_detection_rows', return_value=raw_rows), \
             patch.object(job_refinement_service, '_draft_exists', return_value=True), \
             patch.object(job_refinement_service, '_create_run', return_value=('run-1', True)), \
             patch.object(job_refinement_service, '_finish_run') as finish_run, \
             patch.object(job_refinement_service, '_seed_draft_from_rows') as seed_draft:
            result = job_refinement_service.refine_page_to_refined_layer(page, mode='auto')

        self.assertEqual(result['status'], 'skipped')
        self.assertTrue(result['skipped_for_existing_draft'])
        seed_draft.assert_not_called()
        finish_run.assert_called_once()

    def test_missing_openai_key_falls_back_to_raw_draft_seed(self):
        page = {'id': 'page-1', 'job_id': 'job-1', 'page_number': 8}
        raw_rows = [{'id': '11111111-1111-1111-1111-111111111111', 'class': 'exterior_wall'}]

        with patch.object(job_refinement_service.config, 'OPENAI_API_KEY', None), \
             patch.object(job_refinement_service, '_get_raw_detection_rows', return_value=raw_rows), \
             patch.object(job_refinement_service, '_draft_exists', return_value=False), \
             patch.object(job_refinement_service, '_create_run', return_value=('run-1', True)), \
             patch.object(job_refinement_service, '_finish_run') as finish_run, \
             patch.object(job_refinement_service, '_seed_draft_from_rows', return_value=1) as seed_draft:
            result = job_refinement_service.refine_page_to_refined_layer(page, mode='auto')

        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['draft_seeded'], 1)
        self.assertEqual(result['draft_seed_source'], 'raw')
        seed_draft.assert_called_once_with('job-1', 'page-1', raw_rows, source='raw')
        finish_run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
