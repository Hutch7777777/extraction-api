"""
Tests for POST /recalculate-job/<job_id> — recalc-only endpoint that
re-runs aggregate_detections_for_recalc() for an existing job without
re-importing a PDF and without touching detections.

Run with:  python3 -m unittest discover tests -v
"""

import unittest
from unittest.mock import patch

import app as app_module


class TestRecalculateJobEndpoint(unittest.TestCase):
    JOB_ID = 'job-123'

    def setUp(self):
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()

    @patch('database.get_job', return_value=None)
    def test_unknown_job_returns_404(self, _get_job):
        resp = self.client.post(f'/recalculate-job/{self.JOB_ID}')
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.get_json()['success'])

    @patch('services.bluebeam_import_service.trigger_recalculation_webhook')
    @patch('services.bluebeam_import_service.aggregate_detections_for_recalc',
           return_value={'job_id': 'job-123', 'facade': {'net_siding_sf': 1300.0}})
    @patch('database.get_job', return_value={'id': 'job-123'})
    def test_dry_run_returns_payload_without_calling_webhook(self, _get_job, _agg, trigger):
        resp = self.client.post(f'/recalculate-job/{self.JOB_ID}?dry_run=true')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body['success'])
        self.assertTrue(body['dry_run'])
        self.assertEqual(body['payload']['facade']['net_siding_sf'], 1300.0)
        trigger.assert_not_called()

    @patch('services.bluebeam_import_service.trigger_recalculation_webhook',
           return_value={'success': True, 'status_code': 200, 'response': {'ok': True}})
    @patch('database.get_job', return_value={'id': 'job-123'})
    def test_recalc_calls_webhook_and_returns_result(self, _get_job, trigger):
        resp = self.client.post(f'/recalculate-job/{self.JOB_ID}')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(body['recalculation']['status_code'], 200)
        trigger.assert_called_once_with(self.JOB_ID)

    @patch('services.bluebeam_import_service.trigger_recalculation_webhook',
           return_value={'success': False, 'error': 'n8n timeout'})
    @patch('database.get_job', return_value={'id': 'job-123'})
    def test_webhook_failure_returns_502(self, _get_job, trigger):
        resp = self.client.post(f'/recalculate-job/{self.JOB_ID}')
        self.assertEqual(resp.status_code, 502)
        body = resp.get_json()
        self.assertFalse(body['success'])
        self.assertEqual(body['recalculation']['error'], 'n8n timeout')


if __name__ == '__main__':
    unittest.main()
