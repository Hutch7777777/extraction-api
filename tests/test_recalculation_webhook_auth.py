import unittest
from unittest.mock import Mock, patch

from config import config
from services.bluebeam_import_service import trigger_recalculation_webhook


class RecalculationWebhookAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.original_url = config.N8N_WEBHOOK_URL
        self.original_secret = config.N8N_WEBHOOK_SECRET
        config.N8N_WEBHOOK_URL = 'https://n8n.example.test'

    def tearDown(self):
        config.N8N_WEBHOOK_URL = self.original_url
        config.N8N_WEBHOOK_SECRET = self.original_secret

    @patch('services.bluebeam_import_service.aggregate_detections_for_recalc')
    @patch('requests.post')
    def test_callback_sends_webhook_secret(self, post, aggregate):
        config.N8N_WEBHOOK_SECRET = 'test-webhook-secret'
        aggregate.return_value = {'job_id': 'job-123'}
        post.return_value = Mock(ok=True, status_code=200, content=b'{}')
        post.return_value.json.return_value = {}

        result = trigger_recalculation_webhook('job-123')

        self.assertTrue(result['success'])
        post.assert_called_once_with(
            'https://n8n.example.test/webhook/approve-detection-editor',
            json={'job_id': 'job-123'},
            headers={'X-Webhook-Secret': 'test-webhook-secret'},
            timeout=120,
        )

    @patch('services.bluebeam_import_service.aggregate_detections_for_recalc')
    @patch('requests.post')
    def test_callback_refuses_to_run_without_secret(self, post, aggregate):
        config.N8N_WEBHOOK_SECRET = None

        result = trigger_recalculation_webhook('job-123')

        self.assertFalse(result['success'])
        self.assertIn('authentication', result['error'])
        aggregate.assert_not_called()
        post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
