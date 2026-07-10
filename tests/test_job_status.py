import unittest

from utils.job_status import JOB_STATUSES, can_transition, validate_job_status


class JobStatusTests(unittest.TestCase):
    def test_status_vocabulary_matches_web_application(self):
        self.assertEqual(JOB_STATUSES, {
            'pending',
            'importing',
            'converting',
            'analyzing',
            'classifying',
            'classified',
            'processing',
            'refining',
            'complete',
            'approved',
            'failed',
        })

    def test_known_transitions(self):
        self.assertTrue(can_transition('pending', 'converting'))
        self.assertTrue(can_transition('converting', 'classifying'))
        self.assertTrue(can_transition('classified', 'processing'))
        self.assertTrue(can_transition('processing', 'refining'))
        self.assertTrue(can_transition('refining', 'complete'))
        self.assertTrue(can_transition('complete', 'approved'))
        self.assertFalse(can_transition('pending', 'approved'))
        self.assertFalse(can_transition('approved', 'processing'))

    def test_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            validate_job_status('completed')


if __name__ == '__main__':
    unittest.main()
