"""Canonical extraction job states shared by repository validation and tests."""


JOB_STATUSES = frozenset({
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


ALLOWED_TRANSITIONS = {
    'pending': {'importing', 'converting', 'analyzing', 'classifying', 'failed'},
    'importing': {'complete', 'failed'},
    'converting': {'analyzing', 'classifying', 'failed'},
    'analyzing': {'classified', 'failed'},
    'classifying': {'classified', 'failed'},
    'classified': {'processing', 'refining', 'failed'},
    'processing': {'refining', 'complete', 'failed'},
    'refining': {'classified', 'complete', 'failed'},
    'complete': {'refining', 'approved', 'failed'},
    'approved': set(),
    'failed': {'pending', 'importing', 'converting', 'classifying', 'processing', 'refining'},
}


def validate_job_status(status):
    if status not in JOB_STATUSES:
        raise ValueError(f"Unknown extraction job status: {status}")
    return status


def can_transition(current, next_status):
    validate_job_status(current)
    validate_job_status(next_status)
    return current == next_status or next_status in ALLOWED_TRANSITIONS[current]
