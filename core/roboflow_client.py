"""
Roboflow object detection client with retry logic for serverless resilience
"""

import time
import requests
from config import config
from utils.detection_classes import normalize_detection_class

# Retry configuration
MAX_RETRIES = 3
RETRY_STATUS_CODES = {500, 502, 503, 504}  # Server errors worth retrying


def _normalize_class_name(value):
    return normalize_detection_class(value)


def _filter_allowed_classes(predictions):
    if not config.ROBOFLOW_ALLOWED_CLASSES:
        return predictions

    allowed = {_normalize_class_name(cls) for cls in config.ROBOFLOW_ALLOWED_CLASSES}
    filtered = []
    for pred in predictions:
        canonical_class = _normalize_class_name(pred.get('class'))
        if canonical_class in allowed:
            filtered.append({**pred, 'class': canonical_class})
    dropped = len(predictions) - len(filtered)
    if dropped:
        print(f"[Roboflow] Filtered {dropped} prediction(s) outside allowed classes", flush=True)
    return filtered


def _extract_predictions(result):
    predictions = []

    if isinstance(result.get('predictions'), list):
        predictions = result['predictions']
    elif 'outputs' in result and len(result['outputs']) > 0:
        output = result['outputs'][0]
        if 'predictions' in output:
            pred_data = output['predictions']
            if isinstance(pred_data, dict) and 'predictions' in pred_data:
                predictions = pred_data['predictions']
            elif isinstance(pred_data, list):
                predictions = pred_data

    return _filter_allowed_classes(predictions)


def _response_error_message(response):
    """Return a useful Roboflow error without leaking credentials."""
    if response.status_code == 401:
        return "Roboflow authorization failed (401). Check ROBOFLOW_API_KEY and workflow access."
    if response.status_code == 403:
        return "Roboflow access denied (403). Check that the API key can access this workflow."

    detail = (response.text or '').strip().replace('\n', ' ')[:200]
    api_key = config.ROBOFLOW_API_KEY
    if api_key and detail:
        detail = detail.replace(api_key, '[redacted]')

    error_msg = f"Roboflow error: {response.status_code}"
    if detail:
        error_msg += f" - {detail}"
    return error_msg


def _post_with_retry(url, *, json=None, params=None, source_label='Roboflow'):
    payload = {
        'json': json,
        'params': params,
    }
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                json=payload['json'],
                params=payload['params'],
                timeout=120
            )

            # Success - parse and return predictions
            if response.status_code == 200:
                if attempt > 0:
                    print(f"[{source_label}] Succeeded on attempt {attempt + 1}", flush=True)

                result = response.json()
                return {"predictions": _extract_predictions(result)}

            # Retryable server error
            if response.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(
                    f"[{source_label}] {response.status_code} error on attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"retrying in {wait_time}s...",
                    flush=True
                )
                time.sleep(wait_time)
                last_error = _response_error_message(response)
                continue

            # Non-retryable error or max retries reached
            error_msg = _response_error_message(response)
            if attempt > 0:
                error_msg += f" (after {attempt + 1} attempts)"
            print(f"[{source_label}] FAILED: {error_msg}", flush=True)
            return {"error": error_msg}

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(
                    f"[{source_label}] Timeout on attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"retrying in {wait_time}s...",
                    flush=True
                )
                time.sleep(wait_time)
                last_error = "Roboflow timeout"
                continue

            print(f"[{source_label}] FAILED: Timeout after {MAX_RETRIES} attempts", flush=True)
            return {"error": f"Roboflow timeout after {MAX_RETRIES} attempts"}

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(
                    f"[{source_label}] Connection error on attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"retrying in {wait_time}s...",
                    flush=True
                )
                time.sleep(wait_time)
                last_error = f"Connection error: {str(e)}"
                continue

            print(f"[{source_label}] FAILED: Connection error after {MAX_RETRIES} attempts", flush=True)
            return {"error": f"Connection failed after {MAX_RETRIES} attempts"}

        except Exception as e:
            # Unexpected errors - don't retry
            print(f"[{source_label}] FAILED: Unexpected error: {str(e)}", flush=True)
            return {"error": str(e)}

    return {"error": last_error or "Max retries exceeded"}


def _detect_with_workflow(image_url):
    if not config.ROBOFLOW_WORKFLOW_URL:
        return {"error": "Roboflow workflow URL is not configured."}

    payload = {
        "api_key": config.ROBOFLOW_API_KEY,
        "inputs": {
            "image": {
                "type": "url",
                "value": image_url
            }
        }
    }
    return _post_with_retry(
        config.ROBOFLOW_WORKFLOW_URL,
        json=payload,
        source_label="Roboflow workflow"
    )


def _model_endpoint(model_id):
    if not model_id or '/' not in model_id:
        return None
    dataset_id, version_id = model_id.rsplit('/', 1)
    if not dataset_id or not version_id:
        return None
    return f"{config.ROBOFLOW_SERVERLESS_URL}/{dataset_id}/{version_id}"


def _detect_with_model(image_url):
    if not config.ROBOFLOW_MODEL_ID:
        return {"error": "Roboflow model mode requires ROBOFLOW_MODEL_ID, e.g. project-name/1."}

    endpoint = _model_endpoint(config.ROBOFLOW_MODEL_ID)
    if not endpoint:
        return {"error": "ROBOFLOW_MODEL_ID must use the format project-name/version, e.g. project-name/1."}

    params = {
        "api_key": config.ROBOFLOW_API_KEY,
        "image": image_url,
        "confidence": config.ROBOFLOW_CONFIDENCE,
        "overlap": config.ROBOFLOW_OVERLAP,
        "max_detections": config.ROBOFLOW_MAX_DETECTIONS,
        "format": "json",
        "disable_active_learning": "true",
    }
    return _post_with_retry(
        endpoint,
        params=params,
        source_label="Roboflow model"
    )


def detect_objects(image_url):
    """
    Run Roboflow detection on an image with automatic retry on failures.

    Supports:
    - workflow: Roboflow Workflow endpoint
    - model: single hosted object-detection model endpoint
    - auto: try workflow, then model if a model is configured
    """
    if not config.ROBOFLOW_API_KEY:
        return {"error": "Roboflow API key is not configured."}

    mode = config.ROBOFLOW_INFERENCE_MODE
    if mode == 'workflow':
        return _detect_with_workflow(image_url)
    if mode == 'model':
        return _detect_with_model(image_url)
    if mode == 'auto':
        workflow_result = _detect_with_workflow(image_url)
        if 'error' not in workflow_result or not config.ROBOFLOW_MODEL_ID:
            return workflow_result
        print("[Roboflow] Workflow failed; trying configured model endpoint", flush=True)
        return _detect_with_model(image_url)

    return {"error": f"Unsupported ROBOFLOW_INFERENCE_MODE '{mode}'. Use workflow, model, or auto."}


def detect_with_roboflow(image_url):
    """
    Alias for detect_objects for backward compatibility.
    """
    return detect_objects(image_url)
