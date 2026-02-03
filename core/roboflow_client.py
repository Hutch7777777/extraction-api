"""
Roboflow object detection client with retry logic for serverless resilience
"""

import time
import requests
from config import config

# Retry configuration
MAX_RETRIES = 3
RETRY_STATUS_CODES = {500, 502, 503, 504}  # Server errors worth retrying


def detect_objects(image_url):
    """
    Run Roboflow detection on an image with automatic retry on failures.

    Args:
        image_url: Public URL to the image

    Returns:
        Dict with 'predictions' list or 'error' string

    Retry behavior:
        - Retries up to 3 times on 5xx errors or timeouts
        - Uses exponential backoff: 1s, 2s, 4s between retries
        - Logs each retry attempt for debugging
    """
    payload = {
        "api_key": config.ROBOFLOW_API_KEY,
        "inputs": {
            "image": {
                "type": "url",
                "value": image_url
            }
        }
    }

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                config.ROBOFLOW_WORKFLOW_URL,
                json=payload,
                timeout=120
            )

            # Success - parse and return predictions
            if response.status_code == 200:
                if attempt > 0:
                    print(f"[Roboflow] Succeeded on attempt {attempt + 1}", flush=True)

                result = response.json()
                predictions = []

                # Extract predictions from nested structure
                if 'outputs' in result and len(result['outputs']) > 0:
                    output = result['outputs'][0]
                    if 'predictions' in output:
                        pred_data = output['predictions']
                        if isinstance(pred_data, dict) and 'predictions' in pred_data:
                            predictions = pred_data['predictions']
                        elif isinstance(pred_data, list):
                            predictions = pred_data

                return {"predictions": predictions}

            # Retryable server error
            if response.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(
                    f"[Roboflow] {response.status_code} error on attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"retrying in {wait_time}s... (image: {image_url[:60]}...)",
                    flush=True
                )
                time.sleep(wait_time)
                last_error = f"Roboflow error: {response.status_code}"
                continue

            # Non-retryable error or max retries reached
            error_msg = f"Roboflow error: {response.status_code}"
            if attempt > 0:
                error_msg += f" (after {attempt + 1} attempts)"
            print(f"[Roboflow] FAILED: {error_msg}", flush=True)
            return {"error": error_msg}

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(
                    f"[Roboflow] Timeout on attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"retrying in {wait_time}s...",
                    flush=True
                )
                time.sleep(wait_time)
                last_error = "Roboflow timeout"
                continue

            print(f"[Roboflow] FAILED: Timeout after {MAX_RETRIES} attempts", flush=True)
            return {"error": f"Roboflow timeout after {MAX_RETRIES} attempts"}

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(
                    f"[Roboflow] Connection error on attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"retrying in {wait_time}s...",
                    flush=True
                )
                time.sleep(wait_time)
                last_error = f"Connection error: {str(e)}"
                continue

            print(f"[Roboflow] FAILED: Connection error after {MAX_RETRIES} attempts", flush=True)
            return {"error": f"Connection failed after {MAX_RETRIES} attempts"}

        except Exception as e:
            # Unexpected errors - don't retry
            print(f"[Roboflow] FAILED: Unexpected error: {str(e)}", flush=True)
            return {"error": str(e)}

    return {"error": last_error or "Max retries exceeded"}


def detect_with_roboflow(image_url):
    """
    Alias for detect_objects for backward compatibility.
    """
    return detect_objects(image_url)
