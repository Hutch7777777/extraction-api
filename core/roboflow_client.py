"""
Roboflow object detection client
"""

import requests
from config import config


def detect_objects(image_url):
    """
    Run Roboflow detection on an image.
    
    Args:
        image_url: Public URL to the image
    
    Returns:
        Dict with 'predictions' list or 'error' string
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
    
    try:
        response = requests.post(
            config.ROBOFLOW_WORKFLOW_URL,
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            return {"error": f"Roboflow error: {response.status_code}"}
        
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
    
    except requests.Timeout:
        return {"error": "Roboflow timeout"}
    except Exception as e:
        return {"error": str(e)}


def detect_with_roboflow(image_url):
    """
    Alias for detect_objects for backward compatibility.
    """
    return detect_objects(image_url)
