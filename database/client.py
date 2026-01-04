"""
Supabase database client
"""

import requests
from config import config


def supabase_request(method, endpoint, data=None, filters=None):
    """
    Generic Supabase REST API request handler.
    
    Args:
        method: HTTP method (GET, POST, PATCH, DELETE)
        endpoint: Table name
        data: Request body for POST/PATCH
        filters: Query parameters
    
    Returns:
        Response JSON or None on error
    """
    url = f"{config.SUPABASE_URL}/rest/v1/{endpoint}"
    
    if filters:
        filter_parts = [f"{k}={v}" for k, v in filters.items()]
        url += "?" + "&".join(filter_parts)
    
    headers = {
        'Authorization': f'Bearer {config.SUPABASE_KEY}',
        'apikey': config.SUPABASE_KEY,
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        elif method == 'PATCH':
            response = requests.patch(url, headers=headers, json=data)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers)
        else:
            return None
        
        if response.status_code >= 400:
            print(f"Supabase {method} error: {response.status_code} - {response.text}", flush=True)
            return None
        
        return response.json() if response.content else []
    
    except Exception as e:
        print(f"Supabase error: {e}", flush=True)
        return None
