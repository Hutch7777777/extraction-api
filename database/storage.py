"""
Supabase storage operations
"""

import requests
from config import config


def upload_to_storage(image_data, filename, content_type='image/jpeg', bucket='extraction-markups'):
    """
    Upload file to Supabase storage.
    
    Args:
        image_data: File bytes
        filename: Path within bucket
        content_type: MIME type
        bucket: Storage bucket name
    
    Returns:
        Public URL or None on error
    """
    if not config.SUPABASE_KEY:
        return None
    
    try:
        upload_url = f"{config.SUPABASE_URL}/storage/v1/object/{bucket}/{filename}"
        headers = {
            'Authorization': f'Bearer {config.SUPABASE_KEY}',
            'Content-Type': content_type,
            'x-upsert': 'true'
        }
        
        response = requests.post(upload_url, headers=headers, data=image_data)
        
        if response.status_code in [200, 201]:
            return f"{config.SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
        
        print(f"Storage upload failed: {response.status_code} - {response.text}", flush=True)
        return None
    
    except Exception as e:
        print(f"Storage upload error: {e}", flush=True)
        return None
