"""
Centralized configuration for Extraction API v4.0
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # Flask
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    PORT = int(os.getenv('PORT', 5050))
    
    # External APIs
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/infer/workflows/exterior-finishes/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables"
    
    # Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://okwtyttfqbfmcqtenize.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    # Processing
    MAX_CONCURRENT_CLAUDE = 3
    BATCH_DELAY_SECONDS = 0.5
    PDF_CHUNK_SIZE = 5
    DEFAULT_DPI = 200
    
    # CORS
    CORS_ORIGINS = ['http://localhost:3000', 'https://*.vercel.app']
    
    # Valid page types
    VALID_PAGE_TYPES = {'elevation', 'schedule', 'floor_plan', 'roof_plan', 'section', 'detail', 'cover', 'site_plan', 'other'}
    
    # Markup colors (RGB)
    MARKUP_COLORS = {
        'window': (0, 120, 255),
        'door': (255, 140, 0),
        'garage': (148, 0, 211),
        'building': (34, 139, 34),
        'exterior wall': (34, 139, 34),
        'exterior_wall': (34, 139, 34),
        'roof': (220, 20, 60),
        'gable': (255, 105, 180),
        'gutter': (0, 255, 255),
    }
    
    # Trade groups for filtering
    TRADE_GROUPS = {
        'siding': ['building', 'exterior wall', 'window', 'door', 'garage'],
        'roofing': ['roof', 'gable'],
        'windows': ['window'],
        'doors': ['door', 'garage'],
        'gutters': ['roof'],
        'all': ['window', 'door', 'garage', 'building', 'exterior wall', 'roof', 'gable']
    }

    # ==========================================
    # Detection Post-Processing Configuration
    # ==========================================

    # Confidence filtering - drop detections below this threshold
    DETECTION_MIN_CONFIDENCE = float(os.getenv('DETECTION_MIN_CONFIDENCE', '0.40'))

    # IoU-based deduplication - merge same-class detections with IoU above this
    DETECTION_IOU_THRESHOLD = float(os.getenv('DETECTION_IOU_THRESHOLD', '0.45'))

    # Garage merging settings
    GARAGE_MERGE_ENABLED = os.getenv('GARAGE_MERGE_ENABLED', 'true').lower() == 'true'
    GARAGE_MERGE_Y_TOLERANCE_PX = int(os.getenv('GARAGE_MERGE_Y_TOLERANCE_PX', '30'))  # Max Y difference for horizontal alignment
    GARAGE_MIN_COMBINED_WIDTH_PX = int(os.getenv('GARAGE_MIN_COMBINED_WIDTH_PX', '200'))  # Min combined width to classify as garage
    GARAGE_MAX_GAP_PX = int(os.getenv('GARAGE_MAX_GAP_PX', '50'))  # Max horizontal gap between adjacent doors

    # Minimum size filters (pixels) - detections smaller than these are dropped
    DETECTION_MIN_SIZE = {
        'window': {'width': 30, 'height': 30},
        'door': {'width': 25, 'height': 50},
        'garage': {'width': 80, 'height': 50},
        'gable': {'width': 40, 'height': 25},
        'building': {'width': 100, 'height': 100},
        'exterior wall': {'width': 50, 'height': 30},
        'exterior_wall': {'width': 50, 'height': 30},
        'roof': {'width': 80, 'height': 40},
        'default': {'width': 20, 'height': 20}  # Fallback for unknown classes
    }

    # Containment filter - drop smaller detection if fully inside larger same-class detection
    CONTAINMENT_FILTER_ENABLED = os.getenv('CONTAINMENT_FILTER_ENABLED', 'true').lower() == 'true'
    CONTAINMENT_THRESHOLD = float(os.getenv('CONTAINMENT_THRESHOLD', '0.90'))  # 90% overlap = contained

    # Logging verbosity for post-processing
    DETECTION_POSTPROCESS_VERBOSE = os.getenv('DETECTION_POSTPROCESS_VERBOSE', 'true').lower() == 'true'


# Singleton instance
config = Config()
