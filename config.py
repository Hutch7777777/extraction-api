"""
Centralized configuration for Extraction API v4.0
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""

    @staticmethod
    def _parse_csv(value, default=None):
        source = value if value is not None else default
        if not source:
            return []
        return [item.strip().lower() for item in source.split(',') if item.strip()]
    
    # Flask
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    PORT = int(os.getenv('PORT', 5050))

    # Inbound API authentication — when set, all requests (except /health
    # and CORS preflight OPTIONS) must send a matching X-API-Key header.
    # When unset, the API runs unauthenticated (a startup warning is logged).
    EXTRACTION_API_KEY = (os.getenv('EXTRACTION_API_KEY') or '').strip() or None
    EXTRACTION_API_SIGNING_SECRET = (
        os.getenv('EXTRACTION_API_SIGNING_SECRET') or ''
    ).strip() or None
    EXTRACTION_SIGNED_REQUEST_MAX_AGE_SECONDS = int(
        os.getenv('EXTRACTION_SIGNED_REQUEST_MAX_AGE_SECONDS', '300')
    )
    _running_on_railway = bool(
        os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('RAILWAY_ENVIRONMENT_ID')
    )
    EXTRACTION_REQUIRE_SIGNED_REQUESTS = os.getenv(
        'EXTRACTION_REQUIRE_SIGNED_REQUESTS',
        'true' if _running_on_railway else 'false'
    ).lower() == 'true'
    
    # External APIs
    ROBOFLOW_API_KEY = (os.getenv('ROBOFLOW_API_KEY') or '').strip() or None
    ANTHROPIC_API_KEY = (os.getenv('ANTHROPIC_API_KEY') or '').strip() or None
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL') or os.getenv('ANTHROPIC_MODEL') or 'claude-sonnet-4-6'
    OPENAI_API_KEY = (os.getenv('OPENAI_API_KEY') or '').strip() or None
    OPENAI_REFINEMENT_MODEL = (
        os.getenv('OPENAI_REFINEMENT_MODEL')
        or os.getenv('OPENAI_VISION_MODEL')
        or 'gpt-5.5'
    )
    OPENAI_REFINEMENT_MAX_OUTPUT_TOKENS = int(os.getenv('OPENAI_REFINEMENT_MAX_OUTPUT_TOKENS', '12000'))
    REFINEMENT_MAX_FACADE_OVERLAP_RATIO = float(os.getenv('REFINEMENT_MAX_FACADE_OVERLAP_RATIO', '0.03'))
    REFINEMENT_CONTEXT_PAGE_TYPES = _parse_csv.__func__(
        os.getenv('REFINEMENT_CONTEXT_PAGE_TYPES'),
        'roof_plan,floor_plan'
    )
    REFINEMENT_MAX_CONTEXT_PAGES = int(os.getenv('REFINEMENT_MAX_CONTEXT_PAGES', '4'))
    REFINEMENT_AUTO_ENABLED = os.getenv('REFINEMENT_AUTO_ENABLED', 'true').lower() == 'true'
    REFINEMENT_AUTO_CLASSES = _parse_csv.__func__(
        os.getenv('REFINEMENT_AUTO_CLASSES'),
        'exterior_wall,gable'
    )
    REFINEMENT_ENABLE_RETURN_PLANE_EXPANSION = os.getenv(
        'REFINEMENT_ENABLE_RETURN_PLANE_EXPANSION',
        'false'
    ).lower() == 'true'
    ROBOFLOW_INFERENCE_MODE = (os.getenv('ROBOFLOW_INFERENCE_MODE') or 'workflow').strip().lower()
    ROBOFLOW_SERVERLESS_URL = (os.getenv('ROBOFLOW_SERVERLESS_URL') or 'https://serverless.roboflow.com').strip().rstrip('/')
    ROBOFLOW_MODEL_ID = (
        (os.getenv('ROBOFLOW_MODEL_ID') or os.getenv('ROBOFLOW_MODEL') or '').strip()
        or None
    )
    ROBOFLOW_WORKFLOW_URL = (
        (os.getenv('ROBOFLOW_WORKFLOW_URL') or '').strip()
        or "https://serverless.roboflow.com/infer/workflows/exterior-finishes/find-windows-garages-exterior-walls-roofs-buildings-doors-and-gables"
    )
    ROBOFLOW_CONFIDENCE = float(os.getenv('ROBOFLOW_CONFIDENCE', '0.40'))
    ROBOFLOW_OVERLAP = float(os.getenv('ROBOFLOW_OVERLAP', '0.30'))
    ROBOFLOW_MAX_DETECTIONS = int(os.getenv('ROBOFLOW_MAX_DETECTIONS', '300'))
    ROBOFLOW_PAGE_TYPES = _parse_csv.__func__(os.getenv('ROBOFLOW_PAGE_TYPES'), 'elevation')
    ROBOFLOW_ALLOWED_CLASSES = _parse_csv.__func__(
        os.getenv('ROBOFLOW_ALLOWED_CLASSES'),
        'window,door,garage,building,exterior_wall,roof,gable'
    )
    
    # Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://okwtyttfqbfmcqtenize.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    
    # Processing
    MAX_CONCURRENT_CLAUDE = 3
    BATCH_DELAY_SECONDS = 0.5
    PDF_CHUNK_SIZE = 5
    DEFAULT_DPI = 200
    DEFAULT_SCALE_RATIO = 48  # 1/4"=1' — single source of truth fallback
    
    # CORS
    CORS_ORIGINS = ['http://localhost:3000', 'https://*.vercel.app']
    
    # Valid page types
    VALID_PAGE_TYPES = {'elevation', 'schedule', 'floor_plan', 'roof_plan', 'section', 'detail', 'cover', 'site_plan', 'other', 'unknown', 'review_needed'}

    # Classification confidence threshold - below this, page is marked for review
    PAGE_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.7
    
    # Markup colors (RGB)
    MARKUP_COLORS = {
        'window': (0, 120, 255),
        'door': (255, 140, 0),
        'garage': (148, 0, 211),
        'building': (34, 139, 34),
        'exterior_wall': (34, 139, 34),
        'roof': (220, 20, 60),
        'gable': (255, 105, 180),
        'gutter': (0, 255, 255),
    }
    
    # Trade groups for filtering
    TRADE_GROUPS = {
        'siding': ['building', 'exterior_wall', 'window', 'door', 'garage'],
        'roofing': ['roof', 'gable'],
        'windows': ['window'],
        'doors': ['door', 'garage'],
        'gutters': ['roof'],
        'all': ['window', 'door', 'garage', 'building', 'exterior_wall', 'roof', 'gable']
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
