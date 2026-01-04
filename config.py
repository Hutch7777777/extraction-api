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
    DEFAULT_DPI = 100
    
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


# Singleton instance
config = Config()
