"""
Core module exports - external API integrations
"""

from core.roboflow_client import detect_objects, detect_with_roboflow
from core.claude_client import (
    claude_client, ClaudeClient,
    classify_page_with_claude, ocr_schedule_with_claude,
    analyze_floor_plan_corners, extract_elevation_dimensions,
    extract_roof_plan_data, parse_dimension_string
)

__all__ = [
    # Roboflow
    'detect_objects', 'detect_with_roboflow',
    
    # Claude
    'claude_client', 'ClaudeClient',
    'classify_page_with_claude', 'ocr_schedule_with_claude',
    'analyze_floor_plan_corners', 'extract_elevation_dimensions',
    'extract_roof_plan_data',
    
    # Utilities
    'parse_dimension_string'
]
