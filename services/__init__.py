"""
Services module exports
"""

from services.pdf_service import convert_pdf_background
from services.classification_service import classify_job_background
from services.extraction_service import process_job_background
from services.markup_service import (
    generate_markup_image,
    generate_markups_for_page,
    generate_markups_for_job,
    generate_comprehensive_markup
)
from services.takeoff_service import (
    calculate_takeoff_for_page,
    calculate_takeoff_for_job
)
from services.cross_ref_service import build_cross_references
from services.floor_plan_service import (
    analyze_floor_plan_for_job,
    analyze_single_floor_plan
)
from services.fusion_service import (
    fuse_page_data,
    fuse_job_data,
    get_fusion_summary
)
from services.roof_service import (
    process_roof_plan,
    process_roof_plans_for_job,
    get_roof_summary,
    calculate_pitch_factor,
    calculate_true_area,
    calculate_soffit_area,
    calculate_roofing_squares
)
from services.linear_service import (
    calculate_linear_elements_for_job,
    get_linear_summary,
    set_corner_counts,
    set_wall_height,
    get_wall_heights_from_ocr
)

__all__ = [
    # PDF
    'convert_pdf_background',
    
    # Classification
    'classify_job_background',
    
    # Extraction
    'process_job_background',
    
    # Markup
    'generate_markup_image',
    'generate_markups_for_page',
    'generate_markups_for_job',
    'generate_comprehensive_markup',
    
    # Takeoff
    'calculate_takeoff_for_page',
    'calculate_takeoff_for_job',
    
    # Cross-reference
    'build_cross_references',
    
    # Floor plan
    'analyze_floor_plan_for_job',
    'analyze_single_floor_plan',
    
    # Fusion (Phase 2)
    'fuse_page_data',
    'fuse_job_data',
    'get_fusion_summary',
    
    # Roof Intelligence (Phase 3)
    'process_roof_plan',
    'process_roof_plans_for_job',
    'get_roof_summary',
    'calculate_pitch_factor',
    'calculate_true_area',
    'calculate_soffit_area',
    'calculate_roofing_squares',
    
    # Linear Elements (Phase 4)
    'calculate_linear_elements_for_job',
    'get_linear_summary',
    'set_corner_counts',
    'set_wall_height',
    'get_wall_heights_from_ocr'
]
