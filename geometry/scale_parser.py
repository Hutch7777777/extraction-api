"""
Scale notation parsing utilities
"""

import re


def parse_scale_notation(notation):
    """
    Parse architectural scale notation to scale ratio.
    
    Examples:
        "1/4" = 1'-0"" → 48.0
        "1/8" = 1'-0"" → 96.0
        "1" = 1'-0"" → 12.0
        "1:48" → 48.0
    
    Args:
        notation: Scale string from drawing
    
    Returns:
        Scale ratio (real inches per paper inch) or None
    """
    if not notation:
        return None
    
    # Normalize quotes and apostrophes
    notation = notation.strip().upper()
    notation = notation.replace('"', '"').replace('"', '"')
    notation = notation.replace("'", "'").replace("'", "'")
    
    # Pattern 1: Fractional (1/4" = 1'-0")
    frac_pattern = r'(\d+)/(\d+)\s*["\"]?\s*=\s*1\s*[\'\']?\s*-?\s*0?\s*["\"]?'
    match = re.search(frac_pattern, notation)
    if match:
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        paper_inches = numerator / denominator
        return 12 / paper_inches  # 12 inches per foot / paper inches
    
    # Pattern 2: Whole number (1" = 1'-0")
    whole_pattern = r'(\d+)\s*["\"]?\s*=\s*(\d+)\s*[\'\']?\s*-?\s*0?\s*["\"]?'
    match = re.search(whole_pattern, notation)
    if match:
        paper_inches = float(match.group(1))
        real_feet = float(match.group(2))
        return (real_feet * 12) / paper_inches
    
    # Pattern 3: Ratio (1:48)
    ratio_pattern = r'1\s*:\s*(\d+)'
    match = re.search(ratio_pattern, notation)
    if match:
        return float(match.group(1))
    
    return None


# Common scale lookup table
COMMON_SCALES = {
    '1/16" = 1\'-0"': 192.0,
    '1/8" = 1\'-0"': 96.0,
    '3/16" = 1\'-0"': 64.0,
    '1/4" = 1\'-0"': 48.0,
    '3/8" = 1\'-0"': 32.0,
    '1/2" = 1\'-0"': 24.0,
    '3/4" = 1\'-0"': 16.0,
    '1" = 1\'-0"': 12.0,
    '1-1/2" = 1\'-0"': 8.0,
    '3" = 1\'-0"': 4.0,
}
