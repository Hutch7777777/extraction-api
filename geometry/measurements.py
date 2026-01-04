"""
Measurement calculation utilities
"""


def calculate_derived_measurements(width_in, height_in, qty, element_type='window'):
    """
    Calculate derived measurements for an opening.
    
    Args:
        width_in: Width in inches
        height_in: Height in inches
        qty: Quantity
        element_type: 'window' or 'door'
    
    Returns:
        Dict with calculated measurements
    """
    measurements = {
        'head_trim_lf': round((width_in * qty) / 12, 2),
        'jamb_trim_lf': round((height_in * 2 * qty) / 12, 2),
        'casing_lf': round(((width_in * 2) + (height_in * 2)) * qty / 12, 2),
        'rough_opening_width': width_in + 1,
        'rough_opening_height': height_in + 1,
        'head_flashing_lf': round((width_in + 4) * qty / 12, 2),
        'area_sf': round((width_in * height_in * qty) / 144, 2),
        'perimeter_lf': round(((width_in * 2) + (height_in * 2)) * qty / 12, 2),
    }
    
    if element_type == 'window':
        measurements['sill_trim_lf'] = round((width_in * qty) / 12, 2)
        measurements['sill_pan_lf'] = round((width_in + 4) * qty / 12, 2)
    else:
        measurements['sill_trim_lf'] = 0
        measurements['sill_pan_lf'] = 0
    
    return measurements


def pixels_to_real_inches(pixel_value, dpi, scale_ratio):
    """
    Convert pixel measurement to real-world inches.
    
    Args:
        pixel_value: Measurement in pixels
        dpi: Image DPI
        scale_ratio: Drawing scale ratio
    
    Returns:
        Real-world measurement in inches
    """
    inches_per_pixel = (1.0 / dpi) * scale_ratio
    return pixel_value * inches_per_pixel


def pixels_to_real_feet(pixel_value, dpi, scale_ratio):
    """
    Convert pixel measurement to real-world feet.
    
    Args:
        pixel_value: Measurement in pixels
        dpi: Image DPI
        scale_ratio: Drawing scale ratio
    
    Returns:
        Real-world measurement in feet
    """
    return pixels_to_real_inches(pixel_value, dpi, scale_ratio) / 12


def calculate_area_sf(width_px, height_px, dpi, scale_ratio, is_triangle=False):
    """
    Calculate area in square feet from pixel dimensions.
    
    Args:
        width_px: Width in pixels
        height_px: Height in pixels
        dpi: Image DPI
        scale_ratio: Drawing scale ratio
        is_triangle: If True, calculates triangle area (for gables)
    
    Returns:
        Area in square feet
    """
    inches_per_pixel = (1.0 / dpi) * scale_ratio
    sqft_per_sq_pixel = (inches_per_pixel ** 2) / 144
    
    area = width_px * height_px * sqft_per_sq_pixel
    
    if is_triangle:
        area = area / 2
    
    return area


def calculate_perimeter_lf(width_px, height_px, dpi, scale_ratio):
    """
    Calculate perimeter in linear feet from pixel dimensions.
    
    Args:
        width_px: Width in pixels
        height_px: Height in pixels
        dpi: Image DPI
        scale_ratio: Drawing scale ratio
    
    Returns:
        Perimeter in linear feet
    """
    inches_per_pixel = (1.0 / dpi) * scale_ratio
    width_in = width_px * inches_per_pixel
    height_in = height_px * inches_per_pixel
    
    return (width_in * 2 + height_in * 2) / 12
