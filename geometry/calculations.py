"""
Detection measurement calculations
"""

from config import config


def calculate_real_measurements(predictions, scale_ratio, dpi=None):
    """
    Convert pixel-based predictions to real-world measurements.
    
    Args:
        predictions: List of Roboflow predictions
        scale_ratio: Drawing scale ratio
        dpi: Image DPI (defaults to config value)
    
    Returns:
        Dict with items, counts, and areas by class
    """
    if dpi is None:
        dpi = config.DEFAULT_DPI
    
    if not scale_ratio or scale_ratio <= 0:
        scale_ratio = 48  # Default 1/4" scale
        scale_warning = "Using default scale"
    else:
        scale_warning = None
    
    inches_per_pixel = 1.0 / dpi
    real_inches_per_pixel = inches_per_pixel * scale_ratio
    sqft_per_sq_pixel = (real_inches_per_pixel ** 2) / 144
    
    results = {
        'scale_used': scale_ratio,
        'scale_warning': scale_warning,
        'dpi': dpi,
        'items': {
            'windows': [], 'doors': [], 'garages': [],
            'buildings': [], 'roofs': [], 'gables': []
        },
        'counts': {
            'window': 0, 'door': 0, 'garage': 0,
            'building': 0, 'roof': 0, 'gable': 0
        },
        'areas': {
            'window_sqft': 0, 'door_sqft': 0, 'garage_sqft': 0,
            'building_sqft': 0, 'roof_sqft': 0, 'gable_sqft': 0
        }
    }
    
    for pred in predictions:
        class_name = pred.get('class', '').lower()
        width_px = pred.get('width', 0)
        height_px = pred.get('height', 0)
        
        width_inches = width_px * real_inches_per_pixel
        height_inches = height_px * real_inches_per_pixel
        area_sqft = width_px * height_px * sqft_per_sq_pixel
        
        item = {
            'width_inches': round(width_inches, 1),
            'height_inches': round(height_inches, 1),
            'area_sqft': round(area_sqft, 1),
            'pixel_x': pred.get('x', 0),
            'pixel_y': pred.get('y', 0),
            'pixel_width': width_px,
            'pixel_height': height_px,
            'confidence': pred.get('confidence', 0)
        }
        
        # Map to correct list (handles plural naming)
        list_key = class_name + 's'
        if list_key in results['items']:
            results['items'][list_key].append(item)
        
        if class_name in results['counts']:
            results['counts'][class_name] += 1
            results['areas'][class_name + '_sqft'] += area_sqft
    
    # Calculate derived areas
    results['areas']['gross_wall_sqft'] = round(results['areas']['building_sqft'], 1)
    results['areas']['openings_sqft'] = round(
        sum(results['areas'][k] for k in ['window_sqft', 'door_sqft', 'garage_sqft']), 1
    )
    results['areas']['net_siding_sqft'] = round(
        results['areas']['gross_wall_sqft'] - results['areas']['openings_sqft'], 1
    )
    
    return results
