"""
Detection Post-Processing Module

Filters and cleans Roboflow predictions before database insertion:
1. Confidence filtering - drop low-confidence detections
2. IoU-based deduplication - merge overlapping same-class detections
3. Garage merging - combine adjacent door detections into garage
4. Minimum size filtering - drop tiny detections
5. Containment filtering - drop smaller detections fully inside larger ones
"""

import logging
from typing import List, Dict, Any, Tuple
from config import config

logger = logging.getLogger(__name__)


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Boxes use center x,y format with width/height (Roboflow format):
    {'x': center_x, 'y': center_y, 'width': w, 'height': h}

    Returns:
        IoU value between 0.0 and 1.0
    """
    # Convert center coords to corner coords
    x1_min = box1['x'] - box1['width'] / 2
    x1_max = box1['x'] + box1['width'] / 2
    y1_min = box1['y'] - box1['height'] / 2
    y1_max = box1['y'] + box1['height'] / 2

    x2_min = box2['x'] - box2['width'] / 2
    x2_max = box2['x'] + box2['width'] / 2
    y2_min = box2['y'] - box2['height'] / 2
    y2_max = box2['y'] + box2['height'] / 2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_containment(inner: Dict, outer: Dict) -> float:
    """
    Calculate what fraction of the inner box is contained within the outer box.

    Returns:
        Fraction of inner box area that overlaps with outer (0.0 to 1.0)
    """
    # Convert center coords to corner coords
    inner_x_min = inner['x'] - inner['width'] / 2
    inner_x_max = inner['x'] + inner['width'] / 2
    inner_y_min = inner['y'] - inner['height'] / 2
    inner_y_max = inner['y'] + inner['height'] / 2

    outer_x_min = outer['x'] - outer['width'] / 2
    outer_x_max = outer['x'] + outer['width'] / 2
    outer_y_min = outer['y'] - outer['height'] / 2
    outer_y_max = outer['y'] + outer['height'] / 2

    # Calculate intersection
    inter_x_min = max(inner_x_min, outer_x_min)
    inter_x_max = min(inner_x_max, outer_x_max)
    inter_y_min = max(inner_y_min, outer_y_min)
    inter_y_max = min(inner_y_max, outer_y_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    inner_area = inner['width'] * inner['height']

    if inner_area <= 0:
        return 0.0

    return inter_area / inner_area


def filter_by_confidence(predictions: List[Dict], min_confidence: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter predictions by confidence threshold.

    Returns:
        Tuple of (kept_predictions, filtered_predictions)
    """
    kept = []
    filtered = []

    for pred in predictions:
        conf = pred.get('confidence', 1.0)
        if conf >= min_confidence:
            kept.append(pred)
        else:
            filtered.append(pred)

    return kept, filtered


def filter_by_size(predictions: List[Dict], min_sizes: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter predictions by minimum size requirements.

    Returns:
        Tuple of (kept_predictions, filtered_predictions)
    """
    kept = []
    filtered = []
    default_min = min_sizes.get('default', {'width': 20, 'height': 20})

    for pred in predictions:
        cls = (pred.get('class') or '').lower().strip()
        min_size = min_sizes.get(cls, default_min)

        width = pred.get('width', 0)
        height = pred.get('height', 0)

        if not width or not height:
            logger.warning(
                f"Detection missing dimensions: class={cls}, "
                f"width={width}, height={height}, "
                f"confidence={pred.get('confidence', 'N/A')}"
            )

        if width >= min_size['width'] and height >= min_size['height']:
            kept.append(pred)
        else:
            filtered.append(pred)

    return kept, filtered


def deduplicate_by_iou(predictions: List[Dict], iou_threshold: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove duplicate detections using IoU-based Non-Maximum Suppression.
    For same-class detections with IoU > threshold, keep only the higher-confidence one.

    Returns:
        Tuple of (kept_predictions, suppressed_predictions)
    """
    if not predictions:
        return [], []

    # Sort by confidence descending
    sorted_preds = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)

    kept = []
    suppressed = []
    used_indices = set()

    for i, pred in enumerate(sorted_preds):
        if i in used_indices:
            continue

        kept.append(pred)
        pred_class = (pred.get('class') or '').lower().strip()

        # Mark overlapping same-class detections as suppressed
        for j in range(i + 1, len(sorted_preds)):
            if j in used_indices:
                continue

            other = sorted_preds[j]
            other_class = (other.get('class') or '').lower().strip()

            # Only suppress same-class detections
            if pred_class == other_class:
                iou = calculate_iou(pred, other)
                if iou > iou_threshold:
                    used_indices.add(j)
                    suppressed.append(other)

    return kept, suppressed


def filter_contained(predictions: List[Dict], containment_threshold: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove smaller detections that are fully contained within larger same-class detections.

    Returns:
        Tuple of (kept_predictions, filtered_predictions)
    """
    if not predictions:
        return [], []

    # Sort by area descending (larger first)
    def get_area(p):
        return p.get('width', 0) * p.get('height', 0)

    sorted_preds = sorted(predictions, key=get_area, reverse=True)

    kept = []
    filtered = []
    kept_indices = set()

    for i, pred in enumerate(sorted_preds):
        pred_class = (pred.get('class') or '').lower().strip()
        is_contained = False

        # Check if this detection is contained within any larger kept detection
        for j in kept_indices:
            larger = sorted_preds[j]
            larger_class = (larger.get('class') or '').lower().strip()

            if pred_class == larger_class:
                containment = calculate_containment(pred, larger)
                if containment >= containment_threshold:
                    is_contained = True
                    break

        if is_contained:
            filtered.append(pred)
        else:
            kept.append(pred)
            kept_indices.add(i)

    return kept, filtered


def merge_adjacent_doors_to_garage(
    predictions: List[Dict],
    y_tolerance_px: int,
    min_combined_width_px: int,
    max_gap_px: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Merge horizontally adjacent 'door' detections into a single 'garage' detection.

    Criteria for merging:
    - Both are 'door' class
    - Similar Y center (within y_tolerance_px)
    - Horizontally adjacent (gap < max_gap_px)
    - Combined width >= min_combined_width_px

    Returns:
        Tuple of (processed_predictions, merged_door_groups)
    """
    # Separate doors from other detections
    doors = []
    others = []

    for pred in predictions:
        cls = (pred.get('class') or '').lower().strip()
        if cls == 'door':
            doors.append(pred)
        else:
            others.append(pred)

    if len(doors) < 2:
        return predictions, []

    # Sort doors by X position
    doors_sorted = sorted(doors, key=lambda d: d.get('x', 0))

    merged_groups = []
    used_indices = set()
    new_garages = []

    for i, door1 in enumerate(doors_sorted):
        if i in used_indices:
            continue

        # Try to form a group starting with this door
        group = [door1]
        group_indices = {i}

        # Find adjacent doors
        for j in range(i + 1, len(doors_sorted)):
            if j in used_indices:
                continue

            door2 = doors_sorted[j]
            last_in_group = group[-1]

            # Check Y alignment
            y_diff = abs(door2.get('y', 0) - last_in_group.get('y', 0))
            if y_diff > y_tolerance_px:
                continue

            # Check horizontal adjacency (gap between right edge of last and left edge of next)
            last_right = last_in_group.get('x', 0) + last_in_group.get('width', 0) / 2
            next_left = door2.get('x', 0) - door2.get('width', 0) / 2
            gap = next_left - last_right

            if gap <= max_gap_px:
                group.append(door2)
                group_indices.add(j)

        # Check if group should become a garage
        if len(group) >= 2:
            # Calculate combined bounding box
            min_x = min(d.get('x', 0) - d.get('width', 0) / 2 for d in group)
            max_x = max(d.get('x', 0) + d.get('width', 0) / 2 for d in group)
            min_y = min(d.get('y', 0) - d.get('height', 0) / 2 for d in group)
            max_y = max(d.get('y', 0) + d.get('height', 0) / 2 for d in group)

            combined_width = max_x - min_x
            combined_height = max_y - min_y

            if combined_width >= min_combined_width_px:
                # Create merged garage detection
                avg_confidence = sum(d.get('confidence', 0) for d in group) / len(group)

                garage = {
                    'class': 'garage',
                    'x': (min_x + max_x) / 2,
                    'y': (min_y + max_y) / 2,
                    'width': combined_width,
                    'height': combined_height,
                    'confidence': avg_confidence,
                    'merged_from': [d.get('detection_id') for d in group if d.get('detection_id')],
                    'merge_count': len(group)
                }

                new_garages.append(garage)
                merged_groups.append(group)
                used_indices.update(group_indices)

    # Build final list: others + unmerged doors + new garages
    unmerged_doors = [d for i, d in enumerate(doors_sorted) if i not in used_indices]
    result = others + unmerged_doors + new_garages

    return result, merged_groups


def postprocess_detections(
    predictions: List[Dict],
    verbose: bool = None
) -> Dict[str, Any]:
    """
    Main entry point for detection post-processing.

    Applies all filters in sequence:
    1. Confidence filter
    2. Minimum size filter
    3. IoU deduplication
    4. Containment filter
    5. Garage merging

    Args:
        predictions: Raw Roboflow predictions
        verbose: Override config verbosity setting

    Returns:
        Dict with:
        - 'predictions': Cleaned predictions list
        - 'stats': Filtering statistics
        - 'filtered': Details of what was filtered
    """
    if verbose is None:
        verbose = config.DETECTION_POSTPROCESS_VERBOSE

    original_count = len(predictions)
    stats = {
        'original_count': original_count,
        'confidence_filtered': 0,
        'size_filtered': 0,
        'iou_suppressed': 0,
        'containment_filtered': 0,
        'doors_merged_to_garages': 0,
        'final_count': 0
    }
    filtered_details = {
        'low_confidence': [],
        'too_small': [],
        'duplicates': [],
        'contained': [],
        'merged_door_groups': []
    }

    current = predictions

    # 1. Confidence filter
    current, low_conf = filter_by_confidence(current, config.DETECTION_MIN_CONFIDENCE)
    stats['confidence_filtered'] = len(low_conf)
    filtered_details['low_confidence'] = [
        {'class': p.get('class'), 'confidence': p.get('confidence')} for p in low_conf
    ]
    if verbose and low_conf:
        print(f"[PostProcess] Dropped {len(low_conf)} low-confidence detections (< {config.DETECTION_MIN_CONFIDENCE})", flush=True)

    # 2. Minimum size filter
    current, too_small = filter_by_size(current, config.DETECTION_MIN_SIZE)
    stats['size_filtered'] = len(too_small)
    filtered_details['too_small'] = [
        {'class': p.get('class'), 'width': p.get('width'), 'height': p.get('height')} for p in too_small
    ]
    if verbose and too_small:
        print(f"[PostProcess] Dropped {len(too_small)} undersized detections", flush=True)

    # 3. IoU deduplication
    current, duplicates = deduplicate_by_iou(current, config.DETECTION_IOU_THRESHOLD)
    stats['iou_suppressed'] = len(duplicates)
    filtered_details['duplicates'] = [
        {'class': p.get('class'), 'confidence': p.get('confidence')} for p in duplicates
    ]
    if verbose and duplicates:
        print(f"[PostProcess] Suppressed {len(duplicates)} duplicate detections (IoU > {config.DETECTION_IOU_THRESHOLD})", flush=True)

    # 4. Containment filter
    if config.CONTAINMENT_FILTER_ENABLED:
        current, contained = filter_contained(current, config.CONTAINMENT_THRESHOLD)
        stats['containment_filtered'] = len(contained)
        filtered_details['contained'] = [
            {'class': p.get('class'), 'width': p.get('width'), 'height': p.get('height')} for p in contained
        ]
        if verbose and contained:
            print(f"[PostProcess] Dropped {len(contained)} contained detections", flush=True)

    # 5. Garage merging
    if config.GARAGE_MERGE_ENABLED:
        current, merged_groups = merge_adjacent_doors_to_garage(
            current,
            config.GARAGE_MERGE_Y_TOLERANCE_PX,
            config.GARAGE_MIN_COMBINED_WIDTH_PX,
            config.GARAGE_MAX_GAP_PX
        )
        doors_merged = sum(len(g) for g in merged_groups)
        stats['doors_merged_to_garages'] = doors_merged
        filtered_details['merged_door_groups'] = [
            [{'class': d.get('class'), 'x': d.get('x')} for d in g] for g in merged_groups
        ]
        if verbose and doors_merged:
            print(f"[PostProcess] Merged {doors_merged} door detections into {len(merged_groups)} garage(s)", flush=True)

    stats['final_count'] = len(current)

    if verbose:
        removed = original_count - len(current)
        print(f"[PostProcess] Summary: {original_count} raw → {len(current)} final ({removed} removed)", flush=True)

    return {
        'predictions': current,
        'stats': stats,
        'filtered': filtered_details
    }
