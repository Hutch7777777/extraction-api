"""
Bluebeam PDF import service.

Parses Bluebeam-edited PDFs and generates a diff of changes compared to
the original detections. This enables round-trip workflows where users
can edit annotations in Bluebeam and import changes back.

Uses PyMuPDF (fitz) to read PDF annotations and extract the round-trip
metadata embedded in the NM (Name) field during export.
"""

import io
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("WARNING: pymupdf not installed. Install with: pip install pymupdf", flush=True)

from database import supabase_request


class ChangeType(str, Enum):
    """Types of changes detected during import"""
    MATCHED = "matched"      # Annotation matches original detection (no changes)
    MODIFIED = "modified"    # Annotation exists but position/size changed
    DELETED = "deleted"      # Original detection has no matching annotation
    ADDED = "added"          # New annotation without original detection ID


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates"""
    x: float  # Center X
    y: float  # Center Y
    w: float  # Width
    h: float  # Height

    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y, 'w': self.w, 'h': self.h}

    @classmethod
    def from_dict(cls, d: Dict) -> 'BoundingBox':
        return cls(
            x=float(d.get('x', 0)),
            y=float(d.get('y', 0)),
            w=float(d.get('w', 0)),
            h=float(d.get('h', 0))
        )

    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bbox"""
        # Convert center-based to corner-based
        ax1, ay1 = self.x - self.w/2, self.y - self.h/2
        ax2, ay2 = self.x + self.w/2, self.y + self.h/2
        bx1, by1 = other.x - other.w/2, other.y - other.h/2
        bx2, by2 = other.x + other.w/2, other.y + other.h/2

        # Intersection
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = self.w * self.h
        area_b = other.w * other.h
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class ImportedAnnotation:
    """Represents an annotation parsed from a Bluebeam PDF"""
    page_number: int
    pdf_rect: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in PDF coords
    bbox: BoundingBox  # Converted to pixel coords
    subject: Optional[str] = None  # Bluebeam "Subject" field (class name)
    contents: Optional[str] = None  # Bluebeam "Contents" field (measurement)
    roundtrip_metadata: Optional[Dict] = None  # Parsed EST: JSON from NM field

    @property
    def detection_id(self) -> Optional[str]:
        """Get original detection ID from roundtrip metadata"""
        if self.roundtrip_metadata:
            return self.roundtrip_metadata.get('det_id')
        return None

    @property
    def original_bbox(self) -> Optional[BoundingBox]:
        """Get original bbox from roundtrip metadata"""
        if self.roundtrip_metadata and 'bbox' in self.roundtrip_metadata:
            return BoundingBox.from_dict(self.roundtrip_metadata['bbox'])
        return None

    def to_dict(self) -> Dict:
        return {
            'page_number': self.page_number,
            'pdf_rect': self.pdf_rect,
            'bbox': self.bbox.to_dict(),
            'subject': self.subject,
            'contents': self.contents,
            'detection_id': self.detection_id,
            'has_roundtrip_metadata': self.roundtrip_metadata is not None
        }


@dataclass
class ChangeRecord:
    """Records a single change detected during import"""
    change_type: ChangeType
    detection_id: Optional[str]
    page_id: Optional[str]
    page_number: int
    detection_class: Optional[str]
    original_bbox: Optional[Dict] = None
    imported_bbox: Optional[Dict] = None
    bbox_shift: Optional[Dict] = None  # {dx, dy, dw, dh}
    iou: Optional[float] = None
    annotation_subject: Optional[str] = None
    annotation_contents: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'change_type': self.change_type.value,
            'detection_id': self.detection_id,
            'page_id': self.page_id,
            'page_number': self.page_number,
            'detection_class': self.detection_class,
            'original_bbox': self.original_bbox,
            'imported_bbox': self.imported_bbox,
            'bbox_shift': self.bbox_shift,
            'iou': self.iou,
            'annotation_subject': self.annotation_subject,
            'annotation_contents': self.annotation_contents
        }


def parse_roundtrip_metadata(nm_field: str) -> Optional[Dict]:
    """
    Parse round-trip metadata from PDF annotation NM (Name) field.

    The metadata is stored as JSON prefixed with "EST:" during export.
    Example: EST:{"v":1,"det_id":"abc-123","page_id":"def-456",...}

    Args:
        nm_field: The NM field value from the PDF annotation

    Returns:
        Parsed JSON dict or None if not our metadata
    """
    if not nm_field or not isinstance(nm_field, str):
        return None

    # Look for our prefix
    if not nm_field.startswith('EST:'):
        return None

    json_str = nm_field[4:]  # Remove "EST:" prefix

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[Bluebeam Import] Failed to parse roundtrip metadata: {e}", flush=True)
        return None


def extract_annotations_from_pdf(
    pdf_bytes: bytes,
    page_dimensions: Dict[int, Tuple[int, int]]
) -> List[ImportedAnnotation]:
    """
    Extract all rectangle annotations from a PDF.

    Args:
        pdf_bytes: The PDF file contents
        page_dimensions: Dict mapping page_number -> (image_width, image_height)
                        Used to convert PDF coords back to pixel coords

    Returns:
        List of ImportedAnnotation objects
    """
    if fitz is None:
        raise RuntimeError("pymupdf not installed")

    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    annotations = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1
        pdf_rect = page.rect

        # Get image dimensions for coordinate conversion
        img_dims = page_dimensions.get(page_number)
        if img_dims:
            img_width, img_height = img_dims
            scale_x = img_width / pdf_rect.width if pdf_rect.width > 0 else 1.0
            scale_y = img_height / pdf_rect.height if pdf_rect.height > 0 else 1.0
        else:
            # Fallback: assume 1:1 mapping
            scale_x = 1.0
            scale_y = 1.0

        # Iterate through annotations
        for annot in page.annots():
            annot_type = annot.type[0]  # Type code

            # We're interested in rectangle/square annotations (type 4 = Square)
            # and polygon annotations (type 5 = Polygon)
            if annot_type not in [4, 5, 6]:  # Square, Polygon, PolyLine
                continue

            # Get annotation rect
            rect = annot.rect

            # Convert PDF coordinates to pixel coordinates (center-based)
            pdf_x1, pdf_y1 = rect.x0, rect.y0
            pdf_x2, pdf_y2 = rect.x1, rect.y1

            # Scale to image pixels
            px_x1 = pdf_x1 * scale_x
            px_y1 = pdf_y1 * scale_y
            px_x2 = pdf_x2 * scale_x
            px_y2 = pdf_y2 * scale_y

            # Convert to center-based bbox
            bbox = BoundingBox(
                x=(px_x1 + px_x2) / 2,
                y=(px_y1 + px_y2) / 2,
                w=px_x2 - px_x1,
                h=px_y2 - px_y1
            )

            # Get annotation metadata
            info = annot.info
            subject = info.get('subject', '')
            contents = info.get('content', '')
            nm_field = info.get('name', '')

            # Parse round-trip metadata from NM field
            roundtrip_metadata = parse_roundtrip_metadata(nm_field)

            annotations.append(ImportedAnnotation(
                page_number=page_number,
                pdf_rect=(rect.x0, rect.y0, rect.x1, rect.y1),
                bbox=bbox,
                subject=subject,
                contents=contents,
                roundtrip_metadata=roundtrip_metadata
            ))

    doc.close()
    return annotations


def fetch_current_detections(job_id: str) -> Dict[str, Dict]:
    """
    Fetch current detections for a job from Supabase.

    Returns:
        Dict mapping detection_id -> detection data
    """
    detections = supabase_request('GET', 'extraction_detections_draft', filters={
        'job_id': f'eq.{job_id}',
        'status': 'neq.deleted'
    })

    if not detections:
        # Try the view instead
        detections = supabase_request('GET', 'extraction_detection_details', filters={
            'job_id': f'eq.{job_id}',
            'status': 'neq.deleted'
        })

    if not detections:
        return {}

    return {det['id']: det for det in detections}


def fetch_page_dimensions(job_id: str) -> Dict[int, Tuple[int, int]]:
    """
    Fetch page dimensions for all pages in a job.

    Returns:
        Dict mapping page_number -> (image_width, image_height)
    """
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'order': 'page_number.asc'
    })
    print(f"[Bluebeam Import] Pages query returned: {len(pages) if pages else 0} pages", flush=True)

    if not pages:
        return {}

    # Debug: show available fields from first page
    if pages and len(pages) > 0:
        print(f"[Bluebeam Import] First page keys: {list(pages[0].keys())}", flush=True)
        print(f"[Bluebeam Import] First page values: original_width={pages[0].get('original_width')}, original_height={pages[0].get('original_height')}, image_width={pages[0].get('image_width')}, image_height={pages[0].get('image_height')}, width={pages[0].get('width')}, height={pages[0].get('height')}", flush=True)

    result = {}
    for page in pages:
        page_num = page.get('page_number', 0)
        width = page.get('original_width') or page.get('image_width') or page.get('width') or 0
        height = page.get('original_height') or page.get('image_height') or page.get('height') or 0
        if page_num and width and height:
            result[page_num] = (width, height)

    # Fallback: if no dimensions found but pages exist, use 1:1 mapping
    if not result and pages:
        print(f"[Bluebeam Import] WARNING: No dimension fields found, using 1:1 coordinate mapping fallback", flush=True)
        for page in pages:
            page_num = page.get('page_number', 0)
            if page_num:
                result[page_num] = (1, 1)

    return result


def fetch_page_id_mapping(job_id: str) -> Dict[int, str]:
    """
    Fetch mapping from page_number to page_id.

    Returns:
        Dict mapping page_number -> page_id
    """
    pages = supabase_request('GET', 'extraction_pages', filters={
        'job_id': f'eq.{job_id}',
        'order': 'page_number.asc'
    })

    if not pages:
        return {}

    return {page['page_number']: page['id'] for page in pages}


def compute_diff(
    annotations: List[ImportedAnnotation],
    detections: Dict[str, Dict],
    page_id_map: Dict[int, str],
    modification_threshold: float = 0.8
) -> List[ChangeRecord]:
    """
    Compare imported annotations against current detections.

    Args:
        annotations: Parsed annotations from imported PDF
        detections: Current detections keyed by ID
        page_id_map: Mapping from page_number to page_id
        modification_threshold: IoU threshold below which we consider modified (default 0.8)

    Returns:
        List of ChangeRecord objects describing all changes
    """
    changes = []
    matched_detection_ids = set()

    # Process each annotation
    for annot in annotations:
        det_id = annot.detection_id
        page_id = page_id_map.get(annot.page_number)

        if det_id and det_id in detections:
            # We have round-trip metadata and the detection still exists
            det = detections[det_id]
            matched_detection_ids.add(det_id)

            # Get original bbox from detection
            original_bbox = BoundingBox(
                x=float(det.get('pixel_x', 0)),
                y=float(det.get('pixel_y', 0)),
                w=float(det.get('pixel_width', 0)),
                h=float(det.get('pixel_height', 0))
            )

            # Calculate IoU between original and imported bbox
            iou = annot.bbox.iou(original_bbox)

            # Calculate shift
            bbox_shift = {
                'dx': annot.bbox.x - original_bbox.x,
                'dy': annot.bbox.y - original_bbox.y,
                'dw': annot.bbox.w - original_bbox.w,
                'dh': annot.bbox.h - original_bbox.h
            }

            # Determine if this is a match or modification
            if iou >= modification_threshold:
                change_type = ChangeType.MATCHED
            else:
                change_type = ChangeType.MODIFIED

            changes.append(ChangeRecord(
                change_type=change_type,
                detection_id=det_id,
                page_id=page_id,
                page_number=annot.page_number,
                detection_class=det.get('class'),
                original_bbox=original_bbox.to_dict(),
                imported_bbox=annot.bbox.to_dict(),
                bbox_shift=bbox_shift,
                iou=round(iou, 4),
                annotation_subject=annot.subject,
                annotation_contents=annot.contents
            ))
        else:
            # New annotation (no round-trip metadata or detection was deleted)
            changes.append(ChangeRecord(
                change_type=ChangeType.ADDED,
                detection_id=None,
                page_id=page_id,
                page_number=annot.page_number,
                detection_class=annot.subject.lower().replace(' ', '_') if annot.subject else 'unknown',
                original_bbox=None,
                imported_bbox=annot.bbox.to_dict(),
                bbox_shift=None,
                iou=None,
                annotation_subject=annot.subject,
                annotation_contents=annot.contents
            ))

    # Find deleted detections (in DB but no matching annotation)
    for det_id, det in detections.items():
        if det_id not in matched_detection_ids:
            # Get page number from page_id_map (reverse lookup)
            page_number = None
            page_id = det.get('page_id')
            for pnum, pid in page_id_map.items():
                if pid == page_id:
                    page_number = pnum
                    break

            changes.append(ChangeRecord(
                change_type=ChangeType.DELETED,
                detection_id=det_id,
                page_id=page_id,
                page_number=page_number or 0,
                detection_class=det.get('class'),
                original_bbox={
                    'x': det.get('pixel_x'),
                    'y': det.get('pixel_y'),
                    'w': det.get('pixel_width'),
                    'h': det.get('pixel_height')
                },
                imported_bbox=None,
                bbox_shift=None,
                iou=None,
                annotation_subject=None,
                annotation_contents=None
            ))

    return changes


def import_bluebeam_pdf(
    pdf_bytes: bytes,
    job_id: str,
    modification_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Import a Bluebeam-edited PDF and generate a diff against current detections.

    This is the main entry point for the import workflow.

    Args:
        pdf_bytes: The uploaded PDF file contents
        job_id: The extraction job UUID to compare against
        modification_threshold: IoU threshold for considering a detection modified (default 0.8)

    Returns:
        Dict with success status, summary statistics, and detailed changes
    """
    if fitz is None:
        return {
            'success': False,
            'error': 'pymupdf not installed. Run: pip install pymupdf'
        }

    print(f"[Bluebeam Import] Starting import for job {job_id}", flush=True)

    # 1. Fetch page dimensions for coordinate conversion
    page_dimensions = fetch_page_dimensions(job_id)
    if not page_dimensions:
        return {
            'success': False,
            'error': f'No pages found for job {job_id}'
        }

    print(f"[Bluebeam Import] Found {len(page_dimensions)} pages", flush=True)

    # 2. Fetch page ID mapping
    page_id_map = fetch_page_id_mapping(job_id)

    # 3. Extract annotations from PDF
    try:
        annotations = extract_annotations_from_pdf(pdf_bytes, page_dimensions)
        print(f"[Bluebeam Import] Extracted {len(annotations)} annotations from PDF", flush=True)
    except Exception as e:
        print(f"[Bluebeam Import] Error extracting annotations: {e}", flush=True)
        return {
            'success': False,
            'error': f'Failed to parse PDF: {str(e)}'
        }

    # Count annotations with round-trip metadata
    with_metadata = sum(1 for a in annotations if a.roundtrip_metadata)
    print(f"[Bluebeam Import] {with_metadata}/{len(annotations)} annotations have round-trip metadata", flush=True)

    # 4. Fetch current detections
    detections = fetch_current_detections(job_id)
    print(f"[Bluebeam Import] Found {len(detections)} current detections", flush=True)

    # 5. Compute diff
    changes = compute_diff(annotations, detections, page_id_map, modification_threshold)

    # 6. Summarize results
    summary = {
        'matched': 0,
        'modified': 0,
        'deleted': 0,
        'added': 0
    }

    for change in changes:
        summary[change.change_type.value] += 1

    # Group changes by page
    changes_by_page = {}
    for change in changes:
        page_num = change.page_number
        if page_num not in changes_by_page:
            changes_by_page[page_num] = {
                'matched': [],
                'modified': [],
                'deleted': [],
                'added': []
            }
        changes_by_page[page_num][change.change_type.value].append(change.to_dict())

    print(f"[Bluebeam Import] Diff complete: {summary}", flush=True)

    return {
        'success': True,
        'job_id': job_id,
        'summary': {
            **summary,
            'total_annotations': len(annotations),
            'total_detections': len(detections),
            'annotations_with_metadata': with_metadata
        },
        'changes_by_page': changes_by_page,
        'changes': [c.to_dict() for c in changes],
        'import_timestamp': datetime.now(timezone.utc).isoformat()
    }
