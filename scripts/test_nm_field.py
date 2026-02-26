#!/usr/bin/env python3
"""
Test script for verifying Bluebeam round-trip NM field preservation.

Usage:
    python scripts/test_nm_field.py <path_to_bluebeam_exported_pdf>

This script:
1. Opens a Bluebeam-exported PDF (either fresh export or after Bluebeam editing)
2. Reads all annotations
3. Prints the NM (Name) field for each annotation
4. Confirms the embedded JSON is readable and parseable
5. Reports statistics on how many annotations have valid round-trip metadata

Use this to verify that:
- The NM field is being written correctly during export
- Bluebeam preserves the NM field after user edits
- The JSON is parseable for import operations
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: pymupdf not installed. Run: pip install pymupdf")
    sys.exit(1)


# Round-trip metadata prefix
ROUNDTRIP_PREFIX = "EST:"


def parse_roundtrip_metadata(nm_value: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse round-trip metadata from NM field value.

    Args:
        nm_value: The NM field string from the annotation

    Returns:
        Tuple of (success, parsed_data, error_message)
    """
    if not nm_value:
        return False, None, "NM field is empty"

    if not nm_value.startswith(ROUNDTRIP_PREFIX):
        return False, None, f"NM field doesn't start with '{ROUNDTRIP_PREFIX}'"

    json_str = nm_value[len(ROUNDTRIP_PREFIX):]

    try:
        data = json.loads(json_str)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON parse error: {e}"


def analyze_pdf_annotations(pdf_path: str) -> Dict[str, Any]:
    """
    Analyze all annotations in a PDF and extract round-trip metadata.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with analysis results
    """
    doc = fitz.open(pdf_path)

    results = {
        'file': pdf_path,
        'total_pages': len(doc),
        'annotations': [],
        'stats': {
            'total_annotations': 0,
            'with_nm_field': 0,
            'with_valid_roundtrip': 0,
            'without_nm_field': 0,
            'invalid_roundtrip': 0,
            'by_class': defaultdict(int),
            'by_type': defaultdict(int),
        }
    }

    for page_num in range(len(doc)):
        page = doc[page_num]

        for annot in page.annots():
            results['stats']['total_annotations'] += 1

            annot_info = {
                'page': page_num + 1,
                'type': annot.type[1] if annot.type else 'unknown',
                'rect': list(annot.rect),
                'subject': annot.info.get('subject', ''),
                'contents': annot.info.get('content', '')[:100] if annot.info.get('content') else '',
                'author': annot.info.get('title', ''),
                'nm_field': None,
                'roundtrip_data': None,
                'roundtrip_error': None,
            }

            results['stats']['by_type'][annot_info['type']] += 1

            # Get NM field
            nm_value = annot.info.get('name', '')
            annot_info['nm_field'] = nm_value

            if nm_value:
                results['stats']['with_nm_field'] += 1

                # Try to parse round-trip metadata
                success, data, error = parse_roundtrip_metadata(nm_value)

                if success:
                    results['stats']['with_valid_roundtrip'] += 1
                    annot_info['roundtrip_data'] = data

                    # Track by class
                    det_class = data.get('class', 'unknown')
                    results['stats']['by_class'][det_class] += 1
                else:
                    results['stats']['invalid_roundtrip'] += 1
                    annot_info['roundtrip_error'] = error
            else:
                results['stats']['without_nm_field'] += 1

            results['annotations'].append(annot_info)

    doc.close()
    return results


def print_results(results: Dict[str, Any], verbose: bool = False):
    """Print analysis results in a readable format."""

    print("=" * 70)
    print(f"BLUEBEAM ROUND-TRIP METADATA ANALYSIS")
    print("=" * 70)
    print(f"File: {results['file']}")
    print(f"Total Pages: {results['total_pages']}")
    print()

    stats = results['stats']
    print("ANNOTATION STATISTICS:")
    print("-" * 40)
    print(f"  Total annotations:        {stats['total_annotations']}")
    print(f"  With NM field:            {stats['with_nm_field']}")
    print(f"  With valid round-trip:    {stats['with_valid_roundtrip']}")
    print(f"  Without NM field:         {stats['without_nm_field']}")
    print(f"  Invalid round-trip data:  {stats['invalid_roundtrip']}")
    print()

    if stats['with_valid_roundtrip'] > 0:
        print("BY DETECTION CLASS:")
        print("-" * 40)
        for cls, count in sorted(stats['by_class'].items()):
            print(f"  {cls}: {count}")
        print()

    print("BY ANNOTATION TYPE:")
    print("-" * 40)
    for typ, count in sorted(stats['by_type'].items()):
        print(f"  {typ}: {count}")
    print()

    # Print details for annotations with round-trip data
    roundtrip_annots = [a for a in results['annotations'] if a['roundtrip_data']]

    if roundtrip_annots:
        print("ROUND-TRIP METADATA SAMPLES (first 5):")
        print("-" * 40)
        for annot in roundtrip_annots[:5]:
            data = annot['roundtrip_data']
            print(f"  Page {annot['page']}: {annot['subject']}")
            print(f"    Detection ID: {data.get('det_id', 'N/A')}")
            print(f"    Page ID:      {data.get('page_id', 'N/A')}")
            print(f"    Job ID:       {data.get('job_id', 'N/A')}")
            print(f"    Class:        {data.get('class', 'N/A')}")
            bbox = data.get('bbox', {})
            print(f"    Original BBox: x={bbox.get('x')}, y={bbox.get('y')}, w={bbox.get('w')}, h={bbox.get('h')}")
            print(f"    Export Time:  {data.get('export_ts', 'N/A')}")
            print()

    # Print annotations without round-trip data (likely labels or user-added)
    non_roundtrip = [a for a in results['annotations'] if not a['roundtrip_data'] and a['nm_field']]
    if non_roundtrip and verbose:
        print("ANNOTATIONS WITH NM FIELD BUT NO ROUND-TRIP DATA:")
        print("-" * 40)
        for annot in non_roundtrip[:5]:
            print(f"  Page {annot['page']}: {annot['type']} - NM: {annot['nm_field'][:80]}...")
            if annot['roundtrip_error']:
                print(f"    Error: {annot['roundtrip_error']}")
        print()

    # Summary
    print("=" * 70)
    if stats['with_valid_roundtrip'] == stats['total_annotations']:
        print("SUCCESS: All annotations have valid round-trip metadata!")
    elif stats['with_valid_roundtrip'] > 0:
        pct = (stats['with_valid_roundtrip'] / stats['total_annotations']) * 100
        print(f"PARTIAL: {stats['with_valid_roundtrip']}/{stats['total_annotations']} ({pct:.1f}%) annotations have valid round-trip metadata")
        print("Note: FreeText labels and user-added annotations won't have round-trip data")
    else:
        print("WARNING: No annotations have valid round-trip metadata!")
        print("This PDF may not have been exported with round-trip support enabled.")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_nm_field.py <path_to_pdf> [-v|--verbose]")
        print()
        print("Examples:")
        print("  python scripts/test_nm_field.py /path/to/bluebeam_export.pdf")
        print("  python scripts/test_nm_field.py ~/Downloads/my_export.pdf --verbose")
        sys.exit(1)

    pdf_path = sys.argv[1]
    verbose = '-v' in sys.argv or '--verbose' in sys.argv

    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Analyzing: {pdf_path}")
    print()

    results = analyze_pdf_annotations(pdf_path)
    print_results(results, verbose=verbose)

    # Return exit code based on results
    if results['stats']['with_valid_roundtrip'] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
