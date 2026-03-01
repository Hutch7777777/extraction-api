#!/usr/bin/env python3
"""
BAX File Loader for Supabase
Parses Bluebeam BAX/XML files and loads annotations into the database.

Usage:
    pip install requests
    python3 load_bax_to_supabase.py /path/to/file.bax [--project-name "My Project"]
    
    # Batch load multiple files:
    python3 load_bax_to_supabase.py ~/Downloads/*.bax
"""

import xml.etree.ElementTree as ET
import re
import sys
import os
import json
import argparse
import hashlib
from datetime import datetime
import requests

# ===========================================
# CONFIGURATION - UPDATE THESE
# ===========================================
SUPABASE_URL = "https://okwtyttfqbfmcqtenize.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9rd3R5dHRmcWJmbWNxdGVuaXplIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjQ0NjA1MSwiZXhwIjoyMDc4MDIyMDUxfQ.ZsCSC60_9f04O1ra9niD3YG7FgjVKH2Yoii-cP-pOv8"  # Replace with your service role key

# ===========================================
# PARSING FUNCTIONS
# ===========================================

def parse_measurement(contents):
    """Parse measurement value and unit from contents string."""
    if not contents:
        return None, None, None
    
    contents = contents.strip()
    
    # Area: "123 sf" or "123.45 sf"
    area_match = re.match(r'^([\d,]+\.?\d*)\s*sf$', contents, re.IGNORECASE)
    if area_match:
        value = float(area_match.group(1).replace(',', ''))
        return value, 'sf', 'area'
    
    # Count: just a number like "45" or "123"
    count_match = re.match(r'^(\d+)$', contents)
    if count_match:
        value = float(count_match.group(1))
        return value, 'ct', 'count'
    
    # Length: "3'-6\"" or "10'-0\"" - store total inches
    length_match = re.match(r"^(\d+)'-(\d+)\"?$", contents)
    if length_match:
        feet = int(length_match.group(1))
        inches = int(length_match.group(2))
        value = feet * 12 + inches  # Total inches
        return value, 'in', 'length'
    
    # Length feet only: "3'" - convert to inches
    feet_match = re.match(r"^(\d+)'$", contents)
    if feet_match:
        value = int(feet_match.group(1)) * 12
        return value, 'in', 'length'
    
    # Length inches only: "36\""
    inches_match = re.match(r'^(\d+)"$', contents)
    if inches_match:
        value = int(inches_match.group(1))
        return value, 'in', 'length'
    
    return None, None, None


def parse_bax_file(file_path):
    """Parse a BAX file and extract annotations."""
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # Parse XML
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return None, []
    
    # Extract pages and annotations
    pages = []
    annotations = []
    
    # Find all pages - BAX format has Index attribute and child elements
    for page in root.findall('.//Page'):
        # Get page index from attribute
        page_idx = int(page.get('Index', 0))
        
        # Get page label from child element
        page_label = page.findtext('Label', f'Page {page_idx + 1}')
        
        page_info = {
            'page_index': page_idx,
            'page_label': page_label,
            'width': None,
            'height': None
        }
        
        # Get page dimensions from child elements
        width_el = page.findtext('Width')
        height_el = page.findtext('Height')
        if width_el:
            try:
                page_info['width'] = float(width_el)
            except:
                pass
        if height_el:
            try:
                page_info['height'] = float(height_el)
            except:
                pass
        
        pages.append(page_info)
        
        # Find annotations on this page - they're direct children named <Annotation>
        for annotation_el in page.findall('Annotation'):
            subject = annotation_el.findtext('Subject', '').strip()
            contents = annotation_el.findtext('Contents', '').strip()
            
            # Skip empty or non-measurement annotations
            if not subject or subject in ['Arrow', 'Line', 'Text', 'Callout', 'Note']:
                continue
            
            # Parse measurement
            value, unit, mtype = parse_measurement(contents)
            
            # Get other attributes
            author = annotation_el.findtext('Author', '')
            created = annotation_el.findtext('CreationDate', '')
            modified = annotation_el.findtext('ModDate', '')
            color = annotation_el.findtext('Color', '')
            bluebeam_id = annotation_el.findtext('ID', '')
            type_internal = annotation_el.findtext('TypeInternal', '')
            raw_data = annotation_el.findtext('Raw', '')
            
            annotation = {
                'page_index': page_idx,
                'page_label': page_label,
                'subject': subject,
                'contents': contents,
                'measurement_value': value,
                'measurement_unit': unit,
                'annotation_type': mtype or 'area',  # Default to area
                'author': author,
                'color': color,
                'bluebeam_id': bluebeam_id,
                'type_internal': type_internal,
                'raw_data': raw_data,
                'bluebeam_created_at': parse_bluebeam_date(created),
                'bluebeam_modified_at': parse_bluebeam_date(modified),
            }
            
            annotations.append(annotation)
    
    return pages, annotations


def deduplicate_count_markups(annotations):
    """
    Deduplicate Bluebeam Count markups to prevent N×N overcounting.

    Bluebeam "Count" markups export each click-point as a separate <Annotation>
    in the XML, but each annotation carries the GROUP TOTAL as its <Contents> value.
    So 6 clicks = 6 annotations each saying "6", which would create 6 rows × value 6 = 36.
    The correct answer is just 6 (one row with value 6).

    This function keeps only one entry per (subject, page_index, contents) group
    for annotations where the subject contains "Count".
    """
    count_annotations = []
    non_count_annotations = []

    for ann in annotations:
        subject = ann.get('subject', '') or ''
        # Count subjects contain "Count" in the subject name (e.g., "4/4 x 3 Trim Count")
        if 'count' in subject.lower():
            count_annotations.append(ann)
        else:
            non_count_annotations.append(ann)

    # Deduplicate count annotations: keep one per (subject, page_index, contents)
    # All annotations in the same count group have identical subject and contents
    seen = set()
    deduped_counts = []
    for ann in count_annotations:
        key = (ann.get('subject'), ann.get('page_index'), ann.get('contents'))
        if key not in seen:
            seen.add(key)
            deduped_counts.append(ann)

    if count_annotations:
        original_count = len(count_annotations)
        deduped_count = len(deduped_counts)
        if original_count != deduped_count:
            print(f"  Deduplicated Count markups: {original_count} → {deduped_count} (removed {original_count - deduped_count} duplicates)")

    return non_count_annotations + deduped_counts


def parse_bluebeam_date(date_str):
    """Parse Bluebeam date format to ISO."""
    if not date_str:
        return None
    try:
        # ISO format: 2024-11-06T09:56:12.0000000Z
        if 'T' in date_str:
            # Already ISO format, just clean it up
            return date_str.replace('.0000000Z', 'Z').replace('.0000000', '')
        
        # Old Bluebeam format: D:20240607143022-07'00'
        match = re.match(r"D:(\d{14})", date_str)
        if match:
            dt = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
            return dt.isoformat()
    except:
        pass
    return None


# ===========================================
# SUPABASE FUNCTIONS
# ===========================================

def supabase_request(method, endpoint, data=None, params=None):
    """Make a request to Supabase REST API."""
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    
    if method == 'GET':
        response = requests.get(url, headers=headers, params=params)
    elif method == 'POST':
        response = requests.post(url, headers=headers, json=data)
    elif method == 'PATCH':
        response = requests.patch(url, headers=headers, json=data, params=params)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    response.raise_for_status()
    return response.json() if response.content else None


def create_project(project_name, project_code, file_path, trade='siding'):
    """Create a new bluebeam_project record."""
    data = {
        'project_name': project_name,
        'project_code': project_code,
        'bax_file_path': file_path,
        'trade': trade,
        'status': 'pending',
        'is_training_data': True,
        'created_by': 'bax_loader'
    }
    
    result = supabase_request('POST', 'bluebeam_projects', data)
    return result[0] if result else None


def create_pages(project_id, pages, annotations):
    """Create bluebeam_pages records only for pages with annotations."""
    if not pages:
        return []
    
    # Get unique page indices that have annotations
    pages_with_annotations = set(a['page_index'] for a in annotations)
    
    # Deduplicate pages by page_index and only include those with annotations
    seen_indices = set()
    page_records = []
    for page in pages:
        idx = page['page_index']
        if idx in pages_with_annotations and idx not in seen_indices:
            seen_indices.add(idx)
            page_records.append({
                'bluebeam_project_id': project_id,
                'page_index': idx,
                'page_label': page['page_label'],
                'width': page.get('width'),
                'height': page.get('height'),
            })
    
    if not page_records:
        return []
    
    result = supabase_request('POST', 'bluebeam_pages', page_records)
    return result or []


def create_annotations(project_id, annotations, page_lookup):
    """Create bluebeam_annotations records in batches."""
    if not annotations:
        return 0
    
    batch_size = 100
    total_created = 0
    
    for i in range(0, len(annotations), batch_size):
        batch = annotations[i:i + batch_size]
        
        records = []
        for anno in batch:
            page_id = page_lookup.get(anno['page_index'])
            
            record = {
                'bluebeam_project_id': project_id,
                'bluebeam_page_id': page_id,
                'page_index': anno['page_index'],
                'page_label': anno['page_label'],
                'subject': anno['subject'],
                'contents': anno['contents'],
                'measurement_value': anno['measurement_value'],
                'measurement_unit': anno['measurement_unit'],
                'annotation_type': anno['annotation_type'],
                'author': anno.get('author'),
                'color': anno.get('color'),
                'bluebeam_id': anno.get('bluebeam_id'),
                'type_internal': anno.get('type_internal'),
                'raw_data': anno.get('raw_data'),
                'bluebeam_created_at': anno.get('bluebeam_created_at'),
                'bluebeam_modified_at': anno.get('bluebeam_modified_at'),
            }
            records.append(record)
        
        try:
            supabase_request('POST', 'bluebeam_annotations', records)
            total_created += len(records)
            print(f"    Inserted batch {i//batch_size + 1}: {len(records)} annotations")
        except Exception as e:
            print(f"    Error inserting batch: {e}")
    
    return total_created


def mark_project_completed(project_id, annotation_count, page_count):
    """Mark project as completed."""
    data = {
        'status': 'completed',
        'processed_at': datetime.utcnow().isoformat(),
        'total_annotations': annotation_count,
        'pages_with_annotations': page_count
    }
    
    supabase_request('PATCH', 'bluebeam_projects', data, params={'id': f'eq.{project_id}'})


# ===========================================
# MAIN
# ===========================================

def generate_project_code(file_path):
    """Generate a project code from filename."""
    basename = os.path.basename(file_path)
    # Remove extension and clean up
    code = os.path.splitext(basename)[0]
    # Replace spaces and special chars
    code = re.sub(r'[^\w\-]', '_', code)
    return code[:50]  # Limit length


def load_bax_file(file_path, project_name=None, trade='siding'):
    """Load a single BAX file into Supabase."""
    
    print(f"\n{'='*60}")
    print(f"Loading: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"  ERROR: File not found: {file_path}")
        return False
    
    # Parse the BAX file
    print("  Parsing BAX file...")
    pages, annotations = parse_bax_file(file_path)
    
    if annotations is None:
        print("  ERROR: Failed to parse file")
        return False
    
    print(f"  Found {len(pages)} pages, {len(annotations)} annotations")

    if not annotations:
        print("  WARNING: No annotations found, skipping")
        return False

    # Deduplicate Count markups to prevent N×N overcounting
    annotations = deduplicate_count_markups(annotations)

    # Generate project info
    if not project_name:
        project_name = os.path.splitext(os.path.basename(file_path))[0]
    project_code = generate_project_code(file_path)
    
    # Create project
    print(f"  Creating project: {project_name}")
    project = create_project(project_name, project_code, file_path, trade)
    
    if not project:
        print("  ERROR: Failed to create project")
        return False
    
    project_id = project['id']
    print(f"  Project ID: {project_id}")
    
    # Create pages
    print("  Creating pages...")
    created_pages = create_pages(project_id, pages, annotations)
    
    # Build page lookup (page_index -> page_id)
    page_lookup = {}
    for page in created_pages:
        page_lookup[page['page_index']] = page['id']
    
    # Create annotations
    print("  Creating annotations...")
    annotation_count = create_annotations(project_id, annotations, page_lookup)
    
    # Mark completed
    print("  Marking project completed...")
    pages_with_annotations = len(set(a['page_index'] for a in annotations))
    mark_project_completed(project_id, annotation_count, pages_with_annotations)
    
    # Summary
    print(f"\n  ✅ SUCCESS!")
    print(f"     Project: {project_name}")
    print(f"     Annotations: {annotation_count}")
    print(f"     Pages: {pages_with_annotations}")
    
    # Show subject breakdown
    subject_counts = {}
    for anno in annotations:
        subj = anno['subject']
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
    
    print(f"\n  Subject breakdown:")
    for subj, count in sorted(subject_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {count:4d} x {subj}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Load BAX files into Supabase')
    parser.add_argument('files', nargs='+', help='BAX file(s) to load')
    parser.add_argument('--project-name', '-n', help='Project name (for single file)')
    parser.add_argument('--trade', '-t', default='siding', help='Trade (default: siding)')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Parse only, do not upload')
    
    args = parser.parse_args()
    
    # Check API key
    if SUPABASE_KEY == "YOUR_SERVICE_ROLE_KEY_HERE":
        print("ERROR: Please update SUPABASE_KEY in the script with your service role key")
        print("       Find it in Supabase Dashboard > Settings > API > service_role key")
        sys.exit(1)
    
    print("="*60)
    print("BAX FILE LOADER")
    print("="*60)
    print(f"Files to process: {len(args.files)}")
    print(f"Trade: {args.trade}")
    
    success_count = 0
    fail_count = 0
    
    for file_path in args.files:
        # Expand user path
        file_path = os.path.expanduser(file_path)
        
        try:
            if args.dry_run:
                print(f"\n[DRY RUN] Parsing: {file_path}")
                pages, annotations = parse_bax_file(file_path)
                if annotations:
                    # Apply deduplication to show accurate count
                    annotations = deduplicate_count_markups(annotations)
                    print(f"  Would load {len(annotations)} annotations from {len(pages)} pages")
                    success_count += 1
                else:
                    print(f"  No annotations found")
                    fail_count += 1
            else:
                project_name = args.project_name if len(args.files) == 1 else None
                if load_bax_file(file_path, project_name, args.trade):
                    success_count += 1
                else:
                    fail_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1
    
    print("\n" + "="*60)
    print(f"COMPLETE: {success_count} succeeded, {fail_count} failed")
    print("="*60)


if __name__ == "__main__":
    main()
