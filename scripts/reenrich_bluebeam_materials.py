#!/usr/bin/env python3
"""
Re-enrich Bluebeam Import Detections with Material IDs

This script fixes detections that have bluebeam_content but are missing
assigned_material_id. This can happen when:
1. The SKU resolution code wasn't deployed when the import ran
2. The bluebeam_subject_mappings were updated after import
3. New pricing_items were added after import

Usage:
    python scripts/reenrich_bluebeam_materials.py <job_id>
    python scripts/reenrich_bluebeam_materials.py --all  # Process all jobs with missing materials

The script:
1. Loads bluebeam_subject_mappings with suggested_sku values
2. Resolves each SKU to pricing_items.id
3. Updates detections where bluebeam_content matches a mapping subject
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import supabase_request


def get_mappings_with_skus():
    """Load bluebeam_subject_mappings with suggested_sku values."""
    mappings = supabase_request('GET', 'bluebeam_subject_mappings?active=eq.true&suggested_sku=not.is.null&select=bluebeam_subject,suggested_sku')
    if not mappings:
        print("No mappings with suggested_sku found")
        return {}

    return {m['bluebeam_subject']: m['suggested_sku'] for m in mappings}


def resolve_skus_to_pricing_ids(skus):
    """Resolve SKUs to pricing_items.id."""
    if not skus:
        return {}

    unique_skus = list(set(skus))
    sku_filter = ','.join(f'"{s}"' for s in unique_skus)
    items = supabase_request('GET', f'pricing_items?sku=in.({sku_filter})&active=eq.true&select=id,sku')

    if not items:
        print(f"No pricing_items found for SKUs: {unique_skus}")
        return {}

    return {item['sku']: item['id'] for item in items}


def reenrich_job(job_id):
    """Re-enrich a single job's detections with material IDs."""
    print(f"\n=== Processing job {job_id} ===")

    # Get detections with bluebeam_content but no assigned_material_id
    detections = supabase_request(
        'GET',
        f'extraction_detections_draft?job_id=eq.{job_id}&bluebeam_content=not.is.null&assigned_material_id=is.null&select=id,bluebeam_content'
    )

    if not detections:
        print(f"No detections need enrichment for job {job_id}")
        return {'job_id': job_id, 'updated': 0, 'skipped': 0}

    print(f"Found {len(detections)} detections missing assigned_material_id")

    # Load mappings and resolve SKUs
    mappings = get_mappings_with_skus()
    if not mappings:
        print("No subject->SKU mappings available")
        return {'job_id': job_id, 'updated': 0, 'skipped': len(detections)}

    print(f"Loaded {len(mappings)} subject mappings with SKUs")

    pricing_ids = resolve_skus_to_pricing_ids(list(mappings.values()))
    print(f"Resolved {len(pricing_ids)} SKUs to pricing_items.id")

    # Build subject -> pricing_id lookup
    subject_to_pricing = {}
    for subject, sku in mappings.items():
        if sku in pricing_ids:
            subject_to_pricing[subject] = pricing_ids[sku]

    print(f"Final mapping: {len(subject_to_pricing)} subjects -> pricing_items")

    # Update detections
    updated = 0
    skipped = 0

    for det in detections:
        subject = det['bluebeam_content']
        pricing_id = subject_to_pricing.get(subject)

        if pricing_id:
            result = supabase_request(
                'PATCH',
                f'extraction_detections_draft?id=eq.{det["id"]}',
                {'assigned_material_id': pricing_id}
            )
            if result is not None:
                updated += 1
            else:
                print(f"  Failed to update detection {det['id']}")
                skipped += 1
        else:
            skipped += 1

    print(f"Updated: {updated}, Skipped (no mapping): {skipped}")
    return {'job_id': job_id, 'updated': updated, 'skipped': skipped}


def find_jobs_needing_enrichment():
    """Find all jobs that have detections with bluebeam_content but missing material IDs."""
    # Query for distinct job_ids where detections have bluebeam_content but no assigned_material_id
    result = supabase_request(
        'GET',
        'extraction_detections_draft?bluebeam_content=not.is.null&assigned_material_id=is.null&select=job_id'
    )

    if not result:
        return []

    # Get unique job IDs
    job_ids = list(set(d['job_id'] for d in result))
    return job_ids


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/reenrich_bluebeam_materials.py <job_id>")
        print("  python scripts/reenrich_bluebeam_materials.py --all")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == '--all':
        job_ids = find_jobs_needing_enrichment()
        if not job_ids:
            print("No jobs need enrichment")
            sys.exit(0)

        print(f"Found {len(job_ids)} jobs needing enrichment")

        total_updated = 0
        total_skipped = 0

        for job_id in job_ids:
            result = reenrich_job(job_id)
            total_updated += result['updated']
            total_skipped += result['skipped']

        print(f"\n=== Summary ===")
        print(f"Jobs processed: {len(job_ids)}")
        print(f"Total updated: {total_updated}")
        print(f"Total skipped: {total_skipped}")
    else:
        job_id = arg
        result = reenrich_job(job_id)
        print(f"\nResult: {result}")


if __name__ == '__main__':
    main()
