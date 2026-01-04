# Extraction API - Development Guide

## Overview

This document explains how to make changes to the modular Extraction API v4.0. Use this as a reference when working with Claude or developing independently.

---

## Architecture Summary

```
extraction-api/
├── app.py                    # Flask entry point - ROUTING ONLY
├── config.py                 # All configuration settings
│
├── core/                     # External API integrations
│   ├── roboflow_client.py    # Object detection
│   └── claude_client.py      # Vision AI (OCR, classification)
│
├── services/                 # Business logic (THE MAIN WORK HAPPENS HERE)
│   ├── pdf_service.py        # PDF → images conversion
│   ├── classification_service.py  # Page type detection
│   ├── extraction_service.py # Main orchestration
│   ├── markup_service.py     # Visual markup generation
│   ├── takeoff_service.py    # Measurement calculations
│   ├── cross_ref_service.py  # Schedule ↔ detection matching
│   └── floor_plan_service.py # Corner analysis
│
├── database/                 # Database layer
│   ├── client.py             # Raw Supabase requests
│   ├── storage.py            # File uploads
│   └── repositories/         # CRUD operations by table
│       ├── job_repository.py
│       ├── page_repository.py
│       └── detection_repository.py
│
├── geometry/                 # Pure calculation functions
│   ├── scale_parser.py       # "1/4" = 1'-0"" → 48.0
│   ├── measurements.py       # Derived measurements
│   └── calculations.py       # Pixel → real-world conversion
│
└── utils/                    # Shared utilities
    └── validation.py         # Input validation
```

---

## Key Principles

### 1. app.py is for ROUTING ONLY

**DO NOT** add business logic to `app.py`. It should only:
- Define endpoints
- Parse request data
- Call service functions
- Return JSON responses

**Good:**
```python
@app.route('/calculate-takeoff', methods=['POST'])
def calculate_takeoff():
    from services.takeoff_service import calculate_takeoff_for_job
    data = request.json
    return jsonify(calculate_takeoff_for_job(data['job_id']))
```

**Bad:**
```python
@app.route('/calculate-takeoff', methods=['POST'])
def calculate_takeoff():
    # DON'T put calculation logic here!
    pages = supabase_request('GET', 'extraction_pages', ...)
    for page in pages:
        # ... 100 lines of calculation code
    return jsonify(results)
```

### 2. Services contain business logic

Each service handles one domain:
- `pdf_service.py` - PDF conversion
- `classification_service.py` - Page classification
- `extraction_service.py` - Main extraction workflow
- `markup_service.py` - Image markup generation
- `takeoff_service.py` - Measurement calculations
- `cross_ref_service.py` - Schedule matching
- `floor_plan_service.py` - Corner analysis

### 3. Database operations go through repositories

**DO NOT** call `supabase_request()` directly in services. Use repositories:

**Good:**
```python
from database import get_page, update_page, get_elevation_pages

page = get_page(page_id)
update_page(page_id, {'status': 'complete'})
```

**Acceptable** (for complex queries not in repositories):
```python
from database import supabase_request

# Complex query that doesn't fit repository pattern
results = supabase_request('GET', 'extraction_elevation_calcs', filters={
    'job_id': f'eq.{job_id}',
    'order': 'elevation_name'
})
```

### 4. Geometry functions are pure

Functions in `geometry/` should:
- Take inputs, return outputs
- Have no side effects
- Not access the database
- Be easily testable

```python
# Good - pure function
def parse_scale_notation(notation):
    # ... parsing logic
    return scale_ratio

# Bad - impure function
def parse_scale_notation(notation, page_id):
    result = parse(notation)
    update_page(page_id, {'scale_ratio': result})  # NO! Side effect!
    return result
```

---

## How to Make Common Changes

### Adding a New Endpoint

1. **Create service function** in appropriate service file:
   ```python
   # services/my_service.py
   def my_new_function(job_id):
       # Business logic here
       return {"success": True, "data": result}
   ```

2. **Add route** in `app.py`:
   ```python
   @app.route('/my-endpoint', methods=['POST'])
   def my_endpoint():
       from services.my_service import my_new_function
       data = request.json
       return jsonify(my_new_function(data['job_id']))
   ```

### Modifying Calculation Logic

1. **Find the relevant service** (usually `takeoff_service.py` or `geometry/`)
2. **Make changes** to the specific function
3. **Test locally** before deploying

Example: Changing how gross facade is calculated:
```python
# services/takeoff_service.py, line ~80
# BEFORE:
gross_facade_sf = areas['building_area_sf'] - areas['roof_area_sf']

# AFTER (if you want to include gables):
gross_facade_sf = areas['building_area_sf'] - areas['roof_area_sf'] + areas['gable_area_sf']
```

### Adding a New Detection Class

1. **Update config.py** with new color:
   ```python
   MARKUP_COLORS = {
       # ... existing colors
       'soffit': (139, 69, 19),  # Brown
   }
   ```

2. **Update TRADE_GROUPS** if needed:
   ```python
   TRADE_GROUPS = {
       'siding': ['building', 'exterior wall', 'window', 'door', 'garage', 'soffit'],
       # ...
   }
   ```

3. **Update takeoff calculations** in `services/takeoff_service.py`

### Adding a New External API

1. **Create client** in `core/`:
   ```python
   # core/new_api_client.py
   import requests
   from config import config
   
   def call_new_api(data):
       response = requests.post(config.NEW_API_URL, json=data)
       return response.json()
   ```

2. **Add config** in `config.py`:
   ```python
   NEW_API_KEY = os.getenv('NEW_API_KEY')
   NEW_API_URL = "https://api.example.com/v1"
   ```

3. **Export** in `core/__init__.py`:
   ```python
   from core.new_api_client import call_new_api
   ```

### Adding a New Database Table

1. **Create repository** in `database/repositories/`:
   ```python
   # database/repositories/new_table_repository.py
   from database.client import supabase_request
   
   def get_item(item_id):
       result = supabase_request('GET', 'new_table', filters={'id': f'eq.{item_id}'})
       return result[0] if result else None
   
   def create_item(data):
       return supabase_request('POST', 'new_table', data)
   ```

2. **Export** in `database/__init__.py`:
   ```python
   from database.repositories.new_table_repository import get_item, create_item
   ```

---

## Testing Changes

### Local Testing

```bash
# Start server
cd ~/Documents/extraction-api
python3 app.py

# Test in another terminal
curl http://localhost:5050/health
curl http://localhost:5050/list-jobs
curl -X POST http://localhost:5050/my-endpoint -H "Content-Type: application/json" -d '{"job_id": "xxx"}'
```

### Testing Individual Modules

```python
# In Python REPL
>>> from geometry import parse_scale_notation
>>> parse_scale_notation('1/4" = 1\'-0"')
48.0

>>> from database import list_jobs
>>> len(list_jobs())
6
```

---

## Deployment

```bash
# 1. Test locally first!
python3 app.py
curl http://localhost:5050/health

# 2. Commit and push
git add .
git commit -m "Description of changes"
git push origin main

# 3. Railway auto-deploys - verify
curl https://extraction-api-production.up.railway.app/health
```

---

## File-by-File Reference

### app.py
- **Purpose**: Flask routing only
- **When to modify**: Adding/removing endpoints
- **Never add**: Business logic, calculations, database queries

### config.py
- **Purpose**: All configuration in one place
- **When to modify**: Adding new config values, changing colors, trade groups
- **Contains**: API keys, URLs, constants

### core/roboflow_client.py
- **Purpose**: Roboflow object detection
- **Key function**: `detect_objects(image_url)` → `{"predictions": [...]}`

### core/claude_client.py
- **Purpose**: Claude Vision API for OCR and classification
- **Key functions**:
  - `classify_page(image_url)` → page type, scale, elevation name
  - `extract_schedule(image_url)` → windows/doors list
  - `analyze_floor_plan_corners(image_url)` → corner counts

### services/pdf_service.py
- **Purpose**: Convert PDF to images
- **Key function**: `convert_pdf_background(job_id, pdf_url)`

### services/classification_service.py
- **Purpose**: Classify all pages in a job
- **Key function**: `classify_job_background(job_id)`

### services/extraction_service.py
- **Purpose**: Main extraction orchestration
- **Key function**: `process_job_background(job_id, scale_override, generate_markups)`
- **Calls**: Roboflow detection, measurements, markups, cross-refs

### services/markup_service.py
- **Purpose**: Generate visual markup images
- **Key functions**:
  - `generate_markups_for_page(page_id, trades)`
  - `generate_markups_for_job(job_id, trades)`
  - `generate_comprehensive_markup(page_id)`

### services/takeoff_service.py
- **Purpose**: Calculate measurements from detections
- **Key functions**:
  - `calculate_takeoff_for_page(page_id)` → stores in extraction_detection_details, extraction_elevation_calcs
  - `calculate_takeoff_for_job(job_id)` → aggregates into extraction_job_totals
- **Critical formula**: `gross_facade = building - roof`

### services/cross_ref_service.py
- **Purpose**: Match schedule data with detections
- **Key function**: `build_cross_references(job_id)`
- **Stores**: extraction_cross_refs, extraction_takeoff_summary

### services/floor_plan_service.py
- **Purpose**: Analyze floor plans for corners
- **Key function**: `analyze_floor_plan_for_job(job_id)`

### geometry/scale_parser.py
- **Purpose**: Parse scale notation
- **Key function**: `parse_scale_notation("1/4\" = 1'-0\"")` → `48.0`

### geometry/measurements.py
- **Purpose**: Derived measurements
- **Key function**: `calculate_derived_measurements(width_in, height_in, qty, element_type)`
- **Returns**: head_trim_lf, jamb_trim_lf, sill_trim_lf, area_sf, etc.

### geometry/calculations.py
- **Purpose**: Convert predictions to real measurements
- **Key function**: `calculate_real_measurements(predictions, scale_ratio, dpi)`

### database/client.py
- **Purpose**: Raw Supabase REST API
- **Key function**: `supabase_request(method, endpoint, data, filters)`

### database/storage.py
- **Purpose**: File uploads to Supabase storage
- **Key function**: `upload_to_storage(image_data, filename, content_type)`

### database/repositories/*.py
- **Purpose**: CRUD operations for specific tables
- **Pattern**: `get_X()`, `create_X()`, `update_X()`, `delete_X()`, `list_X()`

---

## Common Gotchas

1. **Import errors**: Use relative imports within packages
   ```python
   # In services/takeoff_service.py
   from database import get_page  # Good
   from database.repositories.page_repository import get_page  # Also works
   ```

2. **Circular imports**: If you get these, move the import inside the function
   ```python
   def my_function():
       from services.other_service import other_function  # Deferred import
       return other_function()
   ```

3. **Missing __init__.py**: Every folder needs an `__init__.py` file

4. **Background threads**: Services called with `threading.Thread()` run asynchronously
   ```python
   # This returns immediately, work happens in background
   threading.Thread(target=process_job_background, args=(job_id,)).start()
   ```

5. **Database filters**: Supabase uses PostgREST syntax
   ```python
   filters={'job_id': f'eq.{job_id}', 'status': 'eq.complete', 'order': 'page_number'}
   ```
