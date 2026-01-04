# Extraction API v4.0 - Context for Claude

## Quick Reference

**Repository**: https://github.com/Hutch7777777/extraction-api
**Production URL**: https://extraction-api-production.up.railway.app
**Local Path**: ~/Documents/extraction-api

## Architecture (Modular v4.0)

```
app.py              → Flask routing ONLY (no business logic)
config.py           → All configuration
core/               → External APIs (Roboflow, Claude Vision)
services/           → Business logic (PDF, classification, extraction, markup, takeoff)
database/           → Supabase client + repositories
geometry/           → Pure calculation functions
utils/              → Validation helpers
```

## Key Files & Their Purpose

| File | Purpose | When to Modify |
|------|---------|----------------|
| `app.py` | Flask endpoints | Adding/removing routes |
| `config.py` | Settings, colors, trade groups | Config changes |
| `services/extraction_service.py` | Main workflow orchestration | Changing extraction flow |
| `services/takeoff_service.py` | Measurement calculations | Calculation formula changes |
| `services/markup_service.py` | Visual markup generation | Markup appearance changes |
| `services/cross_ref_service.py` | Schedule ↔ detection matching | Cross-reference logic |
| `core/claude_client.py` | Claude Vision API | OCR prompts, classification |
| `core/roboflow_client.py` | Roboflow detection | Detection workflow changes |
| `geometry/scale_parser.py` | Scale notation parsing | Adding scale formats |
| `geometry/measurements.py` | Derived measurements | Trim/flashing formulas |

## Database Tables Used

| Table | Purpose |
|-------|---------|
| `extraction_jobs` | Job metadata and status |
| `extraction_pages` | Individual pages from PDF |
| `extraction_detection_details` | Individual detections per page |
| `extraction_elevation_calcs` | Aggregated calcs per elevation |
| `extraction_job_totals` | Aggregated calcs per job |
| `extraction_cross_refs` | Schedule ↔ detection links |
| `extraction_takeoff_summary` | Final takeoff summary |

## API Endpoints

| Endpoint | Method | Service Function |
|----------|--------|------------------|
| `/start-job` | POST | `pdf_service.convert_pdf_background()` |
| `/classify-job` | POST | `classification_service.classify_job_background()` |
| `/process-job` | POST | `extraction_service.process_job_background()` |
| `/generate-markups` | POST | `markup_service.generate_markups_for_job()` |
| `/comprehensive-markup` | POST | `markup_service.generate_comprehensive_markup()` |
| `/calculate-takeoff` | POST | `takeoff_service.calculate_takeoff_for_job()` |
| `/job-takeoff` | GET | Direct DB query |
| `/cross-reference` | POST | `cross_ref_service.build_cross_references()` |
| `/analyze-floor-plan` | POST | `floor_plan_service.analyze_floor_plan_for_job()` |

## Critical Formulas

```python
# Gross facade calculation (roof excluded from siding)
gross_facade_sf = building_area_sf - roof_area_sf

# Net siding calculation
net_siding_sf = gross_facade_sf - window_area - door_area - garage_area + gable_area

# Gable area (triangle)
gable_area_sf = (width * height) / 144 / 2

# Gable rake length
rake_lf = sqrt((width_ft/2)² + height_ft²) * 2

# Scale conversion
real_inches = pixels * (1/dpi) * scale_ratio
```

## Making Changes

### Adding a New Endpoint
1. Create function in appropriate `services/*.py`
2. Add route in `app.py` that calls the service
3. Test locally, then push to GitHub

### Modifying Calculations
1. Find relevant function in `services/takeoff_service.py` or `geometry/`
2. Make changes
3. Test with existing job data

### Changing OCR/Classification
1. Modify prompts in `core/claude_client.py`
2. Update parsing logic if response format changes

## Import Patterns

```python
# From services
from services.takeoff_service import calculate_takeoff_for_job

# From database
from database import get_page, update_page, get_elevation_pages

# From geometry
from geometry import parse_scale_notation, calculate_derived_measurements

# From config
from config import config
```

## Testing

```bash
# Local
cd ~/Documents/extraction-api
python3 app.py
curl http://localhost:5050/health

# Production
curl https://extraction-api-production.up.railway.app/health
```

## Deployment

```bash
git add .
git commit -m "Description"
git push origin main
# Railway auto-deploys
```

## Environment Variables (Railway)

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_KEY` - Supabase service role key
- `ANTHROPIC_API_KEY` - Claude API key
- `ROBOFLOW_API_KEY` - Roboflow API key
- `PORT` - Usually auto-set by Railway
