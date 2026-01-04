# Extraction API v4.0 - Modular Architecture

## Overview

This is the restructured extraction API with a modular architecture. The original 1,400-line `app.py` has been split into focused modules:

```
extraction-api-v2/
├── app.py                          # Flask entry point (routing only)
├── config.py                       # Centralized configuration
│
├── core/                           # External API clients
│   ├── roboflow_client.py          # Roboflow detection
│   └── claude_client.py            # Claude Vision API
│
├── services/                       # Business logic
│   ├── pdf_service.py              # PDF conversion
│   ├── classification_service.py   # Page classification
│   ├── extraction_service.py       # Main extraction orchestration
│   ├── markup_service.py           # Image markup generation
│   ├── takeoff_service.py          # Takeoff calculations
│   ├── cross_ref_service.py        # Schedule/detection matching
│   └── floor_plan_service.py       # Corner analysis
│
├── database/                       # Database layer
│   ├── client.py                   # Supabase client
│   ├── storage.py                  # File storage
│   └── repositories/               # CRUD operations
│       ├── job_repository.py
│       ├── page_repository.py
│       └── detection_repository.py
│
├── geometry/                       # Calculation utilities
│   ├── scale_parser.py             # Scale notation parsing
│   ├── measurements.py             # Derived measurements
│   └── calculations.py             # Detection calculations
│
└── utils/                          # Shared utilities
    └── validation.py               # Input validation
```

## Installation

1. **Copy files to your extraction-api directory:**
   ```bash
   # Backup existing app.py
   cd ~/Documents/extraction-api
   cp app.py app.py.backup
   
   # Copy new structure (from extracted zip)
   cp -r extraction-api-v2/* ~/Documents/extraction-api/
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install flask flask-cors python-dotenv requests pillow pdf2image anthropic
   ```

3. **Verify environment variables** in `.env`:
   ```env
   ROBOFLOW_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   SUPABASE_URL=https://okwtyttfqbfmcqtenize.supabase.co
   SUPABASE_KEY=your_key
   PORT=5050
   ```

4. **Test locally:**
   ```bash
   python app.py
   # Then: curl http://localhost:5050/health
   ```

## Key Changes

### Before (Monolithic)
- Single 1,400-line `app.py`
- All functions mixed together
- Hard to test individual components
- Difficult to modify without breaking things

### After (Modular)
- `app.py` reduced to ~500 lines (routing only)
- Logic separated into focused modules
- Each module can be tested independently
- Changes are isolated and safe

## API Endpoints

All endpoints remain the same - this is a **non-breaking refactor**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/start-job` | POST | Start extraction job |
| `/classify-job` | POST | Classify pages |
| `/process-job` | POST | Process pages |
| `/generate-markups` | POST | Generate markups |
| `/comprehensive-markup` | POST | Comprehensive markup |
| `/calculate-takeoff` | POST | Calculate takeoff |
| `/job-takeoff` | GET | Get takeoff totals |
| `/cross-reference` | POST | Build cross-refs |
| `/takeoff-summary` | GET | Get summary |
| `/cross-refs` | GET | Get cross-ref data |
| `/analyze-floor-plan` | POST | Analyze corners |
| `/generate-facade-markup` | POST | Facade markup |
| `/job-status` | GET | Job status |
| `/list-jobs` | GET | List jobs |
| `/parse-scale` | POST | Parse scale notation |

## Testing

Test each module independently:

```python
# Test config
from config import config
print(config.SUPABASE_URL)

# Test database
from database import list_jobs
print(list_jobs())

# Test geometry
from geometry import parse_scale_notation
print(parse_scale_notation('1/4" = 1\'-0"'))  # → 48.0

# Test full app
# curl http://localhost:5050/health
```

## Deployment

Deploy to Railway the same way as before - just push to GitHub:

```bash
git add .
git commit -m "Refactor: Modular architecture v4.0"
git push origin main
```

Railway will detect `app.py` and run it automatically.

## Next Steps

With this modular foundation, you can now:

1. **Add unit tests** for each module
2. **Enhance OCR** in `core/claude_client.py`
3. **Add schedule extraction** in `services/schedule_service.py`
4. **Implement tag matching** in `services/matching_service.py`
5. **Add linear elements** in `services/linear_service.py`

Each enhancement is isolated and won't break existing functionality.
