# Construction Plan Extraction API

Unified API combining Command A (OCR) + Roboflow (Detection) for construction plan analysis.

## Endpoints

### `GET /health`
Health check endpoint.

### `POST /extract`
Full extraction pipeline - PDF or images → Command A classification → Roboflow detection → Combined output.

**Request:**
```json
{
  "pdf_url": "https://example.com/plans.pdf",
  "scale_config": {
    "scale_inches": 0.1875,
    "scale_feet": 1,
    "dpi": 144,
    "original_width": 5185,
    "original_height": 3456
  },
  "project_id": "project-123"
}
```

**Or with image URLs:**
```json
{
  "image_urls": [
    "https://example.com/page1.png",
    "https://example.com/page2.png"
  ],
  "scale_config": {...},
  "project_id": "project-123"
}
```

### `POST /classify`
Command A classification only.

**Request:**
```json
{
  "image_url": "https://example.com/page.png"
}
```

### `POST /detect`
Roboflow detection only.

**Request:**
```json
{
  "image_url": "https://example.com/page.png",
  "scale_config": {...}
}
```

### `POST /markup`
Generate siding markup visualization.

**Request:**
```json
{
  "image_url": "https://example.com/page.png",
  "predictions": [...],
  "calculations": {...}
}
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run
python app.py
```

## Docker

```bash
# Build
docker build -t extraction-api .

# Run
docker run -p 5050:5050 --env-file .env extraction-api
```

## Railway Deployment

1. Create new project in Railway
2. Connect GitHub repo (or use CLI)
3. Add environment variables in Railway dashboard:
   - `COHERE_API_KEY`
   - `ROBOFLOW_API_KEY`
   - `ROBOFLOW_MODEL`
4. Deploy

```bash
# Using Railway CLI
railway login
railway init
railway up
```

## Output Format

The `/extract` endpoint returns HOVER-compatible measurements:

```json
{
  "project_id": "project-123",
  "pages": [...],
  "elevation_pages": [9, 10, 11],
  "schedule_data": {
    "windows": [{"tag": "W1", "width": "3'-0\"", "height": "4'-0\""}],
    "doors": [{"tag": "D1", "width": "3'-0\"", "height": "6'-8\""}]
  },
  "combined_measurements": {
    "net_siding_sqft": 2219.5,
    "gross_wall_sqft": 3740.5,
    "window_sqft": 1297.6,
    "door_sqft": 223.4,
    "window_count": 48,
    "door_count": 6,
    "window_perimeter_lf": 672.0,
    "door_perimeter_lf": 116.0
  }
}
```
# v2
