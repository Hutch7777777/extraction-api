"""
Claude Vision API client - Enhanced with Elevation OCR
"""

import base64
import json
import re
import time
import requests
from config import config


def parse_dimension_string(dim_text):
    """
    Parse architectural dimension string to inches.
    
    Handles formats:
    - "12'-6" or 12'-6" -> 150 inches
    - "12' 6" -> 150 inches  
    - "12'-6 1/2" -> 150.5 inches
    - "36" -> 36 inches
    - "3'-0" -> 36 inches
    - "9'-0" -> 108 inches
    
    Returns:
        float: Value in inches, or None if parsing fails
    """
    if not dim_text:
        return None
    
    # Clean up the string - normalize quotes and apostrophes
    text = str(dim_text).strip().upper()
    # Replace various quote characters with standard ones
    for char in ['"', '"', '″', "''", "''"]:
        text = text.replace(char, '"')
    for char in ["'", "'", "′"]:
        text = text.replace(char, "'")
    
    try:
        # Pattern: feet and inches like 12'-6" or 12' 6"
        match = re.match(r"(\d+)['\-]\s*(\d+)(?:\s*(\d+)/(\d+))?\s*\"?", text)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            # Handle fractions like 1/2
            if match.group(3) and match.group(4):
                inches += int(match.group(3)) / int(match.group(4))
            return feet * 12 + inches
        
        # Pattern: feet only like 12'-0" or 12'
        match = re.match(r"(\d+)['\-]\s*0?\s*\"?", text)
        if match:
            return int(match.group(1)) * 12
        
        # Pattern: inches only like 36"
        match = re.match(r"(\d+)(?:\s*(\d+)/(\d+))?\s*\"", text)
        if match:
            inches = int(match.group(1))
            if match.group(2) and match.group(3):
                inches += int(match.group(2)) / int(match.group(3))
            return inches
        
        # Pattern: decimal feet like 9.5'
        match = re.match(r"(\d+\.?\d*)\s*['\s]", text)
        if match:
            return float(match.group(1)) * 12
        
        return None
    except:
        return None


class ClaudeClient:
    """Claude Vision API wrapper"""
    
    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-sonnet-4-20250514"
    
    def _make_request(self, image_base64, prompt, max_tokens=1000, media_type="image/png"):
        """Make API request to Claude"""
        if not self.api_key:
            return {"error": "No API key configured"}
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_json(self, text):
        """Extract JSON from Claude's response"""
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None
    
    def _download_and_encode(self, image_url):
        """Download image and encode as base64"""
        response = requests.get(image_url, timeout=30)
        return base64.b64encode(response.content).decode('utf-8')
    
    def classify_page(self, image_url):
        """
        Classify an architectural page.
        
        Returns:
            Dict with page_type, confidence, elevation_name, scale_notation, etc.
        """
        prompt = '''Analyze this architectural drawing. Return ONLY valid JSON:

{
  "page_type": "elevation" | "schedule" | "floor_plan" | "section" | "detail" | "cover" | "site_plan" | "other",
  "confidence": 0.0 to 1.0,
  "elevation_name": "front" | "rear" | "left" | "right" | "north" | "south" | "east" | "west" | null,
  "scale_notation": "exact scale text from drawing or null",
  "scale_location": "where scale was found",
  "contains_schedule": true | false,
  "schedule_type": "window" | "door" | "window_and_door" | null,
  "notes": "brief description"
}'''
        
        try:
            image_base64 = self._download_and_encode(image_url)
            result = self._make_request(image_base64, prompt, max_tokens=500)
            
            if 'error' in result:
                return {"page_type": "other", "confidence": 0, "error": result['error']}
            
            text = result.get('content', [{}])[0].get('text', '{}')
            parsed = self._extract_json(text)
            
            if parsed:
                # Normalize page type
                from utils.validation import normalize_page_type
                parsed['page_type'] = normalize_page_type(parsed.get('page_type'))
                
                # Parse scale if found
                if parsed.get('scale_notation'):
                    from geometry import parse_scale_notation
                    parsed['scale_ratio'] = parse_scale_notation(parsed['scale_notation'])
                
                return parsed
            
            return {"page_type": "other", "confidence": 0}
        
        except Exception as e:
            return {"page_type": "other", "confidence": 0, "error": str(e)}
    
    def extract_schedule(self, image_url):
        """
        Extract window/door schedule from page.
        
        Returns:
            Dict with windows, doors lists
        """
        prompt = '''Extract the window and door schedule from this architectural drawing.

Return ONLY valid JSON:
{
  "windows": [{"tag": "W1", "width_inches": 36, "height_inches": 48, "type": "DH", "qty": 4, "notes": ""}],
  "doors": [{"tag": "D1", "width_inches": 36, "height_inches": 80, "type": "Entry", "qty": 1, "notes": ""}],
  "raw_text": "additional text"
}'''
        
        try:
            image_base64 = self._download_and_encode(image_url)
            result = self._make_request(image_base64, prompt, max_tokens=4000)
            
            if 'error' in result:
                return {"error": result['error']}
            
            text = result.get('content', [{}])[0].get('text', '{}')
            parsed = self._extract_json(text)
            
            return parsed if parsed else {"error": "Failed to parse response"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_floor_plan_corners(self, image_url):
        """
        Count corners from floor plan.
        
        Returns:
            Dict with outside_corners, inside_corners counts
        """
        prompt = '''Analyze this architectural floor plan. Focus ONLY on the EXTERIOR BUILDING PERIMETER (the outermost walls that define the building footprint - typically shown as thick black lines).

TASK: Trace the exterior perimeter and count corners.

DEFINITIONS:
- OUTSIDE CORNER (convex): Where two exterior walls meet and point OUTWARD (away from building interior). These are the "bump out" corners.
- INSIDE CORNER (concave): Where two exterior walls meet and point INWARD (into the building). These are the "cut in" corners, like where an L-shape occurs.

DO NOT COUNT:
- Interior wall corners (rooms, closets, etc.)
- Window or door openings
- Garage door openings
- Porch railings or deck edges (only count if they have siding)

If there are multiple floor plans or units on this sheet, analyze each separately.

Return ONLY valid JSON in this exact format:
{
  "floor_plans": [
    {
      "name": "Unit/Floor Plan Name",
      "outside_corners": <integer>,
      "inside_corners": <integer>,
      "confidence": "high/medium/low",
      "notes": "any observations about complexity"
    }
  ],
  "total_outside_corners": <sum of all>,
  "total_inside_corners": <sum of all>
}'''
        
        try:
            image_base64 = self._download_and_encode(image_url)
            result = self._make_request(image_base64, prompt, max_tokens=1000)
            
            if 'error' in result:
                return {"error": result['error']}
            
            text = result.get('content', [{}])[0].get('text', '{}')
            
            # Clean response
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            return json.loads(text.strip())
        
        except Exception as e:
            return {"error": str(e)}
    
    def extract_elevation_dimensions(self, image_url):
        """
        Extract dimension text and annotations from an elevation drawing.
        
        This method uses Claude Vision to OCR architectural elevation drawings,
        extracting:
        - Wall heights (floor-to-floor, plate heights)
        - Window/door callout tags (W-1, D-2, etc.)
        - Dimension strings with their locations
        - Level markers (T.O. PLATE, T.O. SLAB, etc.)
        - Eave and ridge heights
        
        Args:
            image_url: URL to the elevation image
        
        Returns:
            Dict with extracted dimension data
        """
        prompt = '''Analyze this architectural ELEVATION drawing and extract ALL dimension and annotation information.

IMPORTANT: This is an elevation view (side view of building), NOT a floor plan.

EXTRACT THE FOLLOWING:

1. WALL HEIGHTS - Vertical dimensions showing:
   - Floor-to-floor heights
   - Plate heights (like 9'-0", 8'-1 1/4")
   - Story heights
   Look for vertical dimension lines or text like 9'-0", PLATE HT., etc.

2. WINDOW/DOOR CALLOUTS - Tags near openings:
   - Window tags: W-1, W1, A9.06, 3050, etc.
   - Door tags: D-1, D1, etc.
   - Note the approximate pixel position (x, y from top-left corner)

3. DIMENSION TEXT - Any dimension strings visible:
   - Horizontal dimensions (widths)
   - Vertical dimensions (heights)
   - Include the raw text exactly as shown
   - Note if horizontal or vertical
   - Estimate pixel position

4. LEVEL MARKERS - Reference points with heights:
   - T.O. PLATE (Top of Plate)
   - T.O. SUBFLOOR
   - T.O. SLAB
   - T.O. FDN (Top of Foundation)
   - FIN. FLR (Finished Floor)
   Include the height text shown (like 19'-4 1/4")

5. BUILDING HEIGHTS:
   - EAVE HEIGHT (where roof meets wall)
   - RIDGE HEIGHT (peak of roof)
   - Overall building height

Return ONLY valid JSON in this exact format:
{
  "wall_heights": [
    {"value_text": "9'-0", "value_inches": 108, "label": "1ST FLOOR", "confidence": 0.95}
  ],
  "dimension_text": [
    {"raw_text": "12'-6", "value_inches": 150, "x_pixel": 450, "y_pixel": 320, "orientation": "horizontal", "confidence": 0.9}
  ],
  "element_callouts": [
    {"tag": "W-1", "x_pixel": 200, "y_pixel": 400, "element_type": "window"},
    {"tag": "D-1", "x_pixel": 600, "y_pixel": 500, "element_type": "door"}
  ],
  "level_markers": [
    {"label": "T.O. PLATE 2ND FLR", "height_text": "19'-4 1/4", "height_inches": 232.25, "y_pixel": 180}
  ],
  "eave_height_ft": 18.5,
  "ridge_height_ft": 24.0,
  "average_wall_height_ft": 9.0,
  "total_building_height_ft": 24.0,
  "extraction_confidence": 0.85,
  "notes": "any relevant observations"
}

IMPORTANT NOTES:
- Convert all dimensions to inches for value_inches fields
- For feet-inches notation: 9'-6" = 114 inches
- Pixel coordinates are approximate (estimate from top-left)
- Set confidence 0.0-1.0 based on text clarity
- If no data found for a category, return empty array []
- If heights cannot be determined, set to null'''

        start_time = time.time()
        
        try:
            image_base64 = self._download_and_encode(image_url)
            result = self._make_request(image_base64, prompt, max_tokens=4000)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if 'error' in result:
                return {
                    "error": result['error'],
                    "wall_heights": [],
                    "dimension_text": [],
                    "element_callouts": [],
                    "level_markers": [],
                    "extraction_confidence": 0
                }
            
            text = result.get('content', [{}])[0].get('text', '{}')
            
            # Clean response - handle markdown code blocks
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            parsed = json.loads(text.strip())
            
            # Post-process: validate and clean dimension values
            parsed = self._validate_ocr_results(parsed)
            
            # Add processing metadata
            parsed['processing_time_ms'] = processing_time
            parsed['raw_response'] = result
            
            return parsed
        
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parse error: {str(e)}",
                "raw_text": text if 'text' in dir() else None,
                "wall_heights": [],
                "dimension_text": [],
                "element_callouts": [],
                "level_markers": [],
                "extraction_confidence": 0
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "wall_heights": [],
                "dimension_text": [],
                "element_callouts": [],
                "level_markers": [],
                "extraction_confidence": 0
            }
    
    def _validate_ocr_results(self, data):
        """
        Validate and clean OCR extraction results.
        
        - Ensures arrays exist
        - Validates dimension conversions
        - Calculates missing values
        """
        # Ensure required arrays exist
        data.setdefault('wall_heights', [])
        data.setdefault('dimension_text', [])
        data.setdefault('element_callouts', [])
        data.setdefault('level_markers', [])
        
        # Validate wall heights - recalculate inches from text if needed
        for wh in data.get('wall_heights', []):
            if wh.get('value_text') and not wh.get('value_inches'):
                wh['value_inches'] = parse_dimension_string(wh['value_text'])
        
        # Validate dimension text
        for dt in data.get('dimension_text', []):
            if dt.get('raw_text') and not dt.get('value_inches'):
                dt['value_inches'] = parse_dimension_string(dt['raw_text'])
        
        # Validate level markers
        for lm in data.get('level_markers', []):
            if lm.get('height_text') and not lm.get('height_inches'):
                lm['height_inches'] = parse_dimension_string(lm['height_text'])
        
        # Calculate average wall height if we have wall heights but no average
        if data.get('wall_heights') and not data.get('average_wall_height_ft'):
            heights_inches = [wh.get('value_inches') for wh in data['wall_heights'] if wh.get('value_inches')]
            if heights_inches:
                avg_inches = sum(heights_inches) / len(heights_inches)
                data['average_wall_height_ft'] = round(avg_inches / 12, 2)
        
        # Ensure confidence is present
        data.setdefault('extraction_confidence', 0.5)
        
        return data


# Singleton instance
claude_client = ClaudeClient()


# Backward-compatible function aliases
def classify_page_with_claude(image_url):
    """Classify page using Claude Vision"""
    return claude_client.classify_page(image_url)


def ocr_schedule_with_claude(image_url):
    """Extract schedule using Claude Vision"""
    return claude_client.extract_schedule(image_url)


def analyze_floor_plan_corners(image_url):
    """Analyze floor plan corners using Claude Vision"""
    return claude_client.analyze_floor_plan_corners(image_url)


def extract_elevation_dimensions(image_url):
    """Extract dimensions from elevation drawing using Claude Vision"""
    return claude_client.extract_elevation_dimensions(image_url)
