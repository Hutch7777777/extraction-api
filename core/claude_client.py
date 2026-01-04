"""
Claude Vision API client
"""

import base64
import json
import re
import requests
from config import config


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
        prompt = """Analyze this architectural drawing. Return ONLY valid JSON:

{
  "page_type": "elevation" | "schedule" | "floor_plan" | "section" | "detail" | "cover" | "site_plan" | "other",
  "confidence": 0.0 to 1.0,
  "elevation_name": "front" | "rear" | "left" | "right" | "north" | "south" | "east" | "west" | null,
  "scale_notation": "exact scale text from drawing or null",
  "scale_location": "where scale was found",
  "contains_schedule": true | false,
  "schedule_type": "window" | "door" | "window_and_door" | null,
  "notes": "brief description"
}"""
        
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
        prompt = """Extract the window and door schedule from this architectural drawing.

Return ONLY valid JSON:
{
  "windows": [{"tag": "W1", "width_inches": 36, "height_inches": 48, "type": "DH", "qty": 4, "notes": ""}],
  "doors": [{"tag": "D1", "width_inches": 36, "height_inches": 80, "type": "Entry", "qty": 1, "notes": ""}],
  "raw_text": "additional text"
}"""
        
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
        prompt = """Analyze this architectural floor plan. Focus ONLY on the EXTERIOR BUILDING PERIMETER (the outermost walls that define the building footprint - typically shown as thick black lines).

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
}"""
        
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
