# Wrapper for attribute extraction
from models.gemini_vision import GeminiVision

_vision = None

def _get_vision():
    global _vision
    if _vision is None:
        _vision = GeminiVision()
    return _vision

def extract_attributes(image_path):
    return _get_vision().extract_attributes(image_path)
