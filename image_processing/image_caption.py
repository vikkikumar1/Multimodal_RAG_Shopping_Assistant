from models.gemini_vision import GeminiVision
from utils.logger import setup_logger

logger = setup_logger(__name__)
_vision = None

def _get_vision():
    global _vision
    if _vision is None:
        _vision = GeminiVision()
    return _vision

def generate_caption(image_path):
    return _get_vision().generate_caption(image_path)

def extract_attributes(image_path):
    return _get_vision().extract_attributes(image_path)