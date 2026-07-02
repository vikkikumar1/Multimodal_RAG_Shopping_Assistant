import os
from PIL import Image
import logging
from utils.config import UPLOAD_DIR
from models.genai_client import configure, generate_with_image

logger = logging.getLogger(__name__)


class GeminiVision:
    def __init__(self, api_key=None, model_name=None):
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
        configure(api_key)
        self.model_name = model_name or os.getenv('VISION_MODEL', 'models/gemini-2.0-flash')
        # If you need higher quality, use 'gemini-1.5-pro'

    def generate_caption(self, image_path, prompt="Describe this product image in detail."):
        try:
            img = Image.open(image_path)
            return generate_with_image(prompt, img, model_name=self.model_name)
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return "No description available."

    def extract_attributes(self, image_path):
        prompt = """You are a product analyst. List key attributes of this product in a concise, structured way:
- Category
- Color
- Material
- Style
- Any visible text/logo
- Estimated price range
Return as plain text with bullet points."""
        try:
            img = Image.open(image_path)
            return generate_with_image(prompt, img, model_name=self.model_name)
        except Exception as e:
            logger.error(f"Attribute extraction error: {e}")
            return "Could not extract attributes."