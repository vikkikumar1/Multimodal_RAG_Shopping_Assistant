import os
import time
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency at module import time
def _import_genai():
    try:
        import google.generativeai as genai
        return genai
    except Exception:
        return None


def configure(api_key=None):
    genai = _import_genai()
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logger.warning('GOOGLE_API_KEY not set; genai calls will fail if attempted')
    if genai is not None and api_key:
        try:
            genai.configure(api_key=api_key)
            logger.debug('Configured genai client')
        except Exception as e:
            logger.warning(f'genai configure failed: {e}')


def generate_text(prompt, model_name='gemini-pro', temperature=0.7, max_tokens=512, retries=2, backoff=1):
    """Generate text using Google Generative AI. Falls back to a mock if unavailable."""
    # Mock support for tests/local dev
    if os.getenv('MOCK_GENAI') == '1' or os.getenv('TEST_MODE') == '1':
        return "(mocked genai response)"

    genai = _import_genai()
    if genai is None:
        raise RuntimeError('google.generativeai not available')

    model = genai.GenerativeModel(model_name)
    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    'temperature': temperature,
                    'max_output_tokens': max_tokens
                }
            )
            return getattr(response, 'text', str(response))
        except Exception as e:
            logger.warning(f'genai generate_text attempt {attempt} failed: {e}')
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise


def generate_with_image(prompt, image, model_name='gemini-pro-vision', retries=2, backoff=1):
    """Generate content using an image (PIL Image or path)."""
    if os.getenv('MOCK_GENAI') == '1' or os.getenv('TEST_MODE') == '1':
        return "(mocked vision response)"

    genai = _import_genai()
    if genai is None:
        raise RuntimeError('google.generativeai not available')

    model = genai.GenerativeModel(model_name)
    for attempt in range(retries + 1):
        try:
            # The original code passed [prompt, img]
            response = model.generate_content([prompt, image])
            return getattr(response, 'text', str(response))
        except Exception as e:
            logger.warning(f'genai generate_with_image attempt {attempt} failed: {e}')
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise
