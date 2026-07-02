import os
from utils.logger import setup_logger
from models.genai_client import configure, generate_text

logger = setup_logger(__name__)

class GeminiLLM:
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or os.getenv('LLM_MODEL', 'models/gemini-2.0-flash')
        configure(api_key)

    def generate(self, prompt, temperature=0.7, max_tokens=1024):
        try:
            return generate_text(prompt, model_name=self.model_name, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            error_msg = str(e)
            # Check if it's a quota/rate-limit error (429)
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.warning(f"Gemini rate limit hit: {error_msg}")
                return "⏳ Gemini API rate limit reached. Please wait and try again later."
            else:
                logger.error(f"LLM generation error: {e}")
                return "Sorry, I couldn't generate a response due to an unexpected error."

    def generate_with_context(self, context, query, system_prompt=None):
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser query: {query}"
        else:
            full_prompt = f"Context:\n{context}\n\nUser query: {query}"
        return self.generate(full_prompt)