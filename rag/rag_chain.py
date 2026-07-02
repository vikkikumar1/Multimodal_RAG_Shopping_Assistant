from rag.retriever import Retriever
from rag.context_builder import build_context
from models.gemini_llm import GeminiLLM
from models.prompts import RAG_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGChain:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = GeminiLLM()

    def run(self, text_query=None, image_path=None, k=5):
        # Retrieve
        results, caption, attributes = self.retriever.retrieve_hybrid(text_query, image_path, k)
        # Build context
        context = build_context(results)
        # Build prompt
        user_query = text_query if text_query else "Find similar products based on this image."
        prompt = RAG_PROMPT_TEMPLATE.format(
            system_prompt=RAG_SYSTEM_PROMPT,
            context=context,
            query=user_query
        )
        # Generate response
        response = self.llm.generate(prompt)
        return {
            'response': response,
            'retrieved_products': results,
            'image_caption': caption,
            'attributes': attributes,
            'context': context
        }