# If you want to separate generation from chain
from models.gemini_llm import GeminiLLM
from models.prompts import RAG_SYSTEM_PROMPT

def generate_response(context, query):
    llm = GeminiLLM()
    prompt = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser Query: {query}\n\nResponse:"
    return llm.generate(prompt)