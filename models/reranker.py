from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=5):
        # candidates: list of dicts with 'metadata' and 'text' (combined text)
        pairs = [[query, cand['text']] for cand in candidates]
        scores = self.model.predict(pairs)
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        reranked = [candidates[i] for i in sorted_indices]
        return reranked