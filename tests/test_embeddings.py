import pytest
from embeddings.text_embeddings import TextEmbedder

def test_text_embedder():
    embedder = TextEmbedder()
    emb = embedder.model.encode(["test"])
    assert emb.shape[1] == 384