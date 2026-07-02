import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import os
from utils.config import EMBEDDINGS_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TextEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        logger.info(f"Loaded text embedding model {model_name} with dimension {self.dimension}")

    def embed_texts(self, texts, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i+batch_size]
            batch_embeds = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeds)
        return np.vstack(embeddings)

    def embed_products(self, df, text_column='combined_text'):
        texts = df[text_column].astype(str).tolist()
        embeddings = self.embed_texts(texts)
        # Save embeddings and ids
        np.save(EMBEDDINGS_DIR / 'product_text_embeddings.npy', embeddings)
        with open(EMBEDDINGS_DIR / 'product_ids.pkl', 'wb') as f:
            pickle.dump(df['id'].tolist(), f)
        logger.info(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_DIR}")
        return embeddings

if __name__ == "__main__":
    from utils.data_loader import load_product_data
    df = load_product_data()
    embedder = TextEmbedder()
    embedder.embed_products(df)