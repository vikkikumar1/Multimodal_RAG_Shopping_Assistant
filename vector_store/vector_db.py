import faiss
import numpy as np
import pickle
import os
from utils.config import FAISS_INDEX_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FAISSVectorStore:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.metadata = []  # list of dicts with product info
        self.ids = []

    def build_index(self, embeddings, metadata_list, ids_list):
        """Build FAISS index from embeddings and store metadata."""
        if embeddings.shape[1] != self.dimension:
            logger.warning(f"Embedding dimension {embeddings.shape[1]} doesn't match {self.dimension}, adjusting")
            self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata_list
        self.ids = ids_list
        # Save
        faiss.write_index(self.index, str(FAISS_INDEX_DIR / 'index.faiss'))
        with open(FAISS_INDEX_DIR / 'metadata.pkl', 'wb') as f:
            pickle.dump((self.metadata, self.ids), f)
        logger.info(f"Built index with {len(metadata_list)} items, saved to {FAISS_INDEX_DIR}")
        return self.index

    def load_index(self):
        """Load existing index and metadata."""
        index_path = FAISS_INDEX_DIR / 'index.faiss'
        meta_path = FAISS_INDEX_DIR / 'metadata.pkl'
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError("FAISS index files not found. Run generate_embeddings first.")
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, 'rb') as f:
            self.metadata, self.ids = pickle.load(f)
        self.dimension = self.index.d
        logger.info(f"Loaded index with {len(self.metadata)} items")
        return self.index, self.metadata, self.ids

    def search(self, query_embedding, k=5):
        """Search top k similar products."""
        if self.index is None:
            self.load_index()
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    'metadata': self.metadata[idx],
                    'distance': float(dist),
                    'id': self.ids[idx] if idx < len(self.ids) else None
                })
        return results