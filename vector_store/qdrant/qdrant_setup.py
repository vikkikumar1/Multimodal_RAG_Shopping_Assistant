from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

class QdrantVectorStore:
    def __init__(self, host="localhost", port=6333, collection_name="products"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, dimension=384):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE)
        )

    def upsert_vectors(self, ids, vectors, payloads):
        points = [
            models.PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, k=5, filter_conditions=None):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=k,
            query_filter=filter_conditions
        )