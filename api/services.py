from rag.rag_chain import RAGChain

class RAGService:
    def __init__(self):
        self.rag_chain = RAGChain()

    def process_query(self, text_query=None, image_path=None, k=5):
        return self.rag_chain.run(text_query, image_path, k)