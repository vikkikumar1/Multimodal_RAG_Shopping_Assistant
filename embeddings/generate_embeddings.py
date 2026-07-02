# Master script to generate all embeddings (text and optionally image)
from utils.data_loader import load_product_data
from embeddings.text_embeddings import TextEmbedder
# from embeddings.image_embeddings import ImageEmbedder
from utils.logger import setup_logger
from vector_store.vector_db import FAISSVectorStore
from utils.config import FAISS_INDEX_DIR
import numpy as np
import os

logger = setup_logger(__name__)

def main():
    logger.info("Loading product data...")
    df = load_product_data()
    logger.info("Generating text embeddings...")
    text_embedder = TextEmbedder()
    embeddings = text_embedder.embed_products(df)

    # Optionally build FAISS index automatically if the env flag is set
    if os.getenv('AUTO_BUILD_FAISS', '1') == '1':
        logger.info('Building FAISS index from generated embeddings...')
        try:
            # metadata and ids are saved by data_loader
            metadata_path = "data/processed/product_metadata.json"
            ids = [r.get('id') for r in df.to_dict(orient='records')]
            FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            store = FAISSVectorStore(dimension=embeddings.shape[1])
            store.build_index(embeddings, df.to_dict(orient='records'), ids)
            np.save(FAISS_INDEX_DIR / 'embeddings.npy', embeddings)
            logger.info('FAISS index built successfully.')
        except Exception as e:
            logger.error(f'Failed to build FAISS index: {e}')
    
    # Optional: uncomment to generate image embeddings if you have images
    # from utils.config import RAW_DATA_DIR
    # image_dir = RAW_DATA_DIR / "images"
    # if image_dir.exists():
    #     logger.info("Generating image embeddings...")
    #     img_embedder = ImageEmbedder()
    #     img_embedder.embed_all_images(image_dir)
    # else:
    #     logger.warning("No image directory found, skipping image embeddings")
    logger.info("Done.")

if __name__ == "__main__":
    main()