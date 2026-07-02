"""Build FAISS index from saved embeddings and processed metadata.

Usage:
    python scripts/build_index.py --embeddings path/to/emb.npy --metadata path/to/metadata.json

If no paths provided, this script will look for:
- embeddings: `embeddings/generated/product_text_embeddings.npy`
- metadata: `data/processed/product_metadata.json` or `data/processed/cleaned_products.csv`

It will write the FAISS index and metadata to the directory configured in `utils.config.FAISS_INDEX_DIR`.
"""
import argparse
import numpy as np
import pickle
import json
import pandas as pd
from pathlib import Path
from utils.config import EMBEDDINGS_DIR, PROCESSED_DATA_DIR, FAISS_INDEX_DIR
from vector_store.vector_db import FAISSVectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, help='Path to embeddings .npy file')
    parser.add_argument('--metadata', type=str, help='Path to metadata json or csv')
    args = parser.parse_args()

    emb_path = Path(args.embeddings) if args.embeddings else EMBEDDINGS_DIR / 'product_text_embeddings.npy'
    if not emb_path.exists():
        # fallback to existing faiss_index embeddings
        fallback = FAISS_INDEX_DIR / 'embeddings.npy'
        if fallback.exists():
            emb_path = fallback
        else:
            raise FileNotFoundError(f"Embeddings file not found at {emb_path} and no fallback available.")

    embeddings = np.load(str(emb_path))

    # Load metadata
    if args.metadata:
        meta_path = Path(args.metadata)
    else:
        meta_json = PROCESSED_DATA_DIR / 'product_metadata.json'
        csv_path = PROCESSED_DATA_DIR / 'cleaned_products.csv'
        if meta_json.exists():
            meta_path = meta_json
        elif csv_path.exists():
            meta_path = csv_path
        else:
            raise FileNotFoundError('No metadata found in processed data directory.')

    if meta_path.suffix.lower() in ['.json']:
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        df = pd.read_csv(meta_path)
        metadata = df.to_dict(orient='records')

    ids = [m.get('id') for m in metadata]

    # Ensure FAISS index dir exists
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Build index
    store = FAISSVectorStore(dimension=embeddings.shape[1])
    store.build_index(embeddings, metadata, ids)

    # Save a copy of raw embeddings for reproducibility
    np.save(FAISS_INDEX_DIR / 'embeddings.npy', embeddings)
    with open(FAISS_INDEX_DIR / 'metadata.pkl', 'rb') as f:
        # metadata.pkl is already written by build_index; show path
        pass

    print(f"✅ Built FAISS index with {len(metadata)} items at {FAISS_INDEX_DIR}")


if __name__ == '__main__':
    main()
