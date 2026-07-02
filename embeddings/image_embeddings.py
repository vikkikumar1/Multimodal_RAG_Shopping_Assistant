# If you want to also embed product images for direct visual similarity
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.config import RAW_DATA_DIR, EMBEDDINGS_DIR
import pickle

class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding.cpu().numpy().flatten()

    def embed_all_images(self, image_dir, output_path=None):
        image_paths = list(Path(image_dir).glob("*.[jp][pn]g")) + list(Path(image_dir).glob("*.jpeg"))
        embeddings = []
        ids = []
        for img_path in tqdm(image_paths, desc="Embedding images"):
            emb = self.embed_image(img_path)
            embeddings.append(emb)
            ids.append(img_path.stem)  # assume filename is product id
        embeddings = np.array(embeddings)
        if output_path is None:
            output_path = EMBEDDINGS_DIR / "product_image_embeddings.npy"
        np.save(output_path, embeddings)
        with open(EMBEDDINGS_DIR / "image_product_ids.pkl", 'wb') as f:
            pickle.dump(ids, f)
        return embeddings