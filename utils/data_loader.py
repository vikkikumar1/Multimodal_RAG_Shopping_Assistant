import pandas as pd
import numpy as np
import json
from pathlib import Path
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.helper import clean_text

def find_image_path(product_id, image_dir):
    """
    Check if an image exists for the given product ID.
    Supports common extensions and returns the first match.
    """
    if not image_dir.exists():
        return ""
    pid = str(product_id)
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
        candidate = image_dir / f"{pid}{ext}"
        if candidate.exists():
            return str(candidate)
    return ""


def create_sample_products():
    """Fallback sample data if no CSV is found."""
    categories = ["Electronics", "Clothing", "Home", "Books", "Toys"]
    data = []
    for i in range(1, 21):
        data.append({
            "id": i,
            "title": f"Product {i}",
            "description": f"This is a great product from category {categories[i % 5]}.",
            "price": round(10 + i * 15.5, 2),
            "category": categories[i % 5],
            "image_url": f"https://picsum.photos/seed/{i}/200/200",
            "image_path": ""
        })
    return pd.DataFrame(data)


def load_product_data():
    """
    Load product data from CSV, clean, augment, and save processed files.
    Returns a DataFrame ready for embedding generation.
    """
    csv_path = RAW_DATA_DIR / "products.csv"

    if not csv_path.exists():
        print("⚠️ products.csv not found. Using sample data.")
        df = create_sample_products()
    else:
        df = pd.read_csv(csv_path, on_bad_lines='skip')

    # --- Build title and description ---
    if 'productDisplayName' in df.columns:
        df['title'] = df['productDisplayName'].fillna('').astype(str)
        text_cols = ['gender', 'masterCategory', 'subCategory', 'articleType',
                     'baseColour', 'season', 'usage']
        existing_text_cols = [col for col in text_cols if col in df.columns]
        # Build description from available columns, fill NaN with empty string
        df['description'] = df[existing_text_cols].fillna('').agg(' '.join, axis=1).astype(str)
        if 'year' in df.columns:
            df['description'] = df['description'] + " " + df['year'].astype(str)
    else:
        # Fallback: assume CSV has 'title' and 'description'
        df['title'] = df.get('title', '').fillna('').astype(str)
        df['description'] = df.get('description', '').fillna('').astype(str)

    # Ensure no NaN remains
    df['title'] = df['title'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str)

    # Clean text (optional: clean before combining? we do it after combining)
    # But for safety, we clean both fields
    df['title'] = df['title'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)

    # Create combined text for embedding
    df['combined_text'] = df['title'] + " " + df['description']
    # Remove rows with empty combined_text (after stripping)
    df = df[df['combined_text'].str.strip() != '']

    # --- Add price if missing ---
    if 'price' not in df.columns:
        if 'masterCategory' in df.columns:
            category_prices = {'Apparel': 35, 'Accessories': 25, 'Footwear': 55, 'Clothes': 30}
            df['price'] = df['masterCategory'].map(category_prices).fillna(30)
        else:
            df['price'] = 30
        df['price'] = df['price'] + np.random.randint(-5, 15, len(df))
        df['price'] = df['price'].clip(lower=5)

    # --- Category ---
    df['category'] = df.get('masterCategory', df.get('category', 'General')).fillna('General')

    # --- Link local images ---
    image_dir = RAW_DATA_DIR / 'images'
    df['image_path'] = df['id'].apply(
        lambda pid: find_image_path(pid, image_dir) if image_dir.exists() else ""
    ).astype(str)
    # For API serving: set image_url to a static route (will be mounted)
    df['image_url'] = df['image_path'].apply(
        lambda p: f"/static/images/{Path(p).name}" if p else "https://via.placeholder.com/150"
    )

    # --- Save processed data ---
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "cleaned_products.csv", index=False)

    # Save metadata for FAISS (list of dicts)
    metadata = df.to_dict(orient='records')
    with open(PROCESSED_DATA_DIR / "product_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Processed {len(df)} products. Data saved to {PROCESSED_DATA_DIR}")
    return df