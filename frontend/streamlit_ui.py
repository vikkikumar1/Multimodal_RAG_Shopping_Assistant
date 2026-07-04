import sys
from pathlib import Path

# ---- CRITICAL: Add project root to Python path ----
# Must be before any imports that use project modules (utils, rag, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---- Now import project modules ----
import os
import tempfile
import streamlit as st
from PIL import Image
from rag.rag_chain import RAGChain
from utils.config import FAISS_INDEX_DIR, PROCESSED_DATA_DIR

# ---- Generate FAISS index if missing (one-time on Render) ----
if not (FAISS_INDEX_DIR / "index.faiss").exists():
    print("🔄 FAISS index not found. Generating embeddings...")
    os.system("python -m embeddings.generate_embeddings")

# ---- Page config ----
st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")

@st.cache_resource
def get_rag():
    """Singleton RAG chain instance."""
    return RAGChain()

def display_product_card(meta, col):
    """Display a single product card in the given column."""
    with col:
        # 1. Try local image file first
        img_path = meta.get('image_path')
        if img_path and os.path.exists(img_path):
            st.image(img_path, width=150)
        else:
            # Fallback to web URL
            img_url = meta.get('image_url', 'https://via.placeholder.com/150')
            st.image(img_url, width=150)

        # 2. Title and price
        st.write(f"**{meta.get('title', 'Product')}**")
        st.write(f"💰 ${meta.get('price', 'N/A')}")

        # 3. Extra attributes (if available)
        extra = []
        for attr in ['baseColour', 'masterCategory']:
            if attr in meta and meta[attr]:
                extra.append(f"{attr.replace('master', '')}: {meta[attr]}")
        if extra:
            st.caption(", ".join(extra))

        # 4. Short description
        desc = meta.get('description', '')
        st.caption(desc[:80] + ("..." if len(desc) > 80 else ""))

def main():
    st.title("🛍️ AI Shopping Assistant")
    st.markdown("Upload an image or type a query to find similar products.")

    try:
        rag = get_rag()
    except Exception as e:
        st.error(f"Failed to initialize RAG backend: {e}")
        return

    # Sidebar inputs
    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader(
            "Upload product image",
            type=["jpg", "jpeg", "png"]
        )
        text_query = st.text_area(
            "Or enter text query",
            placeholder="e.g., 'Show me similar products under $500'"
        )
        k = st.slider("Number of results", 1, 10, 5)
        search_btn = st.button("🔍 Search")

    if not search_btn:
        return

    # Validate inputs
    if not text_query and not uploaded_file:
        st.error("Please provide a query or upload an image.")
        return

    # Save uploaded image temporarily
    image_path = None
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name
        # Show uploaded image
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(Image.open(image_path), width=200)
        with col2:
            st.write("📷 Uploaded image ready.")

    # Process query
    with st.spinner("🔍 Searching for products..."):
        result = rag.run(
            text_query=text_query,
            image_path=image_path,
            k=k
        )

    # Show assistant response
    st.subheader("🤖 Assistant Response")
    st.write(result.get('response', 'No response generated.'))

    # Show recommended products
    st.subheader("📦 Recommended Products")
    products = result.get('retrieved_products', [])
    if products:
        # Display in a responsive grid (max 4 columns)
        cols = st.columns(min(k, 4))
        for idx, item in enumerate(products):
            meta = item.get('metadata', {})
            col = cols[idx % len(cols)]
            display_product_card(meta, col)
    else:
        st.info("No matching products found.")

    # Show image analysis (if image was uploaded)
    if uploaded_file:
        with st.expander("🔍 Image Analysis"):
            if result.get('image_caption'):
                st.write("**Caption:**", result['image_caption'])
            if result.get('attributes'):
                st.write("**Attributes:**", result['attributes'])

    # Clean up temporary image file (ignore permission errors on Windows)
    if image_path and os.path.exists(image_path):
        try:
            os.unlink(image_path)
        except PermissionError:
            # The file may still be in use; we'll skip deletion.
            pass

if __name__ == "__main__":
    main()
