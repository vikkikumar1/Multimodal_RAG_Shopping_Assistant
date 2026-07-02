from vector_store.vector_db import FAISSVectorStore
from embeddings.text_embeddings import TextEmbedder
from image_processing.image_caption import generate_caption, extract_attributes
from utils.logger import setup_logger
import re
from typing import Optional, Dict, List, Any

logger = setup_logger(__name__)

class Retriever:
    def __init__(self):
        self.vector_store = FAISSVectorStore()
        self.text_embedder = TextEmbedder()
        try:
            self.vector_store.load_index()
        except FileNotFoundError:
            logger.error("FAISS index not found. Run generate_embeddings.py first.")
            self.vector_store = None

        # Pre‑define known attribute values for matching (you can extend these)
        self.known_colours = {'blue', 'black', 'white', 'red', 'green', 'yellow', 'purple',
                              'pink', 'orange', 'grey', 'gray', 'brown', 'navy', 'maroon'}
        self.known_categories = {'apparel', 'footwear', 'accessories', 'sporting goods',
                                 'personal care', 'clothing', 'shoes', 'bags', 'watches',
                                 'electronics', 'home', 'books', 'toys'}
        self.known_seasons = {'summer', 'fall', 'winter', 'spring', 'autumn'}
        self.known_genders = {'men', 'women', 'unisex', 'boys', 'girls'}
        self.known_usages = {'casual', 'formal', 'sports', 'party', 'office'}

    def retrieve_from_text(self, query: str, k: int = 5) -> List[Dict]:
        if not self.vector_store:
            logger.warning("Vector store unavailable; returning empty results")
            return []
        try:
            query_embedding = self.text_embedder.model.encode([query])[0]
            results = self.vector_store.search(query_embedding, k)
            return results
        except Exception as e:
            logger.error(f"Text embedding or search failed: {e}")
            return []

    def retrieve_from_image(self, image_path: str, k: int = 5):
        caption = generate_caption(image_path)
        attributes = extract_attributes(image_path)
        combined = f"{caption}. {attributes}"
        results = self.retrieve_from_text(combined, k)
        return results, caption, attributes

    @staticmethod
    def parse_price_filter(query: str) -> Optional[float]:
        """Extract max price (e.g., 'under 3000' or 'below 500')."""
        if not query:
            return None
        match = re.search(r'under\s*(\d+)', query, re.IGNORECASE) or \
                re.search(r'below\s*(\d+)', query, re.IGNORECASE) or \
                re.search(r'less than\s*(\d+)', query, re.IGNORECASE) or \
                re.search(r'<=?\s*(\d+)', query, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def parse_filters(self, query: str) -> Dict[str, Any]:
        """
        Parse colour, category, season, gender, and usage from the query.
        Returns a dict with keys: 'price_max', 'colour', 'category', 'season',
        'gender', 'usage' (each may be None or a list of values).
        """
        filters = {
            'price_max': self.parse_price_filter(query),
            'colour': None,
            'category': None,
            'season': None,
            'gender': None,
            'usage': None
        }

        if not query:
            return filters

        # Lowercase and split into words for easier matching
        words = set(re.findall(r'\b[a-z]+\b', query.lower()))

        # Match colour
        for colour in self.known_colours:
            if colour in words or colour in query.lower():
                filters['colour'] = colour
                break

        # Match category
        for cat in self.known_categories:
            if cat in words or cat in query.lower():
                filters['category'] = cat
                break

        # Match season
        for season in self.known_seasons:
            if season in words or season in query.lower():
                filters['season'] = season
                break

        # Match gender
        for gender in self.known_genders:
            if gender in words or gender in query.lower():
                filters['gender'] = gender
                break

        # Match usage
        for usage in self.known_usages:
            if usage in words or usage in query.lower():
                filters['usage'] = usage
                break

        return filters

    def apply_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Filter the retrieved products based on the parsed criteria."""
        if not results:
            return results

        filtered = []
        for item in results:
            meta = item.get('metadata', {})
            match = True

            # Price filter
            max_price = filters.get('price_max')
            if max_price is not None:
                price = meta.get('price', float('inf'))
                if price > max_price:
                    match = False

            # Colour filter
            colour = filters.get('colour')
            if colour and match:
                meta_colour = meta.get('baseColour', '').lower()
                if colour not in meta_colour:
                    match = False

            # Category filter (check masterCategory or subCategory)
            category = filters.get('category')
            if category and match:
                meta_cat = meta.get('masterCategory', '').lower()
                meta_sub = meta.get('subCategory', '').lower()
                if category not in meta_cat and category not in meta_sub:
                    match = False

            # Season filter
            season = filters.get('season')
            if season and match:
                meta_season = meta.get('season', '').lower()
                if season not in meta_season:
                    match = False

            # Gender filter
            gender = filters.get('gender')
            if gender and match:
                meta_gender = meta.get('gender', '').lower()
                if gender not in meta_gender:
                    match = False

            # Usage filter
            usage = filters.get('usage')
            if usage and match:
                meta_usage = meta.get('usage', '').lower()
                if usage not in meta_usage:
                    match = False

            if match:
                filtered.append(item)

        return filtered

    def retrieve_hybrid(self, text_query: Optional[str] = None,
                        image_path: Optional[str] = None,
                        k: int = 5) -> tuple[List[Dict], Optional[str], Optional[str]]:
        results = []
        caption = None
        attributes = None

        # 1. Retrieve based on input(s)
        if image_path and text_query:
            caption = generate_caption(image_path)
            combined = f"{text_query} {caption}"
            results = self.retrieve_from_text(combined, k)
        elif image_path:
            results, caption, attributes = self.retrieve_from_image(image_path, k)
        elif text_query:
            results = self.retrieve_from_text(text_query, k)
        else:
            return [], None, None

        # 2. Parse filters from the text query (if any) and apply them
        if text_query:
            filters = self.parse_filters(text_query)
            results = self.apply_filters(results, filters)

        return results, caption, attributes