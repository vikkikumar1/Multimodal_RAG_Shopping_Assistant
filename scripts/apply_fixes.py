from pathlib import Path

replacements = [
    (
        Path('image_processing/image_caption.py'),
        "from models.gemini_vision import GeminiVision\nfrom utils.logger import setup_logger\n\nlogger = setup_logger(__name__)\n_vision = None\n\ndef _get_vision():\n    global _vision\n    if _vision is None:\n        _vision = GeminiVision()\n    return _vision\n\n",
        "from models.gemini_vision import GeminiVision\nfrom utils.logger import setup_logger\n\nlogger = setup_logger(__name__)\n_vision = None\n\ndef _get_vision():\n    global _vision\n    if _vision is None:\n        _vision = GeminiVision()\n    return _vision\n\n"
    ),
    (
        Path('image_processing/image_caption.py'),
        "def generate_caption(image_path):\n    return _get_vision().generate_caption(image_path)\n\ndef extract_attributes(image_path):\n    return vision.extract_attributes(image_path)\n",
        "def generate_caption(image_path):\n    return _get_vision().generate_caption(image_path)\n\ndef extract_attributes(image_path):\n    return _get_vision().extract_attributes(image_path)\n"
    ),
    (
        Path('image_processing/attribute_extractor.py'),
        "# Wrapper for attribute extraction\nfrom models.gemini_vision import GeminiVision\n\ndef extract_attributes(image_path):\n    vision = GeminiVision()\n    return vision.extract_attributes(image_path)\n",
        "# Wrapper for attribute extraction\nfrom models.gemini_vision import GeminiVision\n\n_vision = None\n\ndef _get_vision():\n    global _vision\n    if _vision is None:\n        _vision = GeminiVision()\n    return _vision\n\ndef extract_attributes(image_path):\n    return _get_vision().extract_attributes(image_path)\n"
    ),
    (
        Path('utils/logger.py'),
        "import logging\nimport sys\n\ndef setup_logger(name=__name__, level=logging.INFO):\n    logger = logging.getLogger(name)\n    logger.setLevel(level)\n    handler = logging.StreamHandler(sys.stdout)\n    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))\n    logger.addHandler(handler)\n    return logger\n",
        "import logging\nimport sys\n\ndef setup_logger(name=__name__, level=logging.INFO):\n    logger = logging.getLogger(name)\n    if logger.handlers:\n        return logger\n    logger.setLevel(level)\n    handler = logging.StreamHandler(sys.stdout)\n    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))\n    logger.addHandler(handler)\n    return logger\n"
    ),
    (
        Path('tests/test_rag.py'),
        "from rag.rag_chain import RAGChain\n\ndef test_rag_chain():\n    chain = RAGChain()\n    result = chain.run(text_query=\"test\")\n    assert 'response' in result\n",
        "from rag.context_builder import build_context\n\ndef test_build_context_empty():\n    assert build_context([]) == \"No products found.\"\n\ndef test_build_context_products():\n    products = [\n        {\n            'metadata': {\n                'title': 'Test Product',\n                'price': 19.99,\n                'description': 'A sample product description.',\n                'image_url': 'https://example.com/image.jpg'\n            }\n        }\n    ]\n    context = build_context(products)\n    assert 'Test Product' in context\n    assert '$19.99' in context\n"
    ),
    (
        Path('frontend/pages/product_search.py'),
        "import streamlit as st\nfrom frontend.streamlit_ui import main\nmain()\n",
        "import streamlit as st\nfrom frontend.streamlit_ui import main\n\nif __name__ == '__main__':\n    main()\n"
    )
]

for path, old, new in replacements:
    text = path.read_text()
    if old not in text:
        print(f"Pattern not found in {path}")
        continue
    path.write_text(text.replace(old, new))
    print(f"Updated {path}")
