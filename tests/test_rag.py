from rag.context_builder import build_context


def test_build_context_empty():
    assert build_context([]) == "No products found."


def test_build_context_products():
    products = [
        {
            'metadata': {
                'title': 'Test Product',
                'price': 19.99,
                'description': 'A sample product description.',
                'image_url': 'https://example.com/image.jpg'
            }
        }
    ]
    context = build_context(products)
    assert 'Test Product' in context
    assert '$19.99' in context