def build_context(retrieved_products):
    if not retrieved_products:
        return "No products found."
    context = "Here are some products that match your query:\n"
    for i, item in enumerate(retrieved_products, 1):
        meta = item['metadata']
        title = meta.get('title', 'Product')
        price = meta.get('price', 'N/A')
        desc = meta.get('description', '')
        context += f"{i}. **{title}** - ${price}\n"
        context += f"   {desc[:150]}...\n"
        if 'image_url' in meta:
            context += f"   Image: {meta['image_url']}\n"
        context += "\n"
    return context