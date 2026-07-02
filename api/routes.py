from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from api.schemas import QueryResponse
from api.services import RAGService
import tempfile
import shutil
import os

router = APIRouter()
_service = None

def get_service():
    global _service
    if _service is None:
        _service = RAGService()
    return _service

@router.post("/search", response_model=QueryResponse)
async def search(
    query: str = Form(None),
    image: UploadFile = File(None),
    k: int = 5
):
    if not query and not image:
        raise HTTPException(status_code=400, detail="Provide query or image")
    image_path = None
    if image:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(image.file, tmp)
            image_path = tmp.name
    result = get_service().process_query(text_query=query, image_path=image_path, k=k)
    # Convert products to schema
    products = []
    for item in result['retrieved_products']:
        meta = item['metadata']
        products.append({
            "id": item.get('id'),
            "title": meta.get('title'),
            "description": meta.get('description'),
            "price": meta.get('price'),
            "category": meta.get('category'),
            "image_url": meta.get('image_url'),
            "distance": item['distance']
        })
    return QueryResponse(
        response=result['response'],
        products=products,
        caption=result.get('image_caption'),
        attributes=result.get('attributes')
    )