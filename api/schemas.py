from pydantic import BaseModel
from typing import Optional, List

class Product(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    distance: Optional[float] = None

class QueryResponse(BaseModel):
    response: str
    products: List[Product]
    caption: Optional[str] = None
    attributes: Optional[str] = None