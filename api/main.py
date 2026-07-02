from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from utils.logger import setup_logger
from utils.config import RAW_DATA_DIR
import os

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    version="1.0",
    description="AI shopping assistant API that searches products via text and images.",
)

# --- CORS (optional, uncomment if needed) ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, restrict to your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# --- Mount static images folder ---
image_dir = RAW_DATA_DIR / "images"
if image_dir.exists():
    app.mount("/static/images", StaticFiles(directory=str(image_dir)), name="static")
    logger.info(f"Serving static images from {image_dir}")
else:
    logger.warning(f"Image directory not found: {image_dir}. Static serving disabled.")

# --- Include API routes ---
app.include_router(router, prefix="/api/v1")

# --- Health check ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "static_images_available": image_dir.exists()}

# --- Optional: root redirect ---
@app.get("/")
async def root():
    return {
        "message": "Multimodal RAG API is running.",
        "docs": "/docs",
        "health": "/health"
    }

# --- Run with uvicorn (only when executed directly) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto‑reload during development
        log_level="info"
    )