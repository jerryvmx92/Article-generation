"""Main entry point for the Article Generation system."""

import asyncio
import traceback
from typing import List
import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from article_generation.integration.content_manager import ContentManager

# Load environment variables
load_dotenv()

# Define base directory for generated content
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLES_DIR = os.path.join(BASE_DIR, "generated_articles")
IMAGES_DIR = os.path.join(BASE_DIR, "generated_images")

# Get base URL from environment variable or use default
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Create necessary directories
os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(
    title="Article Generation API",
    description="API for generating SEO-optimized articles with AI-generated images",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount the static directories
app.mount("/articles", StaticFiles(directory=ARTICLES_DIR), name="articles")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

class GenerateRequest(BaseModel):
    topic: str
    keywords: List[str]
    num_images: int = 1
    aspect_ratio: str = "16:9"

class GenerateResponse(BaseModel):
    article_path: str
    images_path: str

@app.get("/")
async def root():
    """Redirect root path to API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/generate", response_model=GenerateResponse)
async def generate_content(request: GenerateRequest) -> dict:
    """Generate an article with images based on the given topic and keywords."""
    try:
        content_manager = ContentManager()
        content = await content_manager.generate_content(
            topic=request.topic,
            keywords=request.keywords,
            num_images=request.num_images,
            aspect_ratio=request.aspect_ratio,
        )
        paths = content_manager.save_content(content)
        
        # Convert local paths to URLs using environment-based base URL
        response = {
            "article_path": f"{BASE_URL}/articles/{os.path.basename(paths['article_path'])}",
            "images_path": f"{BASE_URL}/images/{os.path.basename(paths['images_path'])}"
        }
        return response
    except Exception as e:
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("Error details:", error_details)  # Print error details to server console
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}\nTraceback: {traceback.format_exc()}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    ) 