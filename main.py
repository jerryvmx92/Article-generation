"""Main entry point for the Article Generation system."""

import asyncio
import traceback
from typing import List
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from article_generation.integration.content_manager import ContentManager

app = FastAPI(
    title="Article Generation API",
    description="API for generating SEO-optimized articles with AI-generated images",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount the static directories
app.mount("/articles", StaticFiles(directory="generated_articles"), name="articles")
app.mount("/images", StaticFiles(directory="generated_images"), name="images")

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
        # Ensure directories exist
        os.makedirs("generated_articles", exist_ok=True)
        os.makedirs("generated_images", exist_ok=True)
        
        content_manager = ContentManager()
        content = await content_manager.generate_content(
            topic=request.topic,
            keywords=request.keywords,
            num_images=request.num_images,
            aspect_ratio=request.aspect_ratio,
        )
        paths = content_manager.save_content(content)
        
        # Convert local paths to URLs
        base_url = "https://article-generation-22zt.onrender.com"
        response = {
            "article_path": f"{base_url}/articles/{os.path.basename(paths['article_path'])}",
            "images_path": f"{base_url}/images/{os.path.basename(paths['images_path'])}"
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