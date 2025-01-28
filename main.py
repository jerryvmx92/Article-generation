"""Main entry point for the Article Generation system."""

import asyncio
import traceback
from typing import List
import os
import sys
from dotenv import load_dotenv
import urllib.parse

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from article_generation.integration.content_manager import ContentManager

# Load environment variables
load_dotenv()

# Configure Loguru
# Remove default logger
logger.remove()
# Add console logger with custom format
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
# Add file logger with rotation
logger.add(
    "logs/article_generation_{time}.log",
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG",
    enqueue=True,
    backtrace=True,
    diagnose=True
)

# Define base directory for generated content
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLES_DIR = os.path.join(BASE_DIR, "generated_articles")
IMAGES_DIR = os.path.join(BASE_DIR, "generated_images")

# Get base URL from environment variable or use default
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Create necessary directories
os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
logger.info(f"Created/verified directories: {ARTICLES_DIR}, {IMAGES_DIR}")

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
logger.info("Static directories mounted successfully")

class GenerateRequest(BaseModel):
    topic: str
    keywords: List[str]
    num_images: int = 1
    aspect_ratio: str = "16:9"

class GenerateResponse(BaseModel):
    article_path: str
    images_path: str
    article_content: str
    images_content: str

@app.get("/")
async def root():
    """Redirect root path to API documentation."""
    logger.debug("Root endpoint accessed, redirecting to docs")
    return RedirectResponse(url="/docs")

@app.post("/generate", response_model=GenerateResponse)
@logger.catch
async def generate_content(request: GenerateRequest) -> dict:
    """Generate an article with images based on the given topic and keywords."""
    logger.info(f"Received generation request for topic: {request.topic}")
    logger.debug(f"Request details: {request.dict()}")
    
    try:
        content_manager = ContentManager()
        logger.debug("Content manager initialized")
        
        logger.info("Starting content generation")
        content = await content_manager.generate_content(
            topic=request.topic,
            keywords=request.keywords,
            num_images=request.num_images,
            aspect_ratio=request.aspect_ratio,
        )
        logger.success("Content generation completed successfully")
        
        logger.debug("Saving generated content to files")
        paths = content_manager.save_content(content)
        logger.debug(f"Content saved to paths: {paths}")
        
        # Read the generated files
        try:
            with open(paths['article_path'], 'r', encoding='utf-8') as f:
                article_content = f.read()
            with open(paths['images_path'], 'r', encoding='utf-8') as f:
                images_content = f.read()
            logger.debug("Successfully read generated files")
        except Exception as e:
            logger.error(f"Error reading generated files: {str(e)}")
            article_content = ""
            images_content = ""
        
        # Convert local paths to URLs using environment-based base URL
        # Use URL-safe filenames
        article_filename = urllib.parse.quote(os.path.basename(paths['article_path']))
        images_filename = urllib.parse.quote(os.path.basename(paths['images_path']))
        
        response = {
            "article_path": f"{BASE_URL}/articles/{article_filename}",
            "images_path": f"{BASE_URL}/images/{images_filename}",
            "article_content": article_content,
            "images_content": images_content
        }
        logger.info(f"Generation completed for topic: {request.topic}")
        logger.debug(f"Response URLs: {response['article_path']}, {response['images_path']}")
        return response
        
    except Exception as e:
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        logger.error(f"Error during content generation: {str(e)}")
        logger.exception("Full error details:")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}\nTraceback: {traceback.format_exc()}"
        )

if __name__ == "__main__":
    logger.info("Starting Article Generation API")
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    ) 