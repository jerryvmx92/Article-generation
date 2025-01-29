"""Main entry point for the Article Generation system."""

import asyncio
import traceback
from typing import List, Optional
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import urllib.parse

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
    num_images: int = Field(default=1, ge=1, description="Number of images to generate")
    aspect_ratio: str = Field(default="16:9", pattern="^[0-9]+:[0-9]+$", description="Image aspect ratio in format width:height")

class ImageMetadata(BaseModel):
    url: str
    caption: str
    alt_text: str
    aspect_ratio: str

class ArticleSection(BaseModel):
    heading: str
    content: str
    images: Optional[List[ImageMetadata]] = None

class ArticleMetadata(BaseModel):
    title: str
    description: str
    keywords: List[str]
    author: str = "AI Content Generator"
    date_generated: datetime = Field(default_factory=datetime.now)
    reading_time: str
    language: str = "en"

class Article(BaseModel):
    metadata: ArticleMetadata
    introduction: str
    sections: List[ArticleSection]
    conclusion: str
    raw_markdown: str
    file_url: str

class GenerateResponse(BaseModel):
    article: Article
    images_file_url: str
    status: str = "success"
    generation_time: float

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
    
    start_time = datetime.now()
    
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
        
        # Parse the markdown content to create structured article
        article = parse_markdown_to_article(
            markdown_content=article_content,
            file_url=f"{BASE_URL}/articles/{article_filename}",
            metadata={
                "title": request.topic,
                "keywords": request.keywords,
                "description": content.get('description', ''),
                "language": content.get('language', 'en'),
            }
        )
        
        response = {
            "article": article,
            "images_file_url": f"{BASE_URL}/images/{images_filename}",
            "status": "success",
            "generation_time": (datetime.now() - start_time).total_seconds()
        }
        
        logger.info(f"Generation completed for topic: {request.topic}")
        logger.debug(f"Response URLs: {response['article'].file_url}, {response['images_file_url']}")
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

def parse_markdown_to_article(markdown_content: str, file_url: str, metadata: dict) -> Article:
    """Parse markdown content into structured Article object."""
    try:
        # Split content into sections
        sections = []
        current_section = None
        introduction = ""
        conclusion = ""
        
        lines = markdown_content.split('\n')
        reading_time = estimate_reading_time(markdown_content)
        
        for line in lines:
            if line.startswith('# '):  # Main title, skip
                continue
            elif line.startswith('## '):  # Section heading
                if current_section:
                    sections.append(current_section)
                current_section = ArticleSection(
                    heading=line.replace('## ', '').strip(),
                    content='',
                    images=[]
                )
            elif current_section is None and line.strip():  # Introduction
                introduction += line + '\n'
            elif current_section and line.strip():  # Section content
                current_section.content += line + '\n'
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        # If the last section is "Conclusion", move it to conclusion field
        if sections and sections[-1].heading.lower() == "conclusion":
            conclusion = sections[-1].content
            sections = sections[:-1]
        
        article = Article(
            metadata=ArticleMetadata(
                title=metadata['title'],
                description=metadata['description'],
                keywords=metadata['keywords'],
                language=metadata['language'],
                reading_time=reading_time
            ),
            introduction=introduction.strip(),
            sections=sections,
            conclusion=conclusion.strip(),
            raw_markdown=markdown_content,
            file_url=file_url
        )
        
        return article
    except Exception as e:
        logger.error(f"Error parsing markdown to article: {str(e)}")
        raise

def estimate_reading_time(content: str) -> str:
    """Estimate reading time based on word count."""
    words = len(content.split())
    minutes = max(1, round(words / 200))  # Assuming 200 words per minute
    return f"{minutes} min read"

if __name__ == "__main__":
    logger.info("Starting Article Generation API")
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    ) 