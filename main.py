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

class Article(BaseModel):
    title: str
    introduction: str
    body: str
    conclusion: str
    image_url: str
    article_url: str
    markdown: str

class GenerateResponse(BaseModel):
    article: Article
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
    max_retries = 3
    base_delay = 1  # Base delay in seconds
    
    try:
        content_manager = ContentManager()
        logger.debug("Content manager initialized")
        
        # Implement retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting content generation (attempt {attempt + 1}/{max_retries})")
                content = await content_manager.generate_content(
                    topic=request.topic,
                    keywords=request.keywords,
                    num_images=request.num_images,
                    aspect_ratio=request.aspect_ratio,
                )
                logger.success("Content generation completed successfully")
                break
            except Exception as e:
                if 'overloaded_error' in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API overloaded. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                raise  # Re-raise the exception if it's not an overload error or we're out of retries
        
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
            raise HTTPException(
                status_code=500,
                detail=f"Error reading generated files: {str(e)}"
            )
        
        # Convert local paths to URLs using environment-based base URL
        article_filename = urllib.parse.quote(os.path.basename(paths['article_path']))
        images_filename = urllib.parse.quote(os.path.basename(paths['images_path']))
        
        article_url = f"{BASE_URL}/articles/{article_filename}"
        image_url = f"{BASE_URL}/images/{images_filename}"
        
        # Parse the markdown content
        article = parse_markdown_to_simple_article(
            markdown_content=article_content,
            image_url=image_url,
            article_url=article_url,
            title=request.topic
        )
        
        response = {
            "article": article,
            "status": "success",
            "generation_time": (datetime.now() - start_time).total_seconds()
        }
        
        logger.info(f"Generation completed for topic: {request.topic}")
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

def parse_markdown_to_simple_article(markdown_content: str, image_url: str, article_url: str, title: str) -> Article:
    """Parse markdown content into a simple article structure."""
    try:
        introduction = []
        body = []
        conclusion = []
        in_conclusion = False
        in_introduction = False
        
        # Remove YAML frontmatter and schema markup
        content_parts = markdown_content.split('---')
        if len(content_parts) >= 3:
            # Get the YAML frontmatter
            yaml_content = content_parts[1]
            # Remove schema markup section if present
            yaml_lines = [line for line in yaml_content.split('\n') 
                        if not line.strip().startswith('Schema Markup:') 
                        and not ('"@' in line or '{' in line or '}' in line)]
            # Reconstruct the frontmatter without schema
            content_parts[1] = '\n'.join(yaml_lines)
            # Get the main content
            main_content = '---'.join(content_parts[2:]).strip()
        else:
            main_content = markdown_content.strip()
        
        # Process content line by line
        cleaned_lines = []
        in_code_block = False
        in_schema_block = False
        
        for line in main_content.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check for schema block start/end
            if 'Schema Markup:' in line or '"@context"' in line:
                in_schema_block = True
                continue
            if in_schema_block and '}' in line:
                in_schema_block = False
                continue
            
            # Skip if in schema block
            if in_schema_block:
                continue
                
            # Handle code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # Skip if in code block
            if in_code_block:
                continue
                
            # Skip HTML blocks
            if any(skip in line for skip in ['<div', '<meta', '</div>']):
                continue
                
            # Skip image suggestions
            if '[Sugerencia de imagen:' in line or 'Texto alt:' in line:
                continue
                
            # Clean up markdown links that aren't images
            if '[' in line and ']' in line and not line.startswith('!['):
                # Keep only the text inside the brackets
                parts = line.split('[')
                cleaned_parts = []
                for part in parts:
                    if ']' in part:
                        # Extract text inside brackets and remove the link part
                        text = part.split(']')[0]
                        cleaned_parts.append(text)
                    else:
                        cleaned_parts.append(part)
                line = ' '.join(cleaned_parts)
            
            cleaned_lines.append(line)
        
        # Parse sections
        for line in cleaned_lines:
            if line.startswith('# '):
                continue
            elif line.startswith('## '):
                if 'introducci' in line.lower() or 'introduction' in line.lower():
                    in_introduction = True
                    in_conclusion = False
                elif 'conclusi' in line.lower():
                    in_conclusion = True
                    in_introduction = False
                else:
                    in_introduction = False
                    in_conclusion = False
                    body.append(line)
            elif line.startswith('### '):
                # Keep section titles but remove numbers if present
                if any(c.isdigit() for c in line):
                    line = '### ' + ''.join(c for c in line[4:] if not c.isdigit()).strip('. ')
                body.append(line)
            else:
                if in_conclusion:
                    conclusion.append(line)
                elif in_introduction:
                    introduction.append(line)
                else:
                    body.append(line)
        
        # Create clean markdown
        clean_markdown = f"# {title}\n\n"
        if introduction:
            clean_markdown += "## Introduction\n\n"
            clean_markdown += '\n'.join(introduction).strip() + '\n\n'
        clean_markdown += '\n'.join(body).strip() + '\n\n'
        if conclusion:
            clean_markdown += f"## Conclusion\n\n"
            clean_markdown += '\n'.join(conclusion).strip() + '\n'
        
        # Add links to the markdown
        clean_markdown += f"\n---\n\n"
        clean_markdown += f"[View Full Article]({article_url}) | [View Images]({image_url})\n"
        
        return Article(
            title=title,
            introduction='\n'.join(introduction).strip(),
            body='\n'.join(body).strip(),
            conclusion='\n'.join(conclusion).strip(),
            image_url=image_url,
            article_url=article_url,
            markdown=clean_markdown
        )
        
    except Exception as e:
        logger.error(f"Error parsing markdown to article: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting Article Generation API")
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    ) 