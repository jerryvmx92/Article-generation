# Article Generation System

A system for generating SEO-optimized articles using LLMs and AI image generation.

## Features

- Generate SEO-optimized articles using Anthropic's Claude
- Create relevant images using Flux Pro API
- FastAPI-based REST API
- Automatic content saving and organization

## Setup

1. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

2. For development, install additional dependencies:
```bash
uv pip install -r requirements-dev.txt
```

3. Create a `.env` file with your API keys and configuration:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

1. Start the API server:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

3. API Documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### POST /generate

Generate an article with images.

Request body:
```json
{
    "topic": "Your topic here",
    "keywords": ["keyword1", "keyword2"],
    "num_images": 1,
    "aspect_ratio": "16:9"
}
```

Response:
```json
{
    "article_path": "path/to/generated/article.md",
    "images_path": "path/to/generated/images.txt"
}
```

## Project Structure

```
article_generation/
├── llm/                    # LLM-based article generation
├── image_gen/             # Image generation using Flux Pro
└── integration/           # Integration of articles and images
```

## Environment Variables

See `.env.example` for all available configuration options.
