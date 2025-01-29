# Article Generation System

[![Test](https://github.com/jerryvmx92/Article-generation/actions/workflows/test.yml/badge.svg)](https://github.com/jerryvmx92/Article-generation/actions/workflows/test.yml)
[![Lint](https://github.com/jerryvmx92/Article-generation/actions/workflows/lint.yml/badge.svg)](https://github.com/jerryvmx92/Article-generation/actions/workflows/lint.yml)
[![Security](https://github.com/jerryvmx92/Article-generation/actions/workflows/security.yml/badge.svg)](https://github.com/jerryvmx92/Article-generation/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/jerryvmx92/Article-generation/branch/master/graph/badge.svg)](https://codecov.io/gh/jerryvmx92/Article-generation)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

## Testing

The project uses pytest for testing. Here are the different ways to run tests:

1. Run all tests:
```bash
pytest tests/
```

2. Run tests with verbose output:
```bash
pytest tests/ -v
```

3. Run specific test categories:
```bash
# Run only integration tests
pytest tests/ -v -m integration

# Run all tests except integration tests
pytest tests/ -v -m "not integration"
```

4. Run tests with debug output:
```bash
pytest tests/ -v --log-cli-level=DEBUG
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interaction with external APIs (requires valid API keys)
- **Content Quality Tests**: Validate article structure and content requirements

### Environment Setup for Tests

1. For running integration tests, you need a valid Anthropic API key in your `.env` file:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

2. Integration tests will be skipped if:
   - No API key is provided
   - The API key is set to "test_api_key"
   - The API key is invalid

3. Test Configuration:
   - Model: claude-3-opus-20240229
   - Max Tokens: 4096
   - Temperature: 0.7
   - Min Article Length: 1200 words
   - Max Article Length: 3000 words

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
