# Article Generation System

A system for generating SEO-optimized articles using LLMs and AI image generation, with built-in evaluation framework for systematic improvement.

## Features

- Generate SEO-optimized articles using Anthropic's Claude
- Create relevant images using Flux Pro API
- FastAPI-based REST API
- Automatic content saving and organization
- Built-in evaluation framework for systematic improvement:
  - A/B testing of different prompt templates
  - Automated metric tracking (structure, content, SEO scores)
  - Interactive dashboard for visualizing results
  - Human feedback integration

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Modern Python package installer and resolver
- Anthropic API key for Claude
- Flux Pro API key for image generation

## Installation

1. Install uv if you haven't already:
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/Article-generation.git
cd Article-generation
```

3. Create and activate a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

5. Create a `.env` file with your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Project Structure

```
Article-generation/
├── article_generation/
│   ├── llm/                # LLM-based article generation
│   ├── image_gen/         # Image generation using Flux Pro
│   ├── integration/       # Integration of articles and images
│   ├── experimentation/   # Evaluation framework
│   │   ├── experiment.py  # Core experiment functionality
│   │   ├── dashboard.py   # Streamlit dashboard
│   │   └── feedback.py    # Human feedback management
│   └── evaluation/
│       └── evaluator.py   # Article evaluation metrics
├── experiments/           # Experiment configuration files
├── traces/               # Generated article traces
├── tests/               # Test suite
├── main.py              # FastAPI server
├── run_dashboard.py     # Evaluation dashboard
└── run_experiment.py    # Experiment runner
```

## Usage

### Article Generation API

1. Start the API server:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

3. API Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### POST /generate
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

### Evaluation Framework

The project includes a comprehensive evaluation framework based on [Hamel Husain's approach](https://hamel.dev/blog/posts/evals/#level-3-ab-testing) with three levels:

1. **Unit Tests**: Fast, automated tests for basic validation
2. **Human & Model Eval**: Detailed quality assessment
3. **A/B Testing**: Compare different prompt variants

To use the evaluation framework:

1. Create an experiment configuration in `experiments/` directory
2. Run experiments:
```bash
python run_experiment.py
```

3. View results in the dashboard:
```bash
streamlit run run_dashboard.py
```

The dashboard provides:
- Experiment overview
- Metric analysis over time
- Statistical comparisons between variants
- Human feedback integration

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test categories:
```bash
# Run only integration tests
pytest tests/ -v -m integration

# Run all tests except integration tests
pytest tests/ -v -m "not integration"
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
