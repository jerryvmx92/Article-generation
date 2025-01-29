"""Shared pytest fixtures for Article Generation system tests."""

import pytest
import os
from dotenv import load_dotenv
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture(autouse=True)
def load_env():
    """Load environment variables before each test."""
    print("\n=== Loading Environment Variables ===")
    load_dotenv()
    
    # Print loaded environment variables
    print("\nLoaded environment variables:")
    for key in ['ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL', 'MAX_TOKENS', 'TEMPERATURE']:
        value = os.getenv(key)
        if value:
            if 'API_KEY' in key:
                print(f"{key}: {'*' * 10}{value[-5:]}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: Not set")

@pytest.fixture
def base_dir():
    """Get the base directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def test_data_dir(base_dir):
    """Get the test data directory."""
    test_data_path = os.path.join(base_dir, "tests", "data")
    os.makedirs(test_data_path, exist_ok=True)
    return test_data_path

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test that requires API access"
    )

# Test data
TEST_ARTICLES = [
    {
        "topic": "Industrial Safety Protocols",
        "keywords": ["safety", "industrial", "protocols"],
        "expected_sections": ["Introduction", "Body Content", "Conclusion"],
        "min_words": 1200,
        "max_words": 3000
    },
    {
        "topic": "Corrosion Prevention Methods",
        "keywords": ["corrosion", "prevention", "maintenance"],
        "expected_sections": ["Introduction", "Body Content", "Conclusion"],
        "min_words": 1200,
        "max_words": 3000
    }
]

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_api_key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    monkeypatch.setenv("MAX_TOKENS", "4096")
    monkeypatch.setenv("TEMPERATURE", "0.7")
    monkeypatch.setenv("MIN_ARTICLE_LENGTH", "1200")
    monkeypatch.setenv("MAX_ARTICLE_LENGTH", "3000")

@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    mock_content = MagicMock()
    mock_content.text = """# Test Article

## Introduction
This is a test introduction that provides context about the topic.

## Body Content
This is the main content of the article with detailed information.

### Key Points
- Point 1
- Point 2
- Point 3

## Conclusion
This is the conclusion summarizing the key points."""

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response

@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    """Create a mock Anthropic client."""
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(return_value=mock_anthropic_response)
    
    mock_client = MagicMock()
    mock_client.messages = mock_messages
    return mock_client

@pytest.fixture
def test_article_data() -> List[Dict]:
    """Return test article data."""
    return TEST_ARTICLES 