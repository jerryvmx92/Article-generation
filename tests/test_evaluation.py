"""Tests for the evaluation module."""

import os
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from article_generation.evaluation.trace_logger import TraceLogger
from article_generation.evaluation.evaluator import ArticleEvaluator

@pytest.fixture
def trace_logger(tmp_path):
    """Create a trace logger that uses a temporary directory."""
    return TraceLogger(trace_dir=str(tmp_path))

@pytest.fixture
def mock_article():
    """Create a mock article for testing."""
    return {
        "title": "Test Article",
        "keywords": ["test", "evaluation"],
        "content": """# Test Article

## Introduction
This is a test introduction.

## Body Content
This is the main content.

## Conclusion
This is the conclusion."""
    }

@pytest.fixture
def mock_evaluation():
    """Create a mock evaluation response."""
    return {
        "structure_score": 8,
        "structure_strengths": ["Clear section headers", "Proper markdown formatting"],
        "structure_improvements": ["Could use more subsections"],
        "structure_recommendations": ["Add H3 headers for subsections"],
        "content_score": 7,
        "content_strengths": ["Clear writing", "Good organization"],
        "content_improvements": ["Could be more detailed"],
        "content_recommendations": ["Add more specific examples"],
        "seo_score": 9,
        "seo_strengths": ["Keywords well incorporated", "Good header structure"],
        "seo_improvements": ["Could use more keyword variations"],
        "seo_recommendations": ["Add related keywords"],
        "regional_score": 6,
        "regional_strengths": ["Mentions Coatzacoalcos"],
        "regional_improvements": ["Could be more locally focused"],
        "regional_recommendations": ["Add more local context"],
        "overall_score": 7.5,
        "main_issues": ["Needs more detail", "Could be more locally focused"],
        "top_recommendations": [
            "Add more specific examples",
            "Increase local context",
            "Add subsections"
        ]
    }

class TestTraceLogger:
    """Test suite for TraceLogger."""
    
    def test_initialization(self, tmp_path):
        """Test TraceLogger initialization."""
        logger = TraceLogger(trace_dir=str(tmp_path))
        assert os.path.exists(logger.success_dir)
        assert os.path.exists(logger.error_dir)
    
    def test_log_successful_trace(self, trace_logger, mock_article):
        """Test logging a successful generation."""
        path = trace_logger.log_trace(
            title=mock_article["title"],
            keywords=mock_article["keywords"],
            prompt="Test prompt",
            response=mock_article
        )
        
        assert os.path.exists(path)
        assert path.startswith(trace_logger.success_dir)
        
        with open(path) as f:
            trace = json.load(f)
            assert trace["title"] == mock_article["title"]
            assert trace["keywords"] == mock_article["keywords"]
            assert trace["error"] is None
    
    def test_log_error_trace(self, trace_logger, mock_article):
        """Test logging a failed generation."""
        error = Exception("Test error")
        path = trace_logger.log_trace(
            title=mock_article["title"],
            keywords=mock_article["keywords"],
            prompt="Test prompt",
            response={},
            error=error
        )
        
        assert os.path.exists(path)
        assert path.startswith(trace_logger.error_dir)
        
        with open(path) as f:
            trace = json.load(f)
            assert trace["error"] == str(error)
    
    def test_get_traces(self, trace_logger, mock_article):
        """Test retrieving traces."""
        # Create some test traces
        trace_logger.log_trace(
            title=mock_article["title"],
            keywords=mock_article["keywords"],
            prompt="Test prompt",
            response=mock_article
        )
        
        trace_logger.log_trace(
            title=mock_article["title"],
            keywords=mock_article["keywords"],
            prompt="Test prompt",
            response={},
            error=Exception("Test error")
        )
        
        # Test getting all traces
        all_traces = trace_logger.get_traces(success_only=False)
        assert len(all_traces) == 2
        
        # Test getting only successful traces
        success_traces = trace_logger.get_traces(success_only=True)
        assert len(success_traces) == 1
        assert success_traces[0]["error"] is None
        
        # Test date filtering
        future_traces = trace_logger.get_traces(
            start_date=datetime.now() + timedelta(days=1)
        )
        assert len(future_traces) == 0

class TestArticleEvaluator:
    """Test suite for ArticleEvaluator."""
    
    def test_initialization(self):
        """Test ArticleEvaluator initialization."""
        evaluator = ArticleEvaluator(api_key="test_key")
        assert evaluator.api_key == "test_key"
        assert evaluator.model == "claude-3-opus-20240229"
        assert evaluator.temperature == 0.3  # Lower temperature for evaluation
    
    def test_initialization_without_api_key(self, monkeypatch):
        """Test initialization fails without API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError) as exc_info:
            ArticleEvaluator()
        assert "API key must be provided" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_evaluate_article(self, mock_article, mock_evaluation):
        """Test article evaluation."""
        evaluator = ArticleEvaluator(api_key="test_key")
        
        # Mock the API response with proper JSON formatting
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(mock_evaluation))]  # Use json.dumps for proper formatting
        
        mock_messages = MagicMock()
        mock_messages.create = AsyncMock(return_value=mock_response)
        evaluator.client.messages = mock_messages
        
        # Evaluate article
        result = await evaluator.evaluate_article(
            content=mock_article["content"],
            title=mock_article["title"],
            keywords=mock_article["keywords"]
        )
        
        # Verify evaluation
        assert result == mock_evaluation
        assert result["overall_score"] == 7.5
        assert len(result["top_recommendations"]) == 3
        
        # Verify API call
        mock_messages.create.assert_called_once()
        call_args = mock_messages.create.call_args[1]
        assert call_args["model"] == evaluator.model
        assert call_args["temperature"] == evaluator.temperature
    
    @pytest.mark.asyncio
    async def test_evaluate_article_error(self, mock_article):
        """Test article evaluation with API error."""
        evaluator = ArticleEvaluator(api_key="test_key")
        
        # Mock an API error
        mock_messages = MagicMock()
        mock_messages.create = AsyncMock(side_effect=Exception("API Error"))
        evaluator.client.messages = mock_messages
        
        # Evaluate article
        result = await evaluator.evaluate_article(
            content=mock_article["content"],
            title=mock_article["title"],
            keywords=mock_article["keywords"]
        )
        
        # Verify error handling
        assert "error" in result
        assert result["error"] == "API Error" 