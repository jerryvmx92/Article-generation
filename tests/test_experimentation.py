"""Tests for the experimentation module."""

import os
import pytest
import json
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from article_generation.experimentation.experiment import Experiment, Variant, Trial
from article_generation.experimentation.feedback import FeedbackManager, FeedbackCriteria, FeedbackResponse

@pytest.fixture
def test_experiment(tmp_path):
    """Create a test experiment."""
    return Experiment(
        name="test_experiment",
        description="Test experiment",
        metrics=["structure_score", "content_score"],
        experiment_dir=str(tmp_path)
    )

@pytest.fixture
def test_feedback_manager(tmp_path):
    """Create a test feedback manager."""
    return FeedbackManager(feedback_dir=str(tmp_path))

class TestExperiment:
    """Test suite for Experiment class."""
    
    def test_initialization(self, test_experiment):
        """Test experiment initialization."""
        assert test_experiment.name == "test_experiment"
        assert test_experiment.description == "Test experiment"
        assert test_experiment.metrics == ["structure_score", "content_score"]
        assert os.path.exists(test_experiment.experiment_dir)
    
    def test_add_variant(self, test_experiment):
        """Test adding a variant."""
        variant_id = test_experiment.add_variant(
            name="test_variant",
            prompt_template="Test prompt {title}",
            model="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=4096
        )
        
        assert variant_id in test_experiment.variants
        variant = test_experiment.variants[variant_id]
        assert variant.name == "test_variant"
        assert variant.prompt_template == "Test prompt {title}"
        assert variant.model == "claude-3-opus-20240229"
    
    def test_record_trial(self, test_experiment):
        """Test recording a trial."""
        # Add a variant first
        variant_id = test_experiment.add_variant(
            name="test_variant",
            prompt_template="Test prompt",
            model="claude-3-opus-20240229"
        )
        
        # Record a trial
        trial_id = test_experiment.record_trial(
            variant_id=variant_id,
            title="Test Article",
            keywords=["test"],
            metrics={
                "structure_score": 8,
                "content_score": 7
            },
            evaluation={
                "strengths": ["Good structure"],
                "improvements": ["Add more detail"]
            }
        )
        
        assert len(test_experiment.trials) == 1
        trial = next(t for t in test_experiment.trials if t.id == trial_id)
        assert trial.variant_id == variant_id
        assert trial.title == "Test Article"
        assert trial.metrics["structure_score"] == 8
    
    def test_analyze_results(self, test_experiment):
        """Test analyzing experiment results."""
        # Add control variant
        control_id = test_experiment.add_variant(
            name="control",
            prompt_template="Control prompt",
            model="claude-3-opus-20240229"
        )
        
        # Add test variant
        test_id = test_experiment.add_variant(
            name="test",
            prompt_template="Test prompt",
            model="claude-3-opus-20240229"
        )
        
        # Add trials for control
        for score in [7, 8, 7, 8, 7]:
            test_experiment.record_trial(
                variant_id=control_id,
                title="Control Article",
                keywords=["test"],
                metrics={
                    "structure_score": score,
                    "content_score": score
                },
                evaluation={}
            )
        
        # Add trials for test variant
        for score in [8, 9, 8, 9, 8]:
            test_experiment.record_trial(
                variant_id=test_id,
                title="Test Article",
                keywords=["test"],
                metrics={
                    "structure_score": score,
                    "content_score": score
                },
                evaluation={}
            )
        
        # Analyze results
        results = test_experiment.analyze_results("structure_score", control_id)
        
        assert results["metric"] == "structure_score"
        assert results["control"] == control_id
        assert len(results["variants"]) == 2
        
        test_variant = results["variants"][test_id]
        assert test_variant["mean"] > results["variants"][control_id]["mean"]
        assert test_variant["significant"] == True
    
    def test_get_best_variant(self, test_experiment):
        """Test getting the best variant."""
        # Add variants
        variant1_id = test_experiment.add_variant(
            name="variant1",
            prompt_template="Prompt 1",
            model="claude-3-opus-20240229"
        )
        
        variant2_id = test_experiment.add_variant(
            name="variant2",
            prompt_template="Prompt 2",
            model="claude-3-opus-20240229"
        )
        
        # Add trials with variant1 performing better
        test_experiment.record_trial(
            variant_id=variant1_id,
            title="Article 1",
            keywords=["test"],
            metrics={
                "structure_score": 9,
                "content_score": 9
            },
            evaluation={}
        )
        
        test_experiment.record_trial(
            variant_id=variant2_id,
            title="Article 2",
            keywords=["test"],
            metrics={
                "structure_score": 7,
                "content_score": 7
            },
            evaluation={}
        )
        
        best_variant = test_experiment.get_best_variant("structure_score")
        assert best_variant == variant1_id

class TestFeedbackManager:
    """Test suite for FeedbackManager class."""
    
    def test_initialization(self, test_feedback_manager):
        """Test feedback manager initialization."""
        assert os.path.exists(test_feedback_manager.feedback_dir)
        assert isinstance(test_feedback_manager.criteria, dict)
        assert isinstance(test_feedback_manager.responses, list)
    
    def test_add_criterion(self, test_feedback_manager):
        """Test adding a feedback criterion."""
        criterion_id = test_feedback_manager.add_criterion(
            name="Content Quality",
            description="Evaluate the quality of the content",
            scale=[
                {"1": "Poor"},
                {"3": "Average"},
                {"5": "Excellent"}
            ],
            weight=2.0
        )
        
        assert criterion_id in test_feedback_manager.criteria
        criterion = test_feedback_manager.criteria[criterion_id]
        assert criterion.name == "Content Quality"
        assert criterion.weight == 2.0
    
    def test_record_feedback(self, test_feedback_manager):
        """Test recording feedback."""
        # Add a criterion first
        criterion_id = test_feedback_manager.add_criterion(
            name="Content Quality",
            description="Evaluate the quality of the content",
            scale=[
                {"1": "Poor"},
                {"3": "Average"},
                {"5": "Excellent"}
            ]
        )
        
        # Record feedback
        response_id = test_feedback_manager.record_feedback(
            article_id="test_article",
            evaluator_id="test_evaluator",
            ratings={criterion_id: 5},
            comments="Excellent article"
        )
        
        assert len(test_feedback_manager.responses) == 1
        response = next(r for r in test_feedback_manager.responses if r.id == response_id)
        assert response.article_id == "test_article"
        assert response.ratings[criterion_id] == 5
    
    def test_calculate_article_score(self, test_feedback_manager):
        """Test calculating article scores."""
        # Add criteria
        quality_id = test_feedback_manager.add_criterion(
            name="Quality",
            description="Content quality",
            scale=[{"1": "Poor"}, {"5": "Excellent"}],
            weight=2.0
        )
        
        structure_id = test_feedback_manager.add_criterion(
            name="Structure",
            description="Article structure",
            scale=[{"1": "Poor"}, {"5": "Excellent"}],
            weight=1.0
        )
        
        # Record multiple feedback responses
        test_feedback_manager.record_feedback(
            article_id="test_article",
            evaluator_id="evaluator1",
            ratings={
                quality_id: 4,
                structure_id: 5
            }
        )
        
        test_feedback_manager.record_feedback(
            article_id="test_article",
            evaluator_id="evaluator2",
            ratings={
                quality_id: 5,
                structure_id: 4
            }
        )
        
        scores = test_feedback_manager.calculate_article_score("test_article")
        
        assert "Quality" in scores
        assert "Structure" in scores
        assert "overall" in scores
        assert scores["Quality"] == 4.5  # Average of 4 and 5
        assert scores["Structure"] == 4.5  # Average of 5 and 4
        # Overall score: ((4.5 * 2) + (4.5 * 1)) / (2 + 1) = 4.5
        assert scores["overall"] == 4.5
    
    def test_get_feedback_stats(self, test_feedback_manager):
        """Test getting feedback statistics."""
        # Add criterion
        criterion_id = test_feedback_manager.add_criterion(
            name="Quality",
            description="Content quality",
            scale=[{"1": "Poor"}, {"5": "Excellent"}]
        )
        
        # Record multiple feedback responses
        for i in range(5):
            test_feedback_manager.record_feedback(
                article_id=f"article_{i}",
                evaluator_id=f"evaluator_{i % 2}",  # 2 evaluators
                ratings={criterion_id: i + 1}  # Ratings from 1 to 5
            )
        
        stats = test_feedback_manager.get_feedback_stats()
        
        assert stats["total_responses"] == 5
        assert stats["unique_articles"] == 5
        assert stats["unique_evaluators"] == 2
        assert "Quality" in stats["criteria_stats"]
        assert stats["criteria_stats"]["Quality"]["mean"] == 3.0  # Average of 1,2,3,4,5 