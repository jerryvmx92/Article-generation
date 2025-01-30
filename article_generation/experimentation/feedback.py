"""Human feedback collection and management for article evaluation."""

import os
import json
import uuid
import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

@dataclass
class FeedbackCriteria:
    """Defines a feedback criterion."""
    id: str
    name: str
    description: str
    scale: List[Dict[str, str]]  # List of {value: description} pairs
    weight: float = 1.0

@dataclass
class FeedbackResponse:
    """Represents a human feedback response."""
    id: str
    article_id: str
    evaluator_id: str
    timestamp: datetime.datetime
    ratings: Dict[str, int]  # criterion_id: rating
    comments: str
    metadata: Dict[str, Any]

class FeedbackManager:
    """Manages human feedback collection and analysis."""
    
    def __init__(self, feedback_dir: Optional[str] = None):
        """Initialize the feedback manager.
        
        Args:
            feedback_dir: Directory to store feedback data (default: ./feedback)
        """
        self.feedback_dir = feedback_dir or os.path.join(os.getcwd(), "feedback")
        self.criteria: Dict[str, FeedbackCriteria] = {}
        self.responses: List[FeedbackResponse] = []
        
        # Create feedback directory if it doesn't exist
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Load existing data
        self._load_feedback_data()
    
    def add_criterion(
        self,
        name: str,
        description: str,
        scale: List[Dict[str, str]],
        weight: float = 1.0
    ) -> str:
        """Add a feedback criterion.
        
        Args:
            name: Name of the criterion
            description: Detailed description
            scale: List of {value: description} pairs defining the rating scale
            weight: Weight of this criterion in overall score (default: 1.0)
            
        Returns:
            The criterion ID
        """
        criterion_id = str(uuid.uuid4())
        criterion = FeedbackCriteria(
            id=criterion_id,
            name=name,
            description=description,
            scale=scale,
            weight=weight
        )
        self.criteria[criterion_id] = criterion
        self._save_feedback_data()
        return criterion_id
    
    def record_feedback(
        self,
        article_id: str,
        evaluator_id: str,
        ratings: Dict[str, int],
        comments: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record human feedback for an article.
        
        Args:
            article_id: ID of the article being evaluated
            evaluator_id: ID of the human evaluator
            ratings: Dictionary mapping criterion IDs to ratings
            comments: Optional feedback comments
            metadata: Additional metadata
            
        Returns:
            The feedback response ID
        """
        # Validate ratings
        missing_criteria = set(self.criteria.keys()) - set(ratings.keys())
        if missing_criteria:
            raise ValueError(f"Missing ratings for criteria: {missing_criteria}")
            
        # Validate rating values
        for criterion_id, rating in ratings.items():
            criterion = self.criteria[criterion_id]
            valid_values = [int(v) for v in criterion.scale]
            if rating not in valid_values:
                raise ValueError(
                    f"Invalid rating {rating} for criterion {criterion.name}. "
                    f"Must be one of: {valid_values}"
                )
        
        response = FeedbackResponse(
            id=str(uuid.uuid4()),
            article_id=article_id,
            evaluator_id=evaluator_id,
            timestamp=datetime.datetime.now(),
            ratings=ratings,
            comments=comments,
            metadata=metadata or {}
        )
        self.responses.append(response)
        self._save_feedback_data()
        return response.id
    
    def get_article_feedback(self, article_id: str) -> List[FeedbackResponse]:
        """Get all feedback responses for an article.
        
        Args:
            article_id: ID of the article
            
        Returns:
            List of feedback responses
        """
        return [r for r in self.responses if r.article_id == article_id]
    
    def get_evaluator_feedback(self, evaluator_id: str) -> List[FeedbackResponse]:
        """Get all feedback responses from an evaluator.
        
        Args:
            evaluator_id: ID of the evaluator
            
        Returns:
            List of feedback responses
        """
        return [r for r in self.responses if r.evaluator_id == evaluator_id]
    
    def calculate_article_score(self, article_id: str) -> Dict[str, float]:
        """Calculate weighted average scores for an article.
        
        Args:
            article_id: ID of the article
            
        Returns:
            Dictionary containing average scores per criterion and overall score
        """
        responses = self.get_article_feedback(article_id)
        if not responses:
            return {}
            
        scores = {}
        total_weight = sum(c.weight for c in self.criteria.values())
        
        # Calculate average per criterion
        for criterion_id, criterion in self.criteria.items():
            criterion_ratings = [
                r.ratings[criterion_id] 
                for r in responses 
                if criterion_id in r.ratings
            ]
            if criterion_ratings:
                scores[criterion.name] = sum(criterion_ratings) / len(criterion_ratings)
        
        # Calculate weighted overall score
        if scores:
            weighted_sum = sum(
                scores[c.name] * c.weight 
                for c in self.criteria.values() 
                if c.name in scores
            )
            scores["overall"] = weighted_sum / total_weight
        
        return scores
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback.
        
        Returns:
            Dictionary containing feedback statistics
        """
        if not self.responses:
            return {}
            
        stats = {
            "total_responses": len(self.responses),
            "unique_articles": len({r.article_id for r in self.responses}),
            "unique_evaluators": len({r.evaluator_id for r in self.responses}),
            "criteria_stats": {},
            "time_range": {
                "start": min(r.timestamp for r in self.responses),
                "end": max(r.timestamp for r in self.responses)
            }
        }
        
        # Calculate stats per criterion
        for criterion_id, criterion in self.criteria.items():
            all_ratings = [
                r.ratings[criterion_id] 
                for r in self.responses 
                if criterion_id in r.ratings
            ]
            if all_ratings:
                stats["criteria_stats"][criterion.name] = {
                    "count": len(all_ratings),
                    "mean": sum(all_ratings) / len(all_ratings),
                    "min": min(all_ratings),
                    "max": max(all_ratings)
                }
        
        return stats
    
    def _save_feedback_data(self):
        """Save feedback data to disk."""
        data = {
            "criteria": {
                cid: {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "scale": c.scale,
                    "weight": c.weight
                }
                for cid, c in self.criteria.items()
            },
            "responses": [
                {
                    "id": r.id,
                    "article_id": r.article_id,
                    "evaluator_id": r.evaluator_id,
                    "timestamp": r.timestamp.isoformat(),
                    "ratings": r.ratings,
                    "comments": r.comments,
                    "metadata": r.metadata
                }
                for r in self.responses
            ]
        }
        
        path = os.path.join(self.feedback_dir, "feedback_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_feedback_data(self):
        """Load feedback data from disk."""
        path = os.path.join(self.feedback_dir, "feedback_data.json")
        if not os.path.exists(path):
            return
            
        with open(path) as f:
            data = json.load(f)
            
        # Load criteria
        self.criteria = {
            cid: FeedbackCriteria(
                id=c["id"],
                name=c["name"],
                description=c["description"],
                scale=c["scale"],
                weight=c["weight"]
            )
            for cid, c in data["criteria"].items()
        }
        
        # Load responses
        self.responses = [
            FeedbackResponse(
                id=r["id"],
                article_id=r["article_id"],
                evaluator_id=r["evaluator_id"],
                timestamp=datetime.datetime.fromisoformat(r["timestamp"]),
                ratings=r["ratings"],
                comments=r["comments"],
                metadata=r["metadata"]
            )
            for r in data["responses"]
        ] 