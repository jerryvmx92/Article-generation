"""Core experiment functionality for A/B testing article generation."""

import os
import json
import uuid
import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats

@dataclass
class Variant:
    """Represents a variant in an experiment."""
    id: str
    name: str
    prompt_template: str
    model: str
    temperature: float
    max_tokens: int
    metadata: Dict[str, Any]

@dataclass
class Trial:
    """Represents a single trial in an experiment."""
    id: str
    variant_id: str
    timestamp: datetime.datetime
    title: str
    keywords: List[str]
    metrics: Dict[str, float]
    evaluation: Dict[str, Any]
    metadata: Dict[str, Any]

class Experiment:
    """Manages A/B testing experiments for article generation."""
    
    def __init__(
        self,
        name: str,
        description: str,
        metrics: List[str],
        experiment_dir: Optional[str] = None
    ):
        """Initialize an experiment.
        
        Args:
            name: Name of the experiment
            description: Description of what is being tested
            metrics: List of metrics to track (e.g., ["structure_score", "content_score"])
            experiment_dir: Directory to store experiment data (default: ./experiments)
        """
        self.name = name
        self.description = description
        self.metrics = metrics
        self.experiment_dir = experiment_dir or os.path.join(os.getcwd(), "experiments")
        self.variants: Dict[str, Variant] = {}
        self.trials: List[Trial] = []
        
        # Create experiment directory if it doesn't exist
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Load existing data if available
        self._load_experiment()
    
    def add_variant(
        self,
        name: str,
        prompt_template: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a variant to the experiment.
        
        Args:
            name: Name of the variant
            prompt_template: The prompt template to use
            model: The model to use (e.g., "claude-3-opus-20240229")
            temperature: Model temperature
            max_tokens: Maximum tokens
            metadata: Additional metadata about the variant
            
        Returns:
            The variant ID
        """
        variant_id = str(uuid.uuid4())
        variant = Variant(
            id=variant_id,
            name=name,
            prompt_template=prompt_template,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata or {}
        )
        self.variants[variant_id] = variant
        self._save_experiment()
        return variant_id
    
    def record_trial(
        self,
        variant_id: str,
        title: str,
        keywords: List[str],
        metrics: Dict[str, float],
        evaluation: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a trial for a variant.
        
        Args:
            variant_id: ID of the variant used
            title: Article title
            keywords: Article keywords
            metrics: Metric values (must include all experiment metrics)
            evaluation: Full evaluation results
            metadata: Additional metadata about the trial
            
        Returns:
            The trial ID
        """
        # Validate metrics
        missing_metrics = set(self.metrics) - set(metrics.keys())
        if missing_metrics:
            raise ValueError(f"Missing required metrics: {missing_metrics}")
        
        trial = Trial(
            id=str(uuid.uuid4()),
            variant_id=variant_id,
            timestamp=datetime.datetime.now(),
            title=title,
            keywords=keywords,
            metrics=metrics,
            evaluation=evaluation,
            metadata=metadata or {}
        )
        self.trials.append(trial)
        self._save_experiment()
        return trial.id
    
    def analyze_results(self, metric: str, control_variant: str) -> Dict[str, Any]:
        """Analyze experiment results for a specific metric.
        
        Args:
            metric: The metric to analyze
            control_variant: ID of the control variant to compare against
            
        Returns:
            Dictionary containing analysis results
        """
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")
            
        results = {
            "metric": metric,
            "control": control_variant,
            "variants": {},
            "summary": {}
        }
        
        # Get control data
        control_data = [
            t.metrics[metric] 
            for t in self.trials 
            if t.variant_id == control_variant
        ]
        
        if not control_data:
            raise ValueError("No data for control variant")
            
        control_mean = np.mean(control_data)
        control_std = np.std(control_data)
        
        results["variants"][control_variant] = {
            "name": self.variants[control_variant].name,
            "sample_size": len(control_data),
            "mean": float(control_mean),
            "std": float(control_std)
        }
        
        # Compare each variant against control
        for variant_id, variant in self.variants.items():
            if variant_id == control_variant:
                continue
                
            variant_data = [
                t.metrics[metric] 
                for t in self.trials 
                if t.variant_id == variant_id
            ]
            
            if not variant_data:
                continue
                
            # Calculate statistics
            variant_mean = np.mean(variant_data)
            variant_std = np.std(variant_data)
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(control_data, variant_data)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((control_std**2 + variant_std**2) / 2)
            effect_size = (variant_mean - control_mean) / pooled_std
            
            results["variants"][variant_id] = {
                "name": variant.name,
                "sample_size": len(variant_data),
                "mean": float(variant_mean),
                "std": float(variant_std),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "significant": p_value < 0.05,
                "improvement": float(((variant_mean / control_mean) - 1) * 100)
            }
        
        # Summarize results
        results["summary"] = {
            "total_trials": len(self.trials),
            "start_date": min(t.timestamp for t in self.trials),
            "end_date": max(t.timestamp for t in self.trials),
            "significant_improvements": sum(
                1 for v in results["variants"].values()
                if v.get("significant", False) and v.get("improvement", 0) > 0
            )
        }
        
        return results
    
    def get_best_variant(self, metric: str) -> Optional[str]:
        """Get the best performing variant for a metric.
        
        Args:
            metric: The metric to optimize for
            
        Returns:
            ID of the best variant, or None if insufficient data
        """
        if not self.trials:
            return None
            
        variant_means = {}
        for variant_id in self.variants:
            variant_data = [
                t.metrics[metric] 
                for t in self.trials 
                if t.variant_id == variant_id
            ]
            if variant_data:
                variant_means[variant_id] = np.mean(variant_data)
        
        if not variant_means:
            return None
            
        return max(variant_means.items(), key=lambda x: x[1])[0]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert experiment data to a pandas DataFrame.
        
        Returns:
            DataFrame containing all trial data
        """
        data = []
        for trial in self.trials:
            row = {
                "trial_id": trial.id,
                "variant_id": trial.variant_id,
                "variant_name": self.variants[trial.variant_id].name,
                "timestamp": trial.timestamp,
                "title": trial.title,
                "keywords": ",".join(trial.keywords)
            }
            # Add metrics
            row.update(trial.metrics)
            # Add variant info
            variant = self.variants[trial.variant_id]
            row.update({
                "model": variant.model,
                "temperature": variant.temperature
            })
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _save_experiment(self):
        """Save experiment data to disk."""
        experiment_data = {
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
            "variants": {
                vid: {
                    "id": v.id,
                    "name": v.name,
                    "prompt_template": v.prompt_template,
                    "model": v.model,
                    "temperature": v.temperature,
                    "max_tokens": v.max_tokens,
                    "metadata": v.metadata
                }
                for vid, v in self.variants.items()
            },
            "trials": [
                {
                    "id": t.id,
                    "variant_id": t.variant_id,
                    "timestamp": t.timestamp.isoformat(),
                    "title": t.title,
                    "keywords": t.keywords,
                    "metrics": t.metrics,
                    "evaluation": t.evaluation,
                    "metadata": t.metadata
                }
                for t in self.trials
            ]
        }
        
        path = os.path.join(self.experiment_dir, f"{self.name}.json")
        with open(path, "w") as f:
            json.dump(experiment_data, f, indent=2)
    
    def _load_experiment(self):
        """Load experiment data from disk."""
        path = os.path.join(self.experiment_dir, f"{self.name}.json")
        if not os.path.exists(path):
            return
            
        with open(path) as f:
            data = json.load(f)
            
        # Load variants
        self.variants = {
            vid: Variant(
                id=v["id"],
                name=v["name"],
                prompt_template=v["prompt_template"],
                model=v["model"],
                temperature=v["temperature"],
                max_tokens=v["max_tokens"],
                metadata=v["metadata"]
            )
            for vid, v in data["variants"].items()
        }
        
        # Load trials
        self.trials = [
            Trial(
                id=t["id"],
                variant_id=t["variant_id"],
                timestamp=datetime.datetime.fromisoformat(t["timestamp"]),
                title=t["title"],
                keywords=t["keywords"],
                metrics=t["metrics"],
                evaluation=t["evaluation"],
                metadata=t["metadata"]
            )
            for t in data["trials"]
        ] 