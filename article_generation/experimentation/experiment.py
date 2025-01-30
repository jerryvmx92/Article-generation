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
    name: str
    prompt_template: str
    metadata: Dict[str, Any]

@dataclass
class Trial:
    """Represents a single trial in an experiment."""
    id: str
    variant_id: str
    timestamp: datetime.datetime
    metrics: Dict[str, float]
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
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a variant to the experiment.
        
        Args:
            name: Name of the variant
            prompt_template: The prompt template to use
            metadata: Additional metadata about the variant
            
        Returns:
            The variant ID
        """
        variant_id = str(uuid.uuid4())
        variant = Variant(
            name=name,
            prompt_template=prompt_template,
            metadata=metadata or {}
        )
        self.variants[variant_id] = variant
        self._save_experiment()
        return variant_id
    
    def record_trial(
        self,
        variant_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a trial for a variant.
        
        Args:
            variant_id: ID of the variant used
            metrics: Metric values (must include all experiment metrics)
            metadata: Additional metadata about the trial
            
        Returns:
            The trial ID
        """
        # Validate metrics
        missing_metrics = set(self.metrics) - set(metrics.keys())
        if missing_metrics:
            raise ValueError(f"Missing required metrics: {missing_metrics}")
        
        # Ensure variant exists
        if variant_id not in self.variants:
            raise ValueError(f"Unknown variant: {variant_id}")
        
        trial = Trial(
            id=str(uuid.uuid4()),
            variant_id=variant_id,
            timestamp=datetime.datetime.now(),
            metrics=metrics,
            metadata=metadata or {}
        )
        self.trials.append(trial)
        self._save_experiment()
        return trial.id
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.trials:
            return {
                "total_trials": 0,
                "variant_performance": {}
            }
        
        # Get all variant IDs that have trials
        variant_ids = sorted(list(set(t.variant_id for t in self.trials)))
        if not variant_ids:
            return {
                "total_trials": 0,
                "variant_performance": {}
            }
        
        # Find baseline variant ID - look for variant with name "baseline" first
        baseline_id = None
        baseline_trials = []
        
        # First try to find a variant named "baseline"
        for vid, variant in self.variants.items():
            if variant.name.lower() == "baseline":
                baseline_id = vid
                baseline_trials = [t for t in self.trials if t.variant_id == vid]
                break
        
        # If no baseline variant found or it has no trials, use the variant with most trials
        if not baseline_trials:
            variant_trial_counts = {}
            for vid in variant_ids:
                variant_trial_counts[vid] = len([t for t in self.trials if t.variant_id == vid])
            
            baseline_id = max(variant_trial_counts.items(), key=lambda x: x[1])[0]
            baseline_trials = [t for t in self.trials if t.variant_id == baseline_id]
        
        if not baseline_trials:
            raise ValueError("No trials found for any variant")
        
        baseline_data = {
            metric: [t.metrics[metric] for t in baseline_trials]
            for metric in self.metrics
        }
        
        results = {
            "total_trials": len(self.trials),
            "baseline_variant": self.variants[baseline_id].name,
            "variant_performance": {}
        }
        
        # Calculate performance vs baseline for each variant
        for variant_id in variant_ids:
            if variant_id == baseline_id:
                continue
            
            variant_performance = {}
            for metric in self.metrics:
                variant_data = [
                    t.metrics[metric] 
                    for t in self.trials 
                    if t.variant_id == variant_id
                ]
                
                if not variant_data or not baseline_data[metric]:
                    continue
                
                # Calculate relative improvement
                baseline_mean = np.mean(baseline_data[metric])
                variant_mean = np.mean(variant_data)
                
                variant_performance[metric] = (variant_mean / baseline_mean) - 1
            
            if variant_performance:
                results["variant_performance"][self.variants[variant_id].name] = variant_performance
        
        return results
    
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
                "timestamp": trial.timestamp
            }
            # Add metrics
            row.update(trial.metrics)
            # Add metadata
            row.update(trial.metadata)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _save_experiment(self):
        """Save experiment data to disk."""
        experiment_data = {
            "name": self.name,
            "description": self.description,
            "metrics": self.metrics,
            "variants": {
                name: {
                    "name": v.name,
                    "prompt_template": v.prompt_template,
                    "metadata": v.metadata
                }
                for name, v in self.variants.items()
            },
            "trials": [
                {
                    "id": t.id,
                    "variant_id": t.variant_id,
                    "timestamp": t.timestamp.isoformat(),
                    "metrics": t.metrics,
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
            id: Variant(
                name=v["name"],
                prompt_template=v["prompt_template"],
                metadata=v["metadata"]
            )
            for id, v in data["variants"].items()
        }
        
        # Load trials
        self.trials = [
            Trial(
                id=t["id"],
                variant_id=t["variant_id"],
                timestamp=datetime.datetime.fromisoformat(t["timestamp"]),
                metrics=t["metrics"],
                metadata=t["metadata"]
            )
            for t in data["trials"]
        ] 