"""
Experiment Tracking System
========================

Comprehensive experiment management and comparison.
"""

import os
import json
import uuid
import logging
import pickle
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import numpy as np

from .licensing import require_license, LicenseManager


logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Represents a single experiment."""
    id: str
    name: str
    created_at: datetime
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    artifacts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: str = "running"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "config": self.config,
            "metrics": dict(self.metrics),
            "artifacts": self.artifacts,
            "tags": self.tags,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["metrics"] = defaultdict(list, data.get("metrics", {}))
        return cls(**data)


class ExperimentTracker:
    """
    Premium experiment tracking and management system.
    
    Features:
    - Experiment logging and comparison
    - Hyperparameter optimization
    - Model artifact versioning
    - Experiment reproduction
    """
    
    @require_license("experiment_tracking")
    def __init__(
        self,
        experiment_dir: str = "./experiments",
        license_manager: Optional[LicenseManager] = None,
        auto_save: bool = True
    ):
        self.license_manager = license_manager or LicenseManager()
        self.experiment_dir = Path(experiment_dir)
        self.auto_save = auto_save
        self._current_experiment: Optional[Experiment] = None
        self._experiments: Dict[str, Experiment] = {}
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_experiments()
    
    def _load_existing_experiments(self):
        """Load existing experiments from disk."""
        for exp_dir in self.experiment_dir.iterdir():
            if exp_dir.is_dir():
                metadata_file = exp_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        data = json.load(f)
                    experiment = Experiment.from_dict(data)
                    self._experiments[experiment.id] = experiment
    
    def create_experiment(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """Create a new experiment."""
        experiment_id = str(uuid.uuid4())[:8]
        experiment = Experiment(
            id=experiment_id,
            name=name,
            created_at=datetime.now(),
            config=config or {},
            tags=tags or []
        )
        
        self._current_experiment = experiment
        self._experiments[experiment_id] = experiment
        
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created experiment: {name} ({experiment_id})")
        
        if self.auto_save:
            self.save_experiment(experiment_id)
        
        return experiment
    
    def set_current_experiment(self, experiment_id: str):
        """Set the current active experiment."""
        if experiment_id in self._experiments:
            self._current_experiment = self._experiments[experiment_id]
            logger.info(f"Switched to experiment: {experiment_id}")
        else:
            raise ValueError(f"Experiment {experiment_id} not found")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if not self._current_experiment:
            raise RuntimeError("No active experiment")
        
        self._current_experiment.metrics[name].append(value)
        
        if self.auto_save:
            self.save_experiment(self._current_experiment.id)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters."""
        if not self._current_experiment:
            raise RuntimeError("No active experiment")
        
        self._current_experiment.config.update(config)
        
        if self.auto_save:
            self.save_experiment(self._current_experiment.id)
    
    def save_artifact(
        self,
        obj: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save an artifact (model, data, etc.)."""
        if not self._current_experiment:
            raise RuntimeError("No active experiment")
        
        exp_dir = self.experiment_dir / self._current_experiment.id
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        artifact_path = artifacts_dir / name
        with open(artifact_path, "wb") as f:
            pickle.dump(obj, f)
        
        artifact_record = {
            "name": name,
            "path": str(artifact_path.relative_to(exp_dir)),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self._current_experiment.artifacts.append(str(artifact_path))
        
        if self.auto_save:
            self.save_experiment(self._current_experiment.id)
        
        logger.info(f"Saved artifact: {name}")
    
    def load_artifact(self, experiment_id: str, name: str) -> Any:
        """Load an artifact from an experiment."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_dir = self.experiment_dir / experiment_id
        artifact_path = exp_dir / "artifacts" / name
        
        with open(artifact_path, "rb") as f:
            return pickle.load(f)
    
    def save_experiment(self, experiment_id: str):
        """Save experiment to disk."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self._experiments[experiment_id]
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        metadata_file = exp_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> List[Experiment]:
        """List experiments with optional filters."""
        experiments = list(self._experiments.values())
        
        if tags:
            experiments = [
                exp for exp in experiments
                if any(tag in exp.tags for tag in tags)
            ]
        
        if status:
            experiments = [
                exp for exp in experiments
                if exp.status == status
            ]
        
        return experiments
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Returns a comparison report with metrics and config differences.
        """
        experiments = [
            self._experiments[exp_id]
            for exp_id in experiment_ids
            if exp_id in self._experiments
        ]
        
        if not experiments:
            return {}
        
        comparison = {
            "experiments": [exp.to_dict() for exp in experiments],
            "best": {},
            "config_differences": {}
        }
        
        metrics = metrics or list({
            metric
            for exp in experiments
            for metric in exp.metrics
        })
        
        for metric in metrics:
            values = []
            for exp in experiments:
                if metric in exp.metrics and exp.metrics[metric]:
                    values.append((exp.id, np.mean(exp.metrics[metric])))
            
            if values:
                values.sort(key=lambda x: x[1])
                comparison["best"][metric] = {
                    "best_experiment": values[0][0],
                    "best_value": values[0][1],
                    "all_values": {exp_id: val for exp_id, val in values}
                }
        
        return comparison
    
    def get_best_experiment(
        self,
        metric: str,
        maximize: bool = False
    ) -> Optional[Experiment]:
        """
        Find the best experiment based on a metric."""
        experiments = self.list_experiments()
        
        best_exp = None
        best_value = -float('inf') if maximize else float('inf')
        
        for exp in experiments:
            if metric in exp.metrics and exp.metrics[metric]:
                current_value = np.mean(exp.metrics[metric])
                if maximize:
                    if current_value > best_value:
                        best_value = current_value
                        best_exp = exp
                else:
                    if current_value < best_value:
                        best_value = current_value
                        best_exp = exp
        
        return best_exp


class HyperparameterOptimizer:
    """
    Hyperparameter optimization (grid search, random search, Bayesian optimization.
    """
    
    @require_license("experiment_tracking")
    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        tracker: ExperimentTracker,
        license_manager: Optional[LicenseManager] = None
    ):
        self.objective_fn = objective_fn
        self.tracker = tracker
        self.license_manager = license_manager or LicenseManager()
        self._trials: List[Dict[str, Any]] = []
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        experiment_name_prefix: str = "grid_search"
    ) -> Dict[str, Any]:
        """
        Perform grid search over parameter combinations.
        """
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        best_params = None
        best_score = None
        
        for i, combination in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combination))
            
            exp = self.tracker.create_experiment(
                name=f"{experiment_name_prefix}_{i}",
                config=params
            )
            
            score = self.objective_fn(params)
            
            self.tracker.log_metric("score", score)
            self.tracker._current_experiment.status = "completed"
            self.tracker.save_experiment(exp.id)
            
            self._trials.append({
                "params": params,
                "score": score,
                "experiment_id": exp.id
            })
            
            if best_score is None or score < best_score:
                best_score = score
                best_params = params
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "trials": self._trials
        }
    
    def random_search(
        self,
        param_distributions: Dict[str, Callable],
        num_trials: int = 50,
        experiment_name_prefix: str = "random_search"
    ) -> Dict[str, Any]:
        """
        Perform random search over parameter distributions.
        """
        import random
        
        best_params = None
        best_score = None
        
        for i in range(num_trials):
            params = {
                name: sampler()
                for name, sampler in param_distributions.items()
            }
            
            exp = self.tracker.create_experiment(
                name=f"{experiment_name_prefix}_{i}",
                config=params
            )
            
            score = self.objective_fn(params)
            
            self.tracker.log_metric("score", score)
            self.tracker._current_experiment.status = "completed"
            self.tracker.save_experiment(exp.id)
            
            self._trials.append({
                "params": params,
                "score": score,
                "experiment_id": exp.id
            })
            
            if best_score is None or score < best_score:
                best_score = score
                best_params = params
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "trials": self._trials
        }
