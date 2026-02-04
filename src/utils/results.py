"""
Result Management System for PAID-FD Experiments

Features:
- Unique run IDs to prevent overwriting
- Structured result storage with metadata
- Easy result retrieval and comparison
- Git commit tracking for reproducibility
"""

import json
import os
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class RunMetadata:
    """Metadata for an experiment run."""
    method: str
    experiment: str
    run_id: str
    timestamp: str
    git_commit: str
    hostname: str
    python_version: str
    

class ResultManager:
    """
    Manages experiment result storage and retrieval.
    
    Key Features:
    - Each run gets a unique ID (timestamp + config hash)
    - Results are never overwritten
    - Easy to list and compare results
    
    Directory Structure:
        results/
        ├── experiments/
        │   ├── exp1_efficiency/
        │   │   ├── PAID-FD_cifar100_20250203_143000_a1b2c3d4.json
        │   │   └── CSRA_cifar100_20250203_144500_e5f6g7h8.json
        │   ├── exp2_convergence/
        │   └── ...
        ├── checkpoints/
        ├── logs/
        └── figures/
    """
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.base_dir / "experiments",
            self.base_dir / "checkpoints",
            self.base_dir / "logs",
            self.base_dir / "figures"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def generate_run_id(self, config: Dict) -> str:
        """
        Generate a unique run ID from timestamp and config hash.
        
        Format: YYYYMMDD_HHMMSS_<8-char-hash>
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{timestamp}_{config_hash}"
    
    def get_experiment_dir(self, exp_name: str) -> Path:
        """Get or create experiment directory."""
        exp_dir = self.base_dir / "experiments" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def get_checkpoint_dir(self, exp_name: str, method_name: str) -> Path:
        """Get or create checkpoint directory for a method."""
        ckpt_dir = self.base_dir / "checkpoints" / exp_name / method_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir
    
    def save_result(
        self,
        exp_name: str,
        method_name: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> Path:
        """
        Save experiment results to a JSON file.
        
        Args:
            exp_name: Name of the experiment (e.g., "exp2_convergence")
            method_name: Name of the method (e.g., "PAID-FD")
            config: Configuration dictionary
            results: Results dictionary
            run_id: Optional custom run ID (auto-generated if None)
            
        Returns:
            Path to the saved file
        """
        if run_id is None:
            run_id = self.generate_run_id(config)
        
        exp_dir = self.get_experiment_dir(exp_name)
        
        # Build filename: {method}_{dataset}_{run_id}.json
        dataset = config.get("dataset", config.get("data", {}).get("private", {}).get("name", "unknown"))
        filename = f"{method_name}_{dataset}_{run_id}.json"
        filepath = exp_dir / filename
        
        # Gather metadata
        metadata = {
            "method": method_name,
            "experiment": exp_name,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "hostname": self._get_hostname(),
            "python_version": self._get_python_version()
        }
        
        # Combine everything
        full_result = {
            "metadata": metadata,
            "config": config,
            "results": results
        }
        
        # Save with custom encoder for numpy types
        with open(filepath, 'w') as f:
            json.dump(full_result, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Results saved: {filepath}")
        return filepath
    
    def load_result(self, filepath: Union[str, Path]) -> Dict:
        """Load a result file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_results(
        self,
        exp_name: str,
        method_name: Optional[str] = None,
        sort_by: str = "timestamp"
    ) -> List[Path]:
        """
        List all result files for an experiment.
        
        Args:
            exp_name: Experiment name
            method_name: Optional filter by method
            sort_by: Sort key ("timestamp" or "method")
            
        Returns:
            List of result file paths
        """
        exp_dir = self.get_experiment_dir(exp_name)
        
        if method_name:
            pattern = f"{method_name}_*.json"
        else:
            pattern = "*.json"
        
        files = list(exp_dir.glob(pattern))
        
        if sort_by == "timestamp":
            # Extract timestamp from filename for sorting
            files.sort(key=lambda f: f.stem.split('_')[-2] if len(f.stem.split('_')) >= 2 else "")
        elif sort_by == "method":
            files.sort(key=lambda f: f.stem.split('_')[0])
        
        return files
    
    def get_latest_result(
        self,
        exp_name: str,
        method_name: str
    ) -> Optional[Dict]:
        """Get the most recent result for a method."""
        files = self.list_results(exp_name, method_name, sort_by="timestamp")
        if files:
            return self.load_result(files[-1])
        return None
    
    def result_exists(
        self,
        exp_name: str,
        method_name: str
    ) -> bool:
        """Check if any result exists for a method."""
        files = self.list_results(exp_name, method_name)
        return len(files) > 0
    
    def compare_results(
        self,
        exp_name: str,
        metric: str = "final_accuracy"
    ) -> Dict[str, float]:
        """
        Compare latest results across all methods.
        
        Args:
            exp_name: Experiment name
            metric: Metric to compare
            
        Returns:
            Dictionary of {method: metric_value}
        """
        comparison = {}
        files = self.list_results(exp_name)
        
        # Group by method, keep only latest
        method_files = {}
        for f in files:
            method = f.stem.split('_')[0]
            if method not in method_files:
                method_files[method] = f
            else:
                # Keep the newer one
                if f.stat().st_mtime > method_files[method].stat().st_mtime:
                    method_files[method] = f
        
        for method, filepath in method_files.items():
            result = self.load_result(filepath)
            # Navigate to the metric
            value = result.get("results", {}).get("metrics", {}).get(metric)
            if value is None:
                # Try alternative paths
                value = result.get("results", {}).get(metric)
            comparison[method] = value
        
        return comparison
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _get_hostname(self) -> str:
        """Get hostname."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


class ExperimentTracker:
    """
    Tracks progress and results for a single experiment run.
    
    Usage:
        tracker = ExperimentTracker("exp2_convergence", "PAID-FD", config)
        
        for round_idx in range(n_rounds):
            # ... run round ...
            tracker.log_round(round_idx, accuracy, loss, ...)
        
        tracker.log_final_metrics({"final_accuracy": 0.45})
        tracker.save()
    """
    
    def __init__(
        self,
        exp_name: str,
        method_name: str,
        config: Dict[str, Any],
        base_dir: str = "results"
    ):
        self.exp_name = exp_name
        self.method_name = method_name
        self.config = config
        self.result_manager = ResultManager(base_dir)
        self.run_id = self.result_manager.generate_run_id(config)
        
        # Initialize result storage
        self.results = {
            "rounds": [],
            "metrics": {},
            "energy_history": [],
            "participation_history": [],
            "price_history": [],
            "timing": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_seconds": None
            }
        }
        
        self._start_time = datetime.now()
    
    def log_round(
        self,
        round_idx: int,
        accuracy: float,
        loss: float,
        participation_rate: float,
        energy_breakdown: Optional[Dict[str, float]] = None,
        price: Optional[float] = None,
        device_decisions: Optional[List[Dict]] = None,
        extra: Optional[Dict] = None
    ):
        """
        Log results for a single round.
        
        Args:
            round_idx: Round index
            accuracy: Test accuracy
            loss: Training/distillation loss
            participation_rate: Fraction of participating devices
            energy_breakdown: Energy by component {training, inference, communication}
            price: Server's announced price (for Stackelberg methods)
            device_decisions: List of device decisions [{device_id, s, eps, participates}]
            extra: Additional metrics to log
        """
        round_data = {
            "round": round_idx,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "participation_rate": float(participation_rate),
            "timestamp": datetime.now().isoformat()
        }
        
        if energy_breakdown:
            round_data["energy"] = {k: float(v) for k, v in energy_breakdown.items()}
            self.results["energy_history"].append({
                "round": round_idx,
                **round_data["energy"]
            })
        
        if price is not None:
            round_data["price"] = float(price)
            self.results["price_history"].append({
                "round": round_idx,
                "price": float(price)
            })
        
        if device_decisions:
            round_data["n_participants"] = sum(1 for d in device_decisions if d.get("participates", False))
            # Store summary, not full decisions (to save space)
            participating = [d for d in device_decisions if d.get("participates", False)]
            if participating:
                round_data["avg_s"] = float(np.mean([d["s"] for d in participating]))
                round_data["avg_eps"] = float(np.mean([d["eps"] for d in participating]))
            else:
                round_data["avg_s"] = 0.0
                round_data["avg_eps"] = 0.0
        
        if extra:
            round_data.update(extra)
        
        self.results["rounds"].append(round_data)
        self.results["participation_history"].append({
            "round": round_idx,
            "rate": float(participation_rate)
        })
    
    def log_final_metrics(self, metrics: Dict[str, Any]):
        """Log final evaluation metrics."""
        self.results["metrics"] = {
            k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
            for k, v in metrics.items()
        }
    
    def log_device_summary(self, devices: List[Dict]):
        """Log summary of device heterogeneity."""
        self.results["device_summary"] = {
            "n_devices": len(devices),
            "type_distribution": {},
            "lambda_distribution": {},
            "c_comm_stats": {},
            "c_inf_stats": {}
        }
        
        # Compute distributions
        types = [d.get("device_type", "unknown") for d in devices]
        lambdas = [d.get("lambda_i", 0) for d in devices]
        c_comms = [d.get("c_comm", 0) for d in devices]
        c_infs = [d.get("c_inf", 0) for d in devices]
        
        for t in set(types):
            self.results["device_summary"]["type_distribution"][str(t)] = types.count(t)
        
        self.results["device_summary"]["lambda_distribution"] = {
            "mean": float(np.mean(lambdas)),
            "std": float(np.std(lambdas)),
            "min": float(np.min(lambdas)),
            "max": float(np.max(lambdas))
        }
        
        self.results["device_summary"]["c_comm_stats"] = {
            "mean": float(np.mean(c_comms)),
            "std": float(np.std(c_comms))
        }
        
        self.results["device_summary"]["c_inf_stats"] = {
            "mean": float(np.mean(c_infs)),
            "std": float(np.std(c_infs))
        }
    
    def get_best_accuracy(self) -> float:
        """Get best accuracy achieved during training."""
        if not self.results["rounds"]:
            return 0.0
        return max(r["accuracy"] for r in self.results["rounds"])
    
    def get_final_accuracy(self) -> float:
        """Get accuracy from the last round."""
        if not self.results["rounds"]:
            return 0.0
        return self.results["rounds"][-1]["accuracy"]
    
    def get_convergence_round(self, threshold: float = 0.95) -> Optional[int]:
        """
        Get the round at which accuracy reached threshold * final_accuracy.
        
        Returns None if not converged.
        """
        final_acc = self.get_final_accuracy()
        target = threshold * final_acc
        
        for r in self.results["rounds"]:
            if r["accuracy"] >= target:
                return r["round"]
        return None
    
    def save(self) -> Path:
        """Save all results and return the file path."""
        # Record end time
        end_time = datetime.now()
        self.results["timing"]["end_time"] = end_time.isoformat()
        self.results["timing"]["total_seconds"] = (end_time - self._start_time).total_seconds()
        
        # Add summary metrics if not already present
        if "final_accuracy" not in self.results["metrics"]:
            self.results["metrics"]["final_accuracy"] = self.get_final_accuracy()
        if "best_accuracy" not in self.results["metrics"]:
            self.results["metrics"]["best_accuracy"] = self.get_best_accuracy()
        
        # Compute total energy
        if self.results["energy_history"]:
            total_energy = sum(
                sum(e.get(k, 0) for k in ["training", "inference", "communication"])
                for e in self.results["energy_history"]
            )
            self.results["metrics"]["total_energy"] = total_energy
        
        return self.result_manager.save_result(
            exp_name=self.exp_name,
            method_name=self.method_name,
            config=self.config,
            results=self.results,
            run_id=self.run_id
        )
    
    def print_summary(self):
        """Print a summary of the experiment."""
        print(f"\n{'='*50}")
        print(f"Experiment: {self.exp_name}")
        print(f"Method: {self.method_name}")
        print(f"Run ID: {self.run_id}")
        print(f"{'='*50}")
        
        if self.results["rounds"]:
            print(f"Rounds completed: {len(self.results['rounds'])}")
            print(f"Final accuracy: {self.get_final_accuracy():.4f}")
            print(f"Best accuracy: {self.get_best_accuracy():.4f}")
        
        if self.results["metrics"]:
            print(f"\nFinal Metrics:")
            for k, v in self.results["metrics"].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        
        print(f"{'='*50}\n")
