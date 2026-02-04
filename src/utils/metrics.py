"""
Metrics calculation utilities for PAID-FD experiments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AccuracyMetrics:
    """Container for accuracy-related metrics."""
    accuracy: float
    top5_accuracy: float
    per_class_accuracy: Optional[np.ndarray] = None


class MetricsCalculator:
    """
    Calculate various evaluation metrics.
    """
    
    @staticmethod
    def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate classification accuracy.
        
        Args:
            predictions: Predicted class indices or logits
            targets: Ground truth class indices
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)
        return float(np.mean(predictions == targets))
    
    @staticmethod
    def top_k_accuracy(
        logits: np.ndarray, 
        targets: np.ndarray, 
        k: int = 5
    ) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            logits: Prediction logits (n_samples, n_classes)
            targets: Ground truth class indices
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(logits, axis=1)[:, -k:]
        correct = np.array([t in top_k for t, top_k in zip(targets, top_k_preds)])
        return float(np.mean(correct))
    
    @staticmethod
    def per_class_accuracy(
        predictions: np.ndarray,
        targets: np.ndarray,
        n_classes: int
    ) -> np.ndarray:
        """
        Calculate per-class accuracy.
        
        Args:
            predictions: Predicted class indices
            targets: Ground truth class indices
            n_classes: Total number of classes
            
        Returns:
            Array of per-class accuracies
        """
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)
        
        per_class_acc = np.zeros(n_classes)
        for c in range(n_classes):
            mask = targets == c
            if mask.sum() > 0:
                per_class_acc[c] = np.mean(predictions[mask] == c)
        
        return per_class_acc
    
    @staticmethod
    def confusion_matrix(
        predictions: np.ndarray,
        targets: np.ndarray,
        n_classes: int
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            predictions: Predicted class indices
            targets: Ground truth class indices
            n_classes: Total number of classes
            
        Returns:
            Confusion matrix of shape (n_classes, n_classes)
        """
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)
        
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for pred, target in zip(predictions, targets):
            cm[target, pred] += 1
        
        return cm
    
    @staticmethod
    def compute_all(
        predictions: np.ndarray,
        targets: np.ndarray,
        n_classes: int = 100
    ) -> Dict[str, float]:
        """
        Compute all standard metrics.
        
        Returns dictionary with:
        - accuracy
        - top5_accuracy
        - mean_per_class_accuracy
        """
        if predictions.ndim == 1:
            logits = None
            preds = predictions
        else:
            logits = predictions
            preds = predictions.argmax(axis=1)
        
        results = {
            "accuracy": MetricsCalculator.accuracy(preds, targets)
        }
        
        if logits is not None:
            results["top5_accuracy"] = MetricsCalculator.top_k_accuracy(logits, targets, k=5)
        
        per_class = MetricsCalculator.per_class_accuracy(preds, targets, n_classes)
        results["mean_per_class_accuracy"] = float(np.mean(per_class))
        
        return results


class ConvergenceAnalyzer:
    """
    Analyze convergence behavior of training curves.
    """
    
    @staticmethod
    def find_convergence_round(
        accuracies: List[float],
        threshold: float = 0.95
    ) -> Optional[int]:
        """
        Find the round at which accuracy reached threshold * final_accuracy.
        
        Args:
            accuracies: List of accuracies per round
            threshold: Fraction of final accuracy to consider as converged
            
        Returns:
            Round index or None if not converged
        """
        if not accuracies:
            return None
        
        final_acc = accuracies[-1]
        target = threshold * final_acc
        
        for i, acc in enumerate(accuracies):
            if acc >= target:
                return i
        
        return None
    
    @staticmethod
    def compute_auc(accuracies: List[float]) -> float:
        """
        Compute area under the accuracy curve (normalized).
        
        Higher is better - indicates faster convergence.
        """
        if not accuracies:
            return 0.0
        
        n = len(accuracies)
        # Normalize by max possible area
        return np.trapz(accuracies) / n
    
    @staticmethod
    def compute_stability(accuracies: List[float], window: int = 10) -> float:
        """
        Compute training stability as 1 - variance of recent rounds.
        
        Args:
            accuracies: List of accuracies
            window: Number of recent rounds to consider
            
        Returns:
            Stability score (higher is more stable)
        """
        if len(accuracies) < window:
            return 0.0
        
        recent = accuracies[-window:]
        variance = np.var(recent)
        return 1.0 / (1.0 + variance * 100)  # Scale variance


class EnergyAnalyzer:
    """
    Analyze energy consumption patterns.
    """
    
    @staticmethod
    def compute_total_energy(energy_history: List[Dict[str, float]]) -> float:
        """Compute total energy across all rounds."""
        return sum(
            sum(e.get(k, 0) for k in ["training", "inference", "communication", "optimization"])
            for e in energy_history
        )
    
    @staticmethod
    def compute_energy_breakdown(
        energy_history: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute total energy by component."""
        components = ["training", "inference", "communication", "optimization"]
        breakdown = {c: 0.0 for c in components}
        
        for e in energy_history:
            for c in components:
                breakdown[c] += e.get(c, 0)
        
        return breakdown
    
    @staticmethod
    def compute_energy_efficiency(
        energy_history: List[Dict[str, float]],
        accuracies: List[float]
    ) -> float:
        """
        Compute energy efficiency as accuracy_gain / energy_spent.
        
        Higher is better.
        """
        total_energy = EnergyAnalyzer.compute_total_energy(energy_history)
        if total_energy == 0:
            return 0.0
        
        final_acc = accuracies[-1] if accuracies else 0
        return final_acc / total_energy
    
    @staticmethod
    def compare_with_fl(
        fd_energy: Dict[str, float],
        n_rounds: int,
        gradient_size_mb: float = 44.0,
        logit_size_kb: float = 0.4,
        avg_s: float = 1000
    ) -> Dict[str, float]:
        """
        Compare FD energy with hypothetical FL energy.
        
        Returns:
            Dictionary with comparison metrics
        """
        # Estimate FL communication energy (very rough)
        fl_comm_per_round = gradient_size_mb * 1e6 / (logit_size_kb * 1e3 * avg_s)
        fl_comm_total = fl_comm_per_round * fd_energy.get("communication", 0)
        
        fd_total = sum(fd_energy.values())
        fl_total = fd_energy.get("training", 0) + fl_comm_total
        
        return {
            "fd_total": fd_total,
            "fl_estimated_total": fl_total,
            "ratio": fd_total / fl_total if fl_total > 0 else 0,
            "communication_ratio": fd_energy.get("communication", 0) / fl_comm_total if fl_comm_total > 0 else 0
        }
