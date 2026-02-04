"""
Local Differential Privacy (LDP) Mechanisms for PAID-FD

Implements:
- Laplace mechanism for ε-LDP
- Gaussian mechanism for (ε, δ)-DP
- Sensitivity computation for logits

Definition (ε-LDP):
A randomized mechanism M satisfies ε-LDP if for all pairs of inputs x, x'
and all measurable subsets S:
    Pr[M(x) ∈ S] ≤ e^ε × Pr[M(x') ∈ S]
"""

import numpy as np
from typing import Union, Optional, List, Dict
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DPParams:
    """Differential privacy parameters."""
    epsilon: float          # Privacy budget
    delta: float = 0.0      # For (ε,δ)-DP (0 for pure ε-DP)
    sensitivity: float = 1.0  # L1 or L2 sensitivity
    mechanism: str = "laplace"  # 'laplace' or 'gaussian'


class LaplaceDP:
    """
    Laplace mechanism for ε-differential privacy.
    
    For a function f with sensitivity Δ, adding noise from Lap(0, Δ/ε)
    provides ε-differential privacy.
    
    The Laplace distribution: p(x) = (1/2b) × exp(-|x|/b)
    where b = Δ/ε is the scale parameter.
    """
    
    def __init__(self, sensitivity: float = 1.0):
        """
        Args:
            sensitivity: L1 sensitivity of the function being protected
        """
        self.sensitivity = sensitivity
    
    def add_noise(
        self,
        data: Union[np.ndarray, 'torch.Tensor'],
        epsilon: float,
        sensitivity: Optional[float] = None
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Add Laplace noise to data.
        
        Args:
            data: Input data (numpy array or torch tensor)
            epsilon: Privacy budget (larger = less noise = less privacy)
            sensitivity: Override default sensitivity
            
        Returns:
            Noisy data with same type as input
        """
        sens = sensitivity if sensitivity is not None else self.sensitivity
        scale = sens / epsilon
        
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            # PyTorch version
            noise = torch.empty_like(data).exponential_(1/scale)
            sign = torch.empty_like(data).uniform_(-1, 1).sign()
            return data + noise * sign
        else:
            # NumPy version
            noise = np.random.laplace(0, scale, data.shape)
            return data + noise.astype(data.dtype)
    
    def get_noise_scale(self, epsilon: float) -> float:
        """Get the noise scale parameter b = Δ/ε."""
        return self.sensitivity / epsilon
    
    def compute_variance(self, epsilon: float) -> float:
        """
        Compute variance of the Laplace noise.
        
        Var[Lap(0, b)] = 2b²
        """
        scale = self.get_noise_scale(epsilon)
        return 2 * scale ** 2
    
    def estimate_snr(
        self,
        signal_magnitude: float,
        epsilon: float
    ) -> float:
        """
        Estimate signal-to-noise ratio.
        
        SNR = signal² / noise_variance
        """
        variance = self.compute_variance(epsilon)
        return signal_magnitude ** 2 / variance if variance > 0 else float('inf')


class GaussianDP:
    """
    Gaussian mechanism for (ε, δ)-differential privacy.
    
    For a function f with L2 sensitivity Δ₂, adding noise from N(0, σ²)
    where σ = Δ₂ × √(2 ln(1.25/δ)) / ε
    provides (ε, δ)-differential privacy.
    """
    
    def __init__(self, sensitivity: float = 1.0, delta: float = 1e-5):
        """
        Args:
            sensitivity: L2 sensitivity of the function
            delta: Privacy parameter δ (probability of failure)
        """
        self.sensitivity = sensitivity
        self.delta = delta
    
    def add_noise(
        self,
        data: Union[np.ndarray, 'torch.Tensor'],
        epsilon: float,
        sensitivity: Optional[float] = None,
        delta: Optional[float] = None
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Add Gaussian noise to data.
        
        Args:
            data: Input data
            epsilon: Privacy budget
            sensitivity: Override default sensitivity
            delta: Override default delta
            
        Returns:
            Noisy data
        """
        sens = sensitivity if sensitivity is not None else self.sensitivity
        d = delta if delta is not None else self.delta
        
        sigma = self.compute_sigma(epsilon, sens, d)
        
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            noise = torch.randn_like(data) * sigma
            return data + noise
        else:
            noise = np.random.normal(0, sigma, data.shape)
            return data + noise.astype(data.dtype)
    
    def compute_sigma(
        self,
        epsilon: float,
        sensitivity: Optional[float] = None,
        delta: Optional[float] = None
    ) -> float:
        """
        Compute the noise standard deviation.
        
        σ = Δ₂ × √(2 ln(1.25/δ)) / ε
        """
        sens = sensitivity if sensitivity is not None else self.sensitivity
        d = delta if delta is not None else self.delta
        
        return sens * np.sqrt(2 * np.log(1.25 / d)) / epsilon


def compute_sensitivity(
    n_classes: int = 100,
    logit_bound: float = 10.0,
    method: str = "bounded"
) -> float:
    """
    Compute sensitivity for logit vectors.
    
    Args:
        n_classes: Number of output classes
        logit_bound: Maximum absolute value of logits
        method: 'bounded' or 'softmax'
        
    Returns:
        L1 sensitivity for Laplace mechanism
    """
    if method == "bounded":
        # If logits are bounded in [-B, B], changing one input
        # can change each logit by at most 2B
        # L1 sensitivity = 2B (assuming only one logit changes significantly)
        return 2 * logit_bound
    
    elif method == "softmax":
        # After softmax, outputs are probabilities in [0, 1]
        # Changing one input can flip from 0 to 1 or vice versa
        # L1 sensitivity ≤ 2 (sum of absolute changes bounded by 2)
        return 2.0
    
    else:
        raise ValueError(f"Unknown method: {method}")


def add_noise_to_logits(
    logits: Union[np.ndarray, 'torch.Tensor'],
    epsilon: float,
    sensitivity: Optional[float] = None,
    mechanism: str = "laplace",
    clip_bound: Optional[float] = None
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Add differential privacy noise to logits.
    
    Convenience function that:
    1. Optionally clips logits to bound sensitivity
    2. Computes sensitivity if not provided
    3. Adds appropriate noise
    
    Args:
        logits: Logit vectors (N, K) where K is number of classes
        epsilon: Privacy budget
        sensitivity: Pre-computed sensitivity (auto-computed if None)
        mechanism: 'laplace' or 'gaussian'
        clip_bound: Clip logits to [-bound, bound] before adding noise
        
    Returns:
        Noisy logits
    """
    # Determine if torch or numpy
    is_torch = TORCH_AVAILABLE and isinstance(logits, torch.Tensor)
    
    # Clip if requested
    if clip_bound is not None:
        if is_torch:
            logits = torch.clamp(logits, -clip_bound, clip_bound)
        else:
            logits = np.clip(logits, -clip_bound, clip_bound)
    
    # Compute sensitivity if not provided
    if sensitivity is None:
        if clip_bound is not None:
            sensitivity = 2 * clip_bound
        else:
            # Estimate from data
            if is_torch:
                max_val = logits.abs().max().item()
            else:
                max_val = np.abs(logits).max()
            sensitivity = 2 * max_val
    
    # Add noise
    if mechanism == "laplace":
        dp = LaplaceDP(sensitivity=sensitivity)
        return dp.add_noise(logits, epsilon)
    elif mechanism == "gaussian":
        dp = GaussianDP(sensitivity=sensitivity)
        return dp.add_noise(logits, epsilon)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


class PrivacyAccountant:
    """
    Track privacy budget across multiple rounds of training.
    """
    
    def __init__(self, total_budget: float = 10.0):
        """
        Args:
            total_budget: Maximum allowed total privacy budget
        """
        self.total_budget = total_budget
        self.spent = 0.0
        self.history: List[Dict] = []
    
    def spend(self, epsilon: float, description: str = "") -> bool:
        """
        Spend privacy budget.
        
        Args:
            epsilon: Amount to spend
            description: What the budget was used for
            
        Returns:
            True if budget was available, False if would exceed limit
        """
        if self.spent + epsilon > self.total_budget:
            return False
        
        self.spent += epsilon
        self.history.append({
            "epsilon": epsilon,
            "cumulative": self.spent,
            "description": description
        })
        return True
    
    @property
    def remaining(self) -> float:
        """Remaining privacy budget."""
        return self.total_budget - self.spent
    
    def get_summary(self) -> Dict:
        """Get summary of privacy spending."""
        return {
            "total_budget": self.total_budget,
            "spent": self.spent,
            "remaining": self.remaining,
            "n_operations": len(self.history),
            "history": self.history
        }
