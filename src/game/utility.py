"""
Utility and Quality Functions for PAID-FD Stackelberg Game

Quality Function: q_i(s_i, ε_i) = log(1 + s_i × g(ε_i))
where g(ε) = ε / (1 + ε) is the effective information ratio

Device Utility: U_i = p × q_i - c_i × s_i - λ_i × ε_i
Server Utility: U_ES = (γ - p) × Σ q_i
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QualityFunction:
    """
    Quality function for knowledge contribution.
    
    q(s, ε) = log(1 + s × g(ε))
    
    where g(ε) = ε / (1 + ε) represents the effective information ratio
    after differential privacy noise.
    """
    
    @staticmethod
    def g(epsilon: float) -> float:
        """
        Effective information ratio.
        
        g(ε) = ε / (1 + ε)
        
        Properties:
        - g(0) = 0 (infinite noise, no information)
        - g(∞) = 1 (no noise, full information)
        - g is monotonically increasing
        """
        if epsilon <= 0:
            return 0.0
        return epsilon / (1 + epsilon)
    
    @staticmethod
    def q(s: float, epsilon: float) -> float:
        """
        Quality function.
        
        q(s, ε) = log(1 + s × g(ε))
        
        Args:
            s: Upload volume (number of logits)
            epsilon: Privacy budget
            
        Returns:
            Quality value
        """
        if s <= 0 or epsilon <= 0:
            return 0.0
        g_eps = QualityFunction.g(epsilon)
        return np.log(1 + s * g_eps)
    
    @staticmethod
    def dq_ds(s: float, epsilon: float) -> float:
        """
        Partial derivative ∂q/∂s.
        
        ∂q/∂s = g(ε) / (1 + s × g(ε))
        """
        if s < 0 or epsilon <= 0:
            return 0.0
        g_eps = QualityFunction.g(epsilon)
        return g_eps / (1 + s * g_eps)
    
    @staticmethod
    def dq_deps(s: float, epsilon: float) -> float:
        """
        Partial derivative ∂q/∂ε.
        
        ∂q/∂ε = s × g'(ε) / (1 + s × g(ε))
        
        where g'(ε) = 1 / (1 + ε)²
        """
        if s <= 0 or epsilon <= 0:
            return 0.0
        g_eps = QualityFunction.g(epsilon)
        g_prime = 1 / ((1 + epsilon) ** 2)
        return s * g_prime / (1 + s * g_eps)


class UtilityCalculator:
    """
    Calculate utilities for devices and server.
    """
    
    def __init__(self, gamma: float = 10.0):
        """
        Args:
            gamma: Server's valuation coefficient
        """
        self.gamma = gamma
        self.qf = QualityFunction()
    
    def device_utility(
        self,
        p: float,
        s: float,
        epsilon: float,
        c: float,
        lambda_i: float,
        E_train: float = 0.0
    ) -> float:
        """
        Compute device utility.
        
        U_i = p × q(s, ε) - c × s - λ × ε - E_train (if participating)
        
        Args:
            p: Price announced by server
            s: Upload volume
            epsilon: Privacy budget
            c: Aggregate marginal cost (c_inf + c_comm)
            lambda_i: Privacy sensitivity
            E_train: Fixed training cost
            
        Returns:
            Device utility value
        """
        if s <= 0:
            return 0.0
        
        quality = self.qf.q(s, epsilon)
        revenue = p * quality
        cost = c * s + lambda_i * epsilon + E_train
        
        return revenue - cost
    
    def device_utility_gradient(
        self,
        p: float,
        s: float,
        epsilon: float,
        c: float,
        lambda_i: float
    ) -> Tuple[float, float]:
        """
        Compute gradient of device utility.
        
        ∂U/∂s = p × ∂q/∂s - c
        ∂U/∂ε = p × ∂q/∂ε - λ
        
        Returns:
            (∂U/∂s, ∂U/∂ε)
        """
        dU_ds = p * self.qf.dq_ds(s, epsilon) - c
        dU_deps = p * self.qf.dq_deps(s, epsilon) - lambda_i
        
        return dU_ds, dU_deps
    
    def server_utility(
        self,
        p: float,
        qualities: list
    ) -> float:
        """
        Compute server utility.
        
        U_ES = (γ - p) × Σ q_i
        
        Args:
            p: Price
            qualities: List of quality values from participating devices
            
        Returns:
            Server utility value
        """
        total_quality = sum(qualities)
        return (self.gamma - p) * total_quality
    
    def total_payment(
        self,
        p: float,
        qualities: list
    ) -> float:
        """
        Compute total payment to devices.
        
        Σ π_i = p × Σ q_i
        """
        return p * sum(qualities)
    
    def check_participation_condition(
        self,
        utility: float,
        E_train: float = 0.0
    ) -> bool:
        """
        Check if device should participate.
        
        Device participates iff U_i ≥ E_train (individual rationality)
        """
        return utility >= E_train
    
    def find_participation_threshold(
        self,
        c: float,
        lambda_i: float,
        E_train: float,
        p_max: float = 100.0,
        tol: float = 0.01
    ) -> float:
        """
        Find minimum price for device to participate.
        
        Uses bisection to find p such that U_i(s*(p), ε*(p)) = E_train
        
        Args:
            c: Marginal cost
            lambda_i: Privacy sensitivity
            E_train: Training cost
            p_max: Maximum price to consider
            tol: Tolerance
            
        Returns:
            Threshold price
        """
        from .stackelberg import DeviceBestResponse
        
        br = DeviceBestResponse()
        
        p_lo, p_hi = tol, p_max
        
        while p_hi - p_lo > tol:
            p_mid = (p_lo + p_hi) / 2
            decision = br.solve(p_mid, c, lambda_i, E_train)
            
            if decision.participates:
                p_hi = p_mid
            else:
                p_lo = p_mid
        
        return p_hi


def verify_foc_conditions(
    p: float,
    s_star: float,
    eps_star: float,
    c: float,
    lambda_i: float,
    tol: float = 1e-4
) -> dict:
    """
    Verify that FOC (first-order conditions) are satisfied.
    
    At optimum:
    - ∂U/∂s = 0  =>  p × g(ε) / (1 + s × g(ε)) = c
    - ∂U/∂ε = 0  =>  p × s × g'(ε) / (1 + s × g(ε)) = λ
    
    Returns:
        Dictionary with FOC values and whether they're satisfied
    """
    qf = QualityFunction()
    calc = UtilityCalculator()
    
    dU_ds, dU_deps = calc.device_utility_gradient(p, s_star, eps_star, c, lambda_i)
    
    return {
        "dU_ds": dU_ds,
        "dU_deps": dU_deps,
        "foc_s_satisfied": abs(dU_ds) < tol,
        "foc_eps_satisfied": abs(dU_deps) < tol,
        "both_satisfied": abs(dU_ds) < tol and abs(dU_deps) < tol
    }
