"""
Stackelberg Game Solver for PAID-FD

Implements:
- Algorithm 1: Device Best Response (DeviceBR)
- Algorithm 2: Server Optimal Pricing (ServerPricing)

The mechanism achieves:
- Stackelberg Equilibrium
- Incentive Compatibility
- Individual Rationality
- Budget Feasibility
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .utility import QualityFunction, UtilityCalculator


@dataclass
class DeviceDecision:
    """
    Result of device's optimal decision.
    
    Attributes:
        device_id: Device identifier
        s_star: Optimal upload volume
        eps_star: Optimal privacy budget
        quality: Quality contribution q(s*, ε*)
        utility: Device utility value
        participates: Whether device participates
    """
    device_id: int
    s_star: float
    eps_star: float
    quality: float
    utility: float
    participates: bool
    
    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "s": self.s_star,
            "eps": self.eps_star,
            "quality": self.quality,
            "utility": self.utility,
            "participates": self.participates
        }


class DeviceBestResponse:
    """
    Algorithm 1: Device Best Response
    
    Given price p, compute optimal (s*, ε*) for device i.
    
    The algorithm:
    1. Solve cubic equation for ε* using bisection
    2. Compute s* from closed-form solution
    3. Check participation condition
    
    Complexity: O(log(1/δ))
    """
    
    def __init__(
        self,
        delta: float = 1e-6,
        eps_max: float = 10.0,
        s_max: float = 5000.0
    ):
        """
        Args:
            delta: Tolerance for bisection
            eps_max: Maximum privacy budget
            s_max: Maximum upload volume
        """
        self.delta = delta
        self.eps_max = eps_max
        self.s_max = s_max
        self.qf = QualityFunction()
    
    def solve(
        self,
        p: float,
        c: float,
        lambda_i: float,
        E_train: float = 0.0,
        device_id: int = -1
    ) -> DeviceDecision:
        """
        Compute optimal response for device i.
        
        Args:
            p: Price announced by server
            c: Aggregate marginal cost (c_inf + c_comm)
            lambda_i: Privacy sensitivity coefficient
            E_train: Fixed training energy cost
            device_id: Device identifier
            
        Returns:
            DeviceDecision containing (s*, ε*, utility, participates)
        """
        # Handle edge case: price too low
        if p <= 1.0:
            return self._non_participation(device_id)
        
        # Step 1: Solve cubic equation for ε*
        eps_star = self._solve_cubic_bisection(p, lambda_i)
        
        if eps_star is None or eps_star <= 0:
            return self._non_participation(device_id)
        
        # Step 2: Compute s* from closed-form
        # s* = p/c - (1 + ε*)/ε*
        s_star = p / c - (1 + eps_star) / eps_star
        
        # Apply bounds
        if s_star <= 0:
            return self._non_participation(device_id)
        s_star = min(s_star, self.s_max)
        
        # Step 3: Compute utility and check participation
        quality = self.qf.q(s_star, eps_star)
        utility = p * quality - c * s_star - lambda_i * eps_star
        
        # Participation condition: U_i ≥ E_train
        participates = utility >= E_train
        
        if not participates:
            return self._non_participation(device_id)
        
        return DeviceDecision(
            device_id=device_id,
            s_star=s_star,
            eps_star=eps_star,
            quality=quality,
            utility=utility,
            participates=True
        )
    
    def _solve_cubic_bisection(
        self,
        p: float,
        lambda_i: float
    ) -> Optional[float]:
        """
        Solve cubic equation using bisection:
        
        f(ε) = λε³ + λε² + (1-p)ε + 1 = 0
        
        For p > 1, there exists exactly one positive root where f changes
        from positive to negative (at small ε) to positive again (at large ε).
        We want the first positive root.
        
        Returns:
            Positive root ε*, or None if not found
        """
        def f(eps):
            return lambda_i * eps**3 + lambda_i * eps**2 + (1 - p) * eps + 1
        
        # For p > 1, f(0+) ≈ 1 > 0
        # f has a local minimum, then increases to infinity
        # We need to find where f first becomes negative
        
        eps_lo = self.delta
        f_lo = f(eps_lo)
        
        if f_lo < 0:
            # Already negative at lower bound - shouldn't happen for typical params
            return None
        
        # Search for where f becomes negative
        eps_hi = self.delta
        f_hi = f_lo
        
        # Exponential search to find upper bound where f < 0
        while eps_hi < 1000:
            eps_hi = min(eps_hi * 2, 1000)
            f_hi = f(eps_hi)
            if f_hi < 0:
                break
        
        if f_hi >= 0:
            # f never becomes negative - no positive root in range
            # This can happen when lambda is too large relative to p
            return None
        
        # Now we have f(eps_lo) > 0 and f(eps_hi) < 0
        # Bisection to find the root
        while eps_hi - eps_lo > self.delta:
            eps_mid = (eps_lo + eps_hi) / 2
            f_mid = f(eps_mid)
            
            if f_mid > 0:
                eps_lo = eps_mid
            else:
                eps_hi = eps_mid
        
        return (eps_lo + eps_hi) / 2
    
    def _non_participation(self, device_id: int) -> DeviceDecision:
        """Return a non-participating decision."""
        return DeviceDecision(
            device_id=device_id,
            s_star=0.0,
            eps_star=0.0,
            quality=0.0,
            utility=0.0,
            participates=False
        )
    
    def verify_optimality(
        self,
        decision: DeviceDecision,
        p: float,
        c: float,
        lambda_i: float
    ) -> dict:
        """
        Verify that the solution satisfies FOC.
        
        Useful for debugging.
        """
        if not decision.participates:
            return {"verified": True, "reason": "non-participating"}
        
        from .utility import verify_foc_conditions
        return verify_foc_conditions(
            p, decision.s_star, decision.eps_star, c, lambda_i
        )


class ServerPricing:
    """
    Algorithm 2: Server Optimal Pricing
    
    Find optimal price p* using ternary search.
    
    Server utility: U_ES(p) = (γ - p) × Q(p)
    where Q(p) = Σ q_i(s_i*(p), ε_i*(p))
    
    Complexity: O(N × log(γ/δ) × log(1/δ))
    """
    
    def __init__(
        self,
        gamma: float = 10.0,
        delta: float = 0.01,
        budget: float = float('inf')
    ):
        """
        Args:
            gamma: Server's valuation coefficient
            delta: Search tolerance
            budget: Total budget constraint
        """
        self.gamma = gamma
        self.delta = delta
        self.budget = budget
        self.device_br = DeviceBestResponse(delta=delta/10)
        self.qf = QualityFunction()
    
    def solve(
        self,
        devices: list
    ) -> Tuple[float, List[DeviceDecision]]:
        """
        Find optimal price p* and all device decisions.
        
        Args:
            devices: List of device profiles (need c_total and lambda_i)
            
        Returns:
            (p_star, decisions): Optimal price and list of DeviceDecision
        """
        # Ternary search for optimal price
        p_lo, p_hi = self.delta, self.gamma - self.delta
        
        while p_hi - p_lo > self.delta:
            p1 = p_lo + (p_hi - p_lo) / 3
            p2 = p_hi - (p_hi - p_lo) / 3
            
            U1 = self._evaluate_server_utility(p1, devices)
            U2 = self._evaluate_server_utility(p2, devices)
            
            if U1 < U2:
                p_lo = p1
            else:
                p_hi = p2
        
        p_star = (p_lo + p_hi) / 2
        
        # Get final decisions
        decisions = self._get_all_decisions(p_star, devices)
        
        # Check budget constraint
        total_payment = sum(p_star * d.quality for d in decisions if d.participates)
        
        if total_payment > self.budget:
            p_star = self._adjust_for_budget(p_star, devices)
            decisions = self._get_all_decisions(p_star, devices)
        
        return p_star, decisions
    
    def _evaluate_server_utility(
        self,
        p: float,
        devices: list
    ) -> float:
        """
        Evaluate U_ES(p) = (γ - p) × Q(p)
        """
        total_quality = 0.0
        
        for dev in devices:
            # Get device's cost parameters
            c = getattr(dev, 'c_total', getattr(dev, 'c_comm', 1.0) + getattr(dev, 'c_inf', 0.0))
            lambda_i = getattr(dev, 'lambda_i', 0.5)
            
            decision = self.device_br.solve(p, c, lambda_i)
            
            if decision.participates:
                total_quality += decision.quality
        
        return (self.gamma - p) * total_quality
    
    def _get_all_decisions(
        self,
        p: float,
        devices: list
    ) -> List[DeviceDecision]:
        """Get decisions for all devices at price p."""
        decisions = []
        
        for dev in devices:
            dev_id = getattr(dev, 'device_id', 0)
            c = getattr(dev, 'c_total', getattr(dev, 'c_comm', 1.0) + getattr(dev, 'c_inf', 0.0))
            lambda_i = getattr(dev, 'lambda_i', 0.5)
            E_train = getattr(dev, 'E_train', 0.0)
            
            decision = self.device_br.solve(p, c, lambda_i, E_train, dev_id)
            decisions.append(decision)
        
        return decisions
    
    def _adjust_for_budget(
        self,
        p_init: float,
        devices: list
    ) -> float:
        """
        Reduce price via bisection to satisfy budget constraint.
        """
        p_lo, p_hi = self.delta, p_init
        
        while p_hi - p_lo > self.delta:
            p_mid = (p_lo + p_hi) / 2
            decisions = self._get_all_decisions(p_mid, devices)
            
            total_payment = sum(p_mid * d.quality for d in decisions if d.participates)
            
            if total_payment > self.budget:
                p_hi = p_mid
            else:
                p_lo = p_mid
        
        return p_lo
    
    def analyze_price_curve(
        self,
        devices: list,
        n_points: int = 50
    ) -> dict:
        """
        Analyze server utility as a function of price.
        
        Returns data for plotting the "inverted-U" curve.
        """
        prices = np.linspace(self.delta, self.gamma - self.delta, n_points)
        utilities = []
        qualities = []
        participation_rates = []
        
        for p in prices:
            decisions = self._get_all_decisions(p, devices)
            
            Q = sum(d.quality for d in decisions if d.participates)
            U = (self.gamma - p) * Q
            rate = sum(1 for d in decisions if d.participates) / len(devices)
            
            utilities.append(U)
            qualities.append(Q)
            participation_rates.append(rate)
        
        return {
            "prices": prices.tolist(),
            "utilities": utilities,
            "qualities": qualities,
            "participation_rates": participation_rates,
            "optimal_idx": int(np.argmax(utilities)),
            "optimal_price": prices[np.argmax(utilities)]
        }


class StackelbergSolver:
    """
    Complete Stackelberg game solver.
    
    Combines device best response and server pricing into a unified interface.
    
    Usage:
        solver = StackelbergSolver(gamma=10.0)
        result = solver.solve(devices)
        
        print(f"Optimal price: {result['price']}")
        print(f"Participation rate: {result['participation_rate']}")
    """
    
    def __init__(
        self,
        gamma: float = 10.0,
        delta: float = 0.01,
        budget: float = float('inf')
    ):
        self.gamma = gamma
        self.delta = delta
        self.budget = budget
        self.server_pricing = ServerPricing(gamma, delta, budget)
        self.qf = QualityFunction()
    
    def solve(self, devices: list) -> dict:
        """
        Solve the complete Stackelberg game.
        
        Args:
            devices: List of DeviceProfile instances
            
        Returns:
            Dictionary with:
            - price: Optimal price p*
            - decisions: List of DeviceDecision
            - server_utility: Server's utility
            - total_quality: Aggregate quality
            - total_payment: Total payment to devices
            - participation_rate: Fraction of participating devices
            - avg_s: Average upload volume
            - avg_eps: Average privacy budget
        """
        price, decisions = self.server_pricing.solve(devices)
        
        # Compute statistics
        participants = [d for d in decisions if d.participates]
        n_participants = len(participants)
        n_total = len(decisions)
        
        total_quality = sum(d.quality for d in participants)
        total_payment = price * total_quality
        server_utility = (self.gamma - price) * total_quality
        
        avg_s = np.mean([d.s_star for d in participants]) if participants else 0
        avg_eps = np.mean([d.eps_star for d in participants]) if participants else 0
        
        return {
            "price": price,
            "decisions": decisions,
            "server_utility": server_utility,
            "total_quality": total_quality,
            "total_payment": total_payment,
            "participation_rate": n_participants / n_total if n_total > 0 else 0,
            "n_participants": n_participants,
            "avg_s": avg_s,
            "avg_eps": avg_eps
        }
    
    def solve_for_fixed_price(
        self,
        devices: list,
        price: float
    ) -> dict:
        """
        Get device decisions for a fixed price (useful for analysis).
        """
        br = DeviceBestResponse(delta=self.delta/10)
        decisions = []
        
        for dev in devices:
            dev_id = getattr(dev, 'device_id', 0)
            c = getattr(dev, 'c_total', 1.0)
            lambda_i = getattr(dev, 'lambda_i', 0.5)
            
            decision = br.solve(price, c, lambda_i, device_id=dev_id)
            decisions.append(decision)
        
        participants = [d for d in decisions if d.participates]
        total_quality = sum(d.quality for d in participants)
        
        return {
            "price": price,
            "decisions": decisions,
            "total_quality": total_quality,
            "participation_rate": len(participants) / len(decisions) if decisions else 0
        }


def test_stackelberg_solver():
    """Quick test of the Stackelberg solver."""
    from ..devices.heterogeneity import HeterogeneityGenerator
    
    # Generate test devices
    gen = HeterogeneityGenerator(n_devices=20, seed=42)
    devices = gen.generate()
    
    # Solve
    solver = StackelbergSolver(gamma=10.0)
    result = solver.solve(devices)
    
    print("\n" + "="*50)
    print("Stackelberg Solver Test")
    print("="*50)
    print(f"Number of devices: {len(devices)}")
    print(f"Optimal price: {result['price']:.4f}")
    print(f"Server utility: {result['server_utility']:.4f}")
    print(f"Total quality: {result['total_quality']:.4f}")
    print(f"Participation rate: {result['participation_rate']:.2%}")
    print(f"Average s*: {result['avg_s']:.2f}")
    print(f"Average ε*: {result['avg_eps']:.4f}")
    print("="*50 + "\n")
    
    return result


if __name__ == "__main__":
    test_stackelberg_solver()
