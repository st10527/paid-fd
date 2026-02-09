"""
Federated learning methods for PAID-FD.

Available methods:
  - PAIDFD:       Our method (Stackelberg game + adaptive ε + LDP)
  - FixedEpsilon: Ablation (fixed ε, no game)
  - FedMD:        Classic FD baseline (no privacy, no game)
  - FedAvg:       Parameter averaging baseline (no distillation)
"""

from .base import FederatedMethod, RoundResult
from .paid_fd import PAIDFD, PAIDFDConfig
from .fixed_eps import FixedEpsilon, FixedEpsilonConfig
from .fedmd import FedMD, FedMDConfig
from .fedavg import FedAvg, FedAvgConfig

__all__ = [
    "FederatedMethod",
    "RoundResult",
    "PAIDFD",
    "PAIDFDConfig",
    "FixedEpsilon",
    "FixedEpsilonConfig",
    "FedMD",
    "FedMDConfig",
    "FedAvg",
    "FedAvgConfig",
]
