"""
Federated learning methods for PAID-FD.

Available methods:
  - PAIDFD:       Our method (Stackelberg game + adaptive ε + LDP)
  - FixedEpsilon: Ablation (fixed ε, no game)
  - FedMD:        Classic FD baseline (no privacy, no game)
  - FedAvg:       Parameter averaging baseline (no distillation)
  - CSRA:         Reverse auction DPFL baseline (Yang et al., TIFS 2024)
  - FedGMKD:      Prototype-based FL with GMM + DAT (Zhang et al., 2024)
"""

from .base import FederatedMethod, RoundResult
from .paid_fd import PAIDFD, PAIDFDConfig
from .fixed_eps import FixedEpsilon, FixedEpsilonConfig
from .fedmd import FedMD, FedMDConfig
from .fedavg import FedAvg, FedAvgConfig
from .csra import CSRA, CSRAConfig
from .fedgmkd import FedGMKD, FedGMKDConfig

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
    "CSRA",
    "CSRAConfig",
    "FedGMKD",
    "FedGMKDConfig",
]
