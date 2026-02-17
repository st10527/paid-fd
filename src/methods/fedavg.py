"""
FedAvg: Federated Averaging (McMahan et al., AISTATS 2017)

Parameter-averaging FL baseline (non-distillation).
Uploads full model parameters instead of logits.

Key differences from PAID-FD / FD methods:
  - Communication: full model parameters (~44 MB/round) vs logits (~400 KB)
  - No public data needed (aggregation on parameters, not knowledge)
  - No privacy mechanism
  - Simple weighted averaging of model state_dicts

Adaptations from the original paper for fair comparison:
  - Uses momentum SGD (standard practice) instead of vanilla SGD
  - Homogeneous models (all ResNet-18), same as all other methods
  - Same local_epochs=20, local_lr=0.1 across all methods

Reference:
  McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data", AISTATS 2017
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import copy

from .base import FederatedMethod, RoundResult
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class FedAvgConfig:
    """Configuration for FedAvg."""
    # Local training (per round)
    local_epochs: int = 2
    local_lr: float = 0.01
    local_momentum: float = 0.9

    # Participation
    participation_rate: float = 0.5  # Fraction of devices per round

    # Communication (for energy calculation)
    # ResNet-18: ~11.2M params × 4 bytes = ~44 MB
    model_size_bytes: int = 44_000_000


class FedAvg(FederatedMethod):
    """
    Federated Averaging.

    Protocol per round:
      1. Server broadcasts global model to all clients
      2. Each client trains locally for E epochs
      3. Clients upload updated model parameters
      4. Server averages parameters weighted by dataset size

    No public data is used. No privacy mechanism.
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: FedAvgConfig = None,
        n_classes: int = 100,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(server_model, n_classes, device)
        self.config = config or FedAvgConfig()
        self.energy_calc = EnergyCalculator()
        self.rng = np.random.RandomState(42)

    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any = None,  # Not used in FedAvg
        test_loader: Optional[Any] = None
    ) -> RoundResult:
        """Execute one round of FedAvg."""
        self.current_round = round_idx

        # Select participants
        n_devices = len(devices)
        n_participants = max(1, int(n_devices * self.config.participation_rate))
        participant_ids = sorted(
            self.rng.choice(n_devices, size=n_participants, replace=False).tolist()
        )

        # Collect local model updates
        local_state_dicts = []
        local_weights = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}

        global_state = copy.deepcopy(self.server_model.state_dict())

        for dev_id in participant_ids:
            if dev_id not in client_loaders:
                continue

            dev = devices[dev_id]
            local_loader = client_loaders[dev_id]
            n_samples = len(local_loader.dataset)

            # Create local model from global model
            local_model = copy_model(self.server_model, device=self.device)

            # Local training
            self.train_local(
                local_model,
                local_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr
            )

            # Collect local state dict
            local_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            local_weights.append(n_samples)

            # Energy: training + communication (upload full model)
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=n_samples,
                s_i=n_samples,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            total_energy["training"] += energy.get("training", 0)
            total_energy["inference"] += 0  # No inference in FedAvg
            # Communication cost: uploading full model (~44 MB)
            # Much larger than logits in FD methods
            comm_energy = energy.get("communication", 0)
            # Scale by model_size / logit_size ratio
            total_energy["communication"] += comm_energy * 100  # ~100x more than logits

            del local_model

        # Aggregate: weighted average of model parameters
        if local_state_dicts:
            self._fedavg_aggregate(local_state_dicts, local_weights)

        # Evaluate
        accuracy = 0.0
        loss = 0.0
        if test_loader is not None:
            eval_result = self.evaluate(test_loader)
            accuracy = eval_result["accuracy"]
            loss = eval_result["loss"]

        participation_rate = len(participant_ids) / n_devices if n_devices > 0 else 0

        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            n_participants=len(participant_ids),
            energy=total_energy,
            extra={
                "method": "FedAvg",
                "local_epochs": self.config.local_epochs,
                "comm_bytes": self.config.model_size_bytes * len(participant_ids),
            }
        )

        self.round_history.append(result)
        return result

    def _fedavg_aggregate(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ):
        """Weighted average of model parameters."""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Initialize with zeros
        avg_state = {}
        for key in state_dicts[0]:
            avg_state[key] = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)

        # Weighted sum
        for state_dict, weight in zip(state_dicts, normalized_weights):
            for key in avg_state:
                avg_state[key] += weight * state_dict[key].float()

        # Load aggregated state
        self.server_model.load_state_dict(avg_state)

    def aggregate(self, updates: List[Dict], weights: List[float]) -> None:
        """Interface compatibility."""
        pass
