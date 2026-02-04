"""
Fixed-Epsilon Baseline for PAID-FD Ablation Study

This baseline uses a fixed privacy budget ε for all devices,
without the adaptive Stackelberg game mechanism.

Purpose: Demonstrate the value of adaptive privacy allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base import FederatedMethod, RoundResult
from ..privacy.ldp import add_noise_to_logits
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class FixedEpsilonConfig:
    """Configuration for Fixed-Epsilon baseline."""
    # Fixed privacy budget (no adaptation)
    epsilon: float = 1.0         # Fixed ε for all devices
    
    # Training parameters
    local_epochs: int = 1
    local_lr: float = 0.01
    local_momentum: float = 0.9
    
    # Distillation parameters
    distill_epochs: int = 5
    distill_lr: float = 0.001
    temperature: float = 3.0
    
    # Privacy parameters
    clip_bound: float = 5.0
    
    # Participation
    participation_rate: float = 1.0  # Fraction of devices to participate
    samples_per_device: int = 100    # Fixed samples from public data


class FixedEpsilon(FederatedMethod):
    """
    Fixed-Epsilon Federated Distillation Baseline.
    
    Key differences from PAID-FD:
    - No Stackelberg game / no pricing
    - Fixed ε for all devices (no adaptation)
    - Random or full participation
    
    This serves as an ablation to show the benefit of:
    1. Adaptive privacy allocation
    2. Incentive mechanism
    """
    
    def __init__(
        self,
        server_model: nn.Module,
        config: FixedEpsilonConfig = None,
        n_classes: int = 100,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        super().__init__(server_model, n_classes, device)
        
        self.config = config or FixedEpsilonConfig()
        self.energy_calc = EnergyCalculator()
        self.rng = np.random.RandomState(42)
    
    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any,
        test_loader: Optional[Any] = None
    ) -> RoundResult:
        """
        Execute one round of Fixed-Epsilon FD.
        """
        self.current_round = round_idx
        
        # Select participating devices (random subset or all)
        n_devices = len(devices)
        n_participants = int(n_devices * self.config.participation_rate)
        participant_ids = self.rng.choice(
            n_devices, size=n_participants, replace=False
        ).tolist()
        
        # Collect logits
        all_logits = []
        all_weights = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        
        for dev_id in participant_ids:
            if dev_id not in client_loaders:
                continue
            
            dev = devices[dev_id]
            local_loader = client_loaders[dev_id]
            
            # Create and train local model (on same device as server model)
            local_model = copy_model(self.server_model, device=self.device)
            self.train_local(
                local_model,
                local_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr
            )
            
            # Compute logits with fixed sample count
            n_samples = self.config.samples_per_device
            logits = self.compute_logits(local_model, public_loader, n_samples)
            
            # Add noise with FIXED epsilon
            noisy_logits = add_noise_to_logits(
                logits.numpy(),
                epsilon=self.config.epsilon,  # Fixed, not adaptive
                clip_bound=self.config.clip_bound
            )
            noisy_logits = torch.from_numpy(noisy_logits).float()
            
            all_logits.append(noisy_logits)
            all_weights.append(1.0)  # Equal weights
            
            # Compute energy
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=getattr(dev, 'data_size', len(local_loader.dataset)),
                s_i=n_samples,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            for k in ["training", "inference", "communication"]:
                total_energy[k] += energy.get(k, 0)
            
            del local_model
        
        # Aggregate and distill
        if all_logits:
            aggregated_logits = self._aggregate_logits(all_logits, all_weights)
            self._distill_to_server(aggregated_logits, public_loader)
        
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
                "fixed_epsilon": self.config.epsilon,
                "samples_per_device": self.config.samples_per_device
            }
        )
        
        self.round_history.append(result)
        return result
    
    def _aggregate_logits(
        self,
        logits_list: List[torch.Tensor],
        weights: List[float]
    ) -> torch.Tensor:
        """Simple average aggregation."""
        min_len = min(len(l) for l in logits_list)
        truncated = [l[:min_len] for l in logits_list]
        return sum(truncated) / len(truncated)
    
    def _distill_to_server(
        self,
        target_logits: torch.Tensor,
        public_loader: Any
    ):
        """Distill to server model."""
        self.server_model.train()
        optimizer = torch.optim.Adam(
            self.server_model.parameters(),
            lr=self.config.distill_lr
        )
        
        T = self.config.temperature
        target_logits = target_logits.to(self.device)
        n_target = len(target_logits)
        
        for epoch in range(self.config.distill_epochs):
            idx = 0
            for data, _ in public_loader:
                if idx >= n_target:
                    break
                
                batch_size = min(len(data), n_target - idx)
                data = data[:batch_size].to(self.device)
                teacher_logits = target_logits[idx:idx + batch_size]
                
                student_logits = self.server_model(data)
                
                loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                idx += batch_size
    
    def aggregate(self, updates: List[Dict], weights: List[float]) -> None:
        """Interface compatibility."""
        pass
