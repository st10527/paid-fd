"""
PAID-FD: Privacy-Aware Incentive-Driven Federated Distillation

Complete implementation of the PAID-FD method including:
- Stackelberg game-based pricing
- Device best response optimization
- Local differential privacy on logits
- Knowledge distillation aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import copy

from .base import FederatedMethod, RoundResult
from ..game.stackelberg import StackelbergSolver, DeviceDecision
from ..privacy.ldp import add_noise_to_logits
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class PAIDFDConfig:
    """Configuration for PAID-FD."""
    # Game parameters
    gamma: float = 10.0          # Server valuation coefficient
    delta: float = 0.01          # Search tolerance
    budget: float = float('inf') # Budget constraint
    
    # Training parameters
    local_epochs: int = 1        # Local training epochs
    local_lr: float = 0.01       # Local learning rate
    local_momentum: float = 0.9  # SGD momentum
    
    # Distillation parameters
    distill_epochs: int = 5      # Epochs for server distillation
    distill_lr: float = 0.001    # Distillation learning rate
    temperature: float = 3.0     # Distillation temperature
    
    # Privacy parameters
    clip_bound: float = 5.0      # Logit clipping bound
    
    # Public data
    public_samples: int = 1000   # Samples from public data per round


class PAIDFD(FederatedMethod):
    """
    PAID-FD: Privacy-Aware Incentive-Driven Federated Distillation
    
    Protocol per round:
    1. Server computes optimal price p* via ServerPricing
    2. Server broadcasts p* to all devices
    3. Each device locally computes (s_i*, ε_i*) via DeviceBR
    4. Participating devices:
       a. Train local model on private data
       b. Compute logits on public data
       c. Add LDP noise
       d. Upload noisy logits
    5. Server aggregates logits and distills to global model
    
    Usage:
        config = PAIDFDConfig(gamma=10.0)
        method = PAIDFD(server_model, config)
        
        for round_idx in range(n_rounds):
            result = method.run_round(round_idx, devices, client_loaders, public_loader)
            print(f"Round {round_idx}: Acc={result.accuracy:.4f}")
    """
    
    def __init__(
        self,
        server_model: nn.Module,
        config: PAIDFDConfig = None,
        n_classes: int = 100,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        super().__init__(server_model, n_classes, device)
        
        self.config = config or PAIDFDConfig()
        
        # Initialize game solver
        self.solver = StackelbergSolver(
            gamma=self.config.gamma,
            delta=self.config.delta,
            budget=self.config.budget
        )
        
        # Energy calculator
        self.energy_calc = EnergyCalculator()
        
        # Track statistics
        self.price_history = []
        self.participation_history = []
    
    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any,
        test_loader: Optional[Any] = None
    ) -> RoundResult:
        """
        Execute one round of PAID-FD.
        
        Args:
            round_idx: Current round index
            devices: List of DeviceProfile
            client_loaders: Dict mapping device_id -> DataLoader
            public_loader: DataLoader for public dataset
            test_loader: Optional test DataLoader for evaluation
            
        Returns:
            RoundResult with metrics
        """
        self.current_round = round_idx
        
        # Stage 1: Server computes optimal price
        game_result = self.solver.solve(devices)
        price = game_result["price"]
        decisions = game_result["decisions"]
        
        self.price_history.append(price)
        
        # Stage 2: Collect logits from participating devices
        all_logits = []
        all_weights = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        participants = []
        
        for decision in decisions:
            if not decision.participates:
                continue
            
            dev_id = decision.device_id
            dev = devices[dev_id]
            participants.append(dev_id)
            
            # Skip if no data loader for this device
            if dev_id not in client_loaders:
                continue
            
            # Get local data loader
            local_loader = client_loaders[dev_id]
            
            # Create local model copy (on same device as server model)
            local_model = copy_model(self.server_model, device=self.device)
            
            # Local training
            self.train_local(
                local_model,
                local_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr
            )
            
            # Compute logits on public data
            n_samples = min(int(decision.s_star), self.config.public_samples)
            logits = self.compute_logits(local_model, public_loader, n_samples)
            
            # Add LDP noise
            noisy_logits = add_noise_to_logits(
                logits.numpy(),
                epsilon=decision.eps_star,
                clip_bound=self.config.clip_bound
            )
            noisy_logits = torch.from_numpy(noisy_logits).float()
            
            all_logits.append(noisy_logits)
            all_weights.append(decision.s_star)
            
            # Compute energy
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=getattr(dev, 'data_size', len(local_loader.dataset)),
                s_i=n_samples,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            for k in ["training", "inference", "communication"]:
                total_energy[k] += energy.get(k, 0)
            
            # Clean up
            del local_model
        
        # Stage 3: Aggregate and distill
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
        
        # Compute participation rate
        participation_rate = len(participants) / len(devices) if devices else 0
        self.participation_history.append(participation_rate)
        
        # Build result
        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            n_participants=len(participants),
            energy=total_energy,
            extra={
                "price": price,
                "avg_s": game_result["avg_s"],
                "avg_eps": game_result["avg_eps"],
                "server_utility": game_result["server_utility"],
                "total_quality": game_result["total_quality"]
            }
        )
        
        self.round_history.append(result)
        return result
    
    def _aggregate_logits(
        self,
        logits_list: List[torch.Tensor],
        weights: List[float]
    ) -> torch.Tensor:
        """
        Aggregate logits from multiple devices.
        
        Uses weighted average based on upload volume (s_i).
        """
        # Find minimum length (in case of different sizes)
        min_len = min(len(l) for l in logits_list)
        
        # Truncate and stack
        truncated = [l[:min_len] for l in logits_list]
        
        # Weighted average
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        aggregated = sum(
            w * logits for w, logits in zip(normalized_weights, truncated)
        )
        
        return aggregated
    
    def _distill_to_server(
        self,
        target_logits: torch.Tensor,
        public_loader: Any
    ):
        """
        Distill aggregated knowledge to server model.
        
        Uses KL divergence loss with temperature scaling.
        """
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
                
                # Student forward pass
                student_logits = self.server_model(data)
                
                # KL divergence loss with temperature
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
        """
        Aggregate updates (for interface compatibility).
        
        In PAID-FD, aggregation happens via logits in run_round.
        """
        pass  # Aggregation handled in _aggregate_logits
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "n_rounds": len(self.round_history),
            "price_history": self.price_history,
            "participation_history": self.participation_history,
            "best_accuracy": self.get_best_accuracy(),
            "final_accuracy": self.get_final_accuracy(),
            "avg_participation": np.mean(self.participation_history) if self.participation_history else 0
        }


def create_paid_fd(
    model_name: str = "resnet18",
    n_classes: int = 100,
    gamma: float = 10.0,
    device: str = None,
    **config_kwargs
) -> PAIDFD:
    """
    Convenience function to create PAID-FD with specified model.
    
    Args:
        model_name: Name of model ('resnet18', 'cnn', etc.)
        n_classes: Number of classes
        gamma: Server valuation coefficient
        device: Device to use
        **config_kwargs: Additional config parameters
        
    Returns:
        Configured PAIDFD instance
    """
    from ..models import get_model
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = get_model(model_name, n_classes=n_classes)
    config = PAIDFDConfig(gamma=gamma, **config_kwargs)
    
    return PAIDFD(model, config, n_classes, device)
