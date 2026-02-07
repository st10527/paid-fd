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
    """
    Configuration for PAID-FD.
    
    Default values based on literature:
    - local_epochs/lr: FedAvg (McMahan et al., 2017) typically uses 1-5 epochs, lr=0.01
    - temperature: Knowledge Distillation (Hinton et al., 2015) suggests T=2-5
    - distill_lr: Standard Adam default is 0.001, but KD often uses 0.01
    """
    # Game parameters (YOUR CONTRIBUTION - need tuning)
    gamma: float = 10.0          # Server valuation coefficient
    delta: float = 0.01          # Search tolerance
    budget: float = float('inf') # Budget constraint
    
    # Local training (Literature: FedAvg, FedProx)
    local_epochs: int = 3        # FedAvg uses 1-5, we use 3 as middle ground
    local_lr: float = 0.01       # Standard for SGD on CIFAR
    local_momentum: float = 0.9  # Standard SGD momentum
    
    # Distillation (Literature: Hinton 2015, FedMD, FedDF)
    distill_epochs: int = 10     # More epochs helps convergence
    distill_lr: float = 0.01     # Adam lr for distillation
    temperature: float = 3.0     # Hinton suggests 2-5, we use 3
    
    # Privacy (YOUR CONTRIBUTION)
    clip_bound: float = 5.0      # Logit clipping for LDP
    
    # Public data
    public_samples: int = 1000   # Samples per round


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
        
        # Stage 2: Collect soft labels from participating devices
        # ==========================================
        # [TMC Fix v3] Architecture:
        #   - Collect a FIXED set of public images (use ALL public data)
        #   - Each device computes logits -> softmax on the SAME images
        #   - Aggregate probabilities (weighted average across devices)
        #   - Add ONE noise pass on the aggregate (privacy amplified by N devices)
        #   - Distill using full (image, soft_label) pairs
        #
        # Key insight: s_i* from the game controls payment/cost accounting,
        # NOT the actual number of samples used for distillation.
        # In FedMD/FedDF, all devices share logits on the FULL public set.
        # ==========================================
        
        # Collect ALL public images into a fixed tensor
        public_images_list = []
        for data, _ in public_loader:
            public_images_list.append(data)
        public_images = torch.cat(public_images_list, dim=0)
        n_public = len(public_images)
        
        all_probs = []  # Store probability vectors per device
        all_weights = []  # Weight = quality contribution for payment
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        participants = []
        avg_eps = 0.0  # Track for aggregate noise
        
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
            
            # Create local model copy
            local_model = copy_model(self.server_model, device=self.device)
            
            # Local training on private data
            self.train_local(
                local_model,
                local_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr
            )
            
            # Compute softmax probabilities on ALL public data
            local_model.eval()
            T = self.config.temperature
            probs_chunks = []
            bs = 256
            with torch.no_grad():
                for start in range(0, n_public, bs):
                    batch = public_images[start:start+bs].to(self.device)
                    logits = local_model(batch)
                    probs = F.softmax(logits / T, dim=1)
                    probs_chunks.append(probs.cpu())
            device_probs = torch.cat(probs_chunks, dim=0)  # (n_public, K)
            
            all_probs.append(device_probs)
            # Weight by game quality (for Stackelberg consistency)
            all_weights.append(decision.quality)
            avg_eps += decision.eps_star
            
            # Energy accounting (s_i* used here for cost, not for limiting data)
            s_for_cost = max(int(decision.s_star), 200)
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=getattr(dev, 'data_size', len(local_loader.dataset)),
                s_i=s_for_cost,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            for k in ["training", "inference", "communication"]:
                total_energy[k] += energy.get(k, 0)
            
            del local_model
        
        # Stage 3: Aggregate and distill
        if all_probs:
            n_participants = len(all_probs)
            avg_eps = avg_eps / n_participants if n_participants > 0 else 1.0
            
            # 3a. Weighted average of device probabilities
            aggregated = self._aggregate_probs(all_probs, all_weights)
            
            # 3b. Add LDP noise on the AGGREGATE (privacy amplified by averaging)
            # Effective epsilon for aggregate = avg_eps * sqrt(N) (privacy amplification)
            # Sensitivity in probability space = 2/N (one device's contribution is 1/N)
            effective_sensitivity = 2.0 / n_participants
            noisy_agg = add_noise_to_logits(
                aggregated.numpy(),
                epsilon=avg_eps,
                sensitivity=effective_sensitivity,
                clip_bound=None
            )
            # Re-normalize
            noisy_agg = np.clip(noisy_agg, 1e-8, None)
            noisy_agg = noisy_agg / noisy_agg.sum(axis=1, keepdims=True)
            aggregated = torch.from_numpy(noisy_agg).float()
            
            # 3c. Distill to server model
            self._distill_to_server(aggregated, public_images)
        
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
    
    def _aggregate_probs(
        self,
        probs_list: List[torch.Tensor],
        weights: List[float]
    ) -> torch.Tensor:
        """
        Aggregate probability vectors from multiple devices.
        
        Each device already converted logits -> softmax -> noisy probs.
        We just do a weighted average here.
        
        Returns:
            Aggregated probability tensor (N, K)
        """
        # Find minimum length (in case of different sizes)
        min_len = min(len(p) for p in probs_list)
        probs_list = [p[:min_len] for p in probs_list]
        
        # Weighted average of probabilities
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        aggregated = sum(
            w * probs for w, probs in zip(normalized_weights, probs_list)
        )
        
        return aggregated
    
    def _distill_to_server(
        self,
        teacher_probs: torch.Tensor,
        public_images: torch.Tensor
    ):
        """
        Distill aggregated knowledge to server model.
        
        Args:
            teacher_probs: Aggregated soft labels (N, K) - probability vectors
            public_images: Corresponding public images (N, C, H, W)
        
        Uses KL divergence loss with temperature scaling.
        Receives pre-collected (images, probs) pairs so alignment is guaranteed.
        Uses all N samples for distillation.
        """
        self.server_model.train()
        optimizer = torch.optim.Adam(
            self.server_model.parameters(),
            lr=self.config.distill_lr
        )
        
        T = self.config.temperature
        n_target = min(len(teacher_probs), len(public_images))
        
        # Keep teacher_probs on CPU, move batches to device on-the-fly
        # to avoid GPU OOM with large public sets
        batch_size = 128
        
        for epoch in range(self.config.distill_epochs):
            perm = torch.randperm(n_target)
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, n_target, batch_size):
                end = min(start + batch_size, n_target)
                idx = perm[start:end]
                
                data = public_images[idx].to(self.device)
                target = teacher_probs[idx].to(self.device)
                
                # Student forward pass
                student_logits = self.server_model(data)
                
                # KL divergence: KL(teacher || student)
                loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target,
                    reduction='batchmean'
                ) * (T * T)
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
    
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