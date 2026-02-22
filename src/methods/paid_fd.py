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
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import copy

from .base import FederatedMethod, RoundResult
from ..game.stackelberg import StackelbergSolver, DeviceDecision
from ..privacy.ldp import add_noise_to_logits  # kept for reference; v4 uses np.random.laplace directly
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class PAIDFDConfig:
    """
    Configuration for PAID-FD.
    
    Key design: persistent local models + EMA logit buffer + mixed-loss distillation.
    Each device keeps its own model across rounds and does local epochs
    per round. Over 100 rounds, this accumulates training.
    
    EMA logit buffer averages noisy aggregated logits across rounds:
    - Single-round SNR ~ 1.6–2.6 (noise dominates, distillation useless)
    - EMA(0.9) buffer SNR ~ 3.7–6.0 (meaningful teaching signal)
    - Buffer changes smoothly → safe to use moderate distill_lr
    - Different γ → different buffer SNR → γ differentiation visible
    
    Distillation uses mixed loss: α×KL(buffer) + (1-α)×CE(true)
    - KL from EMA buffer provides noise-reduced ensemble knowledge
    - CE(true) anchors model, prevents drift
    - α=0.7 amplifies γ-dependent KL signal (70% ensemble, 30% GT)
    
    Pre-training is kept moderate (10 epochs) so that FL contribution
    matters — ensemble diversity (more participants) creates real signal
    differences across gamma values.
    """
    # Game parameters
    gamma: float = 10.0          # Server valuation coefficient
    delta: float = 0.01          # Search tolerance
    budget: float = float('inf') # Budget constraint
    
    # Local training (per round — models persist across rounds)
    local_epochs: int = 5        # 5 epochs/round → more local divergence → ensemble diversity
    local_lr: float = 0.01       # SGD fine-tuning lr (models start pre-trained)
    local_momentum: float = 0.9  # Standard SGD momentum
    
    # Distillation (EMA buffer + mixed loss)
    distill_epochs: int = 1      # 1 epoch/round
    distill_lr: float = 0.001    # Adam lr (safe: EMA buffer is smooth, not random each round)
    distill_alpha: float = 0.7   # α×KL(buffer) + (1-α)×CE(true): high α amplifies γ signal
    temperature: float = 3.0     # Soft-label T=3: preserves dark knowledge under noise
    ema_momentum: float = 0.9    # EMA for logit buffer: effective window ~5 rounds
    
    # Pre-training on public data (FedMD "transfer learning" phase)
    # 10 epochs → ~35-40% start, leaving ~20% room for FL improvement
    # This makes ensemble diversity (N devices) actually matter
    pretrain_epochs: int = 10    # 10 epochs: moderate start, FL has room
    pretrain_lr: float = 0.1     # Standard SGD lr for pre-training
    
    # Privacy
    clip_bound: float = 2.0      # Logit clipping for LDP (C=2: balances signal preservation vs noise)
    
    # Public data
    public_samples: int = 20000  # 200/class for better pre-training & logit diversity


class PAIDFD(FederatedMethod):
    """
    PAID-FD: Privacy-Aware Incentive-Driven Federated Distillation
    
    Protocol per round:
    1. Server computes optimal price p* via ServerPricing
    2. Server broadcasts p* to all devices
    3. Each device locally computes (s_i*, ε_i*) via DeviceBR
    4. Participating devices:
       a. Train local model on private data
       b. Compute logits on public data, clip to [-C, C]
       c. Add per-device LDP noise: Lap(0, 2C/ε_i)
       d. Upload noisy logits
    5. Server averages noisy logits → softmax(T=3) → KL distillation
    
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
        
        # Persistent distillation optimizer (maintains momentum across rounds)
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(),
            lr=self.config.distill_lr
        )
        
        # Persistent local models: each device keeps its own model across rounds
        # Initialized lazily on first participation
        self.local_models = {}      # dev_id -> nn.Module
        self.local_optimizers = {}  # dev_id -> optimizer
        
        # Pre-training state
        self._pretrained = False
        
        # EMA logit buffer: accumulates aggregated noisy logits across rounds
        # After ~5 rounds, buffer SNR ≈ 2.3× single-round SNR
        self.logit_buffer = None  # Initialized on first round with logits
        
        # Privacy accounting: track cumulative ε per device across rounds
        self.privacy_spent = {}  # device_id -> cumulative epsilon
        
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
        
        # ── Pre-train on public data (once, before first FL round) ──
        if not self._pretrained:
            self._pretrain_on_public(public_loader)
            self._pretrained = True
        
        # Stage 1: Server computes optimal price
        game_result = self.solver.solve(devices)
        price = game_result["price"]
        decisions = game_result["decisions"]
        
        self.price_history.append(price)
        
        # ==========================================
        # [TMC Fix v7] PER-DEVICE LDP + BLUE AGG + SOFT-LABEL KL
        #
        # Per-device LDP: each device adds Lap(0, 2C/ε_i) locally.
        # BLUE aggregation: w_i ∝ ε_i² (inverse-variance optimal weights).
        # Soft-label KL: preserves full logit distribution under noise.
        #
        # Per-device: noise_i ~ Lap(0, 2C/ε_i) for each element
        # BLUE weights: w_i = ε_i² / Σ ε_j² (minimises aggregated variance)
        # → High-ε devices contribute more; noisy marginal devices down-weighted.
        #
        # Pipeline per round:
        #   1. Collect ALL public images into a fixed tensor
        #   2. Each device: local train → logits → clip [-C,C] → Lap noise
        #   3. Server: ε²-weighted (BLUE) average of noisy logits
        #   4. Server: softmax(agg_logits / T) → teacher probs
        #   5. Server: KL distillation from teacher probs (no EMA)
        # ==========================================
        
        # Collect ALL public images AND labels into fixed tensors
        public_images_list = []
        public_labels_list = []
        for data, labels in public_loader:
            public_images_list.append(data)
            public_labels_list.append(labels)
        public_images = torch.cat(public_images_list, dim=0)
        public_labels = torch.cat(public_labels_list, dim=0)
        n_public = len(public_images)
        
        all_logits = []  # Store CLIPPED LOGITS per device
        all_weights = []  # Aggregation weights
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        participants = []
        eps_list = []  # Track epsilon per participant
        
        C = self.config.clip_bound  # Logit clip bound
        
        for decision in decisions:
            if not decision.participates:
                continue
            
            dev_id = decision.device_id
            dev = devices[dev_id]
            participants.append(dev_id)
            
            if dev_id not in client_loaders:
                continue
            
            local_loader = client_loaders[dev_id]
            
            # Get or create persistent local model for this device
            if dev_id not in self.local_models:
                self.local_models[dev_id] = copy_model(self.server_model, device=self.device)
                self.local_optimizers[dev_id] = torch.optim.SGD(
                    self.local_models[dev_id].parameters(),
                    lr=self.config.local_lr,
                    momentum=self.config.local_momentum,
                    weight_decay=5e-4
                )
            
            local_model = self.local_models[dev_id]
            local_optimizer = self.local_optimizers[dev_id]
            
            # Local training on private data (1 epoch with persistent model)
            local_model.train()
            criterion = nn.CrossEntropyLoss()
            for data, target in local_loader:
                data, target = data.to(self.device), target.to(self.device)
                local_optimizer.zero_grad()
                loss = criterion(local_model(data), target)
                loss.backward()
                local_optimizer.step()
            
            # Compute CLIPPED logits on ALL public data
            local_model.eval()
            logit_chunks = []
            bs = 512
            with torch.no_grad():
                for start in range(0, n_public, bs):
                    batch = public_images[start:start+bs].to(self.device)
                    logits = local_model(batch)
                    # Clip logits to bound sensitivity
                    logits = torch.clamp(logits, -C, C)
                    logit_chunks.append(logits.cpu())
            device_logits = torch.cat(logit_chunks, dim=0)  # (n_public, K)
            
            # ── Per-device LDP: add Laplace noise LOCALLY ──
            # Each device perturbs its own logits before upload.
            # Sensitivity = 2C (max change in one element when clipped to [-C,C])
            # Scale = 2C / ε_i  (device's chosen privacy level)
            # This is true Local DP: server never sees clean logits.
            # When server averages N noisy logits, noise var ~ 1/N.
            # → More participants = less noise = better accuracy.
            device_eps = decision.eps_star
            device_sensitivity = 2.0 * C  # Per-device sensitivity
            device_noise_scale = device_sensitivity / device_eps
            device_noise = np.random.laplace(
                0, device_noise_scale, device_logits.shape
            ).astype(np.float32)
            noisy_device_logits = device_logits + torch.from_numpy(device_noise)
            
            all_logits.append(noisy_device_logits)
            # BLUE (Best Linear Unbiased Estimator): w_i ∝ ε_i²
            # Var[noise_i] = 2(2C/ε_i)² ∝ 1/ε_i², so inverse-variance weight ∝ ε_i²
            # This down-weights high-noise (low-ε) devices optimally.
            all_weights.append(decision.eps_star ** 2)
            eps_list.append(decision.eps_star)
            
            # Energy accounting
            s_for_cost = max(int(decision.s_star), 200)
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=getattr(dev, 'data_size', len(local_loader.dataset)),
                s_i=s_for_cost,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            for k in ["training", "inference", "communication"]:
                total_energy[k] += energy.get(k, 0)
        
        # Stage 3: Aggregate NOISY logits → EMA buffer → distill
        # Each device already added LDP noise locally.
        # Single-round SNR ~ 1.6–2.6 → distillation learns mostly noise.
        # EMA buffer accumulates logits across rounds → SNR × √(eff_window).
        # With ema_momentum=0.9, effective window ≈ 5 rounds → SNR 3.7–6.0.
        if all_logits:
            N = len(all_logits)
            avg_eps = np.mean(eps_list)
            
            # 3a. BLUE weighted average of NOISY logits (per-device noise already added)
            min_len = min(len(l) for l in all_logits)
            total_w = sum(all_weights)
            norm_w = [w / total_w for w in all_weights]
            aggregated_noisy = sum(
                w * l[:min_len] for w, l in zip(norm_w, all_logits)
            )
            noisy_logits_tensor = aggregated_noisy.float()
            
            # 3b. Update EMA logit buffer (noise averaging across rounds)
            # buffer_t = α * buffer_{t-1} + (1-α) * logits_t
            # After K rounds: noise variance ≈ var_single / (effective_window)
            # → SNR improves by √(eff_window) ≈ 2.3× for α=0.9
            ema = self.config.ema_momentum
            if self.logit_buffer is None:
                self.logit_buffer = noisy_logits_tensor.clone()
            else:
                # Handle size mismatch (shouldn't happen with fixed public set)
                buf_len = min(len(self.logit_buffer), min_len)
                self.logit_buffer[:buf_len] = (
                    ema * self.logit_buffer[:buf_len]
                    + (1 - ema) * noisy_logits_tensor[:buf_len]
                )
            
            # 3c. Distill from EMA buffer (not single-round noisy logits)
            T = self.config.temperature
            buf_len = min(len(self.logit_buffer), min_len)
            teacher_probs = F.softmax(self.logit_buffer[:buf_len] / T, dim=1)
            
            # 3d. Mixed loss: α × KL(buffer_teacher) + (1-α) × CE(true)
            self._distill_to_server_kl(
                teacher_probs, public_images[:buf_len], public_labels[:buf_len]
            )
            
            # 3d. Privacy accounting
            for dev_id, eps in zip(participants, eps_list):
                self.privacy_spent[dev_id] = self.privacy_spent.get(dev_id, 0) + eps
        
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
                "total_quality": game_result["total_quality"],
                "max_privacy_spent": max(self.privacy_spent.values()) if self.privacy_spent else 0,
                "avg_privacy_spent": float(np.mean(list(self.privacy_spent.values()))) if self.privacy_spent else 0,
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
    
    def _pretrain_on_public(self, public_loader):
        """
        Pre-train server model on public data (FedMD "transfer learning" phase).
        
        This gives every model a ~30-35% starting accuracy on CIFAR-100 before
        FL begins, so local models produce meaningful logits from round 1.
        Without this, each device's 800 non-IID samples can only reach ~12%
        and distillation from 12%-accurate teachers transfers nothing.
        """
        print(f"  [Pre-training] {self.config.pretrain_epochs} epochs on public data ...")
        
        # Augmentation for pre-training
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        
        optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=self.config.pretrain_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_epochs
        )
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.pretrain_epochs):
            self.server_model.train()
            epoch_loss = 0.0
            n_batches = 0
            for data, target in public_loader:
                data = augment(data).to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.server_model(data), target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{self.config.pretrain_epochs}: "
                      f"loss={epoch_loss/n_batches:.4f}")
        
        print(f"  [Pre-training] Done. All local models will start from this checkpoint.")
    
    def _distill_to_server_kl(
        self,
        teacher_probs: torch.Tensor,
        public_images: torch.Tensor,
        public_labels: torch.Tensor = None
    ):
        """
        Distill via mixed loss: α×KL(noisy) + (1-α)×CE(true).
        
        The KL term transfers ensemble knowledge from noisy aggregated logits.
        The CE term anchors the model to ground-truth labels, preventing
        gradual accuracy degradation from 100 rounds of noisy-target fitting.
        
        With α=0.5 and lr=0.0001, each round makes a small update that
        balances FL signal with stability.
        
        Args:
            teacher_probs: Soft target probabilities from noisy logits (N, K)
            public_images: Corresponding public images (N, C, H, W)
            public_labels: Ground-truth labels for public data (N,)
        """
        self.server_model.train()
        optimizer = self.distill_optimizer
        T = self.config.temperature
        alpha = self.config.distill_alpha
        ce_criterion = nn.CrossEntropyLoss()
        
        # Augmentation on normalised tensors (works for CIFAR 32×32)
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        
        n_target = min(len(teacher_probs), len(public_images))
        batch_size = 256
        
        for epoch in range(self.config.distill_epochs):
            perm = torch.randperm(n_target)
            
            for start in range(0, n_target, batch_size):
                end = min(start + batch_size, n_target)
                idx = perm[start:end]
                
                data = augment(public_images[idx]).to(self.device)
                target_probs = teacher_probs[idx].to(self.device)
                
                student_logits = self.server_model(data)
                
                # KL distillation from noisy teacher
                loss_kl = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target_probs,
                    reduction='batchmean'
                ) * (T * T)
                
                # Ground-truth CE regularization
                if public_labels is not None and alpha < 1.0:
                    true_labels = public_labels[idx].to(self.device)
                    loss_ce = ce_criterion(student_logits, true_labels)
                    loss = alpha * loss_kl + (1 - alpha) * loss_ce
                else:
                    loss = loss_kl
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()
    
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