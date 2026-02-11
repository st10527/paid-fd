"""
Fixed-Epsilon Baseline for PAID-FD Ablation Study

Uses a fixed privacy budget ε for all devices,
without the adaptive Stackelberg game mechanism.

v4-aligned: Uses the same logit-space aggregation pipeline as PAID-FD:
  1. Each device: local train → compute logits → clip to [-C, C]
  2. Server: simple average of clipped logits (equal weights)
  3. Server: Laplace noise with FIXED ε (sensitivity = 2C/N)
  4. Server: softmax(noisy_logits / T) → teacher probs
  5. Distill server model from teacher

Purpose: Demonstrate the value of adaptive privacy allocation (game).
  - Same pipeline as PAID-FD, but ε is fixed → shows game's benefit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base import FederatedMethod, RoundResult
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class FixedEpsilonConfig:
    """Configuration for Fixed-Epsilon baseline."""
    # Fixed privacy budget (no adaptation)
    epsilon: float = 1.0

    # Local training (same as PAID-FD)
    local_epochs: int = 20
    local_lr: float = 0.1
    local_momentum: float = 0.9

    # Distillation (same as PAID-FD)
    distill_epochs: int = 10
    distill_lr: float = 0.005
    temperature: float = 3.0

    # Privacy
    clip_bound: float = 5.0

    # Participation
    participation_rate: float = 1.0


class FixedEpsilon(FederatedMethod):
    """
    Fixed-Epsilon Federated Distillation.

    Identical to PAID-FD pipeline but with:
      - Fixed ε for ALL devices (no Stackelberg game)
      - Equal weights (no quality-based weighting)
      - All devices participate (or random subset)

    Ablation to show the benefit of:
      1. Adaptive privacy allocation (ε* per device)
      2. Quality-based weighting (s* per device)
      3. Incentive mechanism (price p*)
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
        """Execute one round of Fixed-ε FD."""
        self.current_round = round_idx

        # Select participants
        n_devices = len(devices)
        n_participants = max(1, int(n_devices * self.config.participation_rate))
        if n_participants < n_devices:
            participant_ids = sorted(
                self.rng.choice(n_devices, size=n_participants, replace=False).tolist()
            )
        else:
            participant_ids = list(range(n_devices))

        # ── Collect ALL public images into fixed tensor ──
        public_images_list = []
        for data, _ in public_loader:
            public_images_list.append(data)
        public_images = torch.cat(public_images_list, dim=0)
        n_public = len(public_images)

        C = self.config.clip_bound
        all_logits = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        actual_participants = []

        for dev_id in participant_ids:
            if dev_id not in client_loaders:
                continue

            actual_participants.append(dev_id)
            dev = devices[dev_id]
            local_loader = client_loaders[dev_id]
            local_model = copy_model(self.server_model, device=self.device)

            # Local training
            self.train_local(
                local_model,
                local_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr
            )

            # Compute clipped logits on ALL public data
            local_model.eval()
            logit_chunks = []
            bs = 512
            with torch.no_grad():
                for start in range(0, n_public, bs):
                    batch = public_images[start:start + bs].to(self.device)
                    logits = local_model(batch)
                    logits = torch.clamp(logits, -C, C)
                    logit_chunks.append(logits.cpu())
            device_logits = torch.cat(logit_chunks, dim=0)

            all_logits.append(device_logits)

            # Energy
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=getattr(dev, 'data_size', len(local_loader.dataset)),
                s_i=n_public,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            for k in ["training", "inference", "communication"]:
                total_energy[k] += energy.get(k, 0)

            del local_model

        # ── Aggregate: simple average + FIXED-ε noise ──
        if all_logits:
            N = len(all_logits)
            min_len = min(len(l) for l in all_logits)

            # Simple average (equal weights, no quality-based weighting)
            aggregated_logits = sum(l[:min_len] for l in all_logits) / N

            # Add Laplace noise with FIXED epsilon
            sensitivity_per_elem = 2.0 * C / N
            noise_scale = sensitivity_per_elem / self.config.epsilon
            noise = np.random.laplace(0, noise_scale, aggregated_logits.shape)
            noisy_logits = aggregated_logits.numpy() + noise.astype(np.float32)

            # Convert to teacher probs
            T = self.config.temperature
            noisy_logits_tensor = torch.from_numpy(noisy_logits).float()
            teacher_probs = F.softmax(noisy_logits_tensor / T, dim=1)

            # Distill
            self._distill_to_server(teacher_probs, public_images[:min_len])

        # Evaluate
        accuracy = 0.0
        loss = 0.0
        if test_loader is not None:
            eval_result = self.evaluate(test_loader)
            accuracy = eval_result["accuracy"]
            loss = eval_result["loss"]

        participation_rate = len(actual_participants) / n_devices if n_devices > 0 else 0

        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            n_participants=len(actual_participants),
            energy=total_energy,
            extra={
                "method": f"Fixed-eps-{self.config.epsilon}",
                "fixed_epsilon": self.config.epsilon,
            }
        )

        self.round_history.append(result)
        return result

    def _distill_to_server(
        self,
        teacher_probs: torch.Tensor,
        public_images: torch.Tensor
    ):
        """Distill aggregated knowledge to server model (same as PAID-FD)."""
        self.server_model.train()
        optimizer = torch.optim.Adam(
            self.server_model.parameters(),
            lr=self.config.distill_lr
        )

        T = self.config.temperature
        n_target = min(len(teacher_probs), len(public_images))
        batch_size = 256

        for epoch in range(self.config.distill_epochs):
            perm = torch.randperm(n_target)
            for start in range(0, n_target, batch_size):
                end = min(start + batch_size, n_target)
                idx = perm[start:end]

                data = public_images[idx].to(self.device)
                target = teacher_probs[idx].to(self.device)

                student_logits = self.server_model(data)
                loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target,
                    reduction='batchmean'
                ) * (T * T)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()

    def aggregate(self, updates: List[Dict], weights: List[float]) -> None:
        """Interface compatibility."""
        pass
