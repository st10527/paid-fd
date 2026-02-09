"""
FedMD: Federated Model Distillation (Li & Wang, NeurIPS 2019)

Classic federated distillation baseline.
Same distillation framework as PAID-FD but WITHOUT:
  - Stackelberg game / pricing
  - Local Differential Privacy (no noise)
  - Adaptive participation / quality

All devices participate equally, upload raw logits, simple average.
This is the "upper bound" for FD with no privacy.

Reference:
  Li & Wang, "FedMD: Heterogenous Federated Learning via Model
  Distillation", NeurIPS 2019 Workshop
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
class FedMDConfig:
    """Configuration for FedMD."""
    # Local training
    local_epochs: int = 20
    local_lr: float = 0.1
    local_momentum: float = 0.9

    # Distillation
    distill_epochs: int = 10
    distill_lr: float = 0.005
    temperature: float = 3.0

    # Logit clipping (for stability, not privacy)
    clip_bound: float = 5.0

    # Public data
    public_samples: int = 10000


class FedMD(FederatedMethod):
    """
    FedMD: Federated Model Distillation.

    Protocol per round:
      1. Server broadcasts global model
      2. All devices train locally on private data
      3. All devices compute logits on public data (NO noise)
      4. Server aggregates logits (simple average, equal weights)
      5. Server distills aggregated knowledge into global model

    Key: No privacy, no game, no incentive → pure FD performance upper bound.
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: FedMDConfig = None,
        n_classes: int = 100,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(server_model, n_classes, device)
        self.config = config or FedMDConfig()
        self.energy_calc = EnergyCalculator()

    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any,
        test_loader: Optional[Any] = None
    ) -> RoundResult:
        """Execute one round of FedMD."""
        self.current_round = round_idx

        # ── Collect ALL public images into fixed tensor ──
        public_images_list = []
        for data, _ in public_loader:
            public_images_list.append(data)
        public_images = torch.cat(public_images_list, dim=0)
        n_public = len(public_images)

        C = self.config.clip_bound
        all_logits = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        participants = []

        # ── All devices participate (no selection) ──
        for dev_id, dev in enumerate(devices):
            if dev_id not in client_loaders:
                continue

            participants.append(dev_id)
            local_loader = client_loaders[dev_id]
            local_model = copy_model(self.server_model, device=self.device)

            # Local training
            self.train_local(
                local_model,
                local_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr
            )

            # Compute clipped logits on ALL public data (NO noise)
            local_model.eval()
            logit_chunks = []
            bs = 256
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

        # ── Aggregate: simple average (NO noise) ──
        if all_logits:
            N = len(all_logits)
            min_len = min(len(l) for l in all_logits)
            aggregated_logits = sum(l[:min_len] for l in all_logits) / N

            # Convert to teacher probs via softmax
            T = self.config.temperature
            teacher_probs = F.softmax(aggregated_logits / T, dim=1)

            # Distill to server model
            self._distill_to_server(teacher_probs, public_images[:min_len])

        # Evaluate
        accuracy = 0.0
        loss = 0.0
        if test_loader is not None:
            eval_result = self.evaluate(test_loader)
            accuracy = eval_result["accuracy"]
            loss = eval_result["loss"]

        participation_rate = len(participants) / len(devices) if devices else 0

        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            n_participants=len(participants),
            energy=total_energy,
            extra={
                "method": "FedMD",
                "n_public": n_public,
                "epsilon": float('inf'),  # No privacy
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
        batch_size = 128

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
