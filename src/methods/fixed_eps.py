"""
Fixed-Epsilon v8: Fixed privacy budget baseline.

Standard FD flow (v8): same as PAID-FD v8 but with fixed epsilon for all
devices, no Stackelberg game.  No persistent models, no EMA, no CE anchor.

Purpose: Show the value of adaptive privacy allocation (game mechanism).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .base import FederatedMethod, RoundResult
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class FixedEpsilonConfig:
    """Configuration for Fixed-Epsilon baseline."""
    epsilon: float = 1.0
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9
    distill_epochs: int = 1
    distill_lr: float = 0.001
    temperature: float = 3.0
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.1
    clip_bound: float = 2.0
    participation_rate: float = 1.0
    samples_per_device: int = 100
    public_samples: int = 20000


class FixedEpsilon(FederatedMethod):
    """Fixed-Epsilon v8: Standard FD with fixed privacy budget.

    Identical pipeline to PAID-FD v8 but:
      - Fixed epsilon for ALL devices (no game)
      - Equal weights (no BLUE)
      - All devices participate (or random subset)
    """

    def __init__(self, server_model, config=None, n_classes=100, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or FixedEpsilonConfig()
        self.energy_calc = EnergyCalculator()
        self.rng = np.random.RandomState(42)
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr
        )
        self._pretrained = False

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        """Execute one round of Fixed-eps v8."""
        self.current_round = round_idx

        if not self._pretrained:
            self._pretrain_on_public(public_loader)
            self._pretrained = True

        # Select participants
        n_devices = len(devices)
        n_part = max(1, int(n_devices * self.config.participation_rate))
        if n_part < n_devices:
            participant_ids = sorted(
                self.rng.choice(n_devices, size=n_part, replace=False).tolist()
            )
        else:
            participant_ids = list(range(n_devices))

        # Collect public images
        public_images_list = []
        for data, labels in public_loader:
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

            # v8: Fresh copy each round
            local_model = copy_model(self.server_model, device=self.device)
            local_optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=self.config.local_lr,
                momentum=self.config.local_momentum,
                weight_decay=5e-4
            )

            # Local training
            local_model.train()
            criterion = nn.CrossEntropyLoss()
            for epoch in range(self.config.local_epochs):
                for data, target in local_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    local_optimizer.zero_grad()
                    loss = criterion(local_model(data), target)
                    loss.backward()
                    local_optimizer.step()

            # Compute clipped logits
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

            # Per-device LDP with FIXED epsilon
            noise_scale = 2.0 * C / self.config.epsilon
            noise = np.random.laplace(0, noise_scale, device_logits.shape).astype(np.float32)
            noisy_logits = device_logits + torch.from_numpy(noise)
            all_logits.append(noisy_logits)

            # Energy
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=getattr(dev, 'data_size', len(local_loader.dataset)),
                s_i=n_public,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            for k in ["training", "inference", "communication"]:
                total_energy[k] += energy.get(k, 0)
            del local_model, local_optimizer

        # Aggregate: simple average of noisy logits
        if all_logits:
            N = len(all_logits)
            min_len = min(len(l) for l in all_logits)
            aggregated = sum(l[:min_len] for l in all_logits) / N
            T = self.config.temperature
            teacher_probs = F.softmax(aggregated.float() / T, dim=1)
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

    def _pretrain_on_public(self, public_loader):
        """Pre-train server model on public data."""
        print(f"  [Fixed-eps Pre-training] {self.config.pretrain_epochs} epochs ...")
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=self.config.pretrain_lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.pretrain_epochs
        )
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.config.pretrain_epochs):
            self.server_model.train()
            for data, target in public_loader:
                data = augment(data).to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.server_model(data), target)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{self.config.pretrain_epochs}")
        print(f"  [Fixed-eps Pre-training] Done.")

    def _distill_to_server(self, teacher_probs, public_images):
        """Pure KL distillation."""
        self.server_model.train()
        optimizer = self.distill_optimizer
        T = self.config.temperature
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
                loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target_probs,
                    reduction='batchmean'
                ) * (T * T)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()

    def aggregate(self, updates, weights):
        pass
