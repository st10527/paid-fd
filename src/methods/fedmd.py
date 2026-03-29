"""
FedMD v8.2: Noise-free Federated Distillation baseline.

Standard FD flow: Each round, ALL devices get a fresh copy of the
server model, train locally, upload clean (no noise) logits, server
distills via KL.  No persistent local models, no EMA.

This is the noise-free FD oracle -- upper bound for FD methods.
The ONLY differences vs PAID-FD are:
  - No Stackelberg game / pricing (all devices participate)
  - No LDP noise on logits
  - No BLUE weighting (equal weight average)
  - Denoising disabled by default (no noise to denoise)
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
class FedMDConfig:
    """Configuration for FedMD (noise-free FD oracle)."""
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9
    distill_epochs: int = 5
    distill_lr: float = 0.001
    temperature: float = 3.0
    ce_anchor_alpha: float = 0.0   # v8.2: pure KL by default
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.1
    clip_bound: float = 5.0
    public_samples: int = 20000
    use_denoising: bool = False    # No noise → no need for denoising


class FedMD(FederatedMethod):
    """FedMD v8: Standard FD without noise (oracle upper bound).

    Protocol per round:
      1. Server broadcasts current model to all devices
      2. Each device gets fresh copy, trains locally
      3. Each device computes clipped logits on public data (NO noise)
      4. Server: equal-weight average of clean logits
      5. Server: softmax(avg / T) -> teacher probs
      6. Server: KL distillation
    """

    def __init__(self, server_model, config=None, n_classes=100, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or FedMDConfig()
        self.energy_calc = EnergyCalculator()
        # v8.1: No persistent optimizer (fresh SGD each round)
        self._pretrained = False

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        """Execute one round of FedMD v8."""
        self.current_round = round_idx

        if not self._pretrained:
            self._pretrain_on_public(public_loader)
            self._pretrained = True

        # Collect public images AND labels
        public_images_list = []
        public_labels_list = []
        for data, labels in public_loader:
            public_images_list.append(data)
            public_labels_list.append(labels)
        public_images = torch.cat(public_images_list, dim=0)
        public_labels = torch.cat(public_labels_list, dim=0)
        n_public = len(public_images)

        C = self.config.clip_bound
        all_logits = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        participants = []

        for dev_id, dev in enumerate(devices):
            if dev_id not in client_loaders:
                continue
            participants.append(dev_id)
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

            # Compute clipped logits (NO noise)
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
            del local_model, local_optimizer

        # Aggregate: simple average (NO noise)
        if all_logits:
            N = len(all_logits)
            min_len = min(len(l) for l in all_logits)
            aggregated = sum(l[:min_len] for l in all_logits) / N

            # v8.2: Class-conditional denoising (optional, off by default for FedMD)
            denoised = self._denoise_logits(aggregated, public_labels[:min_len])

            T = self.config.temperature
            teacher_probs = F.softmax(denoised / T, dim=1)
            self._distill_to_server(teacher_probs, public_images[:min_len],
                                    public_labels[:min_len])

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
                "epsilon": float('inf'),
            }
        )
        self.round_history.append(result)
        return result

    def _pretrain_on_public(self, public_loader):
        """Pre-train server model on public data."""
        print(f"  [FedMD Pre-training] {self.config.pretrain_epochs} epochs ...")
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
        print(f"  [FedMD Pre-training] Done.")

    def _denoise_logits(self, aggregated, public_labels):
        """Class-conditional denoising (no-op when use_denoising=False)."""
        if not self.config.use_denoising:
            return aggregated
        denoised = aggregated.clone()
        for c in range(self.n_classes):
            mask = (public_labels == c)
            n_c = mask.sum().item()
            if n_c > 1:
                class_mean = aggregated[mask].mean(dim=0)
                denoised[mask] = class_mean.unsqueeze(0).expand(n_c, -1)
        return denoised

    def _distill_to_server(self, teacher_probs, public_images, public_labels=None):
        """v8.2: KL distillation with fresh SGD (optional CE anchor)."""
        self.server_model.train()
        optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=self.config.distill_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        T = self.config.temperature
        alpha = self.config.ce_anchor_alpha
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        ce_criterion = nn.CrossEntropyLoss()
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

                loss_kl = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target_probs,
                    reduction='batchmean'
                ) * (T * T)

                if alpha > 0 and public_labels is not None:
                    labels = public_labels[idx].to(self.device)
                    loss_ce = ce_criterion(student_logits, labels)
                    loss = alpha * loss_ce + (1.0 - alpha) * loss_kl
                else:
                    loss = loss_kl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()

    def aggregate(self, updates, weights):
        pass
