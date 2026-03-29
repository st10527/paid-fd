#!/usr/bin/env python3
"""Generate v8 source files for PAID-FD, FedMD, and Fixed-eps."""

import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = os.path.join(BASE, "src", "methods")

# ============================================================
# PAID-FD v8
# ============================================================
paid_fd_v8 = '''\
"""
PAID-FD v8: Privacy-Aware Incentive-Driven Federated Distillation
=================================================================

Standard FD flow (v8): Each round, participating devices receive a fresh
copy of the current server model, train locally, upload noisy logits, and the
server distills.  No persistent local models, no EMA buffer, no CE anchor.

Key components retained:
  - Stackelberg game-based pricing (gamma -> price p* -> device decisions)
  - Per-device LDP: Lap(0, 2C/eps_i) on logits
  - BLUE aggregation: weights proportional to eps_i^2 (inverse-variance optimal)
  - Pre-training on public data
  - Soft-label KL distillation with temperature T

Why v8 (standard FD) instead of v7 (persistent models)?
  - v7 persistent local models accumulate training across rounds, eventually
    converging to ~60% regardless of gamma -- masking the game effect.
  - v8 standard FD: each round distillation quality depends on THIS round
    participants, so gamma (which controls participation) directly affects accuracy.
  - This is also closer to the original FedMD paper protocol.
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
from ..privacy.ldp import add_noise_to_logits
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class PAIDFDConfig:
    """Configuration for PAID-FD v8 (Standard FD flow).

    v8 design: Standard FD -- each round, devices get fresh copy of server
    model. No persistent local models, no EMA buffer, no CE anchor.
    """
    # Game parameters
    gamma: float = 10.0
    delta: float = 0.01
    budget: float = float('inf')

    # Local training (per round -- fresh copy each round)
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9

    # Distillation (pure KL, no CE anchor)
    distill_epochs: int = 1
    distill_lr: float = 0.001
    temperature: float = 3.0

    # Pre-training on public data
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.1

    # Privacy
    clip_bound: float = 2.0

    # Public data
    public_samples: int = 20000

    # Ablation flags
    use_blue: bool = True   # BLUE (eps^2-weighted) vs equal weights
    use_ldp: bool = True    # Per-device LDP noise vs clean logits (oracle)


class PAIDFD(FederatedMethod):
    """PAID-FD v8: Standard Federated Distillation with Stackelberg Game.

    Protocol per round:
      1. Server computes optimal price p* via Stackelberg game
      2. Each device decides (participate?, s_i*, eps_i*)
      3. Participating devices:
         a. Receive fresh copy of server model
         b. Train locally on private data (local_epochs, SGD)
         c. Compute logits on public data, clip to [-C, C]
         d. Add per-device LDP noise: Lap(0, 2C/eps_i)
         e. Upload noisy logits
      4. Server: BLUE-weighted average of noisy logits
      5. Server: softmax(aggregated / T) -> teacher probs
      6. Server: KL distillation from teacher probs to server model
    """

    def __init__(self, server_model, config=None, n_classes=100, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(server_model, n_classes, device)
        self.config = config or PAIDFDConfig()
        self.solver = StackelbergSolver(
            gamma=self.config.gamma,
            delta=self.config.delta,
            budget=self.config.budget
        )
        self.energy_calc = EnergyCalculator()
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr
        )
        self._pretrained = False
        self.privacy_spent = {}
        self.price_history = []
        self.participation_history = []

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        """Execute one round of PAID-FD v8 (standard FD flow)."""
        self.current_round = round_idx

        if not self._pretrained:
            self._pretrain_on_public(public_loader)
            self._pretrained = True

        # Stage 1: Stackelberg game
        game_result = self.solver.solve(devices)
        price = game_result["price"]
        decisions = game_result["decisions"]
        self.price_history.append(price)

        # Collect public images
        public_images_list = []
        for data, labels in public_loader:
            public_images_list.append(data)
        public_images = torch.cat(public_images_list, dim=0)
        n_public = len(public_images)

        all_logits = []
        all_weights = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}
        participants = []
        eps_list = []
        C = self.config.clip_bound

        # Stage 2: Devices train & upload noisy logits
        for decision in decisions:
            if not decision.participates:
                continue
            dev_id = decision.device_id
            dev = devices[dev_id]
            participants.append(dev_id)
            if dev_id not in client_loaders:
                continue
            local_loader = client_loaders[dev_id]

            # v8: Fresh copy of server model each round
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

            # Per-device LDP
            device_eps = decision.eps_star
            if self.config.use_ldp:
                noise_scale = 2.0 * C / device_eps
                noise = np.random.laplace(0, noise_scale, device_logits.shape).astype(np.float32)
                noisy_logits = device_logits + torch.from_numpy(noise)
            else:
                noisy_logits = device_logits

            all_logits.append(noisy_logits)
            if self.config.use_blue:
                all_weights.append(device_eps ** 2)
            else:
                all_weights.append(1.0)
            eps_list.append(device_eps)

            # Energy
            s_for_cost = max(int(decision.s_star), 200)
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=getattr(dev, 'data_size', len(local_loader.dataset)),
                s_i=s_for_cost,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            for k in ["training", "inference", "communication"]:
                total_energy[k] += energy.get(k, 0)
            del local_model, local_optimizer

        # Stage 3: Aggregate & distill
        if all_logits:
            avg_eps = np.mean(eps_list)
            min_len = min(len(l) for l in all_logits)
            total_w = sum(all_weights)
            norm_w = [w / total_w for w in all_weights]
            aggregated = sum(w * l[:min_len] for w, l in zip(norm_w, all_logits)).float()

            T = self.config.temperature
            teacher_probs = F.softmax(aggregated / T, dim=1)
            self._distill_to_server(teacher_probs, public_images[:min_len])

            for dev_id, eps in zip(participants, eps_list):
                self.privacy_spent[dev_id] = self.privacy_spent.get(dev_id, 0) + eps

        # Evaluate
        accuracy = 0.0
        loss = 0.0
        if test_loader is not None:
            eval_result = self.evaluate(test_loader)
            accuracy = eval_result["accuracy"]
            loss = eval_result["loss"]

        participation_rate = len(participants) / len(devices) if devices else 0
        self.participation_history.append(participation_rate)

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

    def _pretrain_on_public(self, public_loader):
        """Pre-train server model on public data before FL begins."""
        print(f"  [Pre-training] {self.config.pretrain_epochs} epochs on public data ...")
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
            epoch_loss, n_batches = 0.0, 0
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
        print(f"  [Pre-training] Done.")

    def _distill_to_server(self, teacher_probs, public_images):
        """Pure KL distillation from aggregated teacher probs to server model."""
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

    def get_statistics(self):
        return {
            "n_rounds": len(self.round_history),
            "price_history": self.price_history,
            "participation_history": self.participation_history,
            "best_accuracy": self.get_best_accuracy(),
            "final_accuracy": self.get_final_accuracy(),
            "avg_participation": np.mean(self.participation_history) if self.participation_history else 0
        }


def create_paid_fd(model_name="resnet18", n_classes=100, gamma=10.0,
                   device=None, **config_kwargs):
    """Convenience function to create PAID-FD with specified model."""
    from ..models import get_model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, n_classes=n_classes)
    config = PAIDFDConfig(gamma=gamma, **config_kwargs)
    return PAIDFD(model, config, n_classes, device)
'''

# ============================================================
# FedMD v8 (Standard FD, no persistent models)
# ============================================================
fedmd_v8 = '''\
"""
FedMD v8: Noise-free Federated Distillation baseline.

Standard FD flow (v8): Each round, ALL devices get a fresh copy of the
server model, train locally, upload clean (no noise) logits, server
distills via KL.  No persistent local models, no EMA, no CE anchor.

This is the noise-free FD oracle -- upper bound for FD methods.
The ONLY differences vs PAID-FD are:
  - No Stackelberg game / pricing (all devices participate)
  - No LDP noise on logits
  - No BLUE weighting (equal weight average)
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
    """Configuration for FedMD (noise-free FD)."""
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9
    distill_epochs: int = 5
    distill_lr: float = 0.001
    temperature: float = 3.0
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.1
    clip_bound: float = 5.0
    public_samples: int = 20000


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
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(), lr=self.config.distill_lr
        )
        self._pretrained = False

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        """Execute one round of FedMD v8."""
        self.current_round = round_idx

        if not self._pretrained:
            self._pretrain_on_public(public_loader)
            self._pretrained = True

        # Collect public images
        public_images_list = []
        for data, _ in public_loader:
            public_images_list.append(data)
        public_images = torch.cat(public_images_list, dim=0)
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
            T = self.config.temperature
            teacher_probs = F.softmax(aggregated / T, dim=1)
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
'''

# ============================================================
# Fixed-Epsilon v8 (Standard FD, fixed eps)
# ============================================================
fixed_eps_v8 = '''\
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
'''


# ============================================================
# Write all files
# ============================================================
files = {
    os.path.join(METHODS, "paid_fd.py"): paid_fd_v8,
    os.path.join(METHODS, "fedmd.py"): fedmd_v8,
    os.path.join(METHODS, "fixed_eps.py"): fixed_eps_v8,
}

for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)
    print(f"Wrote {path} ({len(content)} bytes)")

print("\nDone! All 3 methods updated to v8 (standard FD flow).")
