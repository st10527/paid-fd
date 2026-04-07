"""
PAID-FD v9.2: Privacy-Aware Incentive-Driven Federated Distillation
===================================================================

Standard FD flow with self-anchor distillation to prevent noise drift.

Version history:
  v8.0: Fresh copy per round + pure KL → catastrophic drift (ALL configs fail)
  v8.1: Added CE anchor → masks the real problem (CE dominates, γ irrelevant)
  v8.2: Class-conditional denoising → correct fix attempt
  v9.0: Fixed cubic solver (two-root bug), game works but distill still degrades
  v9.1: Swept C,T → degradation ∝ C (noise), not T. Only CE anchor stable.
        Structural dilemma: CE anchor makes γ irrelevant.
  v9.2: Self-anchor distillation (Route 2)
        Before distillation, compute self_logits = server's own predictions.
        Loss = α_sa × KL(noisy_teacher) + (1-α_sa) × KL(self_teacher)
        No ground truth needed. Self-distillation prevents noise drift by
        anchoring to server's own knowledge while still learning from teacher.

Key insight (v9.2): The problem is not noise floor but noise DRIFT — each round
of pure KL on noisy teacher gradually pushes the server away from what it already
knows. Self-anchor provides "gravitational pull" back to server's own knowledge.
Unlike CE anchor, self-anchor doesn't bypass the distillation pathway, so γ
(which controls teacher signal quality) remains relevant.

Pipeline per round:
  1. Stackelberg game → price p*, device decisions
  2. Devices: fresh copy → local train → clip logits → LDP noise → upload
  3. Server: BLUE aggregation (ε²-weighted)
  4. Server: softmax(aggregated / T) → teacher probs
  5. Server: Self-anchor KL distillation with fresh SGD
     - Compute self_logits (server's current knowledge) before training
     - loss = α_sa × KL(teacher) + (1-α_sa) × KL(self) [both scaled by T²]
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
    """Configuration for PAID-FD v9.2 (Standard FD + Self-Anchor).

    v9.2 pipeline: fresh copy → local train → clip → LDP → BLUE →
    softmax(T) → self-anchor KL distill (fresh SGD).
    """
    # Game parameters
    gamma: float = 10.0
    delta: float = 0.01
    budget: float = float('inf')

    # Local training (per round -- fresh copy each round)
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9

    # Distillation
    distill_epochs: int = 1
    distill_lr: float = 0.001
    temperature: float = 3.0
    ce_anchor_alpha: float = 0.0   # α_ce: 0=no CE anchor, >0 adds CE loss
    self_anchor_alpha: float = 0.0 # α_sa: weight on KL(teacher) vs KL(self)
                                    # 0 = pure KL(teacher), 0.5 = equal mix
                                    # loss = α_sa * KL(teacher) + (1-α_sa) * KL(self)
                                    # NOTE: when α_sa=0, falls back to pure teacher (v9.1 behavior)

    # Pre-training on public data
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.1

    # Privacy
    clip_bound: float = 2.0

    # Public data
    public_samples: int = 20000

    # Ablation flags
    use_blue: bool = True        # BLUE (eps^2-weighted) vs equal weights
    use_ldp: bool = True         # Per-device LDP noise vs clean logits (oracle)
    use_denoising: bool = True   # Class-conditional denoising (v8.2 key feature)


class PAIDFD(FederatedMethod):
    """PAID-FD v9.2: Federated Distillation with Game + Self-Anchor.

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
      6. Server: Self-anchor KL distillation (fresh SGD)
         - Before training: forward pass to get self_logits (server's knowledge)
         - loss = α_sa × KL(teacher) + (1-α_sa) × KL(self)
         - Self-anchor prevents noise drift without using ground truth
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
        # v8.1: No persistent optimizer -- fresh SGD each round to prevent
        # Adam state accumulation that caused monotonic degradation in v8.0
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

        # Collect public images AND labels (labels needed for CE anchor)
        public_images_list = []
        public_labels_list = []
        for data, labels in public_loader:
            public_images_list.append(data)
            public_labels_list.append(labels)
        public_images = torch.cat(public_images_list, dim=0)
        public_labels = torch.cat(public_labels_list, dim=0)
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
        distill_diag = {}  # v9.2 diagnostics
        if all_logits:
            avg_eps = np.mean(eps_list)
            min_len = min(len(l) for l in all_logits)
            total_w = sum(all_weights)
            norm_w = [w / total_w for w in all_weights]
            aggregated = sum(w * l[:min_len] for w, l in zip(norm_w, all_logits)).float()

            # v8.2: Class-conditional denoising (post-processing, preserves ε-LDP)
            denoised = self._denoise_logits(aggregated, public_labels[:min_len])

            T = self.config.temperature
            teacher_probs = F.softmax(denoised / T, dim=1)

            # v9.2: Measure pre-distillation accuracy on test set
            pre_distill_acc = None
            if test_loader is not None:
                pre_eval = self.evaluate(test_loader)
                pre_distill_acc = pre_eval["accuracy"]

            distill_diag = self._distill_to_server(
                teacher_probs, public_images[:min_len],
                public_labels[:min_len])

            # v9.2: Measure post-distillation accuracy on test set
            if test_loader is not None and pre_distill_acc is not None:
                post_eval = self.evaluate(test_loader)
                distill_diag["pre_distill_acc"] = pre_distill_acc
                distill_diag["post_distill_acc"] = post_eval["accuracy"]
                distill_diag["distill_delta"] = post_eval["accuracy"] - pre_distill_acc

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

        extra_dict = {
            "price": price,
            "avg_s": game_result["avg_s"],
            "avg_eps": game_result["avg_eps"],
            "server_utility": game_result["server_utility"],
            "total_quality": game_result["total_quality"],
            "max_privacy_spent": max(self.privacy_spent.values()) if self.privacy_spent else 0,
            "avg_privacy_spent": float(np.mean(list(self.privacy_spent.values()))) if self.privacy_spent else 0,
        }
        extra_dict.update(distill_diag)  # v9.2 diagnostics

        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            n_participants=len(participants),
            energy=total_energy,
            extra=extra_dict
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

    def _denoise_logits(self, aggregated, public_labels):
        """Class-conditional denoising of aggregated logits.

        For each class c, replace per-sample logits with the class-conditional
        mean. With n_c samples per class, this reduces LDP noise std by √n_c.

        For CIFAR-100 with 20K public data: n_c ≈ 200 → noise ÷ √200 ≈ 14×.
        SNR improves from ~0.8 (unusable) to ~11 (clean enough for KL distill).

        Privacy: Pure server-side post-processing → ε-LDP preserved by
        post-processing immunity theorem.
        """
        if not self.config.use_denoising:
            return aggregated

        denoised = aggregated.clone()
        for c in range(self.n_classes):
            mask = (public_labels == c)
            n_c = mask.sum().item()
            if n_c > 1:
                class_mean = aggregated[mask].mean(dim=0)  # [n_classes] vector
                denoised[mask] = class_mean.unsqueeze(0).expand(n_c, -1)
        return denoised

    def _distill_to_server(self, teacher_probs, public_images, public_labels=None):
        """v9.2: KL distillation with self-anchor (optional CE anchor).

        Self-anchor (α_sa > 0):
          Before training loop, compute self_logits = server's current predictions.
          loss = α_sa * KL(student || teacher) + (1-α_sa) * KL(student || self)
          Both KL terms scaled by T².

        This prevents noise drift: the server "remembers" its own knowledge and
        only partially trusts the noisy teacher. Unlike CE anchor, self-anchor
        doesn't bypass the distillation pathway → γ remains relevant.

        Fallback: α_sa=0 → pure KL(teacher) (v9.1 behavior).

        Returns:
            dict with diagnostic signals:
              - kl_teacher_self: KL(teacher || self) — disagreement between anchors
              - mean_loss_teacher: average KL loss on noisy teacher
              - mean_loss_self: average KL loss on self-anchor (if used)
        """
        T = self.config.temperature
        alpha_ce = self.config.ce_anchor_alpha
        alpha_sa = self.config.self_anchor_alpha
        n_target = min(len(teacher_probs), len(public_images))
        diagnostics = {}

        # --- Self-anchor: compute server's current predictions BEFORE training ---
        self_teacher_probs = None
        if alpha_sa > 0:
            self.server_model.eval()
            self_logit_chunks = []
            bs = 512
            with torch.no_grad():
                for start in range(0, n_target, bs):
                    end = min(start + bs, n_target)
                    batch = public_images[start:end].to(self.device)
                    logits = self.server_model(batch)
                    self_logit_chunks.append(logits.cpu())
            self_logits_all = torch.cat(self_logit_chunks, dim=0)
            # Convert to soft probabilities at temperature T (same as teacher)
            self_teacher_probs = F.softmax(self_logits_all / T, dim=1)

            # v9.2 diagnostic: KL divergence between teacher and self-teacher
            # Large KL = teacher is very different from server's knowledge (noisy)
            # Small KL = self-anchor adds little new info
            with torch.no_grad():
                kl_ts = F.kl_div(
                    torch.log(self_teacher_probs + 1e-8),
                    teacher_probs[:n_target],
                    reduction='batchmean'
                ).item()
                diagnostics["kl_teacher_self"] = kl_ts

        # --- Training loop ---
        self.server_model.train()
        # v8.2+: Fresh optimizer each round (no Adam state leak from v8.0)
        optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=self.config.distill_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        ce_criterion = nn.CrossEntropyLoss()
        batch_size = 256
        loss_teacher_accum, loss_self_accum, n_batches = 0.0, 0.0, 0
        for epoch in range(self.config.distill_epochs):
            perm = torch.randperm(n_target)
            for start in range(0, n_target, batch_size):
                end = min(start + batch_size, n_target)
                idx = perm[start:end]
                data = augment(public_images[idx]).to(self.device)
                target_probs = teacher_probs[idx].to(self.device)
                student_logits = self.server_model(data)

                # KL distillation loss (noisy teacher)
                loss_kl_teacher = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target_probs,
                    reduction='batchmean'
                ) * (T * T)

                # Self-anchor KL loss (server's own previous knowledge)
                if alpha_sa > 0 and self_teacher_probs is not None:
                    self_probs = self_teacher_probs[idx].to(self.device)
                    loss_kl_self = F.kl_div(
                        F.log_softmax(student_logits / T, dim=1),
                        self_probs,
                        reduction='batchmean'
                    ) * (T * T)
                    loss_distill = alpha_sa * loss_kl_teacher + (1.0 - alpha_sa) * loss_kl_self
                    loss_self_accum += loss_kl_self.item()
                else:
                    loss_distill = loss_kl_teacher

                loss_teacher_accum += loss_kl_teacher.item()
                n_batches += 1

                # CE anchor loss (ground truth on public data) — can combine with self-anchor
                if alpha_ce > 0 and public_labels is not None:
                    labels = public_labels[idx].to(self.device)
                    loss_ce = ce_criterion(student_logits, labels)
                    loss = alpha_ce * loss_ce + (1.0 - alpha_ce) * loss_distill
                else:
                    loss = loss_distill

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()

        # v9.2 diagnostics: mean losses
        if n_batches > 0:
            diagnostics["mean_loss_teacher"] = loss_teacher_accum / n_batches
            if alpha_sa > 0:
                diagnostics["mean_loss_self"] = loss_self_accum / n_batches
        return diagnostics

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
