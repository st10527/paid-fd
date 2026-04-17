"""
PAID-FD v10.1: Persistent Local Models + Persistent Adam + Solver Fix
======================================================================

v10.0 -> v10.1: Restored persistent Adam distillation optimizer.

Version history:
  v7:   Persistent local models + EMA + mixed loss + persistent Adam -> 60%
        But old solver (eps*~0.5)
  v8.x: Removed persistent -> "standard FD" -> self-referential problem
        Device logits ~ server + noise -> distillation learns nothing new
  v9.x: Solver fix works (eps*~3) but without persistent models,
        no real knowledge to transfer -> all configs degrade
  v10.0: Persistent models + solver fix + fresh SGD -> 47% (ceiling)
        CE loss stagnated at ~2.08 for 100 rounds: fresh SGD with 78
        batches/round couldn't accumulate enough gradient information.
  v10.1: Restored persistent Adam from v7. The "Adam state leak" in v8
        was actually self-referential learning from fresh models, not
        Adam's state. With persistent local models, Adam's accumulated
        m/v statistics denoise the gradient signal across rounds.

Key insight: In standard FD with fresh copies, devices return w(t)+noise
(self-referential). With persistent models, device i after 50 rounds has
trained 250 epochs on D_i -- it has REAL local knowledge different from
server. Distillation transfers ensemble knowledge, not noise.

Theory: Persistent models don't change the privacy analysis (each round's
logits are still clipped and noised independently). The game mechanism
controls the quality-privacy tradeoff of these richer local models.

Pipeline per round:
  1. Stackelberg game -> price p*, device decisions
  2. Devices: persistent local model -> train on D_i -> clip logits -> LDP -> upload
  3. Server: BLUE aggregation (eps^2-weighted)
  4. Server: EMA logit buffer (noise averaging across rounds)
  5. Server: softmax(buffer / T) -> teacher probs
  6. Server: Mixed loss distillation (alpha*KL + (1-alpha)*CE) with persistent Adam
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
    """Configuration for PAID-FD v10 (Persistent Models + Solver Fix).

    v10 pipeline: persistent local model -> local train -> clip -> LDP ->
    BLUE -> EMA buffer -> softmax(T) -> mixed loss distill (fresh SGD).
    """
    # Game parameters
    gamma: float = 10.0
    delta: float = 0.01
    budget: float = float('inf')

    # Local training (persistent models -- accumulate across rounds)
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9

    # Distillation
    distill_epochs: int = 1
    distill_lr: float = 0.001
    distill_alpha: float = 0.7  # alpha: weight on KL vs CE. alpha*KL + (1-alpha)*CE
    temperature: float = 3.0

    # EMA logit buffer
    ema_momentum: float = 0.9   # EMA for cross-round noise averaging
                                 # effective window ~ 1/(1-0.9) = 10 rounds

    # Pre-training on public data
    pretrain_epochs: int = 10
    pretrain_lr: float = 0.1

    # Privacy
    clip_bound: float = 2.0

    # Public data
    public_samples: int = 20000

    # Ablation flags
    use_blue: bool = True        # BLUE (eps^2-weighted) vs equal weights
    use_ema: bool = True         # EMA logit buffer vs single-round
    use_mixed_loss: bool = True  # Mixed loss vs pure KL
    use_ldp: bool = True         # LDP noise vs clean logits (oracle)
    persistent_local_models: bool = True  # Keep local models across rounds vs fresh copy each round
    use_denoising: bool = False  # Class-conditional denoising (v8.2, optional)

    # Fair comparison mode: bypass Stackelberg game, use fixed epsilon
    # When > 0, ALL devices participate with this fixed epsilon.
    # Pipeline (persistent models, EMA, mixed loss, persistent Adam) stays identical.
    # This isolates the game's contribution from the pipeline's contribution.
    fixed_epsilon: float = 0.0

    # Legacy compat (v9.x params ignored but accepted)
    ce_anchor_alpha: float = 0.0
    self_anchor_alpha: float = 0.0


class PAIDFD(FederatedMethod):
    """PAID-FD v10.1: Federated Distillation with Persistent Models + Game.

    Key difference from v8/v9: devices maintain persistent local models
    across rounds. After R rounds with E local epochs each, device i has
    trained R*E epochs on its private data D_i. This creates genuine
    local knowledge that differs from the server model.

    v10.1 fix: Restored persistent Adam for distillation (v7's key ingredient).
    Fresh SGD in v10.0 caused CE loss to stagnate at ~2.08 (47% ceiling).

    Protocol per round:
      1. Server computes optimal price p* via Stackelberg game
      2. Each device decides (participate?, s_i*, eps_i*)
      3. Participating devices:
         a. Continue training persistent local model on D_i
         b. Compute logits on public data, clip to [-C, C]
         c. Add per-device LDP noise: Lap(0, 2C/eps_i)
         d. Upload noisy logits
      4. Server: BLUE-weighted average of noisy logits
      5. Server: Update EMA logit buffer (noise averaging)
      6. Server: softmax(buffer / T) -> teacher probs
      7. Server: Mixed loss distillation (alpha*KL + (1-alpha)*CE, persistent Adam)
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

        # Persistent local models: each device keeps its own model across rounds
        self.local_models = {}      # dev_id -> nn.Module
        self.local_optimizers = {}  # dev_id -> optimizer

        # EMA logit buffer: accumulates aggregated logits across rounds
        self.logit_buffer = None

        # Persistent distillation optimizer (maintains momentum across rounds)
        # Adam's accumulated m/v statistics effectively denoise the gradient
        # signal over many rounds — 78 batches/round × 100 rounds = 7800 updates.
        # v8's "Adam state leak" was actually self-referential learning from
        # fresh models, not Adam's state. With persistent local models, this is safe.
        self.distill_optimizer = torch.optim.Adam(
            self.server_model.parameters(),
            lr=self.config.distill_lr
        )

        self._pretrained = False
        self.privacy_spent = {}
        self.cumulative_payment = 0.0
        self.price_history = []
        self.participation_history = []

    def run_round(self, round_idx, devices, client_loaders, public_loader,
                  test_loader=None):
        """Execute one round of PAID-FD v10."""
        self.current_round = round_idx

        if not self._pretrained:
            self._pretrain_on_public(public_loader)
            self._pretrained = True

        # Stage 1: Stackelberg game (or fixed-epsilon bypass)
        if self.config.fixed_epsilon > 0:
            # Fair Fixed-ε mode: skip game, all devices participate
            from src.game.stackelberg import DeviceDecision
            fixed_eps = self.config.fixed_epsilon
            decisions = [
                DeviceDecision(
                    device_id=i, participates=True,
                    s_star=100.0, eps_star=fixed_eps,
                    quality=1.0, utility=0.0
                ) for i in range(len(devices))
            ]
            price = 0.0
            game_result = {
                "price": 0.0, "decisions": decisions,
                "avg_s": 100.0, "avg_eps": fixed_eps,
                "server_utility": 0.0, "total_quality": len(devices),
                "total_payment": 0.0,
            }
        else:
            game_result = self.solver.solve(devices)
            price = game_result["price"]
            decisions = game_result["decisions"]
        self.price_history.append(price)

        # Collect public images and labels
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

            # v10: Persistent local model (initialized from server on first use)
            # Ablation: reset each round if persistent_local_models=False
            if not self.config.persistent_local_models and dev_id in self.local_models:
                del self.local_models[dev_id]
                del self.local_optimizers[dev_id]
            if dev_id not in self.local_models:
                self.local_models[dev_id] = copy_model(
                    self.server_model, device=self.device)
                self.local_optimizers[dev_id] = torch.optim.SGD(
                    self.local_models[dev_id].parameters(),
                    lr=self.config.local_lr,
                    momentum=self.config.local_momentum,
                    weight_decay=5e-4
                )

            local_model = self.local_models[dev_id]
            local_optimizer = self.local_optimizers[dev_id]

            # Local training on private data (persistent model continues)
            local_model.train()
            criterion = nn.CrossEntropyLoss()
            for epoch in range(self.config.local_epochs):
                for data, target in local_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    local_optimizer.zero_grad()
                    loss = criterion(local_model(data), target)
                    loss.backward()
                    local_optimizer.step()

            # Compute clipped logits on public data
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

            # Per-device LDP noise
            device_eps = decision.eps_star
            if self.config.use_ldp:
                noise_scale = 2.0 * C / device_eps
                noise = np.random.laplace(
                    0, noise_scale, device_logits.shape).astype(np.float32)
                noisy_logits = device_logits + torch.from_numpy(noise)
            else:
                noisy_logits = device_logits

            all_logits.append(noisy_logits)
            if self.config.use_blue:
                all_weights.append(device_eps ** 2)
            else:
                all_weights.append(1.0)
            eps_list.append(device_eps)

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

        # Stage 3: Aggregate -> EMA buffer -> distill
        distill_diag = {}
        if all_logits:
            avg_eps = np.mean(eps_list)
            min_len = min(len(l) for l in all_logits)
            total_w = sum(all_weights)
            norm_w = [w / total_w for w in all_weights]
            aggregated = sum(
                w * l[:min_len] for w, l in zip(norm_w, all_logits)).float()

            # Optional denoising (v8.2 feature, off by default in v10)
            if self.config.use_denoising:
                aggregated = self._denoise_logits(aggregated, public_labels[:min_len])

            # EMA logit buffer: noise averaging across rounds
            if self.config.use_ema:
                ema = self.config.ema_momentum
                if self.logit_buffer is None:
                    self.logit_buffer = aggregated.clone()
                else:
                    buf_len = min(len(self.logit_buffer), min_len)
                    self.logit_buffer[:buf_len] = (
                        ema * self.logit_buffer[:buf_len]
                        + (1 - ema) * aggregated[:buf_len]
                    )
                distill_source = self.logit_buffer
            else:
                distill_source = aggregated

            T = self.config.temperature
            buf_len = min(len(distill_source), min_len)
            teacher_probs = F.softmax(distill_source[:buf_len] / T, dim=1)

            # v10 diagnostic: pre-distill accuracy
            pre_distill_acc = None
            if test_loader is not None:
                pre_eval = self.evaluate(test_loader)
                pre_distill_acc = pre_eval["accuracy"]

            # Distill with mixed loss
            if self.config.use_mixed_loss:
                distill_diag = self._distill_to_server(
                    teacher_probs, public_images[:buf_len],
                    public_labels[:buf_len])
            else:
                distill_diag = self._distill_to_server(
                    teacher_probs, public_images[:buf_len],
                    public_labels=None)

            # v10 diagnostic: post-distill accuracy
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

        # Efficiency metrics (v10: the game's value is efficiency, not accuracy)
        round_payment = game_result.get("total_payment", price * game_result["total_quality"])
        self.cumulative_payment += round_payment
        round_eps_total = sum(eps_list) if eps_list else 0  # total ε spent this round
        eps_std = float(np.std(eps_list)) if len(eps_list) > 1 else 0.0

        extra_dict = {
            # Game equilibrium
            "price": price,
            "avg_s": game_result["avg_s"],
            "avg_eps": game_result["avg_eps"],
            "eps_std": eps_std,
            "server_utility": game_result["server_utility"],
            "total_quality": game_result["total_quality"],
            # Cost accounting
            "round_payment": round_payment,
            "cumulative_payment": self.cumulative_payment,
            # Privacy accounting
            "round_eps_total": round_eps_total,
            "max_privacy_spent": max(self.privacy_spent.values()) if self.privacy_spent else 0,
            "avg_privacy_spent": float(np.mean(list(self.privacy_spent.values()))) if self.privacy_spent else 0,
            "n_local_models": len(self.local_models),
        }
        extra_dict.update(distill_diag)

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
        """Class-conditional denoising (optional, from v8.2)."""
        denoised = aggregated.clone()
        for c in range(self.n_classes):
            mask = (public_labels == c)
            n_c = mask.sum().item()
            if n_c > 1:
                class_mean = aggregated[mask].mean(dim=0)
                denoised[mask] = class_mean.unsqueeze(0).expand(n_c, -1)
        return denoised

    def _distill_to_server(self, teacher_probs, public_images, public_labels=None):
        """v10.1: Mixed loss distillation with persistent Adam.

        loss = alpha * KL(student || teacher) * T^2 + (1-alpha) * CE(student, labels)

        Persistent Adam across rounds: accumulated m/v statistics act as a
        gradient denoiser. With only 78 batches/round (20k/256), fresh SGD
        can't make meaningful CE progress -- CE loss stagnated at ~2.08 for
        100 rounds in v10.0. Persistent Adam gave v7 60% accuracy.

        The v8 "Adam state leak" was caused by self-referential learning
        (fresh models), not by Adam's state. With persistent local models
        providing genuine knowledge, optimizer persistence is safe.

        Returns:
            dict with diagnostic signals
        """
        T = self.config.temperature
        alpha = self.config.distill_alpha
        n_target = min(len(teacher_probs), len(public_images))
        diagnostics = {}

        self.server_model.train()
        optimizer = self.distill_optimizer  # persistent Adam
        augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ])
        ce_criterion = nn.CrossEntropyLoss()
        batch_size = 256
        loss_kl_accum, loss_ce_accum, n_batches = 0.0, 0.0, 0

        for epoch in range(self.config.distill_epochs):
            perm = torch.randperm(n_target)
            for start in range(0, n_target, batch_size):
                end = min(start + batch_size, n_target)
                idx = perm[start:end]
                data = augment(public_images[idx]).to(self.device)
                target_probs = teacher_probs[idx].to(self.device)
                student_logits = self.server_model(data)

                # KL distillation loss (ensemble teacher)
                loss_kl = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target_probs,
                    reduction='batchmean'
                ) * (T * T)

                # CE anchor loss (ground truth)
                if public_labels is not None and alpha < 1.0:
                    labels = public_labels[idx].to(self.device)
                    loss_ce = ce_criterion(student_logits, labels)
                    loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
                    loss_ce_accum += loss_ce.item()
                else:
                    loss = loss_kl

                loss_kl_accum += loss_kl.item()
                n_batches += 1

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), 5.0)
                optimizer.step()

        if n_batches > 0:
            diagnostics["mean_loss_kl"] = loss_kl_accum / n_batches
            diagnostics["mean_loss_ce"] = loss_ce_accum / n_batches
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
            "avg_participation": np.mean(self.participation_history) if self.participation_history else 0,
            "cumulative_payment": self.cumulative_payment,
            "n_local_models": len(self.local_models),
        }


def create_paid_fd(model_name="resnet18", n_classes=100, gamma=10.0,
                   device=None, **config_kwargs):
    """Convenience function to create PAID-FD with specified model."""
    from ..models import get_model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, num_classes=n_classes)
    config = PAIDFDConfig(gamma=gamma, **config_kwargs)
    return PAIDFD(model, config, n_classes, device)
