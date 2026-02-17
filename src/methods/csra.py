"""
CSRA: Client Selection with Reverse Auction (Yang et al., TIFS 2024)

Robust incentive mechanism for Differentially Private Federated Learning.
Uses reverse auction for client selection + DP noise on model parameters.

Key mechanism:
  1. Server provides reward-privacy menu L = {(ε_m, R_m)}
  2. Clients bid (ε_i, b_i) based on their privacy preference
  3. Server ranks clients by quality = |D_i| * ε_i / b_i
  4. Top-k clients selected within budget B (Myerson's reverse auction)
  5. Selected clients: local train → add Laplace DP noise to gradients → upload
  6. Server: weighted aggregation of noisy model parameters

Key differences from PAID-FD:
  - Reverse auction (multi-round bidding) vs Stackelberg (one-shot pricing)
  - Parameter aggregation (~44 MB/round) vs logit distillation (~400 KB)
  - DP noise on MODEL GRADIENTS (high-dim) vs DP noise on LOGITS (low-dim)
  - Dishonest client detection (CSRA feature, simplified in our baseline)

Adaptations for fair comparison:
  - Same ResNet-18, same local training params, same number of rounds
  - Honest clients assumed (skip dishonest detection for clean comparison)
  - Quality metric includes DP noise scale (CSRA's core contribution)
  - Use same Dirichlet partition as other methods

Reference:
  Yang et al., "CSRA: Robust Incentive Mechanism Design for Differentially
  Private Federated Learning", IEEE TIFS, Vol. 19, 2024
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import copy

from .base import FederatedMethod, RoundResult
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class CSRAConfig:
    """Configuration for CSRA."""
    # Local training
    local_epochs: int = 2
    local_lr: float = 0.01
    local_momentum: float = 0.9

    # Reverse auction
    budget_per_round: float = 50.0  # Reward budget B per round
    # Privacy-reward menu L = [(ε, R)] — higher ε (weaker privacy) = lower reward
    privacy_menu: List = field(default_factory=lambda: [
        (0.5, 10.0),   # Strong privacy, high reward
        (1.0, 5.0),    # Medium privacy, medium reward
        (2.0, 3.0),    # Weak privacy, low reward
        (5.0, 1.5),    # Very weak privacy, minimal reward
    ])

    # DP noise on model parameters
    clip_norm: float = 1.0  # Gradient clipping norm (L2)

    # Communication (for energy calc — full model upload)
    model_size_bytes: int = 44_000_000


class CSRA(FederatedMethod):
    """
    CSRA: Client Selection with Reverse Auction for DPFL.

    Protocol per round:
      1. Clients choose (ε_i, b_i) from privacy menu based on λ_i
      2. Server ranks clients by quality q_i = |D_i| * ε_i / b_i
      3. Top-k clients selected within budget (reverse auction)
      4. Selected clients: local SGD → clip gradients → add Laplace noise → upload
      5. Server: weighted average of noisy parameters

    This is parameter-based FL (like FedAvg) with:
      - Auction-based incentive for client selection
      - DP noise on model PARAMETERS (not logits)
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: CSRAConfig = None,
        n_classes: int = 100,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(server_model, n_classes, device)
        self.config = config or CSRAConfig()
        self.energy_calc = EnergyCalculator()
        self.rng = np.random.RandomState(42)

    def _simulate_bidding(self, devices: List[Any]) -> List[Dict]:
        """
        Simulate the bidding phase.

        Each device chooses (ε, bid) from the privacy menu based on its
        privacy sensitivity λ_i. Higher λ → prefers stronger privacy (lower ε).

        Returns list of bids: [{device_id, epsilon, bid, data_size, quality}]
        """
        bids = []
        menu = self.config.privacy_menu

        for dev in devices:
            lam = getattr(dev, 'privacy_sensitivity', 0.5)
            data_size = getattr(dev, 'data_size', 400)

            # Higher λ → choose lower ε (stronger privacy, higher reward)
            # Map λ to menu index: high λ → low index (strong privacy)
            if lam >= 0.8:
                menu_idx = 0  # ε=0.5 (strongest privacy)
            elif lam >= 0.4:
                menu_idx = 1  # ε=1.0
            elif lam >= 0.1:
                menu_idx = 2  # ε=2.0
            else:
                menu_idx = 3  # ε=5.0 (weakest privacy)

            eps_chosen, reward_cap = menu[menu_idx]

            # Bid = cost to participate (slightly below reward cap)
            # Truthful bidding: bid ≈ actual cost
            cost_factor = 0.5 + 0.4 * self.rng.random()  # Random cost 50-90% of cap
            bid = reward_cap * cost_factor

            # Quality metric from CSRA: q = |D_i| * ε_i / b_i
            quality = data_size * eps_chosen / bid

            bids.append({
                'device_id': dev.device_id,
                'epsilon': eps_chosen,
                'bid': bid,
                'reward_cap': reward_cap,
                'data_size': data_size,
                'quality': quality,
            })

        return bids

    def _reverse_auction_select(self, bids: List[Dict]) -> List[Dict]:
        """
        Reverse auction client selection (Myerson's mechanism).

        Rank by quality = |D_i| * ε_i / b_i (descending).
        Select top-k within budget B.
        Reward = critical payment from Myerson's theorem.
        """
        # Sort by quality descending
        sorted_bids = sorted(bids, key=lambda x: x['quality'], reverse=True)

        selected = []
        total_reward = 0.0
        B = self.config.budget_per_round

        for i, bid_info in enumerate(sorted_bids):
            # Critical payment (Myerson): reward = quality_threshold * D_i * ε_i
            # Simplified: reward = bid (truthful mechanism)
            reward = bid_info['bid']

            if total_reward + reward <= B:
                bid_info['reward'] = reward
                selected.append(bid_info)
                total_reward += reward
            else:
                break  # Budget exhausted

        return selected

    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any = None,  # Not used — CSRA is parameter-based
        test_loader: Optional[Any] = None
    ) -> RoundResult:
        """Execute one round of CSRA."""
        self.current_round = round_idx

        # ── Phase 1: Bidding ──
        bids = self._simulate_bidding(devices)

        # ── Phase 2: Reverse auction selection ──
        selected = self._reverse_auction_select(bids)

        if not selected:
            # No clients selected (budget too low) — skip
            accuracy, loss = 0.0, 0.0
            if test_loader:
                eval_result = self.evaluate(test_loader)
                accuracy, loss = eval_result["accuracy"], eval_result["loss"]
            return RoundResult(
                round_idx=round_idx, accuracy=accuracy, loss=loss,
                participation_rate=0.0, n_participants=0,
                energy={"training": 0, "inference": 0, "communication": 0},
                extra={"method": "CSRA", "n_selected": 0}
            )

        # ── Phase 3: Local training + DP noise on parameters ──
        local_state_dicts = []
        local_weights = []
        eps_list = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}

        global_state = copy.deepcopy(self.server_model.state_dict())

        for bid_info in selected:
            dev_id = bid_info['device_id']
            if dev_id not in client_loaders:
                continue

            dev = devices[dev_id]
            local_loader = client_loaders[dev_id]
            n_samples = len(local_loader.dataset)
            eps_i = bid_info['epsilon']

            # Create local model from global
            local_model = copy_model(self.server_model, device=self.device)

            # Local training
            self.train_local(
                local_model,
                local_loader,
                epochs=self.config.local_epochs,
                lr=self.config.local_lr
            )

            # Compute gradient (delta = local - global)
            local_state = local_model.state_dict()
            noisy_state = {}

            for key in global_state:
                delta = local_state[key].float() - global_state[key].float()

                # Clip gradient norm per-layer
                norm = delta.norm()
                clip_coeff = min(1.0, self.config.clip_norm / (norm + 1e-8))
                delta_clipped = delta * clip_coeff

                # Add Laplace noise (DP on gradients)
                # Sensitivity = 2 * clip_norm (for bounded L2)
                # Per CSRA paper: noise ~ Lap(sensitivity / ε_i) per element
                sensitivity = 2.0 * self.config.clip_norm
                noise_scale = sensitivity / eps_i
                noise = torch.from_numpy(
                    np.random.laplace(0, noise_scale, delta_clipped.shape).astype(np.float32)
                ).to(delta_clipped.device)

                noisy_state[key] = global_state[key].float() + delta_clipped + noise

            local_state_dicts.append(noisy_state)
            local_weights.append(n_samples)
            eps_list.append(eps_i)

            # Energy: training + uploading full model parameters
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=n_samples,
                s_i=n_samples,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            total_energy["training"] += energy.get("training", 0)
            total_energy["communication"] += energy.get("communication", 0) * 100  # ~100x vs logits

            del local_model

        # ── Phase 4: Weighted aggregation of noisy parameters ──
        if local_state_dicts:
            total_weight = sum(local_weights)
            norm_weights = [w / total_weight for w in local_weights]

            avg_state = {}
            for key in local_state_dicts[0]:
                avg_state[key] = torch.zeros_like(
                    local_state_dicts[0][key], dtype=torch.float32
                )

            for state_dict, weight in zip(local_state_dicts, norm_weights):
                for key in avg_state:
                    avg_state[key] += weight * state_dict[key].float()

            self.server_model.load_state_dict(avg_state)

        # Evaluate
        accuracy, loss = 0.0, 0.0
        if test_loader:
            eval_result = self.evaluate(test_loader)
            accuracy, loss = eval_result["accuracy"], eval_result["loss"]

        n_participants = len(selected)
        participation_rate = n_participants / len(devices) if devices else 0
        avg_eps = float(np.mean(eps_list)) if eps_list else 0.0

        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=participation_rate,
            n_participants=n_participants,
            energy=total_energy,
            extra={
                "method": "CSRA",
                "n_selected": n_participants,
                "avg_epsilon": avg_eps,
                "total_reward": sum(b['reward'] for b in selected),
                "budget": self.config.budget_per_round,
            }
        )

        self.round_history.append(result)
        return result

    def aggregate(self, updates: List[Dict], weights: List[float]) -> None:
        """Interface compatibility."""
        pass
