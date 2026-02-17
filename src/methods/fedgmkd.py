"""
FedGMKD: Federated Learning with Gaussian Mixture Knowledge Distillation
(Zhang et al., 2024)

Prototype-based personalized FL using:
  1. CKF (Cluster Knowledge Fusion): GMM clustering per class → prototypes
  2. DAT (Discrepancy-Aware Aggregation): KL-div based client weighting

Key mechanism per round:
  1. Server sends global model W^r and global prototypes G^r
  2. Clients train locally with composite loss:
     L = CE + alpha * L_feature_align + beta * L_knowledge_align
  3. Clients extract class prototypes via feature extraction + GMM
  4. Clients send model params + prototypes + discrepancy to server
  5. Server aggregates both params and prototypes via DAT weights

Key differences from PAID-FD:
  - Parameter averaging + prototype exchange (NOT logit distillation)
  - No public dataset needed (prototypes replace public data)
  - GMM-based multi-modal class representation
  - Discrepancy-aware aggregation (not equal/size-weighted)
  - No explicit privacy mechanism (no DP noise)

Communication per round: ~44 MB (parameters) + ~400 KB (prototypes)

Adaptations for fair comparison:
  - Same ResNet-18, same local training params, same rounds
  - Same Dirichlet partition settings
  - Use K=2 GMM components per class (paper default)

Reference:
  Zhang et al., "FedGMKD: An Efficient Prototype Federated Learning Framework
  through Knowledge Distillation and Discrepancy-Aware Aggregation", 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import copy
from collections import defaultdict

from .base import FederatedMethod, RoundResult
from ..devices.energy import EnergyCalculator
from ..models.utils import copy_model


@dataclass
class FedGMKDConfig:
    """Configuration for FedGMKD."""
    # Local training
    local_epochs: int = 2
    local_lr: float = 0.01
    local_momentum: float = 0.9

    # Knowledge distillation weights (Eq.13 in paper)
    alpha: float = 0.5   # Feature alignment weight
    beta: float = 0.5    # Knowledge alignment weight
    temperature: float = 3.0  # KD temperature

    # GMM prototype extraction
    n_gmm_components: int = 2  # K clusters per class
    min_samples_per_class: int = 5  # Skip GMM if too few samples

    # DAT aggregation
    dat_temperature: float = 1.0  # Temperature for exp(-d/tau) weighting


class FedGMKD(FederatedMethod):
    """
    FedGMKD: Prototype-based FL with GMM clustering and discrepancy-aware
    aggregation.

    Protocol:
      1. Server → clients: global model W^r + global prototypes G^r
      2. Clients: train with composite loss (CE + feature alignment + KD)
      3. Clients: extract prototypes via GMM per class
      4. Clients → server: noisy params + prototypes + discrepancy
      5. Server: DAT-weighted aggregation of params and prototypes
    """

    def __init__(
        self,
        server_model: nn.Module,
        config: FedGMKDConfig = None,
        n_classes: int = 100,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(server_model, n_classes, device)
        self.config = config or FedGMKDConfig()
        self.energy_calc = EnergyCalculator()

        # Feature dimension from model
        self.feat_dim = 512  # ResNet-18 feature dim before FC

        # Global prototypes: {class_id: (feature_vector, soft_prediction)}
        # Initialized empty; populated after first round
        self.global_prototypes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _extract_features_and_labels(
        self,
        model: nn.Module,
        data_loader: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features, logits, and labels from a data loader.

        Returns:
            (features [N, feat_dim], logits [N, C], labels [N])
        """
        model.eval()
        all_features = []
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                features = model.get_features(data)  # [B, feat_dim]
                logits = model.fc(features)           # [B, C]
                all_features.append(features.cpu())
                all_logits.append(logits.cpu())
                all_labels.append(target)

        features = torch.cat(all_features, dim=0)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return features, logits, labels

    def _compute_prototypes_gmm(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute per-class prototypes using GMM clustering (CKF).

        For each class c:
          1. Gather features of class c
          2. Fit GMM with K components
          3. Prototype feature = weighted mean of cluster centers
          4. Prototype prediction = softmax of avg logits (soft label)

        Returns:
            {class_id: (proto_feature [feat_dim], proto_pred [C])}
        """
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            # Fallback to simple mean if sklearn not available
            return self._compute_prototypes_mean(features, logits, labels)

        prototypes = {}
        K = self.config.n_gmm_components

        for c in range(self.n_classes):
            mask = (labels == c)
            n_c = mask.sum().item()

            if n_c == 0:
                continue

            feat_c = features[mask].numpy()  # [n_c, feat_dim]
            logit_c = logits[mask]            # [n_c, C]

            if n_c < self.config.min_samples_per_class or n_c < K:
                # Too few samples → simple mean
                proto_feat = torch.from_numpy(feat_c.mean(axis=0)).float()
            else:
                # GMM clustering
                n_components = min(K, n_c)
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='diag',
                    max_iter=50,
                    random_state=42,
                    reg_covar=1e-4
                )
                try:
                    gmm.fit(feat_c)
                    # Prototype = weighted mean of cluster centers
                    centers = torch.from_numpy(gmm.means_).float()  # [K, feat_dim]
                    weights = torch.from_numpy(gmm.weights_).float()  # [K]
                    proto_feat = (centers * weights.unsqueeze(1)).sum(dim=0)  # [feat_dim]
                except Exception:
                    proto_feat = torch.from_numpy(feat_c.mean(axis=0)).float()

            # Soft prediction = softmax of mean logits for this class
            T = self.config.temperature
            proto_pred = F.softmax(logit_c.mean(dim=0) / T, dim=0)  # [C]

            prototypes[c] = (proto_feat, proto_pred)

        return prototypes

    def _compute_prototypes_mean(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Fallback: simple mean prototype per class (no GMM)."""
        prototypes = {}
        T = self.config.temperature

        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            proto_feat = features[mask].mean(dim=0)
            proto_pred = F.softmax(logits[mask].mean(dim=0) / T, dim=0)
            prototypes[c] = (proto_feat, proto_pred)

        return prototypes

    def _compute_discrepancy(
        self,
        local_prototypes: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        global_prototypes: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Compute discrepancy between local and global prototypes (Eq.10).

        d_i = (1/|C_i|) * Σ_c KL(q_i^c || q^c)

        where q_i^c = local soft prediction, q^c = global soft prediction
        """
        if not global_prototypes:
            return 0.0

        kl_sum = 0.0
        count = 0

        for c in local_prototypes:
            if c not in global_prototypes:
                continue
            q_local = local_prototypes[c][1]   # [C]
            q_global = global_prototypes[c][1]  # [C]

            # KL(local || global), add small eps for numerical stability
            eps = 1e-8
            kl = F.kl_div(
                (q_global + eps).log(),
                q_local + eps,
                reduction='sum'
            ).item()
            kl_sum += max(kl, 0.0)  # Clamp negative from numerical issues
            count += 1

        return kl_sum / max(count, 1)

    def _aggregate_prototypes_dat(
        self,
        client_prototypes: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        discrepancies: List[float],
        data_sizes: List[int]
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Discrepancy-Aware aggregation of prototypes (DAT, Eq.10).

        Weight_i = n_i * exp(-d_i / tau) / Σ_j n_j * exp(-d_j / tau)
        """
        tau = self.config.dat_temperature
        n = len(client_prototypes)

        if n == 0:
            return {}

        # Compute DAT weights
        raw_weights = []
        for i in range(n):
            w = data_sizes[i] * np.exp(-discrepancies[i] / tau)
            raw_weights.append(w)

        total = sum(raw_weights) + 1e-8
        weights = [w / total for w in raw_weights]

        # Aggregate prototypes per class
        aggregated = {}
        all_classes = set()
        for protos in client_prototypes:
            all_classes.update(protos.keys())

        for c in all_classes:
            feat_sum = torch.zeros(self.feat_dim)
            pred_sum = torch.zeros(self.n_classes)
            w_sum = 0.0

            for i, protos in enumerate(client_prototypes):
                if c in protos:
                    feat_i, pred_i = protos[c]
                    feat_sum += weights[i] * feat_i
                    pred_sum += weights[i] * pred_i
                    w_sum += weights[i]

            if w_sum > 0:
                feat_avg = feat_sum / w_sum
                pred_avg = pred_sum / w_sum
                # Re-normalize prediction to be a valid distribution
                pred_avg = pred_avg / (pred_avg.sum() + 1e-8)
                aggregated[c] = (feat_avg, pred_avg)

        return aggregated

    def _train_local_with_kd(
        self,
        model: nn.Module,
        train_loader: Any,
        global_prototypes: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Local training with composite loss (Eq.13 in paper):
          L = CE(y, ŷ) + alpha * L_fa + beta * L_ka

        L_fa = feature alignment loss (MSE between local features and global
               prototype features for matching classes in the batch)
        L_ka = knowledge alignment loss (KL-div between local soft predictions
               and global prototype soft predictions)
        """
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.local_lr,
            momentum=self.config.local_momentum
        )
        criterion = nn.CrossEntropyLoss()
        T = self.config.temperature
        alpha = self.config.alpha
        beta = self.config.beta

        total_loss = 0.0
        n_batches = 0

        # Pre-move global prototype tensors to device
        global_feats = {}
        global_preds = {}
        for c, (feat, pred) in global_prototypes.items():
            global_feats[c] = feat.to(self.device)
            global_preds[c] = pred.to(self.device)

        for _ in range(self.config.local_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass through feature extractor + classifier
                features = model.get_features(data)  # [B, feat_dim]
                logits = model.fc(features)           # [B, C]

                # ── CE loss ──
                ce_loss = criterion(logits, target)

                # ── Feature alignment loss ──
                fa_loss = torch.tensor(0.0, device=self.device)
                # ── Knowledge alignment loss ──
                ka_loss = torch.tensor(0.0, device=self.device)

                if global_prototypes:
                    unique_classes = target.unique()
                    n_matched = 0

                    for c in unique_classes:
                        c_val = c.item()
                        if c_val not in global_feats:
                            continue

                        mask = (target == c)
                        feat_c = features[mask]  # [n_c, feat_dim]
                        logit_c = logits[mask]    # [n_c, C]

                        # Feature alignment: MSE to prototype feature
                        proto_feat = global_feats[c_val].unsqueeze(0)  # [1, feat_dim]
                        fa_loss += F.mse_loss(feat_c, proto_feat.expand_as(feat_c))

                        # Knowledge alignment: KL to prototype soft prediction
                        local_soft = F.log_softmax(logit_c / T, dim=1)    # [n_c, C]
                        proto_soft = global_preds[c_val].unsqueeze(0).expand_as(local_soft)
                        ka_loss += F.kl_div(local_soft, proto_soft, reduction='batchmean')

                        n_matched += 1

                    if n_matched > 0:
                        fa_loss = fa_loss / n_matched
                        ka_loss = ka_loss / n_matched

                # ── Composite loss ──
                loss = ce_loss + alpha * fa_loss + beta * (T * T) * ka_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any = None,  # Not used — FedGMKD needs no public data
        test_loader: Optional[Any] = None
    ) -> RoundResult:
        """Execute one round of FedGMKD."""
        self.current_round = round_idx

        local_state_dicts = []
        local_weights = []
        client_prototypes = []
        client_discrepancies = []
        client_data_sizes = []
        total_energy = {"training": 0.0, "inference": 0.0, "communication": 0.0}

        global_state = copy.deepcopy(self.server_model.state_dict())

        for dev in devices:
            dev_id = dev.device_id
            if dev_id not in client_loaders:
                continue

            local_loader = client_loaders[dev_id]
            n_samples = len(local_loader.dataset)

            # Create local model from global
            local_model = copy_model(self.server_model, device=self.device)

            # ── Phase 1: Local training with composite loss ──
            if self.global_prototypes:
                # After round 0: train with feature + knowledge alignment
                self._train_local_with_kd(
                    local_model, local_loader, self.global_prototypes
                )
            else:
                # Round 0: pure CE training (no prototypes yet)
                self.train_local(
                    local_model, local_loader,
                    epochs=self.config.local_epochs,
                    lr=self.config.local_lr
                )

            # ── Phase 2: Extract local prototypes via GMM (CKF) ──
            features, logits, labels = self._extract_features_and_labels(
                local_model, local_loader
            )
            local_protos = self._compute_prototypes_gmm(features, logits, labels)
            client_prototypes.append(local_protos)

            # ── Phase 3: Compute discrepancy ──
            disc = self._compute_discrepancy(local_protos, self.global_prototypes)
            client_discrepancies.append(disc)
            client_data_sizes.append(n_samples)

            # Collect model parameters
            local_state_dicts.append(local_model.state_dict())
            local_weights.append(n_samples)

            # Energy: training + uploading parameters + prototypes
            energy = self.energy_calc.compute_total_energy(
                cpu_freq=getattr(dev, 'cpu_freq', 1.0),
                data_size=n_samples,
                s_i=n_samples,
                channel_gain=getattr(dev, 'channel_gain', 1.0)
            )
            total_energy["training"] += energy.get("training", 0)
            # Parameters (~44MB) + prototypes (~400KB) ≈ same as FedAvg
            total_energy["communication"] += energy.get("communication", 0) * 100
            total_energy["inference"] += energy.get("inference", 0)

            del local_model

        # ── Phase 4: Server aggregation ──
        if local_state_dicts:
            # 4a. Compute DAT weights
            tau = self.config.dat_temperature
            raw_weights = []
            for i in range(len(local_state_dicts)):
                w = local_weights[i] * np.exp(
                    -client_discrepancies[i] / tau
                )
                raw_weights.append(w)
            total_w = sum(raw_weights) + 1e-8
            dat_weights = [w / total_w for w in raw_weights]

            # 4b. Aggregate model parameters with DAT weights
            avg_state = {}
            for key in local_state_dicts[0]:
                avg_state[key] = torch.zeros_like(
                    local_state_dicts[0][key], dtype=torch.float32
                )

            for state_dict, weight in zip(local_state_dicts, dat_weights):
                for key in avg_state:
                    avg_state[key] += weight * state_dict[key].float()

            self.server_model.load_state_dict(avg_state)

            # 4c. Aggregate prototypes with DAT weights
            self.global_prototypes = self._aggregate_prototypes_dat(
                client_prototypes,
                client_discrepancies,
                client_data_sizes
            )

        # ── Evaluate ──
        accuracy, loss = 0.0, 0.0
        if test_loader:
            eval_result = self.evaluate(test_loader)
            accuracy, loss = eval_result["accuracy"], eval_result["loss"]

        n_participants = len(local_state_dicts)
        avg_disc = float(np.mean(client_discrepancies)) if client_discrepancies else 0.0

        result = RoundResult(
            round_idx=round_idx,
            accuracy=accuracy,
            loss=loss,
            participation_rate=n_participants / len(devices) if devices else 0,
            n_participants=n_participants,
            energy=total_energy,
            extra={
                "method": "FedGMKD",
                "n_prototypes": len(self.global_prototypes),
                "avg_discrepancy": avg_disc,
                "dat_weights": [round(w, 4) for w in (dat_weights if local_state_dicts else [])],
            }
        )

        self.round_history.append(result)
        return result

    def aggregate(self, updates: List[Dict], weights: List[float]) -> None:
        """Interface compatibility."""
        pass
