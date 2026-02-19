#!/usr/bin/env python3
"""
Diagnose gamma differentiation problem.

Goal: Understand WHY different gamma values produce nearly identical accuracy,
and find parameter adjustments (within theory) to create meaningful separation.

Analysis:
1. Game outputs: how participation, epsilon, noise differ across gamma
2. Noise-to-signal analysis: pseudo-label quality at each gamma
3. Simulate: what pretrain_epochs / alpha / beta create separation
"""
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import StackelbergSolver
from src.data.datasets import load_cifar100_safe_split
from src.data.partition import DirichletPartitioner, create_client_loaders
from src.models import get_model
from src.models.utils import copy_model
from src.utils.seed import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42


def analyze_game_outputs():
    """Part 1: What the Stackelberg game produces at each gamma."""
    print("=" * 70)
    print("Part 1: Stackelberg Game Outputs")
    print("=" * 70)

    set_seed(SEED)
    gen = HeterogeneityGenerator(
        n_devices=50,
        config_path='config/devices/heterogeneity.yaml',
        seed=SEED
    )
    devices = gen.generate()

    lambdas = [d.lambda_i for d in devices]
    c_totals = [d.c_total for d in devices]
    print(f"Device params: lambda=[{min(lambdas):.4f}, {max(lambdas):.4f}], "
          f"c_total=[{min(c_totals):.4f}, {max(c_totals):.4f}]")
    print()

    C = 5.0
    gamma_list = [3, 5, 7, 10, 15, 20]
    game_info = {}

    for gamma in gamma_list:
        solver = StackelbergSolver(gamma=gamma)
        result = solver.solve(devices)
        parts = [d for d in result['decisions'] if d.participates]
        N = len(parts)
        if N == 0:
            print(f"  gamma={gamma:3d}: NO PARTICIPANTS")
            game_info[gamma] = {'N': 0}
            continue

        eps_vals = [d.eps_star for d in parts]
        avg_eps = np.mean(eps_vals)
        sensitivity = 2.0 * C / N
        noise_scale = sensitivity / avg_eps

        game_info[gamma] = {
            'N': N, 'avg_eps': avg_eps,
            'noise_scale': noise_scale, 'price': result['price']
        }

        print(f"  gamma={gamma:3d}: N={N:2d}/50 ({N/50*100:4.0f}%), "
              f"price={result['price']:.3f}, "
              f"avg_eps={avg_eps:.3f}, "
              f"noise_scale={noise_scale:.4f}, "
              f"SNR={C/noise_scale:.1f}")

    return game_info


def analyze_pseudo_label_quality(game_info):
    """Part 2: How noise level affects pseudo-label accuracy at each gamma."""
    print()
    print("=" * 70)
    print("Part 2: Pseudo-Label Quality vs Gamma")
    print("=" * 70)

    set_seed(SEED)
    # Load data
    train_data, public_data, test_data = load_cifar100_safe_split(
        root='./data', n_public=20000, seed=SEED
    )
    public_loader = DataLoader(public_data, batch_size=256, shuffle=False)

    # Partition private data
    all_targets = np.array(train_data.dataset.targets)
    targets = all_targets[train_data.indices]
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=50, min_samples_per_client=10, seed=SEED)
    client_indices = partitioner.partition(train_data, targets)
    client_loaders = create_client_loaders(train_data, client_indices, batch_size=128)

    # Devices
    gen = HeterogeneityGenerator(n_devices=50, config_path='config/devices/heterogeneity.yaml', seed=SEED)
    devices = gen.generate()

    # Collect public images & labels
    pub_imgs, pub_labs = [], []
    for data, labels in public_loader:
        pub_imgs.append(data)
        pub_labs.append(labels)
    pub_imgs = torch.cat(pub_imgs)
    pub_labs = torch.cat(pub_labs)
    n_public = len(pub_imgs)

    C = 5.0

    # Test different pretrain_epochs
    for pretrain_ep in [10, 20, 50]:
        print(f"\n--- Pretrain epochs = {pretrain_ep} ---")

        # Pre-train model
        set_seed(SEED)
        model = get_model('resnet18', num_classes=100).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_ep)
        criterion = nn.CrossEntropyLoss()

        for ep in range(pretrain_ep):
            model.train()
            for data, target in public_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate pretrain accuracy
        model.eval()
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                pred = model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += len(target)
        pretrain_acc = correct / total
        print(f"  Pretrain test acc: {pretrain_acc:.4f}")

        # For each gamma, simulate 1 round of aggregation + noise
        for gamma in [3, 5, 10, 20]:
            if gamma not in game_info or game_info[gamma]['N'] == 0:
                continue

            info = game_info[gamma]
            N = info['N']

            # Solve game to get participation
            solver = StackelbergSolver(gamma=gamma)
            result = solver.solve(devices)
            part_ids = [d.device_id for d in result['decisions'] if d.participates]

            # Each participating device: train 3 epochs, compute logits
            set_seed(SEED)
            all_logits = []
            all_weights = []
            for dev_id in part_ids:
                if dev_id not in client_loaders:
                    continue
                local_model = copy_model(model, device=DEVICE)
                local_opt = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                local_model.train()
                for _ in range(3):  # 3 local epochs
                    for data, target in client_loaders[dev_id]:
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        local_opt.zero_grad()
                        loss = criterion(local_model(data), target)
                        loss.backward()
                        local_opt.step()

                # Compute clipped logits
                local_model.eval()
                chunks = []
                with torch.no_grad():
                    for start in range(0, n_public, 512):
                        batch = pub_imgs[start:start+512].to(DEVICE)
                        logits = local_model(batch)
                        logits = torch.clamp(logits, -C, C)
                        chunks.append(logits.cpu())
                device_logits = torch.cat(chunks)
                all_logits.append(device_logits)
                all_weights.append(1.0)  # uniform weight for analysis

            if not all_logits:
                continue

            N_actual = len(all_logits)
            # Aggregate
            total_w = sum(all_weights)
            norm_w = [w / total_w for w in all_weights]
            agg_logits = sum(w * l for w, l in zip(norm_w, all_logits))

            # Clean argmax (no noise)
            clean_argmax = agg_logits.argmax(dim=1)
            clean_match = (clean_argmax == pub_labs).float().mean().item()

            # Add noise
            avg_eps = info['avg_eps']
            sensitivity = 2.0 * C / N_actual
            noise_scale = sensitivity / avg_eps
            noise = np.random.laplace(0, noise_scale, agg_logits.shape)
            noisy_logits = agg_logits.numpy() + noise.astype(np.float32)
            noisy_tensor = torch.from_numpy(noisy_logits).float()

            # Noisy argmax
            noisy_argmax = noisy_tensor.argmax(dim=1)
            noisy_match_gt = (noisy_argmax == pub_labs).float().mean().item()
            noisy_match_clean = (noisy_argmax == clean_argmax).float().mean().item()

            # Max logit margin (how peaked are logits?)
            sorted_logits, _ = agg_logits.sort(dim=1, descending=True)
            margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()

            print(f"  gamma={gamma:3d}: N={N_actual:2d}, "
                  f"noise_scale={noise_scale:.3f}, "
                  f"margin={margin:.3f}, "
                  f"clean_acc={clean_match:.3f}, "
                  f"noisy_acc={noisy_match_gt:.3f}, "
                  f"noisy_vs_clean={noisy_match_clean:.3f}")


def simulate_fl_signal(game_info):
    """Part 3: Quick simulation - how much does FL contribute vs pretrain?"""
    print()
    print("=" * 70)
    print("Part 3: FL Contribution Analysis")
    print("=" * 70)
    print()
    print("From Phase 1.1 results:")
    print("  All gammas converge to ~60% after 100 rounds")
    print("  Pre-train starts at ~52% test accuracy")
    print("  FL contribution: ~8% regardless of gamma")
    print()
    print("Key insight: mixed-loss alpha=0.3 means only 30% of gradient")
    print("comes from FL signal. And pseudo_acc ~35% means effective")
    print("FL signal = 0.3 * 0.35 = 10.5%")
    print()
    print("Potential parameter adjustments (NO theory changes):")
    print("  1. Reduce pretrain_epochs: 50->10-20 (lower floor, more FL headroom)")
    print("  2. Increase distill_alpha: 0.3->0.5+ (more FL weight)")
    print("  3. Reduce ema_beta: 0.7->0.3 (less smoothing, preserve gamma differences)")
    print("  4. Increase distill_epochs: 1->3 (more FL learning per round)")
    print()
    print("The theory predicts: higher gamma -> more participation -> better quality.")
    print("Parameters need to be set so this quality difference is VISIBLE.")


if __name__ == '__main__':
    game_info = analyze_game_outputs()
    analyze_pseudo_label_quality(game_info)
    simulate_fl_signal(game_info)
