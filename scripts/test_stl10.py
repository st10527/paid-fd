#!/usr/bin/env python3
"""
STL-10 Cross-Domain Quick Test

Tests whether PAID-FD works with STL-10 (unlabeled) as public data
instead of CIFAR-100 safe split.

Cross-domain setup:
  Private: CIFAR-100 train (40k, 100 classes)
  Public:  STL-10 unlabeled (10k samples, resized 96→32)
  Test:    CIFAR-100 test (10k)

Key difference: STL-10 uses ImageNet normalization, CIFAR-100 uses its own.
We need to align normalization for the model to work on both.

Usage:
    python3 -u scripts/test_stl10.py
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

def main():
    print("=" * 60)
    print("  STL-10 Cross-Domain Test for PAID-FD")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ─────────────────────────────────────────────
    # 1. Load datasets
    # ─────────────────────────────────────────────
    print("\n[1] Loading datasets...")

    # Private + Test: CIFAR-100
    from src.data.datasets import load_cifar100
    cifar_train, cifar_test = load_cifar100(root='./data')
    print(f"  CIFAR-100 train: {len(cifar_train)} samples")
    print(f"  CIFAR-100 test:  {len(cifar_test)} samples")

    # Public: STL-10 unlabeled (10k samples, resized to 32x32)
    from src.data.datasets import load_stl10
    stl10_public = load_stl10(root='./data', n_samples=10000, resize_to=32)
    print(f"  STL-10 public:   {len(stl10_public)} samples")

    # Check image shapes and value ranges
    cifar_img, cifar_lbl = cifar_train[0]
    stl_img, stl_lbl = stl10_public[0]
    print(f"\n  CIFAR-100 image: shape={cifar_img.shape}, range=[{cifar_img.min():.3f}, {cifar_img.max():.3f}]")
    print(f"  STL-10   image: shape={stl_img.shape}, range=[{stl_img.min():.3f}, {stl_img.max():.3f}]")

    # Note: different normalizations!
    # CIFAR-100: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
    # STL-10:    mean=[0.485, 0.456, 0.406],    std=[0.229, 0.224, 0.225]
    # This mismatch might affect results. Let's test both as-is and aligned.

    # ─────────────────────────────────────────────
    # 2. Partition private data (Dirichlet)
    # ─────────────────────────────────────────────
    print("\n[2] Partitioning private data...")
    from src.data.partition import DirichletPartitioner, create_client_loaders
    from src.utils.seed import set_seed

    set_seed(42)
    n_devices = 10

    train_targets = np.array(cifar_train.targets)
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=n_devices, seed=42)
    client_indices = partitioner.partition(cifar_train, train_targets)
    client_loaders = create_client_loaders(cifar_train, client_indices, batch_size=32)

    for i in range(min(3, n_devices)):
        print(f"  Client {i}: {len(client_indices[i])} samples")

    # ─────────────────────────────────────────────
    # 3. Create loaders
    # ─────────────────────────────────────────────
    test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)
    public_loader = DataLoader(stl10_public, batch_size=128, shuffle=True)

    # ─────────────────────────────────────────────
    # 4. Create model + PAID-FD
    # ─────────────────────────────────────────────
    print("\n[3] Setting up PAID-FD...")
    from src.models import get_model
    from src.methods.paid_fd import PAIDFD, PAIDFDConfig
    from src.devices.heterogeneity import HeterogeneityGenerator

    model = get_model('resnet18', num_classes=100)

    # Use best known config but with λ=1.0
    config_override = {
        'privacy_sensitivity': {
            'levels': {
                'low': {'value': 0.1, 'ratio': 0.40},
                'medium': {'value': 0.5, 'ratio': 0.40},
                'high': {'value': 1.0, 'ratio': 0.20},
            }
        }
    }
    gen = HeterogeneityGenerator(n_devices=n_devices, config_override=config_override, seed=42)
    devices = gen.generate()
    for dev in devices:
        if dev.device_id in client_indices:
            dev.data_size = len(client_indices[dev.device_id])

    config = PAIDFDConfig(
        gamma=500.0,
        distill_lr=0.005,
        temperature=3.0,
        local_epochs=20,
        local_lr=0.1,
        distill_epochs=10,
        public_samples=10000,
    )

    method = PAIDFD(model, config, n_classes=100, device=device)

    # ─────────────────────────────────────────────
    # 5. Run 15 rounds (quick test)
    # ─────────────────────────────────────────────
    print("\n[4] Running 15 rounds with STL-10 as public data...")
    print("    (Compare with CIFAR-100 safe split baseline)")
    print()

    n_rounds = 15
    accuracies = []

    for r in range(n_rounds):
        result = method.run_round(
            round_idx=r,
            devices=devices,
            client_loaders=client_loaders,
            public_loader=public_loader,
            test_loader=test_loader,
        )
        accuracies.append(result.accuracy * 100)

        if (r + 1) % 5 == 0:
            print(f"  R{r+1}: {result.accuracy*100:.1f}% (loss={result.loss:.3f})")

    print(f"\n{'='*60}")
    print(f"  STL-10 Cross-Domain Results")
    print(f"{'='*60}")
    print(f"  Final accuracy: {accuracies[-1]:.2f}%")
    print(f"  Best accuracy:  {max(accuracies):.2f}%")
    print(f"  Accuracy curve: {[f'{a:.1f}' for a in accuracies]}")
    print()
    print(f"  Compare with CIFAR-100 safe split baseline:")
    print(f"  distill_lr=0.005, λ=1.0 → 43.2% @R25 (steady growth)")
    print(f"  If STL-10 ≥ 25% @R15 → cross-domain is viable!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
