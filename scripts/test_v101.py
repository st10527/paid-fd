#!/usr/bin/env python3
"""Quick 5-round test for v10.1 (persistent Adam fix)."""
import torch, time, sys, numpy as np
from torch.utils.data import DataLoader
from src.methods.paid_fd import create_paid_fd
from src.data.datasets import load_cifar100_safe_split
from src.data.partition import DirichletPartitioner, create_client_loaders
from src.devices.heterogeneity import HeterogeneityGenerator
from src.utils.seed import set_seed

set_seed(42)

print("=== v10.1 Quick Test: 5 rounds, gamma=5 ===")
print("Verifying persistent Adam fixes CE stagnation")
t0 = time.time()

n_devices, n_classes = 20, 100

# Data (same pipeline as full experiment)
train_data, public_data, test_data = load_cifar100_safe_split(
    root='./data', n_public=5000, seed=42)
all_targets = np.array(train_data.dataset.targets)
targets = all_targets[train_data.indices]

partitioner = DirichletPartitioner(alpha=0.5, n_clients=n_devices, min_samples_per_client=10, seed=42)
client_indices = partitioner.partition(train_data, targets)
client_loaders = create_client_loaders(train_data, client_indices, batch_size=128)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
public_loader = DataLoader(public_data, batch_size=128, shuffle=False)

# Devices
generator = HeterogeneityGenerator(n_devices=n_devices,
    config_path='config/devices/heterogeneity.yaml', seed=42)
devices = generator.generate()

# PAID-FD v10.1
method = create_paid_fd('resnet18', n_classes=n_classes, gamma=5.0,
                        pretrain_epochs=3, public_samples=5000)

print(f"Setup done in {time.time()-t0:.0f}s")
print(f"Optimizer type: {type(method.distill_optimizer).__name__}")
print()

for r in range(5):
    t1 = time.time()
    result = method.run_round(r, devices, client_loaders, public_loader, test_loader)
    e = result.extra
    kl = e.get('mean_loss_kl', -1)
    ce = e.get('mean_loss_ce', -1)
    pre = e.get('pre_distill_acc', -1)
    post = e.get('post_distill_acc', -1)
    delta = e.get('distill_delta', 0)
    print(f"R{r}: acc={result.accuracy:.4f} kl={kl:.3f} ce={ce:.3f} "
          f"pre={pre:.4f} post={post:.4f} delta={delta:+.4f} "
          f"part={result.participation_rate:.0%} [{time.time()-t1:.0f}s]")

print(f"\nTotal: {time.time()-t0:.0f}s")
r0_ce = method.round_history[0].extra.get("mean_loss_ce", -1)
r4_ce = method.round_history[4].extra.get("mean_loss_ce", -1)
print(f"CE trend: R0={r0_ce:.3f} -> R4={r4_ce:.3f} (should decrease)")
print(f"Acc trend: R0={method.round_history[0].accuracy:.4f} -> R4={method.round_history[4].accuracy:.4f}")

if r4_ce < r0_ce:
    print("\n✓ CE is decreasing — persistent Adam is working!")
else:
    print("\n✗ CE still stagnant — need further investigation")
