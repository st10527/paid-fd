#!/usr/bin/env python3
"""
Diagnose: How does pre-training strength affect gamma differentiation?

Tests the ACTUAL signal quality difference between N=19 and N=50 device
ensembles at different pre-training levels, using real CIFAR-100 data.

This directly measures the mechanism that should differentiate gamma values:
  More participants → more diverse non-IID local models → better ensemble
"""
import torch
import torch.nn as nn
import numpy as np
import sys, os, time
sys.path.insert(0, '.')

from src.models import get_model
from src.models.utils import copy_model
from src.data.datasets import load_cifar100_safe_split
from src.data.partition import DirichletPartitioner, create_client_loaders
from src.utils.seed import set_seed
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================
# Setup: same data split as Phase 1.1
# ============================================================
set_seed(42)
train_data, public_data, test_data = load_cifar100_safe_split(
    root='./data', n_public=20000, seed=42
)
all_targets = np.array(train_data.dataset.targets)
targets = all_targets[train_data.indices]

partitioner = DirichletPartitioner(alpha=0.5, n_clients=50, min_samples_per_client=10, seed=42)
client_indices = partitioner.partition(train_data, targets)
client_loaders = create_client_loaders(train_data, client_indices, batch_size=128)
public_loader = DataLoader(public_data, batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
])

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(1)
            correct += (pred == target).sum().item()
            total += len(target)
    return correct / total

def pretrain(model, loader, epochs, lr=0.1):
    """Pre-train and return accuracy on test set."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for data, target in loader:
            data = augment(data).to(device)
            target = target.to(device)
            opt.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            opt.step()
        sched.step()
    return evaluate(model, test_loader)

def compute_ensemble_logits(models, loader, clip=5.0):
    """Compute average clipped logits from an ensemble of models."""
    all_logits = []
    for m in models:
        m.eval()
        chunks = []
        with torch.no_grad():
            for data, _ in loader:
                logits = m(data.to(device))
                logits = torch.clamp(logits, -clip, clip)
                chunks.append(logits.cpu())
        all_logits.append(torch.cat(chunks, dim=0))
    
    # Average
    stacked = torch.stack(all_logits, dim=0)  # (N_models, n_samples, K)
    avg_logits = stacked.mean(dim=0)           # (n_samples, K)
    return avg_logits

def get_public_labels(loader):
    """Get ground truth labels for public data."""
    labels = []
    for _, target in loader:
        labels.append(target)
    return torch.cat(labels, dim=0)

# ============================================================
# Test different pre-training levels
# ============================================================
pretrain_epochs_list = [5, 10, 20, 50]
n_local_epochs = 2   # per round (faster)
n_rounds_sim = 5      # 5 rounds (enough to see divergence on CPU)

# Simulate two gamma scenarios:
# gamma=3:  N=19 participants (indices 0-18)
# gamma=10: N=50 participants (indices 0-49)
participant_sets = {
    'γ=3 (N=19)': list(range(19)),
    'γ=10 (N=50)': list(range(50)),
}

public_labels = get_public_labels(public_loader)

print("\n" + "=" * 70)
print("EXPERIMENT: Pre-training strength vs. ensemble quality")
print(f"  Local training: {n_local_epochs} epochs/round × {n_rounds_sim} rounds")
print(f"  Clip bound: C=5.0, Dirichlet α=0.5")
print("=" * 70)

for pt_epochs in pretrain_epochs_list:
    print(f"\n{'='*60}")
    print(f"Pre-training: {pt_epochs} epochs")
    print(f"{'='*60}")
    
    # Pre-train base model
    set_seed(42)
    base_model = get_model('resnet18', num_classes=100).to(device)
    t0 = time.time()
    pt_acc = pretrain(base_model, public_loader, pt_epochs)
    pt_time = time.time() - t0
    print(f"  Pre-train acc (test): {pt_acc*100:.2f}% ({pt_time:.0f}s)")
    
    # Create local models from pre-trained checkpoint
    local_models = {}
    local_optimizers = {}
    for dev_id in range(50):
        local_models[dev_id] = copy_model(base_model, device=device)
        local_optimizers[dev_id] = torch.optim.SGD(
            local_models[dev_id].parameters(),
            lr=0.01, momentum=0.9, weight_decay=5e-4
        )
    
    # Simulate local training for n_rounds_sim rounds
    print(f"  Local training ({n_rounds_sim} rounds × {n_local_epochs} epochs)...", end='', flush=True)
    t0 = time.time()
    criterion = nn.CrossEntropyLoss()
    for round_idx in range(n_rounds_sim):
        for dev_id in range(50):
            if dev_id not in client_loaders:
                continue
            model = local_models[dev_id]
            opt = local_optimizers[dev_id]
            model.train()
            for _ in range(n_local_epochs):
                for data, target in client_loaders[dev_id]:
                    data, target = data.to(device), target.to(device)
                    opt.zero_grad()
                    loss = criterion(model(data), target)
                    loss.backward()
                    opt.step()
    train_time = time.time() - t0
    print(f" done ({train_time:.0f}s)")
    
    # Measure individual model quality
    individual_accs = []
    for dev_id in range(50):
        acc = evaluate(local_models[dev_id], test_loader)
        individual_accs.append(acc)
    print(f"  Individual model acc: mean={np.mean(individual_accs)*100:.1f}%, "
          f"std={np.std(individual_accs)*100:.1f}%, "
          f"min={min(individual_accs)*100:.1f}%, max={max(individual_accs)*100:.1f}%")
    
    # Compute ensemble logits for each participant set
    for label, dev_ids in participant_sets.items():
        models_subset = [local_models[d] for d in dev_ids if d in local_models]
        avg_logits = compute_ensemble_logits(models_subset, public_loader)
        
        # Clean ensemble accuracy (argmax of avg logits vs true labels)
        clean_preds = avg_logits.argmax(dim=1)
        clean_acc = (clean_preds == public_labels).float().mean().item()
        
        # Add noise (same noise level for both, since N×ε ≈ constant)
        noise_scale = 0.62  # approximately 2C/(N*ε) ≈ 0.62 for all gammas
        noise = np.random.laplace(0, noise_scale, avg_logits.shape)
        noisy_logits = avg_logits.numpy() + noise.astype(np.float32)
        noisy_preds = np.argmax(noisy_logits, axis=1)
        noisy_acc = np.mean(noisy_preds == public_labels.numpy())
        
        # Margin: avg gap between top-1 and top-2 logit
        sorted_logits, _ = avg_logits.sort(dim=1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()
        
        print(f"  {label}: clean_agg={clean_acc*100:.1f}%, "
              f"noisy_agg={noisy_acc*100:.1f}%, margin={margin:.2f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
If 'clean_agg' accuracy differs significantly between N=19 and N=50,
then reducing pre-training will create gamma differentiation via 
ENSEMBLE QUALITY rather than noise level (which is constant).

The bigger the clean_agg gap, the more gamma matters.
""")
print("Done.")
