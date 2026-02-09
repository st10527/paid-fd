#!/usr/bin/env python3
"""Quick functional test for CSRA and FedGMKD."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Create tiny synthetic dataset
np.random.seed(42)
torch.manual_seed(42)
N = 200
X = torch.randn(N, 3, 32, 32)
Y = torch.randint(0, 10, (N,))

dataset = TensorDataset(X, Y)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Simulate 3 clients
client_loaders = {}
for i in range(3):
    idx = list(range(i*60, (i+1)*60))
    subset = TensorDataset(X[idx], Y[idx])
    client_loaders[i] = DataLoader(subset, batch_size=32, shuffle=True)

# Mock devices
class MockDevice:
    def __init__(self, did):
        self.device_id = did
        self.privacy_sensitivity = 0.5
        self.data_size = 60
        self.cpu_freq = 1.0
        self.channel_gain = 1.0

devices = [MockDevice(i) for i in range(3)]

# ── Test CSRA ──
print("Testing CSRA...", flush=True)
from src.models.resnet import resnet18
from src.methods.csra import CSRA, CSRAConfig

model = resnet18(num_classes=10)
config = CSRAConfig(local_epochs=1, local_lr=0.01, budget_per_round=100.0)
csra = CSRA(model, config, n_classes=10, device='cpu')
result = csra.run_round(0, devices, client_loaders, None, test_loader)
n_sel = result.extra["n_selected"]
avg_eps = result.extra["avg_epsilon"]
print(f"  CSRA Round 0: acc={result.accuracy*100:.1f}%, n_selected={n_sel}, avg_eps={avg_eps:.2f}")
print("  OK CSRA")

# ── Test FedGMKD (2 rounds) ──
print("Testing FedGMKD...", flush=True)
from src.methods.fedgmkd import FedGMKD, FedGMKDConfig

model2 = resnet18(num_classes=10)
config2 = FedGMKDConfig(local_epochs=1, local_lr=0.01, n_gmm_components=2)
gmkd = FedGMKD(model2, config2, n_classes=10, device='cpu')

# Round 0: no prototypes → pure CE
r0 = gmkd.run_round(0, devices, client_loaders, None, test_loader)
np0 = r0.extra["n_prototypes"]
print(f"  FedGMKD Round 0: acc={r0.accuracy*100:.1f}%, n_protos={np0}")

# Round 1: has prototypes → composite loss
r1 = gmkd.run_round(1, devices, client_loaders, None, test_loader)
np1 = r1.extra["n_prototypes"]
disc = r1.extra["avg_discrepancy"]
print(f"  FedGMKD Round 1: acc={r1.accuracy*100:.1f}%, n_protos={np1}, disc={disc:.4f}")
print("  OK FedGMKD")

print("\nAll tests passed!")
