#!/usr/bin/env python3
"""
Diagnose PAID-FD step by step
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Load data
from src.data.datasets import load_cifar10, load_stl10
from src.data.partition import DirichletPartitioner, create_client_loaders
from src.devices.heterogeneity import HeterogeneityGenerator
from src.models import get_model
from src.privacy.ldp import add_noise_to_logits

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Setup
n_devices = 5
n_classes = 10
seed = 42

print("\n1. Loading data...")
train, test = load_cifar10(root='./data')
public = load_stl10(root='./data')

partitioner = DirichletPartitioner(alpha=0.5, n_clients=n_devices, seed=seed)
client_indices = partitioner.partition(train, train.targets)
client_loaders = create_client_loaders(train, client_indices, batch_size=32)

test_loader = DataLoader(test, batch_size=128)
public_loader = DataLoader(public, batch_size=64, shuffle=False)

print("\n2. Creating server model...")
server_model = get_model('resnet18', num_classes=n_classes).to(device)

# Initial accuracy
server_model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        pred = server_model(data).argmax(1)
        correct += (pred == target).sum().item()
init_acc = correct / len(test) * 100
print(f"   Initial accuracy: {init_acc:.2f}%")

print("\n3. Training local models and collecting logits...")
T = 3.0
all_logits = []
all_probs = []

for client_id in range(n_devices):
    print(f"\n   Client {client_id}:")
    
    # Copy server model
    local_model = get_model('resnet18', num_classes=n_classes).to(device)
    local_model.load_state_dict(server_model.state_dict())
    
    # Local training
    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    local_model.train()
    for epoch in range(3):
        total_loss = 0
        for data, target in client_loaders[client_id]:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    print(f"      Training loss: {total_loss/len(client_loaders[client_id]):.4f}")
    
    # Compute logits on public data
    local_model.eval()
    client_logits = []
    with torch.no_grad():
        for data, _ in public_loader:
            data = data.to(device)
            logits = local_model(data)
            client_logits.append(logits.cpu())
    
    client_logits = torch.cat(client_logits, dim=0)[:1000]  # Take 1000 samples
    print(f"      Logits shape: {client_logits.shape}")
    print(f"      Logits mean: {client_logits.mean():.4f}, std: {client_logits.std():.4f}")
    
    # Check predictions
    preds = client_logits.argmax(dim=1)
    unique_preds = len(torch.unique(preds))
    print(f"      Unique predictions: {unique_preds}/{n_classes}")
    
    all_logits.append(client_logits)
    
    # Also compute probs
    probs = F.softmax(client_logits / T, dim=1)
    all_probs.append(probs)
    
    del local_model

print("\n4. Aggregating (comparing methods)...")

# Method A: Average logits (WRONG)
avg_logits = torch.stack(all_logits).mean(dim=0)
print(f"\n   Method A (avg logits):")
print(f"      Mean: {avg_logits.mean():.4f}, std: {avg_logits.std():.4f}")
preds_A = avg_logits.argmax(dim=1)
print(f"      Unique predictions: {len(torch.unique(preds_A))}")

# Method B: Average probs (CORRECT)
avg_probs = torch.stack(all_probs).mean(dim=0)
print(f"\n   Method B (avg probs):")
print(f"      Mean: {avg_probs.mean():.4f}, std: {avg_probs.std():.4f}")
print(f"      Entropy: {-(avg_probs * torch.log(avg_probs + 1e-10)).sum(dim=1).mean():.4f}")
preds_B = avg_probs.argmax(dim=1)
print(f"      Unique predictions: {len(torch.unique(preds_B))}")

print("\n5. Distilling to server (using avg probs)...")

server_model.train()
optimizer = torch.optim.Adam(server_model.parameters(), lr=0.01)

avg_probs = avg_probs.to(device)

for epoch in range(10):
    idx = 0
    total_loss = 0
    for data, _ in public_loader:
        if idx >= len(avg_probs):
            break
        
        batch_size = min(len(data), len(avg_probs) - idx)
        data = data[:batch_size].to(device)
        target_probs = avg_probs[idx:idx + batch_size]
        
        student_logits = server_model(data)
        
        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            target_probs,
            reduction='batchmean'
        ) * (T * T)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        idx += batch_size
    
    # Evaluate
    server_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = server_model(data).argmax(1)
            correct += (pred == target).sum().item()
    acc = correct / len(test) * 100
    print(f"   Epoch {epoch+1}: acc={acc:.2f}%, loss={total_loss:.4f}")
    server_model.train()

print(f"\nFinal accuracy: {acc:.2f}%")
if acc > 30:
    print("SUCCESS - Distillation works when done correctly!")
else:
    print("STILL FAILING - Need more investigation")
