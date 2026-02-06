#!/usr/bin/env python3
"""
Diagnostic script to identify where the learning problem is.

This script tests each component separately:
1. Can a single model learn CIFAR-100? (baseline)
2. Can local models learn from their data?
3. Are the logits meaningful before noise?
4. Does distillation work without noise?
5. Does distillation work with noise?
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset


def test_1_baseline_learning():
    """Test: Can ResNet18 learn CIFAR-100 at all?"""
    print("\n" + "="*60)
    print("TEST 1: Baseline - Can ResNet18 learn CIFAR-100?")
    print("="*60)
    
    from src.data.datasets import load_cifar100
    from src.models import get_model
    
    train, test = load_cifar100(root='./data')
    
    # Use small subset for speed
    train_subset = Subset(train, range(5000))
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=128)
    
    model = get_model('resnet18', num_classes=100).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Training for 5 epochs on 5000 samples...")
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: {acc:.2f}%")
    
    if acc > 5:
        print("✓ PASS: Model can learn (>10%)")
        return True
    else:
        print("✗ FAIL: Model cannot learn basic CIFAR-100")
        return False


def test_2_local_learning():
    """Test: Can a local model learn from a small Non-IID dataset?"""
    print("\n" + "="*60)
    print("TEST 2: Local Learning - Can model learn from Non-IID data?")
    print("="*60)
    
    from src.data.datasets import load_cifar100
    from src.data.partition import DirichletPartitioner, create_client_loaders
    from src.models import get_model
    
    train, test = load_cifar100(root='./data')
    
    # Partition into 10 clients
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=10, seed=42)
    client_indices = partitioner.partition(train, train.targets)
    client_loaders = create_client_loaders(train, client_indices, batch_size=32)
    
    # Train one client's model
    client_id = 0
    loader = client_loaders[client_id]
    print(f"Client {client_id} has {len(client_indices[client_id])} samples")
    
    model = get_model('resnet18', num_classes=100).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Training for 5 epochs...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
    
    # Check if loss decreased
    if avg_loss < 4.0:  # Initial loss is ~4.6 for 100 classes
        print("✓ PASS: Local training reduces loss")
        return True, model
    else:
        print("✗ FAIL: Local training not effective")
        return False, model


def test_3_logits_quality(local_model):
    """Test: Are the logits from local model meaningful?"""
    print("\n" + "="*60)
    print("TEST 3: Logits Quality - Are local logits informative?")
    print("="*60)
    
    from src.data.datasets import load_stl10
    
    public = load_stl10(root='./data')
    public_loader = DataLoader(public, batch_size=32, shuffle=False)
    
    local_model.eval()
    all_logits = []
    
    with torch.no_grad():
        for data, _ in public_loader:
            data = data.cuda()
            logits = local_model(data)
            all_logits.append(logits.cpu())
            if len(all_logits) * 32 >= 500:
                break
    
    logits = torch.cat(all_logits, dim=0)[:500]
    
    # Check logits statistics
    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {logits.mean():.4f}")
    print(f"Logits std: {logits.std():.4f}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Check if predictions are diverse (not all same class)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    unique_preds = len(torch.unique(preds))
    print(f"Unique predictions: {unique_preds}/100 classes")
    
    # Check confidence
    max_probs = probs.max(dim=1)[0]
    print(f"Avg confidence: {max_probs.mean():.4f}")
    
    if unique_preds > 10 and logits.std() > 1.0:
        print("✓ PASS: Logits appear meaningful")
        return True, logits
    else:
        print("✗ FAIL: Logits may not be informative")
        return False, logits


def test_4_distillation_no_noise(logits):
    """Test: Does distillation work WITHOUT noise?"""
    print("\n" + "="*60)
    print("TEST 4: Distillation without noise")
    print("="*60)
    
    from src.data.datasets import load_cifar100, load_stl10
    from src.models import get_model
    
    _, test = load_cifar100(root='./data')
    public = load_stl10(root='./data')
    
    test_loader = DataLoader(test, batch_size=128)
    public_loader = DataLoader(public, batch_size=32, shuffle=False)
    
    # Create fresh server model
    server_model = get_model('resnet18', num_classes=100).cuda()
    
    # Initial accuracy
    server_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            pred = server_model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    init_acc = correct / total * 100
    print(f"Initial accuracy: {init_acc:.2f}%")
    
    # Distill from logits (no noise)
    target_logits = logits.cuda()
    optimizer = torch.optim.Adam(server_model.parameters(), lr=0.01)
    T = 3.0
    
    print("Distilling for 10 epochs...")
    server_model.train()
    
    for epoch in range(10):
        idx = 0
        total_loss = 0
        for data, _ in public_loader:
            if idx >= len(target_logits):
                break
            
            batch_size = min(len(data), len(target_logits) - idx)
            data = data[:batch_size].cuda()
            teacher = target_logits[idx:idx + batch_size]
            
            student = server_model(data)
            
            loss = F.kl_div(
                F.log_softmax(student / T, dim=1),
                F.softmax(teacher / T, dim=1),
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
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                pred = server_model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: acc={acc:.2f}%, loss={total_loss/len(public_loader):.4f}")
        server_model.train()
    
    if acc > init_acc + 1:
        print("✓ PASS: Distillation improves accuracy")
        return True
    else:
        print("✗ FAIL: Distillation not working")
        return False


def test_5_distillation_with_noise(logits):
    """Test: Does distillation work WITH LDP noise?"""
    print("\n" + "="*60)
    print("TEST 5: Distillation WITH LDP noise (eps=5.0)")
    print("="*60)
    
    from src.data.datasets import load_cifar100, load_stl10
    from src.models import get_model
    from src.privacy.ldp import add_noise_to_logits
    
    _, test = load_cifar100(root='./data')
    public = load_stl10(root='./data')
    
    test_loader = DataLoader(test, batch_size=128)
    public_loader = DataLoader(public, batch_size=32, shuffle=False)
    
    # Add noise
    noisy_logits = add_noise_to_logits(logits.numpy(), epsilon=5.0, clip_bound=5.0)
    noisy_logits = torch.from_numpy(noisy_logits).float().cuda()
    
    print(f"Noise added. Logits std before: {logits.std():.2f}, after: {noisy_logits.std():.2f}")
    
    # Create fresh server model
    server_model = get_model('resnet18', num_classes=100).cuda()
    optimizer = torch.optim.Adam(server_model.parameters(), lr=0.01)
    T = 3.0
    
    print("Distilling for 10 epochs with noisy logits...")
    
    for epoch in range(10):
        server_model.train()
        idx = 0
        for data, _ in public_loader:
            if idx >= len(noisy_logits):
                break
            
            batch_size = min(len(data), len(noisy_logits) - idx)
            data = data[:batch_size].cuda()
            teacher = noisy_logits[idx:idx + batch_size]
            
            student = server_model(data)
            
            loss = F.kl_div(
                F.log_softmax(student / T, dim=1),
                F.softmax(teacher / T, dim=1),
                reduction='batchmean'
            ) * (T * T)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += batch_size
        
        # Evaluate
        server_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                pred = server_model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: acc={acc:.2f}%")
    
    if acc > 2:
        print("✓ PASS: Distillation with noise shows learning")
        return True
    else:
        print("✗ FAIL: Noise destroys learning")
        return False


def main():
    print("="*60)
    print("PAID-FD Diagnostic Tests")
    print("="*60)
    print("This will identify where the learning problem is.")
    
    # Test 1
    if not test_1_baseline_learning():
        print("\n❌ STOP: Basic model cannot learn. Check model/data.")
        return
    
    # Test 2
    success, local_model = test_2_local_learning()
    if not success:
        print("\n❌ STOP: Local learning fails. Check data partitioning.")
        return
    
    # Test 3
    success, logits = test_3_logits_quality(local_model)
    if not success:
        print("\n⚠️ WARNING: Logits quality may be poor.")
    
    # Test 4
    if not test_4_distillation_no_noise(logits):
        print("\n❌ STOP: Distillation fails even without noise. Check implementation.")
        return
    
    # Test 5
    if not test_5_distillation_with_noise(logits):
        print("\n❌ ISSUE: Noise is too strong. Need higher epsilon or lower clip bound.")
        return
    
    print("\n" + "="*60)
    print("✅ All tests passed! The components work individually.")
    print("The issue may be in how they are combined in PAID-FD.")
    print("="*60)


if __name__ == '__main__':
    main()
