#!/usr/bin/env python3
"""
Deep diagnosis of why PAID-FD accuracy is stuck at 15-18% on CIFAR-100.

This script tests each component of the pipeline independently to find
exactly where the bottleneck is.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import copy
import time

def diagnose():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 70)
    
    # ================================================================
    # Test 1: Can ResNet18 learn CIFAR-100 AT ALL with standard training?
    # ================================================================
    print("\n[Test 1] Centralized ResNet-18 on CIFAR-100 (10 epochs, no FL)")
    print("-" * 50)
    
    from src.data.datasets import load_cifar100_safe_split
    from src.models import get_model
    from torch.utils.data import DataLoader
    
    private_set, public_set, test_set = load_cifar100_safe_split(
        root='./data', n_public=20000, seed=42
    )
    
    # Use ALL private data as centralized training set
    train_loader = DataLoader(private_set, batch_size=128, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    model = get_model('resnet18', num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total
        print(f"  Epoch {epoch+1}/10: acc={acc:.4f}")
    
    centralized_acc = acc
    print(f"  >> Centralized baseline: {centralized_acc:.4f}")
    
    # ================================================================
    # Pre-training on public data (FedMD "transfer learning" phase)
    # ================================================================
    print("\n[Pre-training] Pre-train base model on public data (50 epochs, 20k samples)")
    print("-" * 50)
    
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
    ])
    
    pretrain_model = get_model('resnet18', num_classes=100).to(device)
    pt_opt = torch.optim.SGD(pretrain_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    pt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pt_opt, T_max=50)
    
    public_loader = DataLoader(public_set, batch_size=256, shuffle=True,
                               num_workers=4, pin_memory=True)
    
    for epoch in range(50):
        pretrain_model.train()
        for data, target in public_loader:
            data = augment(data).to(device)
            target = target.to(device)
            pt_opt.zero_grad()
            loss = criterion(pretrain_model(data), target)
            loss.backward()
            pt_opt.step()
        pt_sched.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            pretrain_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = pretrain_model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            acc = correct / total
            print(f"  Epoch {epoch+1}/50: acc={acc:.4f}")
    
    pretrain_acc = acc
    print(f"  >> Pre-trained baseline (20k public, 50 epochs): {pretrain_acc:.4f}")
    
    # ================================================================
    # Test 2: Single device fine-tuning FROM PRE-TRAINED (non-IID)
    # ================================================================
    print("\n[Test 2] Single device fine-tuning (960 samples, non-IID, 50 epochs, FROM pre-trained)")
    print("-" * 50)
    
    from src.data.partition import DirichletPartitioner, create_client_loaders
    
    all_targets = np.array(private_set.dataset.targets)
    targets = all_targets[private_set.indices]
    
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=50, min_samples_per_client=10, seed=42)
    client_indices = partitioner.partition(private_set, targets)
    client_loaders = create_client_loaders(private_set, client_indices, batch_size=128)
    
    # Pick a device with average data — start FROM pre-trained model
    dev_id = 0
    local_model = copy.deepcopy(pretrain_model)
    local_opt = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    local_loader = client_loaders[dev_id]
    
    print(f"  Device {dev_id}: {len(client_indices[dev_id])} samples")
    
    for epoch in range(50):
        local_model.train()
        for data, target in local_loader:
            data, target = data.to(device), target.to(device)
            local_opt.zero_grad()
            loss = criterion(local_model(data), target)
            loss.backward()
            local_opt.step()
        
        if (epoch + 1) % 10 == 0:
            local_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = local_model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            acc = correct / total
            print(f"  Epoch {epoch+1}/50: test_acc={acc:.4f}")
    
    single_device_acc = acc
    
    # ================================================================
    # Test 3: Logit quality from local model (fine-tuned from pre-trained)
    # ================================================================
    print("\n[Test 3] Logit quality analysis (pre-trained + fine-tuned model)")
    print("-" * 50)
    
    public_loader = DataLoader(public_set, batch_size=256, shuffle=False,
                               num_workers=4, pin_memory=True)
    
    # Get logits from the fine-tuned local model
    local_model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for data, target in public_loader:
            data = data.to(device)
            logits = local_model(data)
            all_logits.append(logits.cpu())
            all_labels.append(target)
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Check logit statistics
    print(f"  Logit shape: {logits.shape}")
    print(f"  Logit range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  Logit mean: {logits.mean():.4f}, std: {logits.std():.4f}")
    
    # After clipping
    C = 5.0
    clipped = torch.clamp(logits, -C, C)
    print(f"  After clip to [-{C},{C}]:")
    print(f"    range: [{clipped.min():.2f}, {clipped.max():.2f}]")
    print(f"    mean: {clipped.mean():.4f}, std: {clipped.std():.4f}")
    
    # Accuracy of logits as predictions
    pred = logits.argmax(dim=1)
    logit_acc = (pred == labels).float().mean().item()
    print(f"  Logit prediction accuracy on public data: {logit_acc:.4f}")
    
    # ================================================================
    # Test 4: Distillation quality (no noise)
    # ================================================================
    print("\n[Test 4] Distillation from perfect teacher (no noise, no FL)")
    print("-" * 50)
    
    # Use the fine-tuned local model as teacher, distill to a student
    # Student starts from pre-trained checkpoint (same as actual FL)
    teacher_model = local_model
    student_model = copy.deepcopy(pretrain_model)
    student_opt = torch.optim.Adam(student_model.parameters(), lr=0.001)
    
    T = 3.0  # Higher temperature extracts richer inter-class info
    teacher_model.eval()
    
    # Get teacher soft labels on public data
    teacher_logits_list = []
    public_images_list = []
    with torch.no_grad():
        for data, _ in public_loader:
            data = data.to(device)
            logits = teacher_model(data)
            teacher_logits_list.append(logits.cpu())
            public_images_list.append(data.cpu())
    
    teacher_logits = torch.cat(teacher_logits_list, dim=0)
    public_images = torch.cat(public_images_list, dim=0)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    
    print(f"  Teacher probs entropy: {-(teacher_probs * teacher_probs.log().clamp(min=-100)).sum(dim=1).mean():.4f}")
    
    # Augmentation prevents server from memorising fixed public images
    augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
    ])
    
    # Distill for 50 epochs (simulating 10 rounds × 5 distill_epochs)
    n_target = len(teacher_probs)
    for epoch in range(50):
        student_model.train()
        perm = torch.randperm(n_target)
        epoch_loss = 0
        n_batches = 0
        for start in range(0, n_target, 256):
            end = min(start + 256, n_target)
            idx = perm[start:end]
            data = augment(public_images[idx]).to(device)
            target = teacher_probs[idx].to(device)
            
            student_logits = student_model(data)
            loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                target,
                reduction='batchmean'
            ) * (T * T)
            
            student_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)
            student_opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            student_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = student_model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            acc = correct / total
            print(f"  Distill epoch {epoch+1}/50: test_acc={acc:.4f}, loss={epoch_loss/n_batches:.4f}")
    
    distill_acc = acc
    
    # ================================================================
    # Test 5: Distillation with noise + EMA logit buffer (simulating PAID-FD)
    # ================================================================
    print("\n[Test 5] Noisy distillation + EMA logit buffer (100 rounds)")
    print("-" * 50)
    
    # Simulate the REAL FL scenario with EMA logit buffer:
    # - 100 rounds, each round: fresh Laplace noise on aggregated logits
    # - EMA buffer accumulates logits across rounds (noise cancels, signal stays)
    # - Distill from smoothed buffer, NOT raw noisy logits
    N_participants = 35
    avg_eps = 0.5
    T_noisy = 3.0           # Safe to use T=3 because EMA denoises
    ema_beta = 0.7           # EMA smoothing factor
    distill_lr_noisy = 0.001 # Standard lr (safe because EMA gives clean targets)
    
    # Simulate multiple devices
    all_device_logits = []
    for i in range(N_participants):
        noise_per_device = torch.randn_like(teacher_logits) * 2.0
        dev_logits = teacher_logits + noise_per_device
        dev_logits = torch.clamp(dev_logits, -C, C)
        all_device_logits.append(dev_logits)
    
    agg_logits = sum(all_device_logits) / N_participants
    
    sensitivity = 2.0 * C / N_participants
    noise_scale = sensitivity / avg_eps
    
    print(f"  N={N_participants}, avg_eps={avg_eps}, noise_scale={noise_scale:.4f}")
    print(f"  T={T_noisy}, ema_beta={ema_beta}, distill_lr={distill_lr_noisy}")
    print(f"  Signal std: {agg_logits.std():.4f}")
    
    student2 = copy.deepcopy(pretrain_model)
    student2_opt = torch.optim.Adam(student2.parameters(), lr=distill_lr_noisy)
    logit_buffer = None  # EMA logit buffer
    
    for rnd in range(100):
        # Each round: fresh Laplace noise (different realization)
        noise = np.random.laplace(0, noise_scale, agg_logits.shape).astype(np.float32)
        noisy_logits = torch.from_numpy(agg_logits.numpy() + noise).float()
        
        # EMA logit buffer update (noise cancels across rounds)
        if logit_buffer is None:
            logit_buffer = noisy_logits.clone()
        else:
            logit_buffer = ema_beta * logit_buffer + (1 - ema_beta) * noisy_logits
        
        # Convert SMOOTHED logits to teacher probs
        smoothed_probs = F.softmax(logit_buffer / T_noisy, dim=1)
        
        # 1 distill epoch per round
        student2.train()
        perm = torch.randperm(n_target)
        for start in range(0, n_target, 256):
            end = min(start + 256, n_target)
            idx = perm[start:end]
            data = augment(public_images[idx]).to(device)
            target = smoothed_probs[idx].to(device)
            
            s_logits = student2(data)
            loss = F.kl_div(
                F.log_softmax(s_logits / T_noisy, dim=1),
                target,
                reduction='batchmean'
            ) * (T_noisy * T_noisy)
            
            student2_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student2.parameters(), 5.0)
            student2_opt.step()
        
        if (rnd + 1) % 20 == 0:
            student2.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = student2(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            acc = correct / total
            # Also measure how close buffer is to true logits
            buf_err = (logit_buffer - agg_logits).abs().mean().item()
            print(f"  Round {rnd+1}/100: test_acc={acc:.4f}, buffer_err={buf_err:.4f}")
    
    noisy_distill_acc = acc
    
    # ================================================================
    # Test 6: Full FL simulation (small scale)
    # ================================================================
    print("\n[Test 6] Full FL simulation: 10 devices, 30 rounds, no noise")
    print("-" * 50)
    
    server_model = copy.deepcopy(pretrain_model)  # Start from pre-trained
    server_opt = torch.optim.Adam(server_model.parameters(), lr=0.001)
    
    # Create 10 local models — ALL start from pre-trained checkpoint
    local_models = {}
    local_opts = {}
    for i in range(10):
        m = copy.deepcopy(pretrain_model)
        local_models[i] = m
        local_opts[i] = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    for round_idx in range(30):
        # Each device trains 3 epochs (persistent)
        for i in range(10):
            if i not in client_loaders:
                continue
            local_models[i].train()
            for ep in range(3):
                for data, target in client_loaders[i]:
                    data, target = data.to(device), target.to(device)
                    local_opts[i].zero_grad()
                    loss = criterion(local_models[i](data), target)
                    loss.backward()
                    local_opts[i].step()
        
        # Collect logits from all 10 devices on public data
        device_logits_list = []
        for i in range(10):
            if i not in client_loaders:
                continue
            local_models[i].eval()
            chunks = []
            with torch.no_grad():
                for data, _ in public_loader:
                    data = data.to(device)
                    logits = local_models[i](data)
                    chunks.append(logits.cpu())
            device_logits_list.append(torch.cat(chunks, dim=0))
        
        # Average logits (no noise, no clipping)
        avg_logits = sum(device_logits_list) / len(device_logits_list)
        teacher_p = F.softmax(avg_logits / T, dim=1)
        
        # Distill to server model (5 epochs with augmentation)
        server_model.train()
        n_t = len(teacher_p)
        for ep in range(5):
            perm = torch.randperm(n_t)
            for start in range(0, n_t, 256):
                end = min(start + 256, n_t)
                idx = perm[start:end]
                data = augment(public_images[idx]).to(device)
                target = teacher_p[idx].to(device)
                
                s_logits = server_model(data)
                loss = F.kl_div(
                    F.log_softmax(s_logits / T, dim=1),
                    target,
                    reduction='batchmean'
                ) * (T * T)
                
                server_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(server_model.parameters(), 5.0)
                server_opt.step()
        
        # Eval
        if (round_idx + 1) % 5 == 0:
            server_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = server_model(data).argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            acc = correct / total
            print(f"  Round {round_idx+1}/30: server_acc={acc:.4f}")
    
    fl_acc = acc
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print(f"  Pre-train - Public data only (20k, 50 epochs):   {pretrain_acc:.4f}")
    print(f"  Test 1 - Centralized (48k samples, 10 epochs):    {centralized_acc:.4f}")
    print(f"  Test 2 - Single device fine-tuned (from pretrain): {single_device_acc:.4f}")
    print(f"  Test 4 - Distill from 1 teacher (no noise, T=3):  {distill_acc:.4f}")
    print(f"  Test 5 - Noisy distill + EMA (100 rnd, β=0.7):  {noisy_distill_acc:.4f}")
    print(f"  Test 6 - Full FL (10 dev, 30 rnd, pretrain, T=3): {fl_acc:.4f}")
    print()
    print("Bottleneck analysis:")
    if pretrain_acc < 0.20:
        print("  ⚠️  Pre-training too weak — public data insufficient")
    if centralized_acc < 0.30:
        print("  ⚠️  Centralized training is weak — model/lr problem")
    if single_device_acc < pretrain_acc:
        print("  ⚠️  Fine-tuning hurts — catastrophic forgetting on non-IID")
    if distill_acc < pretrain_acc:
        print("  ⚠️  Distillation fails — teacher too weak or pipeline issue")
    if noisy_distill_acc < distill_acc * 0.5:
        print("  ⚠️  Noise destroys signal — DP noise too strong")
    if fl_acc > pretrain_acc * 1.2:
        print("  ✅  FL + distillation improves over pre-training!")
    elif fl_acc > pretrain_acc:
        print("  🟡  FL marginally improves over pre-training")
    else:
        print("  ⚠️  FL doesn't improve over pre-training alone")


if __name__ == '__main__':
    diagnose()
