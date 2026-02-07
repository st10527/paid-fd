#!/usr/bin/env python3
"""
TMC Debug Script - 系統性診斷訓練問題

檢查項目：
1. 數據載入和切割是否正確
2. 標籤分佈是否正常
3. DataLoader 輸出是否正確
4. 模型前向傳播是否正常
5. 本地訓練是否有梯度更新
6. 蒸餾過程是否正常
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
from collections import Counter


def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_data_loading():
    """檢查數據載入"""
    print_section("1. 數據載入檢查")
    
    from src.data.datasets import load_cifar100_safe_split
    
    private_set, public_set, test_set = load_cifar100_safe_split(root='./data')
    
    print(f"Private set type: {type(private_set)}")
    print(f"Public set type: {type(public_set)}")
    print(f"Test set type: {type(test_set)}")
    
    print(f"\nPrivate set size: {len(private_set)}")
    print(f"Public set size: {len(public_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # 檢查 Subset 結構
    if hasattr(private_set, 'indices'):
        print(f"\nPrivate is Subset: YES")
        print(f"  - indices length: {len(private_set.indices)}")
        print(f"  - indices range: [{min(private_set.indices)}, {max(private_set.indices)}]")
        print(f"  - base dataset size: {len(private_set.dataset)}")
    
    # 檢查標籤
    if hasattr(private_set, 'dataset') and hasattr(private_set.dataset, 'targets'):
        all_targets = np.array(private_set.dataset.targets)
        private_targets = all_targets[private_set.indices]
        print(f"\nPrivate targets:")
        print(f"  - unique classes: {len(np.unique(private_targets))}")
        print(f"  - min label: {private_targets.min()}, max label: {private_targets.max()}")
        
        # 標籤分佈
        label_counts = Counter(private_targets)
        print(f"  - samples per class (first 5): {dict(list(label_counts.items())[:5])}")
    
    return private_set, public_set, test_set


def check_single_sample(private_set):
    """檢查單個樣本"""
    print_section("2. 單個樣本檢查")
    
    # 取第一個樣本
    img, label = private_set[0]
    print(f"Sample 0:")
    print(f"  - image shape: {img.shape}")
    print(f"  - image dtype: {img.dtype}")
    print(f"  - image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  - label: {label}")
    print(f"  - label type: {type(label)}")
    
    # 檢查多個樣本的標籤
    labels = [private_set[i][1] for i in range(10)]
    print(f"\nFirst 10 labels: {labels}")
    
    return img, label


def check_partition(private_set):
    """檢查數據分割"""
    print_section("3. 數據分割檢查 (Dirichlet)")
    
    from src.data.partition import DirichletPartitioner
    
    # 提取 targets
    if hasattr(private_set, 'dataset'):
        all_targets = np.array(private_set.dataset.targets)
        train_targets = all_targets[private_set.indices]
    else:
        train_targets = np.array(private_set.targets)
    
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=10, seed=42)
    client_indices = partitioner.partition(private_set, train_targets)
    
    print(f"Number of clients: {len(client_indices)}")
    
    for cid in range(min(3, len(client_indices))):
        indices = client_indices[cid]
        print(f"\nClient {cid}:")
        print(f"  - num samples: {len(indices)}")
        print(f"  - indices range: [{min(indices)}, {max(indices)}]")
        
        # 這些索引應該是 0 到 len(private_set)-1
        # 檢查是否超出範圍
        if max(indices) >= len(private_set):
            print(f"  - ⚠️ ERROR: indices exceed private_set size!")
        
        # 檢查這些索引對應的標籤
        client_labels = train_targets[indices]
        print(f"  - unique classes: {len(np.unique(client_labels))}")
        print(f"  - label distribution: {dict(Counter(client_labels).most_common(5))}")
    
    return client_indices, train_targets


def check_client_loaders(private_set, client_indices):
    """檢查 client DataLoader"""
    print_section("4. Client DataLoader 檢查")
    
    from src.data.partition import create_client_loaders
    
    client_loaders = create_client_loaders(private_set, client_indices, batch_size=32)
    
    print(f"Number of client loaders: {len(client_loaders)}")
    
    # 檢查第一個 client 的 loader
    for cid in range(min(2, len(client_loaders))):
        if cid not in client_loaders:
            continue
            
        loader = client_loaders[cid]
        print(f"\nClient {cid} loader:")
        print(f"  - dataset type: {type(loader.dataset)}")
        print(f"  - dataset size: {len(loader.dataset)}")
        
        # 取一個 batch
        batch_data, batch_labels = next(iter(loader))
        print(f"  - batch data shape: {batch_data.shape}")
        print(f"  - batch labels shape: {batch_labels.shape}")
        print(f"  - batch labels: {batch_labels.tolist()}")
        print(f"  - batch data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        
        # 關鍵檢查：標籤是否在 0-99 範圍內
        if batch_labels.min() < 0 or batch_labels.max() >= 100:
            print(f"  - ⚠️ ERROR: labels out of range [0, 99]!")
    
    return client_loaders


def check_model():
    """檢查模型"""
    print_section("5. 模型檢查")
    
    from src.models import get_model
    
    model = get_model('resnet18', num_classes=100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Model type: {type(model)}")
    print(f"Device: {device}")
    
    # 檢查輸出維度
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    if output.shape[1] != 100:
        print(f"⚠️ ERROR: Output dimension is {output.shape[1]}, expected 100!")
    
    # 檢查參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, device


def check_local_training(model, client_loaders, device):
    """檢查本地訓練"""
    print_section("6. 本地訓練檢查")
    
    import copy
    
    # 複製模型
    local_model = copy.deepcopy(model)
    local_model.train()
    
    # 取第一個 client
    cid = list(client_loaders.keys())[0]
    loader = client_loaders[cid]
    
    # 記錄訓練前的參數
    params_before = {name: p.clone() for name, p in local_model.named_parameters()}
    
    # 訓練一個 epoch
    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = local_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(data)
        
        if batch_idx == 0:
            print(f"First batch:")
            print(f"  - data shape: {data.shape}")
            print(f"  - target: {target.tolist()}")
            print(f"  - output shape: {output.shape}")
            print(f"  - predictions: {pred.tolist()}")
            print(f"  - loss: {loss.item():.4f}")
            
            # 檢查梯度
            grad_norms = []
            for name, p in local_model.named_parameters():
                if p.grad is not None:
                    grad_norms.append((name, p.grad.norm().item()))
            print(f"  - gradient norms (first 3):")
            for name, norm in grad_norms[:3]:
                print(f"      {name}: {norm:.6f}")
    
    print(f"\nAfter 1 epoch on client {cid}:")
    print(f"  - avg loss: {total_loss / len(loader):.4f}")
    print(f"  - accuracy: {100.0 * correct / total:.2f}%")
    
    # 檢查參數是否更新
    params_changed = 0
    for name, p in local_model.named_parameters():
        if not torch.equal(params_before[name], p):
            params_changed += 1
    
    print(f"  - parameters changed: {params_changed} / {len(params_before)}")
    
    if params_changed == 0:
        print("  - ⚠️ ERROR: No parameters were updated!")
    
    return local_model


def check_public_loader(public_set):
    """檢查 public DataLoader"""
    print_section("7. Public DataLoader 檢查")
    
    from torch.utils.data import DataLoader
    
    public_loader = DataLoader(public_set, batch_size=128, shuffle=True)
    
    print(f"Public set size: {len(public_set)}")
    print(f"Number of batches: {len(public_loader)}")
    
    # 取一個 batch
    batch_data, batch_labels = next(iter(public_loader))
    print(f"Batch data shape: {batch_data.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch labels (first 10): {batch_labels[:10].tolist()}")
    print(f"Data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
    
    return public_loader


def check_distillation(model, local_model, public_loader, device):
    """檢查蒸餾過程"""
    print_section("8. 蒸餾過程檢查")
    
    import torch.nn.functional as F
    
    # 計算 local model 在 public data 上的 logits
    local_model.eval()
    model.eval()
    
    with torch.no_grad():
        batch_data, _ = next(iter(public_loader))
        batch_data = batch_data.to(device)
        
        local_logits = local_model(batch_data)
        server_logits = model(batch_data)
        
        print(f"Local logits shape: {local_logits.shape}")
        print(f"Local logits range: [{local_logits.min():.3f}, {local_logits.max():.3f}]")
        print(f"Server logits shape: {server_logits.shape}")
        print(f"Server logits range: [{server_logits.min():.3f}, {server_logits.max():.3f}]")
        
        # 轉換為概率
        T = 3.0
        local_probs = F.softmax(local_logits / T, dim=1)
        server_probs = F.softmax(server_logits / T, dim=1)
        
        print(f"\nWith temperature T={T}:")
        print(f"Local probs range: [{local_probs.min():.6f}, {local_probs.max():.6f}]")
        print(f"Server probs range: [{server_probs.min():.6f}, {server_probs.max():.6f}]")
        
        # 檢查概率分佈是否合理
        print(f"\nLocal probs sum (should be 1.0): {local_probs[0].sum().item():.6f}")
        print(f"Max prob per sample: {local_probs.max(dim=1)[0].mean():.6f}")


def check_full_round():
    """完整運行一輪檢查"""
    print_section("9. 完整訓練輪次檢查")
    
    from src.utils.seed import set_seed
    from src.data.datasets import load_cifar100_safe_split
    from src.data.partition import DirichletPartitioner, create_client_loaders
    from src.devices.heterogeneity import HeterogeneityGenerator
    from src.models import get_model
    from src.methods.paid_fd import PAIDFD, PAIDFDConfig
    from src.game.stackelberg import StackelbergSolver
    from torch.utils.data import DataLoader
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 載入數據
    private_set, public_set, test_set = load_cifar100_safe_split(root='./data')
    
    # 提取 targets
    train_targets = np.array(private_set.dataset.targets)[private_set.indices]
    
    # 分割
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=10, seed=42)
    client_indices = partitioner.partition(private_set, train_targets)
    client_loaders = create_client_loaders(private_set, client_indices, batch_size=32)
    
    # DataLoaders
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    public_loader = DataLoader(public_set, batch_size=128, shuffle=True)
    
    # 設備
    gen = HeterogeneityGenerator(n_devices=10, seed=42)
    devices = gen.generate()
    for dev in devices:
        if dev.device_id in client_indices:
            dev.data_size = len(client_indices[dev.device_id])
    
    # ====== 關鍵診斷: 遊戲參數 ======
    print("\n--- Game Parameters ---")
    solver = StackelbergSolver(gamma=100.0)
    game_result = solver.solve(devices)
    print(f"Optimal price p*: {game_result['price']:.4f}")
    print(f"Participation rate: {game_result['participation_rate']:.2%}")
    print(f"Avg epsilon: {game_result['avg_eps']:.4f}")
    print(f"Avg s*: {game_result['avg_s']:.1f}")
    
    for d in game_result['decisions']:
        if d.participates:
            print(f"  Device {d.device_id}: s*={d.s_star:.1f}, eps*={d.eps_star:.4f}, q={d.quality:.4f}")
    
    # 噪聲分析
    print("\n--- Noise Analysis (v4: Logit Space) ---")
    avg_eps = game_result['avg_eps']
    n_part = game_result['n_participants']
    C = 5.0  # clip_bound
    
    # v4: noise on aggregate LOGITS
    sens_per_elem = 2.0 * C / n_part
    noise_scale = sens_per_elem / avg_eps
    noise_std = np.sqrt(2) * noise_scale
    logit_range = 2 * C  # [-5, +5]
    
    print(f"Avg epsilon: {avg_eps:.4f}")
    print(f"N participants: {n_part}")
    print(f"Clip bound C: {C}")
    print(f"Sensitivity per element (2C/N): {sens_per_elem:.4f}")
    print(f"Noise scale (Laplace): {noise_scale:.4f}")
    print(f"Noise std: {noise_std:.4f}")
    print(f"Logit range: {logit_range}")
    print(f"SNR (logit_range / noise_std): {logit_range / noise_std:.1f}")
    print(f"  -> {'Good' if logit_range / noise_std > 3 else 'Marginal' if logit_range / noise_std > 1 else 'Too noisy'}")
    
    # 模型
    model = get_model('resnet18', num_classes=100)
    
    # PAID-FD
    config = PAIDFDConfig(
        gamma=100.0,
        distill_lr=0.001,
        temperature=3.0,
        local_epochs=20,       # v4: more local training for better teacher
        local_lr=0.1,
        distill_epochs=10,
        public_samples=10000   # v3: use ALL public data
    )
    
    method = PAIDFD(model, config, n_classes=100, device=device)
    
    # 先評估初始準確率
    initial_result = method.evaluate(test_loader)
    print(f"\nInitial accuracy: {initial_result['accuracy']*100:.2f}%")
    
    # 運行訓練輪次
    print("\nRunning training rounds...")
    for r in range(10):
        result = method.run_round(
            round_idx=r,
            devices=devices,
            client_loaders=client_loaders,
            public_loader=public_loader,
            test_loader=test_loader
        )
        print(f"Round {r}: acc={result.accuracy*100:.2f}%, loss={result.loss:.4f}, participants={result.n_participants}")


def main():
    print("\n" + "="*60)
    print("  TMC CIFAR-100 訓練診斷")
    print("="*60)
    
    # 1. 數據載入
    private_set, public_set, test_set = check_data_loading()
    
    # 2. 單個樣本
    check_single_sample(private_set)
    
    # 3. 數據分割
    client_indices, train_targets = check_partition(private_set)
    
    # 4. Client loaders
    client_loaders = check_client_loaders(private_set, client_indices)
    
    # 5. 模型
    model, device = check_model()
    
    # 6. 本地訓練
    local_model = check_local_training(model, client_loaders, device)
    
    # 7. Public loader
    public_loader = check_public_loader(public_set)
    
    # 8. 蒸餾
    check_distillation(model, local_model, public_loader, device)
    
    # 9. 完整輪次
    check_full_round()
    
    print("\n" + "="*60)
    print("  診斷完成")
    print("="*60)


if __name__ == '__main__':
    main()
