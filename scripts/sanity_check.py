#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. 設定專案路徑 (確保能 import src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.data.datasets import load_cifar100
from src.models import get_model

def run_sanity_check():
    print("="*60)
    print("🔬 PAID-FD Sanity Check: Single Batch Overfitting")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 檢查點 1: 資料載入與形狀
    # ---------------------------------------------------------
    print("\n[Step 1] Loading Data...")
    try:
        # 使用修正後的參數名稱 'root'
        train_set, _ = load_cifar100(root='./data')
        loader = DataLoader(train_set, batch_size=32, shuffle=True)
        images, labels = next(iter(loader))
        
        images = images.to(device)
        labels = labels.to(device)
        
        print(f"  ✓ Data Loaded Successfully")
        print(f"  ✓ Image Shape: {images.shape} (Expect: [32, 3, 32, 32])")
        print(f"  ✓ Label Shape: {labels.shape}")
        print(f"  ✓ Label Range: min={labels.min().item()}, max={labels.max().item()} (Expect: 0-99)")
        print(f"  ✓ Pixel Values: min={images.min().item():.2f}, max={images.max().item():.2f}")
        
        if images.shape[1] != 3:
            print("  ⚠️ WARNING: Channel first/last mismatch? PyTorch expects [B, C, H, W]")
            
    except Exception as e:
        print(f"  ❌ Data Loading Failed: {e}")
        return

    # ---------------------------------------------------------
    # 檢查點 2: 模型結構與輸出
    # ---------------------------------------------------------
    print("\n[Step 2] Initializing Model...")
    try:
        model = get_model('resnet18', num_classes=100)
        model = model.to(device)
        model.train()
        
        # Forward pass test
        outputs = model(images)
        print(f"  ✓ Model Output Shape: {outputs.shape} (Expect: [32, 100])")
        
        if outputs.shape[1] != 100:
            print(f"  ❌ CRITICAL: Model output classes mismatch! Got {outputs.shape[1]}, expected 100.")
            return

    except Exception as e:
        print(f"  ❌ Model Initialization Failed: {e}")
        return

    # ---------------------------------------------------------
    # 檢查點 3: 過擬合測試 (Overfit Test)
    # ---------------------------------------------------------
    print("\n[Step 3] Trying to Overfit ONE Batch...")
    print("  Goal: Loss should approach 0.0, Accuracy should reach 100%")
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 51): # 跑 50 次迴圈，只訓練這同一批資料
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 計算準確度
        _, predicted = outputs.max(1)
        acc = predicted.eq(labels).sum().item() / labels.size(0) * 100
        
        if epoch % 5 == 0:
            print(f"  Iter {epoch:02d}: Loss = {loss.item():.6f} | Acc = {acc:.2f}%")
            
        if acc == 100.0 and loss.item() < 0.01:
            print(f"\n  🎉 SUCCESS! Model successfully overfitted the batch.")
            print("  Conclusion: Model architecture and Gradients are working.")
            return

    if acc < 90:
        print(f"\n  ❌ FAILURE: Model failed to overfit even a single batch.")
        print("  Possible causes:")
        print("  1. Learning Rate too small/large (Try lr=0.01 or 0.1)")
        print("  2. Gradient broken (check if .detach() is used wrongly in model)")
        print("  3. Input Normalization is wrong (ImageNet mean/std on CIFAR?)")
    else:
        print(f"\n  ⚠️ Warning: Converged but not perfect (Acc={acc:.2f}%)")

if __name__ == "__main__":
    run_sanity_check()