#!/usr/bin/env python3
"""
Parameter Sweep for PAID-FD

Usage:
    python3 scripts/param_sweep.py --dataset cifar10 --sweep distill_lr --rounds 20
    python3 scripts/param_sweep.py --dataset cifar100 --sweep distill_lr --rounds 30
    python3 scripts/param_sweep.py --dataset cifar10 --sweep all --rounds 20
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np


def run_single_experiment(
    # Dataset
    dataset_name="cifar10",
    n_classes=10,
    # === YOUR CONTRIBUTION: Game parameters (NEED TUNING) ===
    gamma=500.0,            # 【sweep結果】500 最佳 (33.6% @R25)
    lambda_multiplier=1.0,   # 【sweep結果】1.0 最佳 (43.2% @R25)
    # === Literature-based: Fixed values ===
    distill_lr=0.005,       # 【sweep結果】0.005 最佳 (42.1% @R25, 持續上升)
    temperature=3.0,
    n_rounds=20,            # 測試時 20 輪夠了，不用 50
    n_devices=10,
    distill_epochs=10,      # 【確認】10 配合 distill_lr=0.005 效果最好
    local_epochs=20,        # 【v4修正】從 5 改成 20 (讓 local model 真正學到東西)
    local_lr=0.1,           # 【致命修正】從 0.01 改成 0.1 (讓 SGD 真正跑起來！)
    public_samples=10000,   # 【v3修正】使用全部 public data (10k) 做蒸餾
    # Other
    seed=42,
    use_synthetic=False,
):
    """Run a single experiment with given parameters."""
    
    from src.utils.seed import set_seed
    from src.data.partition import DirichletPartitioner, create_client_loaders
    from src.devices.heterogeneity import HeterogeneityGenerator
    from src.models import get_model
    from src.methods.paid_fd import PAIDFD, PAIDFDConfig
    from torch.utils.data import DataLoader
    
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # -----------------------------------------------------------
    # [修正開始] 切換數據源
    # -----------------------------------------------------------
    if dataset_name == 'cifar100':
        print("[TMC] Switching to Safe Mode (CIFAR-100 Only)...")
        
        # Import from src.data.datasets (with proper path)
        from src.data.datasets import load_cifar100_safe_split
        
        # 1. 取得乾淨的三份資料
        private_dataset, public_dataset, test_dataset = load_cifar100_safe_split(root='./data')

        # =======================================================
        # [關鍵修正] Fix UnboundLocalError: assign to variables used outside if/else
        # =======================================================
        train = private_dataset   # 讓後面的程式碼找得到 'train'
        test = test_dataset       # 讓後面的程式碼找得到 'test'
        public = public_dataset   # 讓後面的程式碼找得到 'public'
        
        # [重要] 提取 targets，因為 private_dataset 是 Subset，結構不太一樣
        # 從原始完整資料集中，根據 indices 抓出對應的 labels
        train_targets = np.array(private_dataset.dataset.targets)[private_dataset.indices]
        
        # [重要] 強制設定 n_classes，以免它去讀 STL-10 的設定
        n_classes = 100
    
    else:
        from src.data.datasets import load_dataset, load_stl10
        train, test, n_classes = load_dataset(dataset_name, root='./data')
        public = load_stl10(root='./data')
        train_targets = train.targets
    
    # Partition data
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=n_devices, seed=seed)
    client_indices = partitioner.partition(train, train_targets)
    client_loaders = create_client_loaders(train, client_indices, batch_size=32)
    
    # Create DataLoaders (unified for both branches)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)
    public_loader = DataLoader(public, batch_size=128, shuffle=True)
    
    # Create devices with lambda multiplier
    config_override = {
        'privacy_sensitivity': {
            'levels': {
                'low': {'value': 0.01 * lambda_multiplier, 'ratio': 0.40},
                'medium': {'value': 0.05 * lambda_multiplier, 'ratio': 0.40},
                'high': {'value': 0.10 * lambda_multiplier, 'ratio': 0.20},
            }
        }
    }
    
    gen = HeterogeneityGenerator(n_devices=n_devices, config_override=config_override, seed=seed)
    devices = gen.generate()
    
    for dev in devices:
        if dev.device_id in client_indices:
            dev.data_size = len(client_indices[dev.device_id])
    
    # Create model
    model = get_model('resnet18', num_classes=n_classes)
    
    config = PAIDFDConfig(
        gamma=gamma,
        distill_lr=distill_lr,
        temperature=temperature,
        local_epochs=local_epochs,
        local_lr=local_lr,
        distill_epochs=distill_epochs,
        public_samples=public_samples
    )
    
    method = PAIDFD(model, config, n_classes=n_classes, device=device)
    
    # Get game result
    from src.game.stackelberg import StackelbergSolver
    solver = StackelbergSolver(gamma=gamma)
    game_result = solver.solve(devices)
    print(f"      [eps*={game_result['avg_eps']:.2f}, s*={game_result['avg_s']:.0f}]")
    
    # Run training
    accuracies = []
    losses = []
    print("      ", end="", flush=True)
    
    for round_idx in range(n_rounds):
        result = method.run_round(
            round_idx=round_idx,
            devices=devices,
            client_loaders=client_loaders,
            public_loader=public_loader,
            test_loader=test_loader
        )
        accuracies.append(result.accuracy * 100)
        losses.append(result.loss)
        
        if (round_idx + 1) % 5 == 0:
            print(f"R{round_idx+1}:{result.accuracy*100:.1f}% ", end="", flush=True)
    
    print()
    
    return {
        'final_acc': accuracies[-1],
        'best_acc': max(accuracies),
        'avg_last5': np.mean(accuracies[-5:]) if len(accuracies) >= 5 else np.mean(accuracies),
        'improvement': np.mean(accuracies[-5:]) - np.mean(accuracies[:5]) if len(accuracies) >= 5 else 0,
        'final_loss': losses[-1],
        'price': game_result['price'],
        'avg_s': game_result['avg_s'],
        'avg_eps': game_result['avg_eps'],
        'participation': game_result['participation_rate'],
        'accuracies': accuracies,
        'losses': losses
    }


def sweep_distill_lr(dataset_name, n_classes, n_rounds):
    print("\n" + "="*60)
    print("Sweeping: DISTILL_LR")
    print("="*60)
    
    values = [0.001, 0.005, 0.01, 0.05]
    results = []
    
    for lr in values:
        print(f"\n>> distill_lr={lr}")
        r = run_single_experiment(
            dataset_name=dataset_name, n_classes=n_classes,
            distill_lr=lr, n_rounds=n_rounds
        )
        r['distill_lr'] = lr
        results.append(r)
        print(f"   => Final: {r['final_acc']:.2f}%, Best: {r['best_acc']:.2f}%")
    
    print("\n" + "-"*60)
    print(f"{'LR':>8} {'Final':>8} {'Best':>8} {'Improve':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['distill_lr']:>8} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% {r['improvement']:>+9.2f}%")
    
    return results


def sweep_temperature(dataset_name, n_classes, n_rounds):
    print("\n" + "="*60)
    print("Sweeping: TEMPERATURE")
    print("="*60)
    
    values = [1.0, 2.0, 3.0, 5.0]
    results = []
    
    for temp in values:
        print(f"\n>> temperature={temp}")
        r = run_single_experiment(
            dataset_name=dataset_name, n_classes=n_classes,
            temperature=temp, n_rounds=n_rounds
        )
        r['temperature'] = temp
        results.append(r)
        print(f"   => Final: {r['final_acc']:.2f}%, Best: {r['best_acc']:.2f}%")
    
    print("\n" + "-"*60)
    print(f"{'Temp':>8} {'Final':>8} {'Best':>8} {'Improve':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['temperature']:>8.1f} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% {r['improvement']:>+9.2f}%")
    
    return results


def sweep_distill_epochs(dataset_name, n_classes, n_rounds):
    print("\n" + "="*60)
    print("Sweeping: DISTILL_EPOCHS")
    print("="*60)
    
    values = [1, 3, 5, 10, 20]
    results = []
    
    for de in values:
        print(f"\n>> distill_epochs={de}")
        r = run_single_experiment(
            dataset_name=dataset_name, n_classes=n_classes,
            distill_epochs=de, n_rounds=n_rounds
        )
        r['distill_epochs'] = de
        results.append(r)
        print(f"   => Final: {r['final_acc']:.2f}%, Best: {r['best_acc']:.2f}%")
    
    print("\n" + "-"*60)
    print(f"{'Epochs':>8} {'Final':>8} {'Best':>8} {'Improve':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['distill_epochs']:>8} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% {r['improvement']:>+9.2f}%")
    
    return results


def sweep_gamma(dataset_name, n_classes, n_rounds):
    print("\n" + "="*60)
    print("Sweeping: GAMMA")
    print("="*60)
    
    values = [50, 100, 500, 1000]
    results = []
    
    for gamma in values:
        print(f"\n>> gamma={gamma}")
        r = run_single_experiment(
            dataset_name=dataset_name, n_classes=n_classes,
            gamma=gamma, n_rounds=n_rounds
        )
        r['gamma'] = gamma
        results.append(r)
        print(f"   => Final: {r['final_acc']:.2f}%, Best: {r['best_acc']:.2f}%, eps*={r['avg_eps']:.2f}")
    
    print("\n" + "-"*60)
    print(f"{'Gamma':>8} {'Final':>8} {'Best':>8} {'eps*':>8} {'s*':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['gamma']:>8} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% {r['avg_eps']:>8.2f} {r['avg_s']:>8.1f}")
    
    return results


def sweep_lambda(dataset_name, n_classes, n_rounds):
    print("\n" + "="*60)
    print("Sweeping: LAMBDA (privacy sensitivity)")
    print("="*60)
    
    values = [0.001, 0.01, 0.1, 1.0]
    results = []
    
    for mult in values:
        print(f"\n>> lambda_mult={mult}")
        r = run_single_experiment(
            dataset_name=dataset_name, n_classes=n_classes,
            lambda_multiplier=mult, n_rounds=n_rounds
        )
        r['lambda_mult'] = mult
        results.append(r)
        print(f"   => Final: {r['final_acc']:.2f}%, eps*={r['avg_eps']:.2f}")
    
    print("\n" + "-"*60)
    print(f"{'lambda':>8} {'Final':>8} {'Best':>8} {'eps*':>8} {'s*':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['lambda_mult']:>8.3f} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% {r['avg_eps']:>8.2f} {r['avg_s']:>8.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep for PAID-FD')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use (default: cifar10)')
    parser.add_argument('--sweep', type=str, required=True,
                       choices=['gamma', 'lambda', 'game_params', 'distill_lr', 'temperature', 'distill_epochs', 'all'],
                       help='Which parameter to sweep (game_params = gamma + lambda)')
    parser.add_argument('--rounds', type=int, default=20,
                       help='Number of rounds per experiment (default: 20)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save results to JSON file')
    args = parser.parse_args()
    
    n_classes = 10 if args.dataset == 'cifar10' else 100
    
    print("="*60)
    print(f"PAID-FD Parameter Sweep")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()} ({n_classes} classes)")
    print(f"Rounds: {args.rounds}")
    print(f"Devices: 10")
    print(f"GPU: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    all_results = {'config': {'dataset': args.dataset, 'n_classes': n_classes, 'rounds': args.rounds}}
    
    if args.sweep in ['gamma', 'game_params', 'all']:
        all_results['gamma'] = sweep_gamma(args.dataset, n_classes, args.rounds)
    
    if args.sweep in ['lambda', 'game_params', 'all']:
        all_results['lambda'] = sweep_lambda(args.dataset, n_classes, args.rounds)
    
    if args.sweep in ['distill_lr', 'all']:
        all_results['distill_lr'] = sweep_distill_lr(args.dataset, n_classes, args.rounds)
    
    if args.sweep in ['temperature', 'all']:
        all_results['temperature'] = sweep_temperature(args.dataset, n_classes, args.rounds)
    
    if args.sweep in ['distill_epochs', 'all']:
        all_results['distill_epochs'] = sweep_distill_epochs(args.dataset, n_classes, args.rounds)
    
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.save}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for param_name, results in all_results.items():
        if param_name == 'config':
            continue
        best = max(results, key=lambda x: x['best_acc'])
        key = {'distill_lr': 'distill_lr', 'temperature': 'temperature', 
               'gamma': 'gamma', 'lambda': 'lambda_mult',
               'distill_epochs': 'distill_epochs'}[param_name]
        print(f"  Best {param_name}: {best[key]} (best_acc={best['best_acc']:.2f}%)")


if __name__ == '__main__':
    main()