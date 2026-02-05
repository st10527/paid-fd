#!/usr/bin/env python3
"""
Parameter Sweep for PAID-FD

This script runs experiments with different parameter combinations
to find reasonable parameter ranges before running full experiments.

Usage:
    python3 scripts/param_sweep.py --sweep distill_lr --rounds 30
    python3 scripts/param_sweep.py --sweep gamma --rounds 30
    python3 scripts/param_sweep.py --sweep temperature --rounds 30
    python3 scripts/param_sweep.py --sweep lambda --rounds 30
    python3 scripts/param_sweep.py --sweep all --rounds 30
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np


def run_single_experiment(
    gamma=10.0,
    distill_lr=0.001,
    temperature=3.0,
    lambda_multiplier=1.0,
    n_rounds=30,
    n_devices=30,
    seed=42
):
    """Run a single experiment with given parameters."""
    
    from src.utils.seed import set_seed
    from src.data.datasets import create_synthetic_datasets
    from src.data.partition import DirichletPartitioner, create_client_loaders
    from src.devices.heterogeneity import HeterogeneityGenerator
    from src.models import get_model
    from src.methods.paid_fd import PAIDFD, PAIDFDConfig
    from torch.utils.data import DataLoader
    
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create data
    train, test, public = create_synthetic_datasets(
        n_train=n_devices * 100,
        n_test=1000,
        n_public=2000,
        n_classes=100,
        seed=seed
    )
    
    # Partition
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=n_devices, seed=seed)
    client_indices = partitioner.partition(train, train.targets)
    client_loaders = create_client_loaders(train, client_indices, batch_size=32)
    
    test_loader = DataLoader(test, batch_size=128, shuffle=False)
    public_loader = DataLoader(public, batch_size=32, shuffle=False)
    
    # Create devices with lambda multiplier
    config_override = None
    if lambda_multiplier != 1.0:
        config_override = {
            'privacy_sensitivity': {
                'levels': {
                    'low': {'value': 0.01 * lambda_multiplier, 'ratio': 0.40},
                    'medium': {'value': 0.05 * lambda_multiplier, 'ratio': 0.40},
                    'high': {'value': 0.10 * lambda_multiplier, 'ratio': 0.20},
                }
            }
        }
    
    gen = HeterogeneityGenerator(
        n_devices=n_devices, 
        config_override=config_override,
        seed=seed
    )
    devices = gen.generate()
    
    for dev in devices:
        if dev.device_id in client_indices:
            dev.data_size = len(client_indices[dev.device_id])
    
    # Create model and method
    model = get_model('simple_cnn', num_classes=100)
    
    config = PAIDFDConfig(
        gamma=gamma,
        distill_lr=distill_lr,
        temperature=temperature,
        local_epochs=1,
        distill_epochs=5,
        public_samples=500
    )
    
    method = PAIDFD(model, config, n_classes=100, device=device)
    
    # Run training
    accuracies = []
    for round_idx in range(n_rounds):
        result = method.run_round(
            round_idx=round_idx,
            devices=devices,
            client_loaders=client_loaders,
            public_loader=public_loader,
            test_loader=test_loader
        )
        accuracies.append(result.accuracy * 100)
    
    # Get game result for analysis
    from src.game.stackelberg import StackelbergSolver
    solver = StackelbergSolver(gamma=gamma)
    game_result = solver.solve(devices)
    
    return {
        'final_acc': accuracies[-1],
        'best_acc': max(accuracies),
        'avg_last10': np.mean(accuracies[-10:]),
        'improvement': np.mean(accuracies[-10:]) - np.mean(accuracies[:10]),
        'price': game_result['price'],
        'avg_s': game_result['avg_s'],
        'avg_eps': game_result['avg_eps'],
        'participation': game_result['participation_rate'],
        'accuracies': accuracies
    }


def sweep_gamma(n_rounds=30):
    """Sweep gamma parameter."""
    print("\n" + "="*60)
    print("Sweeping: GAMMA (server valuation)")
    print("="*60)
    
    values = [5, 10, 20, 50, 100]
    results = []
    
    for gamma in values:
        print(f"\nRunning gamma={gamma}...")
        r = run_single_experiment(gamma=gamma, n_rounds=n_rounds)
        r['gamma'] = gamma
        results.append(r)
        print(f"  Acc: {r['final_acc']:.2f}%, price={r['price']:.3f}, "
              f"s*={r['avg_s']:.1f}, ε*={r['avg_eps']:.3f}")
    
    print("\n" + "-"*60)
    print(f"{'Gamma':>8} {'Final':>8} {'Best':>8} {'Price':>8} {'Avg s*':>8} {'Avg ε*':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['gamma']:>8} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% "
              f"{r['price']:>8.3f} {r['avg_s']:>8.1f} {r['avg_eps']:>8.3f}")
    
    return results


def sweep_distill_lr(n_rounds=30):
    """Sweep distillation learning rate."""
    print("\n" + "="*60)
    print("Sweeping: DISTILL_LR (distillation learning rate)")
    print("="*60)
    
    values = [0.0001, 0.001, 0.01, 0.05, 0.1]
    results = []
    
    for lr in values:
        print(f"\nRunning distill_lr={lr}...")
        r = run_single_experiment(distill_lr=lr, n_rounds=n_rounds)
        r['distill_lr'] = lr
        results.append(r)
        print(f"  Acc: {r['final_acc']:.2f}%, improvement: {r['improvement']:+.2f}%")
    
    print("\n" + "-"*60)
    print(f"{'LR':>10} {'Final':>8} {'Best':>8} {'Improve':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['distill_lr']:>10} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% "
              f"{r['improvement']:>+9.2f}%")
    
    return results


def sweep_temperature(n_rounds=30):
    """Sweep distillation temperature."""
    print("\n" + "="*60)
    print("Sweeping: TEMPERATURE (distillation temperature)")
    print("="*60)
    
    values = [1.0, 2.0, 3.0, 5.0, 10.0]
    results = []
    
    for temp in values:
        print(f"\nRunning temperature={temp}...")
        r = run_single_experiment(temperature=temp, n_rounds=n_rounds)
        r['temperature'] = temp
        results.append(r)
        print(f"  Acc: {r['final_acc']:.2f}%, improvement: {r['improvement']:+.2f}%")
    
    print("\n" + "-"*60)
    print(f"{'Temp':>8} {'Final':>8} {'Best':>8} {'Improve':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['temperature']:>8.1f} {r['final_acc']:>7.2f}% {r['best_acc']:>7.2f}% "
              f"{r['improvement']:>+9.2f}%")
    
    return results


def sweep_lambda(n_rounds=30):
    """Sweep lambda (privacy sensitivity) multiplier."""
    print("\n" + "="*60)
    print("Sweeping: LAMBDA multiplier (privacy sensitivity)")
    print("="*60)
    print("Base λ values: [0.01, 0.05, 0.10]")
    print("Multiplier scales all λ values")
    
    values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    for mult in values:
        print(f"\nRunning λ_multiplier={mult} (λ range: [{0.01*mult:.3f}, {0.1*mult:.3f}])...")
        r = run_single_experiment(lambda_multiplier=mult, n_rounds=n_rounds)
        r['lambda_mult'] = mult
        r['lambda_range'] = f"[{0.01*mult:.3f}, {0.1*mult:.3f}]"
        results.append(r)
        print(f"  Acc: {r['final_acc']:.2f}%, ε*={r['avg_eps']:.3f}, s*={r['avg_s']:.1f}")
    
    print("\n" + "-"*60)
    print(f"{'λ_mult':>8} {'λ_range':>18} {'Final':>8} {'ε*':>8} {'s*':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['lambda_mult']:>8.1f} {r['lambda_range']:>18} {r['final_acc']:>7.2f}% "
              f"{r['avg_eps']:>8.3f} {r['avg_s']:>8.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep for PAID-FD')
    parser.add_argument('--sweep', type=str, required=True,
                       choices=['gamma', 'distill_lr', 'temperature', 'lambda', 'all'],
                       help='Which parameter to sweep')
    parser.add_argument('--rounds', type=int, default=30,
                       help='Number of rounds per experiment (default: 30)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save results to JSON file')
    args = parser.parse_args()
    
    print("="*60)
    print("PAID-FD Parameter Sweep")
    print("="*60)
    print(f"Rounds per experiment: {args.rounds}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    all_results = {}
    
    if args.sweep in ['gamma', 'all']:
        all_results['gamma'] = sweep_gamma(args.rounds)
    
    if args.sweep in ['distill_lr', 'all']:
        all_results['distill_lr'] = sweep_distill_lr(args.rounds)
    
    if args.sweep in ['temperature', 'all']:
        all_results['temperature'] = sweep_temperature(args.rounds)
    
    if args.sweep in ['lambda', 'all']:
        all_results['lambda'] = sweep_lambda(args.rounds)
    
    # Save results
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.save}")
    
    print("\n" + "="*60)
    print("Sweep complete!")
    print("="*60)


if __name__ == '__main__':
    main()
