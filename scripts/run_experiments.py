#!/usr/bin/env python3
"""
Unified Experiment Runner for TMC Paper

Runs all methods with consistent settings for fair comparison.
Supports multiple seeds for error bars.

Usage:
  # Phase 1: Parameter sensitivity (PAID-FD only)
  python3 -u scripts/run_experiments.py --phase sensitivity --rounds 50

  # Phase 2: All methods comparison
  python3 -u scripts/run_experiments.py --phase comparison --rounds 50 --seeds 3

  # Phase 3: Privacy-accuracy tradeoff
  python3 -u scripts/run_experiments.py --phase privacy --rounds 50

  # Quick test (5 rounds, 1 seed)
  python3 -u scripts/run_experiments.py --phase comparison --rounds 5 --seeds 1

  # STL-10 cross-domain test
  python3 -u scripts/run_experiments.py --phase comparison --rounds 25 --public stl10
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np

# ════════════════════════════════════════════════════════
# Shared setup
# ════════════════════════════════════════════════════════

def setup_data(public_source='cifar100', n_public=10000, seed=42):
    """
    Setup datasets.

    Args:
        public_source: 'cifar100' (safe split) or 'stl10' (cross-domain)
        n_public: Number of public samples
        seed: Random seed

    Returns:
        (private_dataset, public_dataset, test_dataset, train_targets)
    """
    from torch.utils.data import DataLoader

    if public_source == 'cifar100':
        from src.data.datasets import load_cifar100_safe_split
        private, public, test = load_cifar100_safe_split(root='./data', n_public=n_public, seed=seed)
        train_targets = np.array(private.dataset.targets)[private.indices]
    elif public_source == 'stl10':
        from src.data.datasets import load_cifar100, load_stl10
        cifar_train, test = load_cifar100(root='./data')
        private = cifar_train  # Use full 50k for private
        public = load_stl10(root='./data', n_samples=n_public, resize_to=32)
        train_targets = np.array(cifar_train.targets)
    else:
        raise ValueError(f"Unknown public_source: {public_source}")

    return private, public, test, train_targets


def setup_devices(n_devices=10, lambda_multiplier=1.0, seed=42):
    """Create heterogeneous devices."""
    from src.devices.heterogeneity import HeterogeneityGenerator

    config_override = {
        'privacy_sensitivity': {
            'levels': {
                'low': {'value': 0.1 * lambda_multiplier, 'ratio': 0.40},
                'medium': {'value': 0.5 * lambda_multiplier, 'ratio': 0.40},
                'high': {'value': 1.0 * lambda_multiplier, 'ratio': 0.20},
            }
        }
    }
    gen = HeterogeneityGenerator(n_devices=n_devices, config_override=config_override, seed=seed)
    return gen.generate()


def create_method(method_name, n_classes=100, device='cuda', **kwargs):
    """
    Factory to create a method instance.

    Args:
        method_name: One of 'PAID-FD', 'FedMD', 'FedAvg', 'Fixed-eps-X', 'No-Privacy'
        n_classes: Number of classes
        device: torch device
        **kwargs: Override config parameters

    Returns:
        (method_instance, method_label)
    """
    from src.models import get_model

    model = get_model('resnet18', num_classes=n_classes)

    if method_name == 'PAID-FD':
        from src.methods.paid_fd import PAIDFD, PAIDFDConfig
        config = PAIDFDConfig(
            gamma=kwargs.get('gamma', 500.0),
            distill_lr=kwargs.get('distill_lr', 0.005),
            temperature=kwargs.get('temperature', 3.0),
            local_epochs=kwargs.get('local_epochs', 20),
            local_lr=kwargs.get('local_lr', 0.1),
            distill_epochs=kwargs.get('distill_epochs', 10),
            public_samples=kwargs.get('public_samples', 10000),
        )
        method = PAIDFD(model, config, n_classes=n_classes, device=device)
        return method, 'PAID-FD'

    elif method_name == 'FedMD':
        from src.methods.fedmd import FedMD, FedMDConfig
        config = FedMDConfig(
            local_epochs=kwargs.get('local_epochs', 20),
            local_lr=kwargs.get('local_lr', 0.1),
            distill_lr=kwargs.get('distill_lr', 0.005),
            temperature=kwargs.get('temperature', 3.0),
            distill_epochs=kwargs.get('distill_epochs', 10),
            clip_bound=kwargs.get('clip_bound', 5.0),
        )
        method = FedMD(model, config, n_classes=n_classes, device=device)
        return method, 'FedMD'

    elif method_name == 'FedAvg':
        from src.methods.fedavg import FedAvg, FedAvgConfig
        config = FedAvgConfig(
            local_epochs=kwargs.get('local_epochs', 20),
            local_lr=kwargs.get('local_lr', 0.1),
            participation_rate=kwargs.get('participation_rate', 1.0),
        )
        method = FedAvg(model, config, n_classes=n_classes, device=device)
        return method, 'FedAvg'

    elif method_name == 'No-Privacy':
        # FedMD with no noise = upper bound for FD
        from src.methods.fedmd import FedMD, FedMDConfig
        config = FedMDConfig(
            local_epochs=kwargs.get('local_epochs', 20),
            local_lr=kwargs.get('local_lr', 0.1),
            distill_lr=kwargs.get('distill_lr', 0.005),
            temperature=kwargs.get('temperature', 3.0),
            distill_epochs=kwargs.get('distill_epochs', 10),
        )
        method = FedMD(model, config, n_classes=n_classes, device=device)
        return method, 'No-Privacy'

    elif method_name == 'CSRA':
        from src.methods.csra import CSRA, CSRAConfig
        config = CSRAConfig(
            local_epochs=kwargs.get('local_epochs', 20),
            local_lr=kwargs.get('local_lr', 0.1),
            budget_per_round=kwargs.get('budget_per_round', 50.0),
            clip_norm=kwargs.get('clip_norm', 1.0),
        )
        method = CSRA(model, config, n_classes=n_classes, device=device)
        return method, 'CSRA'

    elif method_name == 'FedGMKD':
        from src.methods.fedgmkd import FedGMKD, FedGMKDConfig
        config = FedGMKDConfig(
            local_epochs=kwargs.get('local_epochs', 20),
            local_lr=kwargs.get('local_lr', 0.1),
            alpha=kwargs.get('alpha', 0.5),
            beta=kwargs.get('beta', 0.5),
            temperature=kwargs.get('temperature', 3.0),
            n_gmm_components=kwargs.get('n_gmm_components', 2),
        )
        method = FedGMKD(model, config, n_classes=n_classes, device=device)
        return method, 'FedGMKD'

    elif method_name.startswith('Fixed-eps-'):
        eps = float(method_name.split('-')[-1])
        from src.methods.fixed_eps import FixedEpsilon, FixedEpsilonConfig
        config = FixedEpsilonConfig(
            epsilon=eps,
            local_epochs=kwargs.get('local_epochs', 20),
            local_lr=kwargs.get('local_lr', 0.1),
            distill_lr=kwargs.get('distill_lr', 0.005),
            temperature=kwargs.get('temperature', 3.0),
            distill_epochs=kwargs.get('distill_epochs', 10),
            clip_bound=kwargs.get('clip_bound', 5.0),
        )
        method = FixedEpsilon(model, config, n_classes=n_classes, device=device)
        return method, f'Fixed-ε={eps}'

    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_single(method_name, devices, client_loaders, public_loader, test_loader,
               n_rounds=50, n_classes=100, device='cuda', **kwargs):
    """Run a single method and return results dict."""
    from src.utils.seed import set_seed
    seed = kwargs.pop('seed', 42)
    set_seed(seed)

    method, label = create_method(method_name, n_classes=n_classes, device=device, **kwargs)

    accuracies = []
    losses = []
    extras = []
    start_time = time.time()

    print(f"    [{label}] ", end="", flush=True)

    for r in range(n_rounds):
        result = method.run_round(
            round_idx=r,
            devices=devices,
            client_loaders=client_loaders,
            public_loader=public_loader,
            test_loader=test_loader,
        )
        accuracies.append(result.accuracy * 100)
        losses.append(result.loss)
        extras.append(result.extra)

        if (r + 1) % 5 == 0:
            print(f"R{r+1}:{result.accuracy*100:.1f}% ", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"({elapsed:.0f}s)")

    return {
        'method': label,
        'seed': seed,
        'accuracies': accuracies,
        'losses': losses,
        'final_acc': accuracies[-1],
        'best_acc': max(accuracies),
        'avg_last5': float(np.mean(accuracies[-5:])),
        'elapsed_sec': elapsed,
        'extras': extras,
    }


# ════════════════════════════════════════════════════════
# Phase 1: Parameter Sensitivity
# ════════════════════════════════════════════════════════

def phase_sensitivity(args):
    """Run gamma and lambda sensitivity sweeps."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Parameter Sensitivity Analysis")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {'phase': 'sensitivity', 'timestamp': datetime.now().isoformat()}

    # ── Gamma sweep ──
    print("\n── Gamma Sensitivity ──")
    gamma_values = [50, 100, 200, 500, 1000]
    gamma_results = []

    for gamma in gamma_values:
        print(f"\n  γ = {gamma}")
        private, public, test, train_targets = setup_data(args.public, seed=42)
        from src.data.partition import DirichletPartitioner, create_client_loaders
        from torch.utils.data import DataLoader
        from src.utils.seed import set_seed
        set_seed(42)

        partitioner = DirichletPartitioner(alpha=0.5, n_clients=10, seed=42)
        client_indices = partitioner.partition(private, train_targets)
        client_loaders = create_client_loaders(private, client_indices, batch_size=32)
        test_loader = DataLoader(test, batch_size=128, shuffle=False)
        public_loader = DataLoader(public, batch_size=128, shuffle=True)

        devices = setup_devices(n_devices=10, lambda_multiplier=1.0, seed=42)
        for dev in devices:
            if dev.device_id in client_indices:
                dev.data_size = len(client_indices[dev.device_id])

        r = run_single('PAID-FD', devices, client_loaders, public_loader, test_loader,
                        n_rounds=args.rounds, device=device, gamma=gamma, seed=42)
        r['gamma'] = gamma
        gamma_results.append(r)

    results['gamma'] = gamma_results

    # ── Lambda sweep ──
    print("\n── Lambda Sensitivity ──")
    lambda_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    lambda_results = []

    for lam in lambda_values:
        print(f"\n  λ = {lam}")
        private, public, test, train_targets = setup_data(args.public, seed=42)
        from src.data.partition import DirichletPartitioner, create_client_loaders
        from torch.utils.data import DataLoader
        from src.utils.seed import set_seed
        set_seed(42)

        partitioner = DirichletPartitioner(alpha=0.5, n_clients=10, seed=42)
        client_indices = partitioner.partition(private, train_targets)
        client_loaders = create_client_loaders(private, client_indices, batch_size=32)
        test_loader = DataLoader(test, batch_size=128, shuffle=False)
        public_loader = DataLoader(public, batch_size=128, shuffle=True)

        devices = setup_devices(n_devices=10, lambda_multiplier=lam, seed=42)
        for dev in devices:
            if dev.device_id in client_indices:
                dev.data_size = len(client_indices[dev.device_id])

        r = run_single('PAID-FD', devices, client_loaders, public_loader, test_loader,
                        n_rounds=args.rounds, device=device, seed=42)
        r['lambda'] = lam
        lambda_results.append(r)

    results['lambda'] = lambda_results

    return results


# ════════════════════════════════════════════════════════
# Phase 2: All Methods Comparison
# ════════════════════════════════════════════════════════

def phase_comparison(args):
    """Run all methods for convergence comparison."""
    print("\n" + "=" * 70)
    print("  PHASE 2: All Methods Comparison")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    methods = ['PAID-FD', 'FedMD', 'FedAvg', 'CSRA', 'FedGMKD', 'Fixed-eps-1.0', 'Fixed-eps-5.0']
    seeds = list(range(42, 42 + args.seeds))

    results = {
        'phase': 'comparison',
        'timestamp': datetime.now().isoformat(),
        'methods': {},
    }

    for method_name in methods:
        print(f"\n{'─'*50}")
        print(f"  Method: {method_name}")
        print(f"{'─'*50}")

        method_runs = []
        for seed in seeds:
            print(f"\n  Seed {seed}:")

            private, public, test, train_targets = setup_data(args.public, seed=seed)
            from src.data.partition import DirichletPartitioner, create_client_loaders
            from torch.utils.data import DataLoader
            from src.utils.seed import set_seed
            set_seed(seed)

            partitioner = DirichletPartitioner(alpha=0.5, n_clients=10, seed=seed)
            client_indices = partitioner.partition(private, train_targets)
            client_loaders = create_client_loaders(private, client_indices, batch_size=32)
            test_loader = DataLoader(test, batch_size=128, shuffle=False)
            public_loader = DataLoader(public, batch_size=128, shuffle=True)

            devices = setup_devices(n_devices=10, lambda_multiplier=1.0, seed=seed)
            for dev in devices:
                if dev.device_id in client_indices:
                    dev.data_size = len(client_indices[dev.device_id])

            r = run_single(method_name, devices, client_loaders, public_loader, test_loader,
                            n_rounds=args.rounds, device=device, seed=seed)
            method_runs.append(r)

        results['methods'][method_name] = method_runs

        # Print summary for this method
        accs = [r['best_acc'] for r in method_runs]
        print(f"\n  >> {method_name}: best_acc = {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")

    return results


# ════════════════════════════════════════════════════════
# Phase 3: Privacy-Accuracy Tradeoff
# ════════════════════════════════════════════════════════

def phase_privacy(args):
    """Fixed-ε sweep to show privacy-accuracy tradeoff."""
    print("\n" + "=" * 70)
    print("  PHASE 3: Privacy-Accuracy Tradeoff")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    results = {
        'phase': 'privacy',
        'timestamp': datetime.now().isoformat(),
        'fixed_eps': [],
        'paid_fd': None,
        'no_privacy': None,
    }

    # Setup data once
    private, public, test, train_targets = setup_data(args.public, seed=42)
    from src.data.partition import DirichletPartitioner, create_client_loaders
    from torch.utils.data import DataLoader
    from src.utils.seed import set_seed

    set_seed(42)
    partitioner = DirichletPartitioner(alpha=0.5, n_clients=10, seed=42)
    client_indices = partitioner.partition(private, train_targets)
    client_loaders = create_client_loaders(private, client_indices, batch_size=32)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)
    public_loader = DataLoader(public, batch_size=128, shuffle=True)

    devices = setup_devices(n_devices=10, lambda_multiplier=1.0, seed=42)
    for dev in devices:
        if dev.device_id in client_indices:
            dev.data_size = len(client_indices[dev.device_id])

    # Run Fixed-ε for each ε
    for eps in eps_values:
        print(f"\n  Fixed-ε = {eps}")
        set_seed(42)
        r = run_single(f'Fixed-eps-{eps}', devices, client_loaders, public_loader, test_loader,
                        n_rounds=args.rounds, device=device, seed=42)
        r['epsilon'] = eps
        results['fixed_eps'].append(r)

    # Run PAID-FD (adaptive)
    print(f"\n  PAID-FD (adaptive)")
    set_seed(42)
    r = run_single('PAID-FD', devices, client_loaders, public_loader, test_loader,
                    n_rounds=args.rounds, device=device, seed=42)
    results['paid_fd'] = r

    # Run No-Privacy (upper bound)
    print(f"\n  No-Privacy (upper bound)")
    set_seed(42)
    r = run_single('No-Privacy', devices, client_loaders, public_loader, test_loader,
                    n_rounds=args.rounds, device=device, seed=42)
    results['no_privacy'] = r

    return results


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='TMC Unified Experiment Runner')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['sensitivity', 'comparison', 'privacy', 'all'],
                        help='Which experiment phase to run')
    parser.add_argument('--rounds', type=int, default=50,
                        help='Number of rounds (default: 50)')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of seeds for error bars (default: 3)')
    parser.add_argument('--public', type=str, default='cifar100',
                        choices=['cifar100', 'stl10'],
                        help='Public data source (default: cifar100)')
    parser.add_argument('--save-dir', type=str, default='results/experiments',
                        help='Directory to save results')
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TMC PAID-FD Experiment Runner")
    print("=" * 70)
    print(f"  Phase:   {args.phase}")
    print(f"  Rounds:  {args.rounds}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Public:  {args.public}")
    print(f"  Device:  {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Save to: {save_dir}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if args.phase in ['sensitivity', 'all']:
        results = phase_sensitivity(args)
        fname = save_dir / f"phase1_sensitivity_{args.public}_{timestamp}.json"
        with open(fname, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved: {fname}")

    if args.phase in ['comparison', 'all']:
        results = phase_comparison(args)
        fname = save_dir / f"phase2_comparison_{args.public}_{timestamp}.json"
        with open(fname, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved: {fname}")

    if args.phase in ['privacy', 'all']:
        results = phase_privacy(args)
        fname = save_dir / f"phase3_privacy_{args.public}_{timestamp}.json"
        with open(fname, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved: {fname}")

    print("\n" + "=" * 70)
    print("  ALL DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
