#!/usr/bin/env python3
"""
PAID-FD Unified Experiment Runner (Figure-Oriented)
====================================================

Master script that runs all 7 phases of experiments and generates
all 11 figures + 1 table for the TMC paper.

Usage:
    # Run all phases
    python scripts/run_all_experiments.py --all
    
    # Run specific phase
    python scripts/run_all_experiments.py --phase 1.1
    python scripts/run_all_experiments.py --phase 2
    
    # Quick test (synthetic data, fewer rounds)
    python scripts/run_all_experiments.py --phase 1.1 --quick
    
    # Specify device
    python scripts/run_all_experiments.py --all --device cuda:0
    
    # Skip phases that already have results
    python scripts/run_all_experiments.py --all --skip-existing
"""

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import yaml
import numpy as np

# ============================================================================
# Utility Functions
# ============================================================================

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  💾 Saved: {path}")


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def result_path(phase: str, seed: int = None) -> str:
    """Get result file path. If seed is given, returns per-seed path for parallel runs."""
    if seed is not None:
        return str(PROJECT_ROOT / "results" / "experiments" / f"{phase}_seed{seed}.json")
    return str(PROJECT_ROOT / "results" / "experiments" / f"{phase}.json")


def result_exists(phase: str) -> bool:
    return Path(result_path(phase)).exists()


def save_phase_results(results: dict, phase: str, seeds: list):
    """Save results. If single seed → per-seed file; if multi-seed → combined file."""
    if len(seeds) == 1:
        save_json(results, result_path(phase, seed=seeds[0]))
    else:
        save_json(results, result_path(phase))


def merge_seed_results(phase: str, seeds: list):
    """
    Merge per-seed result files into a single combined result file.
    
    Per-seed files:  phase1_gamma_seed42.json, phase1_gamma_seed123.json, ...
    Merged file:     phase1_gamma.json
    """
    print(f"\n🔀 Merging results for {phase}...")
    
    # Collect per-seed files
    seed_files = []
    for s in seeds:
        p = result_path(phase, seed=s)
        if Path(p).exists():
            seed_files.append((s, p))
            print(f"  Found: {Path(p).name}")
        else:
            print(f"  ⚠️  Missing: {Path(p).name}")
    
    if not seed_files:
        print("  ❌ No seed files found, nothing to merge.")
        return
    
    # Load first file as template
    first_data = load_json(seed_files[0][1])
    
    # Merge strategy depends on phase structure
    if 'runs' in first_data and isinstance(first_data['runs'], dict):
        # Phase 1.1, 1.2, 2, 3, 5, 6, 7 style: runs[key] = [list of seed runs]
        merged = copy.deepcopy(first_data)
        
        # Check if runs values are lists of run-dicts (multi-key like gamma/method)
        sample_val = next(iter(merged['runs'].values()))
        
        if isinstance(sample_val, list) and len(sample_val) > 0 and isinstance(sample_val[0], dict):
            # runs[key] = [run_dict, ...] — one per seed in original format
            # In per-seed files, each key has exactly 1 run. Merge = concatenate.
            for seed_val, path in seed_files[1:]:
                other = load_json(path)
                for key in other.get('runs', {}):
                    if key in merged['runs']:
                        merged['runs'][key].extend(other['runs'][key])
                    else:
                        merged['runs'][key] = other['runs'][key]
        elif isinstance(sample_val, dict):
            # Nested dict like phase5: runs[level][method] = [runs]
            for seed_val, path in seed_files[1:]:
                other = load_json(path)
                for level_key in other.get('runs', {}):
                    if level_key not in merged['runs']:
                        merged['runs'][level_key] = other['runs'][level_key]
                    else:
                        for method_key in other['runs'][level_key]:
                            if method_key in merged['runs'][level_key]:
                                merged['runs'][level_key][method_key].extend(
                                    other['runs'][level_key][method_key])
                            else:
                                merged['runs'][level_key][method_key] = \
                                    other['runs'][level_key][method_key]
    
    elif 'runs' in first_data and isinstance(first_data['runs'], list):
        # Phase 4 style: runs = [run_dict, run_dict, ...]
        merged = copy.deepcopy(first_data)
        for seed_val, path in seed_files[1:]:
            other = load_json(path)
            merged['runs'].extend(other.get('runs', []))
    
    else:
        # Fallback: just use first file
        merged = first_data
        print("  ⚠️  Unknown structure, using first seed file only.")
    
    # Also merge paid_fd_runs / fixed_eps_runs if present (Phase 3)
    for extra_key in ['paid_fd_runs', 'fixed_eps_runs']:
        if extra_key in first_data:
            if extra_key not in merged:
                merged[extra_key] = copy.deepcopy(first_data[extra_key])
            for seed_val, path in seed_files[1:]:
                other = load_json(path)
                for k in other.get(extra_key, {}):
                    if k in merged[extra_key]:
                        merged[extra_key][k].extend(other[extra_key][k])
                    else:
                        merged[extra_key][k] = other[extra_key][k]
    
    # Save merged result
    out_path = result_path(phase)
    save_json(merged, out_path)
    n_seeds = len(seed_files)
    print(f"  ✅ Merged {n_seeds} seeds → {Path(out_path).name}")


def get_device(device_arg: str) -> str:
    import torch
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def setup_seed(seed: int):
    from src.utils.seed import set_seed
    set_seed(seed)


# ============================================================================
# Core Training Loop (shared across all phases)
# ============================================================================

def run_single_experiment(
    method_name: str,
    config: dict,
    seed: int,
    device: str,
    n_rounds: int = 50,
    save_decisions: bool = False,
    verbose: bool = True
) -> dict:
    """
    Run a single experiment with given method and config.
    
    Returns:
        dict with keys: accuracies, losses, participation_rates, 
                        prices, avg_eps_list, avg_s_list, 
                        device_decisions (if save_decisions),
                        energy_history, elapsed_sec, extras
    """
    import torch
    from src.utils.seed import set_seed
    from src.data.datasets import load_cifar100_safe_split, create_synthetic_datasets
    from src.data.partition import DirichletPartitioner, create_client_loaders
    from src.devices.heterogeneity import HeterogeneityGenerator
    from src.models import get_model
    from torch.utils.data import DataLoader
    
    set_seed(seed)
    
    n_devices = config.get('n_devices', 50)
    synthetic = config.get('synthetic', False)
    
    # ---- Data ----
    if synthetic:
        train_data, test_data, public_data = create_synthetic_datasets(
            n_train=n_devices * 100, n_test=1000, n_public=5000, seed=seed
        )
        targets = np.array(train_data.targets)
    else:
        # Use CIFAR-100 only (safe split: no data leakage)
        # Returns: (private_subset, public_subset, test_set)
        train_data, public_data, test_data = load_cifar100_safe_split(
            root='./data',
            n_public=config.get('public_samples', 20000),
            seed=seed
        )
        # train_data is a Subset — extract targets aligned to its 0-based indexing
        all_targets = np.array(train_data.dataset.targets)
        targets = all_targets[train_data.indices]
    
    partitioner = DirichletPartitioner(
        alpha=config.get('alpha', 0.5),
        n_clients=n_devices,
        min_samples_per_client=10,
        seed=seed
    )
    client_indices = partitioner.partition(train_data, targets)
    client_loaders = create_client_loaders(train_data, client_indices, batch_size=128)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    public_loader = DataLoader(public_data, batch_size=128, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    
    # ---- Devices ----
    het_config = config.get('heterogeneity', {})
    generator = HeterogeneityGenerator(
        n_devices=n_devices,
        config_path=het_config.get('config_file', 'config/devices/heterogeneity.yaml'),
        config_override=het_config.get('overrides', None),
        seed=seed
    )
    devices = generator.generate()
    
    for dev in devices:
        if dev.device_id in client_indices:
            dev.data_size = len(client_indices[dev.device_id])
    
    # ---- Method ----
    model = get_model(config.get('model', 'resnet18'), num_classes=100)
    method = _create_method(method_name, model, config, device)
    
    # ---- Training loop ----
    accuracies = []
    losses = []
    participation_rates = []
    prices = []
    avg_eps_list = []
    avg_s_list = []
    device_decisions_history = []
    energy_history = []
    extras_history = []
    
    start_time = time.time()
    
    for r in range(n_rounds):
        result = method.run_round(
            round_idx=r,
            devices=devices,
            client_loaders=client_loaders,
            public_loader=public_loader,
            test_loader=test_loader
        )
        
        accuracies.append(result.accuracy)
        losses.append(result.loss)
        participation_rates.append(result.participation_rate)
        energy_history.append(result.energy)
        
        extra = result.extra or {}
        extras_history.append(extra)
        prices.append(extra.get('price', 0))
        avg_eps_list.append(extra.get('avg_eps', 0))
        avg_s_list.append(extra.get('avg_s', 0))
        
        if verbose and (r % 10 == 0 or r == n_rounds - 1):
            print(f"    Round {r:3d}/{n_rounds}: acc={result.accuracy:.4f}, "
                  f"loss={result.loss:.4f}, part={result.participation_rate:.2f}")
    
    elapsed = time.time() - start_time
    
    return {
        'method': method_name,
        'seed': seed,
        'n_rounds': n_rounds,
        'accuracies': accuracies,
        'losses': losses,
        'participation_rates': participation_rates,
        'prices': prices,
        'avg_eps': avg_eps_list,
        'avg_s': avg_s_list,
        'energy_history': energy_history,
        'extras': extras_history,
        'final_accuracy': accuracies[-1] if accuracies else 0,
        'best_accuracy': max(accuracies) if accuracies else 0,
        'elapsed_sec': elapsed,
        'config': {k: v for k, v in config.items() 
                   if k not in ['heterogeneity']}  # avoid serializing large objects
    }


def _create_method(method_name: str, model, config: dict, device: str):
    """Create a method instance from name and config."""
    from src.methods import (PAIDFD, FixedEpsilon, FedAvg, FedMD, FedGMKD, CSRA)
    from src.methods.paid_fd import PAIDFDConfig
    from src.methods.fixed_eps import FixedEpsilonConfig
    from src.methods.fedavg import FedAvgConfig
    from src.methods.fedmd import FedMDConfig
    from src.methods.fedgmkd import FedGMKDConfig
    from src.methods.csra import CSRAConfig
    from src.models.utils import copy_model
    
    mc = config.get('method_config', {})
    tc = config  # training config is at top level
    
    local_epochs = tc.get('local_epochs', 3)
    local_lr = tc.get('local_lr', 0.01)
    distill_epochs = tc.get('distill_epochs', 5)
    distill_lr = tc.get('distill_lr', 0.001)
    temperature = tc.get('temperature', 3.0)
    
    m = copy_model(model, device=device)
    
    if method_name == 'PAID-FD':
        cfg = PAIDFDConfig(
            gamma=mc.get('gamma', tc.get('gamma', 10.0)),
            delta=mc.get('delta', 0.01),
            budget=float(mc.get('budget', 'inf')),
            local_epochs=local_epochs,
            local_lr=local_lr,
            local_momentum=tc.get('local_momentum', 0.9),
            distill_epochs=1,       # 1 epoch/round
            distill_lr=0.001,       # Moderate lr (safe: EMA buffer is smooth)
            distill_alpha=mc.get('distill_alpha', 0.7),  # 70% KL + 30% CE
            temperature=3.0,        # Soft-label T=3
            ema_momentum=mc.get('ema_momentum', 0.9),    # EMA logit buffer
            pretrain_epochs=10,     # 10 ep: ~35-40% start, FL has room
            clip_bound=mc.get('clip_bound', 2.0),   # C=2: reduced sensitivity
            public_samples=mc.get('public_samples_per_round', 1000),
        )
        return PAIDFD(m, cfg, 100, device)
    
    elif method_name.startswith('Fixed-eps'):
        eps = float(method_name.split('-')[-1])
        cfg = FixedEpsilonConfig(
            epsilon=eps,
            local_epochs=local_epochs,
            local_lr=local_lr,
            distill_epochs=1,
            distill_lr=0.001,       # Match PAID-FD
            distill_alpha=0.7,      # 70% KL + 30% CE
            temperature=3.0,        # Soft-label T=3
            ema_momentum=0.9,       # EMA logit buffer
            pretrain_epochs=10,     # Match PAID-FD
            clip_bound=2.0,              # Match PAID-FD C=2
            participation_rate=mc.get('participation_rate', 1.0),
            samples_per_device=mc.get('samples_per_device', 100),
        )
        return FixedEpsilon(m, cfg, 100, device)
    
    elif method_name == 'FedAvg':
        cfg = FedAvgConfig(
            local_epochs=mc.get('local_epochs', 5),
            local_lr=mc.get('local_lr', 0.01),
            local_momentum=tc.get('local_momentum', 0.9),
            participation_rate=mc.get('participation_rate', 0.5),
        )
        return FedAvg(m, cfg, 100, device)
    
    elif method_name == 'FedMD':
        cfg = FedMDConfig(
            local_epochs=local_epochs,
            local_lr=local_lr,
            distill_epochs=distill_epochs,
            distill_lr=distill_lr,
            temperature=3.0,  # T=3 for noise-free (richer soft label signal)
            clip_bound=5.0,
        )
        return FedMD(m, cfg, 100, device)
    
    elif method_name == 'FedGMKD':
        cfg = FedGMKDConfig(
            alpha=mc.get('alpha', 0.1),
            beta=mc.get('beta', 1.0),
            tau=mc.get('tau', 0.5),
            participation_rate=mc.get('participation_rate', 0.5),
            local_epochs=mc.get('local_epochs', 5),
            local_lr=mc.get('local_lr', 0.01),
        )
        return FedGMKD(m, cfg, 100, device)
    
    elif method_name == 'CSRA':
        cfg = CSRAConfig(
            budget=mc.get('budget', 100.0),
            epsilon_menu=mc.get('epsilon_menu', [0.5, 1.0, 2.0, 5.0, 10.0]),
            participation_rate=mc.get('participation_rate', 0.5),
            local_epochs=mc.get('local_epochs', 5),
            local_lr=mc.get('local_lr', 0.01),
        )
        return CSRA(m, cfg, 100, device)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ============================================================================
# Phase Runners
# ============================================================================

def _phase1_base_config(gamma, n_devices=50, lambda_mult=None, clip_bound=2.0,
                         quick=False):
    """Build the base config dict for all Phase 1 experiments.
    
    Shared training hyper-parameters (from Fix 17):
    - local_epochs=5, local_lr=0.01, distill_lr=0.001
    - EMA buffer (momentum=0.9), distill_alpha=0.7, T=3.0, C configurable
    """
    config = {
        'n_devices': n_devices, 'gamma': gamma, 'alpha': 0.5,
        'local_epochs': 5, 'local_lr': 0.01, 'local_momentum': 0.9,
        'distill_epochs': 1, 'distill_lr': 0.001, 'temperature': 3.0,
        'public_samples': 20000,
        'synthetic': quick,
        'heterogeneity': {
            'config_file': 'config/devices/heterogeneity.yaml',
        },
        'method_config': {
            'gamma': gamma, 'delta': 0.01,
            'clip_bound': clip_bound,
            'ema_momentum': 0.9,
            'distill_alpha': 0.7,
        }
    }
    if lambda_mult is not None:
        config['heterogeneity']['overrides'] = {
            'privacy_sensitivity': {'lambda_mult': lambda_mult}
        }
    return config


def run_phase1_gamma(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 1.1: Gamma sensitivity analysis.
    
    γ ∈ {3, 5, 7, 10, 15}  (fixed: λ_mult=1.0, N=50, C=2)
    
    γ controls the server's valuation of quality — higher γ means the
    server is willing to pay more, attracting more participants.
    Key metrics: accuracy, participation rate, avg ε, communication cost.
    """
    print("\n" + "=" * 70)
    print("Phase 1.1: Gamma Sensitivity Analysis")
    print("=" * 70)
    
    gamma_values = [3, 5, 7, 10, 15]
    if quick:
        gamma_values = [3, 10]
    
    results = {'phase': 'phase1_gamma', 'gamma_values': gamma_values, 'runs': {}}
    
    for gamma in gamma_values:
        print(f"\n--- Gamma = {gamma} ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = _phase1_base_config(gamma, quick=quick)
            run = run_single_experiment('PAID-FD', config, seed, device, n_rounds)
            runs.append(run)
        results['runs'][str(gamma)] = runs
    
    save_phase_results(results, 'phase1_gamma', seeds)
    return results


def run_phase1_lambda(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 1.2: Lambda sensitivity analysis.
    
    λ_mult ∈ {0.3, 0.5, 1.0, 2.0, 3.0}  (fixed: γ=best from 1.1, N=50, C=2)
    
    λ_mult scales all devices' privacy cost coefficients.
    Higher λ → devices demand more compensation → fewer participants.
    """
    print("\n" + "=" * 70)
    print("Phase 1.2: Lambda Sensitivity Analysis")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    print(f"  Using best γ = {best_gamma} from Phase 1.1")
    
    lambda_values = [0.3, 0.5, 1.0, 2.0, 3.0]
    if quick:
        lambda_values = [0.5, 2.0]
    
    results = {
        'phase': 'phase1_lambda', 'best_gamma': best_gamma,
        'lambda_values': lambda_values, 'runs': {}
    }
    
    for lam in lambda_values:
        print(f"\n--- Lambda_mult = {lam} ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = _phase1_base_config(best_gamma, lambda_mult=lam, quick=quick)
            run = run_single_experiment('PAID-FD', config, seed, device, n_rounds)
            runs.append(run)
        results['runs'][str(lam)] = runs
    
    save_phase_results(results, 'phase1_lambda', seeds)
    return results


def run_phase1_ndevices(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 1.3: Number of devices (N) sensitivity analysis.
    
    N ∈ {10, 20, 30, 40, 50}  (fixed: γ=best from 1.1, λ_mult=1.0, C=2)
    
    More devices → more potential participants → richer ensemble.
    But also more non-IID data splits → harder local learning.
    """
    print("\n" + "=" * 70)
    print("Phase 1.3: N (Number of Devices) Sensitivity Analysis")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    print(f"  Using best γ = {best_gamma} from Phase 1.1")
    
    n_values = [10, 20, 30, 40, 50]
    if quick:
        n_values = [10, 50]
    
    results = {
        'phase': 'phase1_ndevices', 'best_gamma': best_gamma,
        'n_values': n_values, 'runs': {}
    }
    
    for n in n_values:
        print(f"\n--- N = {n} devices ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = _phase1_base_config(best_gamma, n_devices=n, quick=quick)
            run = run_single_experiment('PAID-FD', config, seed, device, n_rounds)
            runs.append(run)
        results['runs'][str(n)] = runs
    
    save_phase_results(results, 'phase1_ndevices', seeds)
    return results


def run_phase1_clipbound(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 1.4: Clip bound (C) sensitivity analysis.
    
    C ∈ {1, 2, 3, 4, 5}  (fixed: γ=best from 1.1, λ_mult=1.0, N=50)
    
    C controls the logit clipping range [-C, C]:
    - Lower C → smaller sensitivity (2C) → less noise per ε
    - But also clips informative logits → signal loss
    - Sweet spot balances noise reduction vs information preservation
    """
    print("\n" + "=" * 70)
    print("Phase 1.4: Clip Bound (C) Sensitivity Analysis")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    print(f"  Using best γ = {best_gamma} from Phase 1.1")
    
    c_values = [1, 2, 3, 4, 5]
    if quick:
        c_values = [1, 5]
    
    results = {
        'phase': 'phase1_clipbound', 'best_gamma': best_gamma,
        'c_values': c_values, 'runs': {}
    }
    
    for c in c_values:
        print(f"\n--- Clip bound C = {c} ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = _phase1_base_config(best_gamma, clip_bound=float(c), quick=quick)
            run = run_single_experiment('PAID-FD', config, seed, device, n_rounds)
            runs.append(run)
        results['runs'][str(c)] = runs
    
    save_phase_results(results, 'phase1_clipbound', seeds)
    return results


def run_phase2(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 2: Convergence & Performance Comparison."""
    print("\n" + "=" * 70)
    print("Phase 2: Convergence & Performance Comparison")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    print(f"  Using best γ = {best_gamma}")
    
    methods = [
        'PAID-FD', 'FedAvg', 'FedMD', 'FedGMKD', 'CSRA',
        'Fixed-eps-1.0', 'Fixed-eps-5.0', 'Fixed-eps-10.0'
    ]
    if quick:
        methods = ['PAID-FD', 'FedAvg', 'FedMD', 'Fixed-eps-5.0']
    
    method_configs = {
        'PAID-FD': {'gamma': best_gamma, 'delta': 0.01},
        'FedAvg': {'participation_rate': 0.5, 'local_epochs': 2, 'local_lr': 0.01},
        'FedMD': {},
        'FedGMKD': {
            'alpha': 0.1, 'beta': 1.0, 'tau': 0.5,
            'participation_rate': 0.5, 'local_epochs': 2, 'local_lr': 0.01
        },
        'CSRA': {
            'budget': 100.0,
            'epsilon_menu': [0.5, 1.0, 2.0, 5.0, 10.0],
            'participation_rate': 0.5, 'local_epochs': 2, 'local_lr': 0.01
        },
        'Fixed-eps-1.0': {'participation_rate': 1.0},
        'Fixed-eps-5.0': {'participation_rate': 1.0},
        'Fixed-eps-10.0': {'participation_rate': 1.0},
    }
    
    results = {
        'phase': 'phase2_convergence', 'best_gamma': best_gamma,
        'methods': methods, 'runs': {}
    }
    
    for method_name in methods:
        print(f"\n--- Method: {method_name} ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = {
                'n_devices': 50, 'gamma': best_gamma, 'alpha': 0.5,
                'local_epochs': 3, 'local_lr': 0.01, 'local_momentum': 0.9,
                'distill_epochs': 5, 'distill_lr': 0.001, 'temperature': 3.0,
                'synthetic': quick,
                'heterogeneity': {
                    'config_file': 'config/devices/heterogeneity.yaml',
                },
                'method_config': method_configs.get(method_name, {})
            }
            run = run_single_experiment(method_name, config, seed, device, n_rounds)
            runs.append(run)
        results['runs'][method_name] = runs
    
    save_phase_results(results, 'phase2_convergence', seeds)
    return results


def run_phase3(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 3: Privacy-Accuracy Tradeoff."""
    print("\n" + "=" * 70)
    print("Phase 3: Privacy-Accuracy Tradeoff")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    
    fixed_eps_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    paid_fd_lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    if quick:
        fixed_eps_values = [1.0, 5.0, 10.0]
        paid_fd_lambda_values = [0.5, 2.0, 10.0]
    
    results = {
        'phase': 'phase3_privacy', 'best_gamma': best_gamma,
        'fixed_eps_values': fixed_eps_values,
        'paid_fd_lambda_values': paid_fd_lambda_values,
        'fixed_eps_runs': {},
        'paid_fd_runs': {}
    }
    
    # Fixed-ε experiments
    for eps in fixed_eps_values:
        method_name = f'Fixed-eps-{eps}'
        print(f"\n--- {method_name} ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = {
                'n_devices': 50, 'alpha': 0.5,
                'local_epochs': 3, 'local_lr': 0.01, 'local_momentum': 0.9,
                'distill_epochs': 5, 'distill_lr': 0.001, 'temperature': 3.0,
                'synthetic': quick,
                'heterogeneity': {
                    'config_file': 'config/devices/heterogeneity.yaml',
                },
                'method_config': {'participation_rate': 1.0}
            }
            run = run_single_experiment(method_name, config, seed, device, n_rounds)
            runs.append(run)
        results['fixed_eps_runs'][str(eps)] = runs
    
    # PAID-FD with varying λ
    for lam in paid_fd_lambda_values:
        print(f"\n--- PAID-FD λ_mult={lam} ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = {
                'n_devices': 50, 'gamma': best_gamma, 'alpha': 0.5,
                'local_epochs': 3, 'local_lr': 0.01, 'local_momentum': 0.9,
                'distill_epochs': 5, 'distill_lr': 0.001, 'temperature': 3.0,
                'synthetic': quick,
                'heterogeneity': {
                    'config_file': 'config/devices/heterogeneity.yaml',
                    'overrides': {
                        'privacy_sensitivity': {'lambda_mult': lam}
                    }
                },
                'method_config': {'gamma': best_gamma, 'delta': 0.01}
            }
            run = run_single_experiment('PAID-FD', config, seed, device, n_rounds)
            runs.append(run)
        results['paid_fd_runs'][str(lam)] = runs
    
    save_phase_results(results, 'phase3_privacy', seeds)
    return results


def run_phase4(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 4: Incentive Mechanism Analysis."""
    print("\n" + "=" * 70)
    print("Phase 4: Incentive Mechanism Analysis")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    
    results = {
        'phase': 'phase4_incentive', 'best_gamma': best_gamma,
        'runs': []
    }
    
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        config = {
            'n_devices': 50, 'gamma': best_gamma, 'alpha': 0.5,
            'local_epochs': 3, 'local_lr': 0.01, 'local_momentum': 0.9,
            'distill_epochs': 5, 'distill_lr': 0.001, 'temperature': 3.0,
            'synthetic': quick,
            'heterogeneity': {
                'config_file': 'config/devices/heterogeneity.yaml',
            },
            'method_config': {'gamma': best_gamma, 'delta': 0.01}
        }
        run = run_single_experiment('PAID-FD', config, seed, device, n_rounds,
                                     save_decisions=True)
        results['runs'].append(run)
    
    save_phase_results(results, 'phase4_incentive', seeds)
    return results


def run_phase5(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 5: Heterogeneity Impact Analysis."""
    print("\n" + "=" * 70)
    print("Phase 5: Heterogeneity Impact Analysis")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    
    het_levels = {
        'Homogeneous': [1.0, 1.0, 1.0],
        'Mild': [0.8, 1.0, 1.2],
        'Strong': [0.5, 1.0, 2.0],
        'Extreme': [0.2, 1.0, 5.0],
    }
    if quick:
        het_levels = {'Homogeneous': [1.0, 1.0, 1.0], 'Strong': [0.5, 1.0, 2.0]}
    
    methods = ['PAID-FD', 'FedMD', 'Fixed-eps-5.0']
    if quick:
        methods = ['PAID-FD', 'FedMD']
    
    method_configs = {
        'PAID-FD': {'gamma': best_gamma, 'delta': 0.01},
        'FedMD': {},
        'Fixed-eps-5.0': {'participation_rate': 1.0},
    }
    
    results = {
        'phase': 'phase5_heterogeneity', 'best_gamma': best_gamma,
        'het_levels': {k: v for k, v in het_levels.items()},
        'methods': methods, 'runs': {}
    }
    
    for level_name, multipliers in het_levels.items():
        results['runs'][level_name] = {}
        for method_name in methods:
            print(f"\n--- {level_name} / {method_name} ---")
            runs = []
            for seed in seeds:
                print(f"  Seed {seed}:")
                config = {
                    'n_devices': 50, 'gamma': best_gamma, 'alpha': 0.5,
                    'local_epochs': 3, 'local_lr': 0.01, 'local_momentum': 0.9,
                    'distill_epochs': 5, 'distill_lr': 0.001, 'temperature': 3.0,
                    'synthetic': quick,
                    'heterogeneity': {
                        'config_file': 'config/devices/heterogeneity.yaml',
                        'overrides': {
                            'cost_parameters': {
                                'c_inf_multipliers': multipliers
                            }
                        }
                    },
                    'method_config': method_configs.get(method_name, {})
                }
                run = run_single_experiment(method_name, config, seed, device, n_rounds)
                runs.append(run)
            results['runs'][level_name][method_name] = runs
    
    save_phase_results(results, 'phase5_heterogeneity', seeds)
    return results


def run_phase6(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 6: Scalability Analysis."""
    print("\n" + "=" * 70)
    print("Phase 6: Scalability Analysis")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    
    n_devices_list = [10, 20, 30, 50, 70, 100]
    if quick:
        n_devices_list = [10, 30, 50]
    
    methods = ['PAID-FD', 'FedMD', 'Fixed-eps-5.0']
    if quick:
        methods = ['PAID-FD', 'FedMD']
    
    method_configs = {
        'PAID-FD': {'gamma': best_gamma, 'delta': 0.01},
        'FedMD': {},
        'Fixed-eps-5.0': {'participation_rate': 1.0},
    }
    
    results = {
        'phase': 'phase6_scalability', 'best_gamma': best_gamma,
        'n_devices_list': n_devices_list, 'methods': methods, 'runs': {}
    }
    
    for n_dev in n_devices_list:
        results['runs'][str(n_dev)] = {}
        for method_name in methods:
            print(f"\n--- N={n_dev} / {method_name} ---")
            runs = []
            for seed in seeds:
                print(f"  Seed {seed}:")
                config = {
                    'n_devices': n_dev, 'gamma': best_gamma, 'alpha': 0.5,
                    'local_epochs': 3, 'local_lr': 0.01, 'local_momentum': 0.9,
                    'distill_epochs': 5, 'distill_lr': 0.001, 'temperature': 3.0,
                    'synthetic': quick,
                    'heterogeneity': {
                        'config_file': 'config/devices/heterogeneity.yaml',
                    },
                    'method_config': method_configs.get(method_name, {})
                }
                run = run_single_experiment(method_name, config, seed, device, n_rounds)
                runs.append(run)
            results['runs'][str(n_dev)][method_name] = runs
    
    save_phase_results(results, 'phase6_scalability', seeds)
    return results


def run_phase7(device: str, seeds: list, n_rounds: int, quick: bool = False):
    """Phase 7: Ablation Study."""
    print("\n" + "=" * 70)
    print("Phase 7: Ablation Study")
    print("=" * 70)
    
    best_gamma = _get_best_gamma()
    
    # Full PAID-FD
    variants = {
        'PAID-FD (Full)': {
            'method': 'PAID-FD',
            'config': {'gamma': best_gamma, 'delta': 0.01}
        },
        'w/o Adaptive ε': {
            'method': 'Fixed-eps-5.0',
            'config': {'participation_rate': 1.0}
        },
        'w/o Price (fixed p)': {
            'method': 'PAID-FD',
            'config': {'gamma': best_gamma, 'delta': 0.01, 'fixed_price': 0.5}
        },
        'w/o Game (random)': {
            'method': 'FedMD',  # FedMD = all participate, no game, no privacy
            'config': {}
        },
    }
    if quick:
        variants = {
            'PAID-FD (Full)': variants['PAID-FD (Full)'],
            'w/o Adaptive ε': variants['w/o Adaptive ε'],
        }
    
    results = {
        'phase': 'phase7_ablation', 'best_gamma': best_gamma,
        'variants': list(variants.keys()), 'runs': {}
    }
    
    for variant_name, spec in variants.items():
        print(f"\n--- {variant_name} ---")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            config = {
                'n_devices': 50, 'gamma': best_gamma, 'alpha': 0.5,
                'local_epochs': 3, 'local_lr': 0.01, 'local_momentum': 0.9,
                'distill_epochs': 5, 'distill_lr': 0.001, 'temperature': 3.0,
                'synthetic': quick,
                'heterogeneity': {
                    'config_file': 'config/devices/heterogeneity.yaml',
                },
                'method_config': spec['config']
            }
            run = run_single_experiment(spec['method'], config, seed, device, n_rounds)
            runs.append(run)
        results['runs'][variant_name] = runs
    
    save_phase_results(results, 'phase7_ablation', seeds)
    return results


# ============================================================================
# Helper: Get best gamma from Phase 1.1 results
# ============================================================================

def _find_phase_result(phase: str) -> str:
    """Find result file: try combined first, then fall back to per-seed files."""
    combined = result_path(phase)
    if Path(combined).exists():
        return combined
    # Fall back to per-seed files
    import glob
    pattern = str(PROJECT_ROOT / "results" / "experiments" / f"{phase}_seed*.json")
    seed_files = sorted(glob.glob(pattern))
    if seed_files:
        print(f"  ℹ️  Using per-seed file: {Path(seed_files[0]).name}")
        return seed_files[0]
    return None


def _get_best_gamma(default: float = 10.0) -> float:
    """Load best gamma from phase 1.1 results, or return default."""
    path = _find_phase_result('phase1_gamma')
    if path is None:
        print(f"  ⚠️  Phase 1.1 results not found. Using default γ={default}")
        return default
    
    data = load_json(path)
    best_gamma = default
    best_acc = 0.0
    
    for gamma_str, runs in data.get('runs', {}).items():
        # Average final accuracy across seeds
        accs = [r['final_accuracy'] for r in runs]
        avg_acc = np.mean(accs)
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_gamma = float(gamma_str)
    
    print(f"  ✓ Best γ = {best_gamma} (avg acc = {best_acc:.4f})")
    return best_gamma


def _get_best_lambda(default: float = 0.1) -> float:
    """Load best lambda from phase 1.2 results, or return default."""
    path = _find_phase_result('phase1_lambda')
    if path is None:
        return default
    
    data = load_json(path)
    best_lam = default
    best_acc = 0.0
    
    for lam_str, runs in data.get('runs', {}).items():
        accs = [r['final_accuracy'] for r in runs]
        avg_acc = np.mean(accs)
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_lam = float(lam_str)
    
    return best_lam


# ============================================================================
# Phase Registry
# ============================================================================

PHASES = {
    '1.1': ('Phase 1.1: Gamma Sensitivity', run_phase1_gamma),
    '1.2': ('Phase 1.2: Lambda Sensitivity', run_phase1_lambda),
    '1.3': ('Phase 1.3: N (Device Count) Sensitivity', run_phase1_ndevices),
    '1.4': ('Phase 1.4: Clip Bound (C) Sensitivity', run_phase1_clipbound),
    '2':   ('Phase 2: Convergence & Performance', run_phase2),
    '3':   ('Phase 3: Privacy-Accuracy Tradeoff', run_phase3),
    '4':   ('Phase 4: Incentive Mechanism', run_phase4),
    '5':   ('Phase 5: Heterogeneity Impact', run_phase5),
    '6':   ('Phase 6: Scalability', run_phase6),
    '7':   ('Phase 7: Ablation Study', run_phase7),
}

PHASE_ORDER = ['1.1', '1.2', '1.3', '1.4', '2', '3', '4', '5', '6', '7']

PHASE_RESULT_FILES = {
    '1.1': 'phase1_gamma',
    '1.2': 'phase1_lambda',
    '1.3': 'phase1_ndevices',
    '1.4': 'phase1_clipbound',
    '2':   'phase2_convergence',
    '3':   'phase3_privacy',
    '4':   'phase4_incentive',
    '5':   'phase5_heterogeneity',
    '6':   'phase6_scalability',
    '7':   'phase7_ablation',
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PAID-FD Complete Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_all_experiments.py --phase 1.1              # Run Phase 1.1 only
  python scripts/run_all_experiments.py --phase 1.1 --quick      # Quick test
  python scripts/run_all_experiments.py --all                    # Run all phases
  python scripts/run_all_experiments.py --all --skip-existing    # Skip completed
  python scripts/run_all_experiments.py --phase 2 --device cuda:0
  python scripts/run_all_experiments.py --phase 1.1 --seeds 42 123 456

Parallel execution (3 terminals, same GPU):
  Terminal 1: python scripts/run_all_experiments.py --all --seed 42  --device cuda:0
  Terminal 2: python scripts/run_all_experiments.py --all --seed 123 --device cuda:0
  Terminal 3: python scripts/run_all_experiments.py --all --seed 456 --device cuda:0
  After all:  python scripts/run_all_experiments.py --merge --all

Phase-parallel (after Phase 1.1 done, skip-existing handles deps):
  Terminal 1: python scripts/run_all_experiments.py --phase 2 --seed 42 --device cuda:0
  Terminal 2: python scripts/run_all_experiments.py --phase 3 --seed 42 --device cuda:0
        """
    )
    parser.add_argument('--phase', type=str, default=None,
                       help='Phase to run (1.1, 1.2, 2, 3, 4, 5, 6, 7)')
    parser.add_argument('--all', action='store_true',
                       help='Run all phases in order')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (synthetic data, fewer configs)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of rounds per experiment (default: 100)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Random seeds (default: 42 123 456)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Run single seed only (for parallel execution across terminals)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip phases that already have results')
    parser.add_argument('--merge', action='store_true',
                       help='Merge per-seed result files into combined files')
    parser.add_argument('--list', action='store_true',
                       help='List all phases and their status')
    args = parser.parse_args()
    
    # --seed overrides --seeds
    if args.seed is not None:
        args.seeds = [args.seed]
    
    device = get_device(args.device)
    print(f"🖥️  Device: {device}")
    print(f"🎲 Seeds: {args.seeds}")
    print(f"🔄 Rounds: {args.rounds}")
    if args.quick:
        print("⚡ Quick mode: synthetic data, reduced configs")
    
    if args.list:
        print("\n📋 Phase Status:")
        for pid in PHASE_ORDER:
            name, _ = PHASES[pid]
            exists = result_exists(PHASE_RESULT_FILES[pid])
            status = "✅ Done" if exists else "⬜ Pending"
            print(f"  {pid}: {name} [{status}]")
        return
    
    # Handle --merge mode
    if args.merge:
        phases_to_merge = PHASE_ORDER if args.all else ([args.phase] if args.phase else PHASE_ORDER)
        for pid in phases_to_merge:
            phase_file = PHASE_RESULT_FILES[pid]
            merge_seed_results(phase_file, args.seeds)
        return
    
    if not args.phase and not args.all:
        parser.print_help()
        return
    
    # Determine phases to run
    if args.all:
        phases_to_run = PHASE_ORDER
    else:
        if args.phase not in PHASES:
            print(f"❌ Unknown phase: {args.phase}")
            print(f"   Available: {', '.join(PHASE_ORDER)}")
            return
        phases_to_run = [args.phase]
    
    # Run phases
    total_start = time.time()
    
    for pid in phases_to_run:
        name, runner = PHASES[pid]
        
        if args.skip_existing and result_exists(PHASE_RESULT_FILES[pid]):
            print(f"\n⏭️  Skipping {name} (results exist)")
            continue
        
        phase_start = time.time()
        try:
            runner(device, args.seeds, args.rounds, args.quick)
            elapsed = time.time() - phase_start
            print(f"\n✅ {name} completed in {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - phase_start
            print(f"\n❌ {name} failed after {elapsed:.0f}s: {e}")
            import traceback
            traceback.print_exc()
            if not args.all:
                raise
    
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"🏁 All done! Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
