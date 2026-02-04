#!/usr/bin/env python3
"""
Unified Experiment Runner for PAID-FD

Usage:
    # Run single method
    python run_experiment.py --config exp2_convergence --method PAID-FD
    
    # Run all methods
    python run_experiment.py --config exp2_convergence --method all
    
    # Quick test with synthetic data
    python run_experiment.py --config exp2_convergence --method PAID-FD --synthetic --rounds 10
    
    # Force rerun (ignore existing results)
    python run_experiment.py --config exp2_convergence --method PAID-FD --force
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Get the project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root for relative paths in configs
os.chdir(PROJECT_ROOT)

import torch
import numpy as np

from src.utils.seed import set_seed
from src.utils.logger import setup_logger, ProgressLogger
from src.utils.results import ResultManager, ExperimentTracker


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if '_base_' in config:
        base_path = Path(config_path).parent / config['_base_']
        base_config = load_config(str(base_path))
        # Merge: config overrides base
        merged = deep_merge(base_config, config)
        del merged['_base_']
        return merged
    
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_data(config: Dict, synthetic: bool = False):
    """Set up datasets and data loaders."""
    from src.data.datasets import load_cifar100, load_stl10, create_synthetic_datasets
    from src.data.partition import DirichletPartitioner, IIDPartitioner, create_client_loaders
    from torch.utils.data import DataLoader
    
    n_devices = config['system']['n_devices']
    seed = config['system']['seed']
    
    if synthetic:
        print("Using synthetic data for quick testing...")
        train_data, test_data, public_data = create_synthetic_datasets(
            n_train=n_devices * 100,
            n_test=1000,
            n_public=2000,
            seed=seed
        )
        targets = train_data.targets
    else:
        # Load real data
        data_config = config['data']
        
        print("Loading CIFAR-100...")
        train_data, test_data = load_cifar100(
            root=data_config['private'].get('root', './data'),
            download=data_config['private'].get('download', True)
        )
        targets = train_data.targets
        
        print("Loading STL-10...")
        public_data = load_stl10(
            root=data_config['public'].get('root', './data'),
            split=data_config['public'].get('split', 'unlabeled'),
            n_samples=data_config['public'].get('n_samples', 5000),
            resize_to=data_config['public'].get('resize_to', 32),
            seed=seed
        )
    
    # Partition data
    partition_config = config['data']['partition']
    if partition_config['method'] == 'dirichlet':
        partitioner = DirichletPartitioner(
            alpha=partition_config.get('alpha', 0.5),
            n_clients=n_devices,
            min_samples_per_client=partition_config.get('min_samples', 10),
            seed=seed
        )
    else:
        partitioner = IIDPartitioner(n_clients=n_devices, seed=seed)
    
    print(f"Partitioning data ({partition_config['method']})...")
    client_indices = partitioner.partition(train_data, targets)
    
    # Create client loaders
    batch_size = config['training'].get('local_batch_size', 32)
    client_loaders = create_client_loaders(train_data, client_indices, batch_size)
    
    # Create test and public loaders
    test_loader = DataLoader(
        test_data,
        batch_size=config['training'].get('eval_batch_size', 128),
        shuffle=False
    )
    
    public_loader = DataLoader(
        public_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Print partition summary
    from src.data.partition import print_partition_summary
    print_partition_summary(client_indices, targets, partition_config['method'])
    
    return {
        'train': train_data,
        'test': test_data,
        'public': public_data,
        'client_indices': client_indices,
        'client_loaders': client_loaders,
        'test_loader': test_loader,
        'public_loader': public_loader
    }


def setup_devices(config: Dict):
    """Set up heterogeneous devices."""
    from src.devices.heterogeneity import HeterogeneityGenerator
    
    n_devices = config['system']['n_devices']
    seed = config['system']['seed']
    het_config = config.get('heterogeneity', {})
    
    # New interface: config loaded from file, with optional overrides
    config_file = het_config.get('config_file', None)
    config_override = het_config.get('overrides', None)
    
    generator = HeterogeneityGenerator(
        n_devices=n_devices,
        config_path=config_file,
        config_override=config_override,
        seed=seed
    )
    
    devices = generator.generate()
    
    stats = generator.get_statistics(devices)
    print(f"\nDevice Statistics:")
    print(f"  Types: {stats['type_distribution']}")
    print(f"  Straggler ratio: {stats['straggler_ratio']:.0%}")
    print(f"  Avg λ: {stats['lambda_stats']['mean']:.3f}")
    
    return devices, generator


def setup_method(method_name: str, config: Dict, device: str):
    """Set up the federated learning method."""
    from src.models import get_model
    from src.methods import PAIDFD, FixedEpsilon
    from src.methods.paid_fd import PAIDFDConfig
    from src.methods.fixed_eps import FixedEpsilonConfig
    
    # Create model
    model_config = config['model']
    model = get_model(
        model_config['name'],
        num_classes=model_config.get('num_classes', 100)
    )
    
    # Create method
    training_config = config['training']
    
    if method_name == 'PAID-FD':
        paid_fd_config = config.get('paid_fd', {})
        method_config = PAIDFDConfig(
            gamma=paid_fd_config.get('gamma', 10.0),
            delta=paid_fd_config.get('delta', 0.01),
            budget=float(paid_fd_config.get('budget', 'inf')),
            local_epochs=training_config.get('local_epochs', 1),
            local_lr=training_config.get('local_lr', 0.01),
            distill_epochs=training_config.get('distill_epochs', 5),
            distill_lr=training_config.get('distill_lr', 0.001),
            temperature=training_config.get('temperature', 3.0),
            clip_bound=paid_fd_config.get('clip_bound', 5.0),
            public_samples=paid_fd_config.get('public_samples_per_round', 1000)
        )
        method = PAIDFD(model, method_config, model_config.get('num_classes', 100), device)
    
    elif method_name.startswith('Fixed-eps'):
        # Parse epsilon from name (e.g., "Fixed-eps-1.0")
        eps = float(method_name.split('-')[-1])
        method_config = FixedEpsilonConfig(
            epsilon=eps,
            local_epochs=training_config.get('local_epochs', 1),
            local_lr=training_config.get('local_lr', 0.01),
            distill_epochs=training_config.get('distill_epochs', 5),
            distill_lr=training_config.get('distill_lr', 0.001),
            temperature=training_config.get('temperature', 3.0),
            clip_bound=5.0,
            participation_rate=1.0,
            samples_per_device=100
        )
        method = FixedEpsilon(model, method_config, model_config.get('num_classes', 100), device)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    return method


def run_training(
    method,
    method_name: str,
    devices: list,
    data: Dict,
    config: Dict,
    tracker: ExperimentTracker,
    logger
):
    """Run the training loop."""
    n_rounds = config['training']['n_rounds']
    eval_every = config['training'].get('eval_every', 5)
    
    progress = ProgressLogger(logger, n_rounds, log_every=eval_every)
    
    # Assign data sizes to devices
    for dev in devices:
        if dev.device_id in data['client_indices']:
            dev.data_size = len(data['client_indices'][dev.device_id])
    
    for round_idx in range(n_rounds):
        # Run one round
        result = method.run_round(
            round_idx=round_idx,
            devices=devices,
            client_loaders=data['client_loaders'],
            public_loader=data['public_loader'],
            test_loader=data['test_loader']
        )
        
        # Log to tracker
        tracker.log_round(
            round_idx=round_idx,
            accuracy=result.accuracy,
            loss=result.loss,
            participation_rate=result.participation_rate,
            energy_breakdown=result.energy,
            price=result.extra.get('price'),
            device_decisions=None  # Skip to save space
        )
        
        # Progress logging
        progress.log(
            round_idx,
            accuracy=result.accuracy,
            loss=result.loss,
            participation=result.participation_rate
        )
    
    # Final metrics
    final_metrics = {
        'final_accuracy': method.get_final_accuracy(),
        'best_accuracy': method.get_best_accuracy(),
        'n_rounds': n_rounds
    }
    
    # Add method-specific stats
    if hasattr(method, 'get_statistics'):
        stats = method.get_statistics()
        final_metrics.update(stats)
    
    tracker.log_final_metrics(final_metrics)
    progress.finish(final_accuracy=final_metrics['final_accuracy'])
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='PAID-FD Experiment Runner')
    parser.add_argument('--config', type=str, required=True,
                       help='Experiment config name (e.g., exp2_convergence)')
    parser.add_argument('--method', type=str, default='PAID-FD',
                       help='Method to run (PAID-FD, Fixed-eps-X, or "all")')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for quick testing')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of rounds')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun even if results exist')
    args = parser.parse_args()
    
    # Determine config path
    config_dir = Path(__file__).parent.parent / 'config' / 'experiments'
    config_path = config_dir / f'{args.config}.yaml'
    
    if not config_path.exists():
        # Try default config
        config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    
    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))
    
    # Override settings
    if args.rounds:
        config['training']['n_rounds'] = args.rounds
    if args.seed:
        config['system']['seed'] = args.seed
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Set seed
    seed = config['system']['seed']
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Determine methods to run
    if args.method.lower() == 'all':
        methods = config.get('methods', ['PAID-FD'])
    else:
        methods = [args.method]
    
    # Check for existing results
    result_manager = ResultManager()
    
    # Setup data and devices (shared across methods)
    print("\n" + "="*60)
    print("Setting up data...")
    print("="*60)
    data = setup_data(config, synthetic=args.synthetic)
    
    print("\n" + "="*60)
    print("Setting up devices...")
    print("="*60)
    devices, _ = setup_devices(config)
    
    # Run each method
    for method_name in methods:
        print("\n" + "="*60)
        print(f"Running: {method_name}")
        print("="*60)
        
        # Check existing results
        if not args.force and result_manager.result_exists(args.config, method_name):
            print(f"Results exist for {method_name}. Use --force to rerun.")
            continue
        
        # Setup method
        method = setup_method(method_name, config, device)
        
        # Setup logging
        logger = setup_logger(method_name)
        
        # Setup tracker
        tracker = ExperimentTracker(
            exp_name=args.config,
            method_name=method_name,
            config=config
        )
        
        # Run training
        try:
            final_metrics = run_training(
                method, method_name, devices, data, config, tracker, logger
            )
            
            # Save results
            filepath = tracker.save()
            tracker.print_summary()
            
            print(f"\n✓ {method_name} completed!")
            print(f"  Final accuracy: {final_metrics['final_accuracy']:.4f}")
            print(f"  Results saved to: {filepath}")
            
        except Exception as e:
            print(f"\n✗ {method_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
