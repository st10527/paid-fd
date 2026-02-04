#!/usr/bin/env python3
"""
Visualization script for PAID-FD experiment results.

Usage:
    python scripts/plot_results.py results/experiments/exp2_convergence/PAID-FD_*.json
    python scripts/plot_results.py results/experiments/exp2_convergence/  # Plot all in directory
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_result(filepath: str) -> Dict:
    """Load a single result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_results_from_dir(dirpath: str) -> List[Dict]:
    """Load all JSON results from a directory."""
    results = []
    for f in Path(dirpath).glob("*.json"):
        try:
            results.append(load_result(str(f)))
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return results


def plot_convergence(results: List[Dict], save_path: str = None):
    """Plot accuracy convergence curves for multiple methods."""
    plt.figure(figsize=(10, 6))
    
    for data in results:
        method = data['metadata']['method']
        rounds = data['results']['rounds']
        
        accuracies = [r['accuracy'] * 100 for r in rounds]
        x = list(range(1, len(accuracies) + 1))
        
        plt.plot(x, accuracies, label=method, linewidth=2)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Convergence Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_loss(results: List[Dict], save_path: str = None):
    """Plot loss curves."""
    plt.figure(figsize=(10, 6))
    
    for data in results:
        method = data['metadata']['method']
        rounds = data['results']['rounds']
        
        losses = [r['loss'] for r in rounds]
        x = list(range(1, len(losses) + 1))
        
        plt.plot(x, losses, label=method, linewidth=2)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_single_experiment(data: Dict, save_dir: str = None):
    """Generate all plots for a single experiment."""
    method = data['metadata']['method']
    rounds = data['results']['rounds']
    
    accuracies = [r['accuracy'] * 100 for r in rounds]
    losses = [r['loss'] for r in rounds]
    x = list(range(1, len(rounds) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Accuracy
    axes[0, 0].plot(x, accuracies, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title(f'{method} - Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Loss
    axes[0, 1].plot(x, losses, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title(f'{method} - Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Participation rate
    if 'participation_rate' in rounds[0]:
        participation = [r['participation_rate'] * 100 for r in rounds]
        axes[1, 0].plot(x, participation, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Participation (%)')
        axes[1, 0].set_title(f'{method} - Participation Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 105])
    
    # 4. Price (if available)
    if 'price' in rounds[0] and rounds[0]['price'] is not None:
        prices = [r['price'] for r in rounds]
        axes[1, 1].plot(x, prices, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Price')
        axes[1, 1].set_title(f'{method} - Optimal Price')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / f"{method}_summary.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(data: Dict):
    """Print text summary of results."""
    method = data['metadata']['method']
    rounds = data['results']['rounds']
    metrics = data['results'].get('metrics', {})
    
    accuracies = [r['accuracy'] * 100 for r in rounds]
    losses = [r['loss'] for r in rounds]
    
    print(f"\n{'='*50}")
    print(f"Method: {method}")
    print(f"{'='*50}")
    print(f"Rounds: {len(rounds)}")
    print(f"Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"Best Accuracy:  {max(accuracies):.2f}%")
    print(f"Final Loss:     {losses[-1]:.4f}")
    print(f"Best Loss:      {min(losses):.4f}")
    
    if 'participation_rate' in rounds[0]:
        avg_part = np.mean([r['participation_rate'] for r in rounds])
        print(f"Avg Participation: {avg_part*100:.1f}%")
    
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Plot PAID-FD experiment results')
    parser.add_argument('path', type=str, help='JSON file or directory path')
    parser.add_argument('--output', '-o', type=str, default='results/figures',
                       help='Output directory for figures')
    parser.add_argument('--no-show', action='store_true', help='Save only, do not show')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Load results
    path = Path(args.path)
    if path.is_file():
        results = [load_result(str(path))]
    elif path.is_dir():
        results = load_results_from_dir(str(path))
    else:
        print(f"Error: {path} not found")
        sys.exit(1)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    print(f"Loaded {len(results)} result(s)")
    
    # Print summaries
    for data in results:
        print_summary(data)
    
    # Plot
    for data in results:
        plot_single_experiment(data, args.output)
    
    # Convergence comparison (if multiple)
    if len(results) > 1:
        plot_convergence(results, f"{args.output}/convergence_comparison.png")
        plot_loss(results, f"{args.output}/loss_comparison.png")


if __name__ == '__main__':
    main()
