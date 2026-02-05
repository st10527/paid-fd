#!/usr/bin/env python3
"""
Quick analysis script for PAID-FD experiment results.

Usage:
    python3 scripts/analyze.py results/experiments/exp2_convergence/PAID-FD_*.json
    python3 scripts/analyze.py path/to/result.json --plot
"""

import json
import argparse
import sys
from pathlib import Path


def load_result(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze(data, show_plot=False):
    """Analyze a single result file."""
    
    metadata = data.get('metadata', {})
    rounds = data['results']['rounds']
    
    accuracies = [r['accuracy'] * 100 for r in rounds]
    losses = [r['loss'] for r in rounds]
    
    print('=' * 60)
    print(f"Method: {metadata.get('method', 'Unknown')}")
    print(f"Dataset: {metadata.get('dataset', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print('=' * 60)
    
    print(f"\nTotal rounds: {len(rounds)}")
    
    # Accuracy progression
    print("\n📈 Accuracy Progression:")
    checkpoints = [0, 9, 24, 49, 99, 149, 199]
    for i in checkpoints:
        if i < len(accuracies):
            print(f"  Round {i+1:3d}: {accuracies[i]:5.2f}%")
    
    print(f"\n  Final: {accuracies[-1]:.2f}%")
    print(f"  Best:  {max(accuracies):.2f}% (Round {accuracies.index(max(accuracies))+1})")
    
    # Loss progression
    print("\n📉 Loss Progression:")
    print(f"  Start: {losses[0]:.4f}")
    print(f"  End:   {losses[-1]:.4f}")
    print(f"  Best:  {min(losses):.4f}")
    
    # Learning check
    print("\n🔍 Learning Analysis:")
    n = min(10, len(accuracies))
    first_n = sum(accuracies[:n]) / n
    last_n = sum(accuracies[-n:]) / n
    improvement = last_n - first_n
    
    print(f"  First {n} rounds avg: {first_n:.2f}%")
    print(f"  Last {n} rounds avg:  {last_n:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    
    if improvement > 10:
        print("  ✅ Good learning progress!")
    elif improvement > 3:
        print("  ⚠️  Learning, but slow. Consider more rounds or tuning.")
    elif improvement > 0:
        print("  ⚠️  Minimal improvement. Check parameters.")
    else:
        print("  ❌ No improvement. Something may be wrong.")
    
    # Game parameters (if available)
    if 'price' in rounds[0] and rounds[0]['price'] is not None:
        print("\n🎮 Stackelberg Game:")
        print(f"  Price: {rounds[0]['price']:.4f}")
        print(f"  Participation: {rounds[0].get('participation_rate', 0)*100:.0f}%")
    
    # Energy (if available)
    if 'energy' in rounds[0]:
        total_energy = sum(sum(r['energy'].values()) for r in rounds)
        print(f"\n⚡ Total Energy: {total_energy:.2f} J")
    
    print('=' * 60)
    
    # Plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Accuracy
            axes[0].plot(range(1, len(accuracies)+1), accuracies, 'b-', linewidth=1.5)
            axes[0].set_xlabel('Round')
            axes[0].set_ylabel('Accuracy (%)')
            axes[0].set_title('Accuracy over Rounds')
            axes[0].grid(True, alpha=0.3)
            
            # Loss
            axes[1].plot(range(1, len(losses)+1), losses, 'r-', linewidth=1.5)
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Loss over Rounds')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            fig_path = Path(filepath).with_suffix('.png')
            plt.savefig(fig_path, dpi=150)
            print(f"\n📊 Plot saved: {fig_path}")
            
            plt.show()
            
        except ImportError:
            print("\n⚠️  matplotlib not installed. Run: pip install matplotlib")


def compare_results(filepaths):
    """Compare multiple result files."""
    
    results = []
    for fp in filepaths:
        data = load_result(fp)
        rounds = data['results']['rounds']
        accuracies = [r['accuracy'] * 100 for r in rounds]
        
        results.append({
            'file': Path(fp).name,
            'method': data.get('metadata', {}).get('method', 'Unknown'),
            'rounds': len(rounds),
            'final_acc': accuracies[-1],
            'best_acc': max(accuracies),
            'final_loss': rounds[-1]['loss'],
        })
    
    print('=' * 80)
    print('Comparison')
    print('=' * 80)
    print(f"{'Method':<15} {'Rounds':>8} {'Final Acc':>12} {'Best Acc':>12} {'Final Loss':>12}")
    print('-' * 80)
    
    for r in sorted(results, key=lambda x: -x['best_acc']):
        print(f"{r['method']:<15} {r['rounds']:>8} {r['final_acc']:>11.2f}% {r['best_acc']:>11.2f}% {r['final_loss']:>12.4f}")
    
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze PAID-FD results')
    parser.add_argument('files', nargs='+', help='JSON result file(s)')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate plots')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare multiple files')
    args = parser.parse_args()
    
    if args.compare and len(args.files) > 1:
        compare_results(args.files)
    else:
        for filepath in args.files:
            if not Path(filepath).exists():
                print(f"File not found: {filepath}")
                continue
            
            data = load_result(filepath)
            analyze(data, show_plot=args.plot)
            
            if len(args.files) > 1:
                print("\n")


if __name__ == '__main__':
    main()
