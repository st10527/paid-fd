#!/usr/bin/env python3
"""Quick analysis of Phase 1.1 gamma sweep results."""
import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data = json.load(open('results/experiments/phase1_gamma.json'))

print('=' * 80)
print('Phase 1.1: Gamma Sensitivity — Summary')
print('=' * 80)

for gamma_str in ['100', '300', '500', '700', '1000']:
    runs = data['runs'].get(gamma_str, [])
    if not runs:
        continue
    
    finals = [r['final_accuracy'] for r in runs]
    bests = [r['best_accuracy'] for r in runs]
    seeds = [r['seed'] for r in runs]
    n_rounds = runs[0]['n_rounds']
    
    print(f'\ngamma = {gamma_str} ({len(runs)} seeds, {n_rounds} rounds)')
    print(f'  Final acc (R{n_rounds}):  {np.mean(finals):.4f} +/- {np.std(finals):.4f}')
    for s, a in zip(seeds, finals):
        print(f'    seed {s}: {a:.4f}')
    print(f'  Best acc:          {np.mean(bests):.4f} +/- {np.std(bests):.4f}')
    for s, a in zip(seeds, bests):
        print(f'    seed {s}: {a:.4f}')
    
    # Show accuracy curve for each seed (every 5 rounds)
    print(f'  Accuracy curve (every 5 rounds):')
    for run in runs:
        accs = run['accuracies']
        milestones = [accs[i] for i in range(0, len(accs), 5)]
        tail = accs[-1] if (len(accs) - 1) % 5 != 0 else None
        curve = ' -> '.join([f'R{i*5}:{v:.3f}' for i, v in enumerate(milestones)])
        if tail is not None:
            curve += f' -> R{len(accs)-1}:{tail:.3f}'
        print(f'    seed {run["seed"]}: {curve}')
    
    # Check if still improving
    for run in runs:
        accs = run['accuracies']
        last5_avg = np.mean(accs[-5:])
        prev5_avg = np.mean(accs[-10:-5])
        delta_val = last5_avg - prev5_avg
        if delta_val > 0.005:
            trend = 'STILL IMPROVING'
        elif delta_val > -0.002:
            trend = 'plateau'
        else:
            trend = 'declining'
        print(f'    seed {run["seed"]} trend: avg(R41-45)={prev5_avg:.4f} -> avg(R46-50)={last5_avg:.4f} (delta={delta_val:+.4f}) [{trend}]')

print()
print('=' * 80)
# Best gamma
best_gamma = None
best_acc = 0
for g, runs in data['runs'].items():
    avg = np.mean([r['final_accuracy'] for r in runs])
    if avg > best_acc:
        best_acc = avg
        best_gamma = g
print(f'BEST gamma = {best_gamma}  (avg final acc = {best_acc:.4f})')

# Also show participation rates and epsilon
print()
print('Participation & Privacy stats:')
for gamma_str in ['100', '300', '500', '700', '1000']:
    runs = data['runs'].get(gamma_str, [])
    if not runs:
        continue
    avg_part = np.mean([np.mean(r['participation_rates']) for r in runs])
    avg_eps = np.mean([np.mean(r['avg_eps']) for r in runs])
    avg_price = np.mean([np.mean(r['prices']) for r in runs])
    print(f'  gamma={gamma_str}: avg_participation={avg_part:.3f}, avg_eps={avg_eps:.3f}, avg_price={avg_price:.4f}')

print('=' * 80)
