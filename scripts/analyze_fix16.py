#!/usr/bin/env python3
"""Analyze Fix 16 GPU results."""
import json
import numpy as np

with open('results/experiments/phase1_gamma_seed42.json') as f:
    data = json.load(f)

runs = data.get('runs', data)
print('Fix 16 GPU Results')
print('=' * 70)

bests = {}
finals = {}

for gamma in ['3', '5', '7', '10']:
    if gamma not in runs:
        continue
    run = runs[gamma]
    if isinstance(run, list):
        run = run[0]
    accs = run.get('accuracies', [])
    losses = run.get('losses', [])
    if not accs:
        continue

    best_acc = max(accs)
    best_r = accs.index(best_acc)
    final_acc = accs[-1]
    bests[gamma] = best_acc
    finals[gamma] = final_acc

    # Trajectory every 10 rounds
    traj = []
    for i in range(0, min(len(accs), 100), 10):
        traj.append(f'R{i}:{accs[i]:.1f}')
    if len(accs) > 90:
        traj.append(f'R{len(accs)-1}:{accs[-1]:.1f}')
    
    print(f'gamma={gamma}: best={best_acc:.2f}%(R{best_r}) final={final_acc:.2f}%')
    print(f'  {" -> ".join(traj)}')

print()
print(f'Best spread:  {max(bests.values()) - min(bests.values()):.2f}%')
print(f'Final spread: {max(finals.values()) - min(finals.values()):.2f}%')

# Config
run0 = runs[list(runs.keys())[0]]
if isinstance(run0, list):
    run0 = run0[0]
cfg = run0.get('config', {})
print(f'\nConfig: distill_lr={cfg.get("distill_lr")}, temperature={cfg.get("temperature")}')
print(f'        clip_bound={cfg.get("clip_bound")}, distill_alpha={cfg.get("distill_alpha")}')
print(f'        local_epochs={cfg.get("local_epochs")}, local_lr={cfg.get("local_lr")}')

# Deeper analysis: round-by-round delta between gamma=3 and gamma=10
print('\n' + '=' * 70)
print('Round-by-round gamma=3 vs gamma=10 gap')
print('=' * 70)
if '3' in runs and '10' in runs:
    r3 = runs['3']
    r10 = runs['10']
    if isinstance(r3, list): r3 = r3[0]
    if isinstance(r10, list): r10 = r10[0]
    a3 = r3.get('accuracies', [])
    a10 = r10.get('accuracies', [])
    n = min(len(a3), len(a10))
    
    for i in range(0, n, 5):
        gap = a10[i] - a3[i]
        print(f'  R{i:3d}: g3={a3[i]:.2f}%  g10={a10[i]:.2f}%  gap={gap:+.2f}%')

# Check: how much does distillation contribute vs local training?
# If we look at early rounds (R0-R5) where local models haven't diverged much
print('\n' + '=' * 70)
print('Early rounds analysis (distillation effect)')
print('=' * 70)
for gamma in ['3', '5', '7', '10']:
    if gamma not in runs:
        continue
    run = runs[gamma]
    if isinstance(run, list):
        run = run[0]
    accs = run.get('accuracies', [])
    if len(accs) < 11:
        continue
    # R0 is post-pretrain, R1-R5 is first few FL rounds
    early_gain = accs[5] - accs[0]
    mid_gain = accs[min(50, len(accs)-1)] - accs[5]
    late_gain = accs[-1] - accs[min(50, len(accs)-1)]
    print(f'  gamma={gamma}: R0-R5={early_gain:+.2f}%  R5-R50={mid_gain:+.2f}%  R50-end={late_gain:+.2f}%')
