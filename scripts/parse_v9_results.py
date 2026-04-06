#!/usr/bin/env python3
"""Parse and display v9.0 gamma sweep results."""
import json

with open('results/experiments/v9_0_phase0_gamma_sweep_fixed_solver.json') as f:
    data = json.load(f)

print('Version:', data.get('version'))
print('Seed:', data.get('seed'))
print('Rounds:', data.get('n_rounds'))
print('Gammas:', data.get('gammas'))
print()

print('=' * 90)
fmt = "  {:<12s} {:>6s} {:>6s} {:>6s} {:>7s} {:>5s} {:>7s} {:>6s}"
print(fmt.format('Config', 'R1', 'Final', 'Best', 'Delta', 'Part', 'Eps', 'Time'))
print('  ' + '-' * 82)

for label, run in data['runs'].items():
    accs = run['accuracies']
    r1 = accs[0]
    final = accs[-1]
    best = max(accs)
    best_r = accs.index(best)
    delta = final - r1
    parts = run.get('participation_rates', [])
    avg_part = sum(parts) / len(parts) if parts else 0
    eps_list = run.get('avg_eps', [])
    avg_eps = sum(eps_list) / len(eps_list) if eps_list else 0
    elapsed = run.get('elapsed_sec', 0)
    tag = '+' if delta > 0.005 else ('~' if delta > -0.02 else '-')
    print("  {:<12s} {:5.1f}% {:5.1f}% {:5.1f}% {:+5.1f}% {:4.0f}% {:7.3f} {:5.0f}s {} (best@R{})".format(
        label, r1*100, final*100, best*100, delta*100, avg_part*100, avg_eps, elapsed, tag, best_r))

print()
print('=' * 90)
print()

# Detailed trajectory every 10 rounds
for label, run in data['runs'].items():
    accs = run['accuracies']
    parts = run.get('participation_rates', [])
    eps_list = run.get('avg_eps', [])
    prices = run.get('prices', [])
    print('  {} trajectory:'.format(label))
    for i in range(0, len(accs), 10):
        p = parts[i] if i < len(parts) else 0
        e = eps_list[i] if i < len(eps_list) else 0
        pr = prices[i] if i < len(prices) else 0
        print('    R{:3d}: acc={:.4f}  part={:.2f}  eps={:.3f}  price={:.3f}'.format(i, accs[i], p, e, pr))
    i = len(accs) - 1
    if i % 10 != 0:
        p = parts[i] if i < len(parts) else 0
        e = eps_list[i] if i < len(eps_list) else 0
        pr = prices[i] if i < len(prices) else 0
        print('    R{:3d}: acc={:.4f}  part={:.2f}  eps={:.3f}  price={:.3f} (final)'.format(i, accs[i], p, e, pr))
    print()

# Gap analysis
finals_data = []
for label, run in data['runs'].items():
    accs = run['accuracies']
    parts = run.get('participation_rates', [])
    avg_part = sum(parts) / len(parts) if parts else 0
    eps_list = run.get('avg_eps', [])
    avg_eps = sum(eps_list) / len(eps_list) if eps_list else 0
    finals_data.append((label, accs[-1], max(accs), avg_part, avg_eps))

active = [(l, f, b, p, e) for l, f, b, p, e in finals_data if p > 0.05]
if len(active) >= 2:
    best_l, best_f = max(active, key=lambda x: x[1])[:2]
    worst_l, worst_f = min(active, key=lambda x: x[1])[:2]
    gap = best_f - worst_f
    print('GAMMA DIFFERENTIATION:')
    print('  Final gap: {:.1f}% ({}={:.1f}% vs {}={:.1f}%)'.format(gap*100, best_l, best_f*100, worst_l, worst_f*100))

    ordered = sorted(active, key=lambda x: float(x[0].split('=')[1]))
    acc_ordered = [f for _, f, _, _, _ in ordered]
    mono = all(acc_ordered[i] <= acc_ordered[i+1] for i in range(len(acc_ordered)-1))
    print('  Monotonic (higher gamma -> higher acc): {}'.format('YES' if mono else 'NO'))
    for l, f, b, p, e in ordered:
        print('    {}: final={:.2f}%  best={:.2f}%  part={:.0f}%  eps={:.3f}'.format(l, f*100, b*100, p*100, e))

# v8.3 comparison
import os
v83 = 'results/experiments/v8_3_phase0_gamma_sweep.json'
if os.path.exists(v83):
    print()
    print('COMPARISON vs v8.3 (wrong root):')
    with open(v83) as f:
        old = json.load(f)
    for label, run in data['runs'].items():
        if label in old.get('runs', {}):
            old_final = old['runs'][label]['accuracies'][-1]
            new_final = run['accuracies'][-1]
            diff = new_final - old_final
            print('  {}: v8.3={:.2f}% -> v9.0={:.2f}% ({:+.1f}%)'.format(
                label, old_final*100, new_final*100, diff*100))
