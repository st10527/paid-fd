#!/usr/bin/env python3
"""Parse Phase 0 gamma sweep results."""
import json

with open('results/experiments/v8_phase0_gamma_sweep.json') as f:
    data = json.load(f)

print("Version:", data.get('version'))
print("Rounds:", data.get('n_rounds'))
print()

header = f"{'gamma':>8} {'final_acc':>10} {'best_acc':>10} {'time':>8} {'part_r0':>8} {'part_r2':>8}"
print(header)
print("-" * len(header))

for label, run in data['runs'].items():
    accs = run['accuracies']
    parts = run['participation_rates']
    final = run['final_accuracy']
    best = run['best_accuracy']
    elapsed = run['elapsed_sec']
    print(f"{label:>8} {final:>10.4f} {best:>10.4f} {elapsed:>7.0f}s {parts[0]:>8.2f} {parts[-1]:>8.2f}")

print()
finals = [r['final_accuracy'] for r in data['runs'].values()]
print(f"Max final: {max(finals):.4f}")
print(f"Min final: {min(finals):.4f}")
print(f"Gap: {(max(finals)-min(finals))*100:.1f}%")

# Show per-round details
print()
print("Per-round details:")
for label, run in data['runs'].items():
    accs = run['accuracies']
    parts = run['participation_rates']
    prices = run.get('prices', [])
    avg_eps = run.get('avg_eps', [])
    
    print(f"\n  {label}:")
    for i in range(len(accs)):
        p_str = f"p={prices[i]:.3f}" if i < len(prices) else ""
        e_str = f"eps={avg_eps[i]:.3f}" if i < len(avg_eps) else ""
        print(f"    R{i}: acc={accs[i]:.4f}, part={parts[i]:.2f}, {p_str}, {e_str}")

# Check extra info
print()
print("Extra info from first gamma run:")
first_run = list(data['runs'].values())[0]
if first_run.get('extras'):
    for i, ext in enumerate(first_run['extras'][:3]):
        print(f"  R{i}: {ext}")
