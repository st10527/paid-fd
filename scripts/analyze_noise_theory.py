#!/usr/bin/env python3
"""Theoretical analysis: why BLUE cannot create gamma differentiation."""
import json, os
import numpy as np

fpath = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments', 'v8_3_phase0_gamma_sweep.json')
data = json.load(open(fpath))

C = 2.0  # clip bound

print("=" * 70)
print("AGGREGATE NOISE VARIANCE vs GAMMA")
print("=" * 70)
print()
print("  gamma  n(part)  avg_eps  eps^2    n*eps^2  Var_BLUE(per dim)")
print("  " + "-" * 60)

rows = []
for label in ['gamma=2', 'gamma=3', 'gamma=5', 'gamma=7', 'gamma=10']:
    run = data['runs'][label]
    gamma = float(label.split('=')[1])
    parts = run.get('participation_rates', [])
    avg_part = np.mean(parts) if parts else 0
    n = round(avg_part * 50)
    eps_list = run.get('avg_eps', [])
    avg_eps = np.mean(eps_list) if eps_list else 0

    if avg_eps > 0 and n > 0:
        eps2 = avg_eps ** 2
        n_eps2 = n * eps2
        # Laplace variance per dim: 2*(sensitivity/eps)^2
        # sensitivity = 2*C for range [-C, C]
        var_per = 2 * (2 * C / avg_eps) ** 2
        var_blue = var_per / n
        print("  %5.0f  %5d    %.3f    %.4f   %6.2f   %8.2f" % (
            gamma, n, avg_eps, eps2, n_eps2, var_blue))
        rows.append((gamma, n, avg_eps, n_eps2, var_blue))
    else:
        print("  %5.0f  %5d    %.3f    (no participation)" % (gamma, n, avg_eps))

if len(rows) >= 2:
    best_neps2 = max(rows, key=lambda r: r[3])
    worst_neps2 = min(rows, key=lambda r: r[3])
    ratio = best_neps2[3] / worst_neps2[3]
    print()
    print("  Best n*eps^2:  gamma=%d (%.2f)" % (best_neps2[0], best_neps2[3]))
    print("  Worst n*eps^2: gamma=%d (%.2f)" % (worst_neps2[0], worst_neps2[3]))
    print("  Ratio: %.2fx -> gamma=%d has %.2fx MORE noise after BLUE" % (
        ratio, int(worst_neps2[0]), ratio))
