#!/usr/bin/env python3
"""
Quick parameter sweep: find (C, lambda_mult) that maximizes gamma differentiation.

Tests: C ∈ {1, 2, 3, 5}, lambda_mult ∈ {0.1, 0.2, 0.3, 0.5, 1.0}
Also tests: uniform avg vs ε²-weighted avg (BLUE)
"""
import numpy as np, sys, os
sys.path.insert(0, '.')

from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import StackelbergSolver

gamma_values = [3, 5, 7, 10]

# Generate base devices (seed=42)
gen = HeterogeneityGenerator(n_devices=50, config_path='config/devices/heterogeneity.yaml', seed=42)
devices_base = gen.generate()
base_lambdas = [d.lambda_i for d in devices_base]

# Simulate logits (same as diagnose_game.py)
np.random.seed(42)
n_samples = 5000
n_classes = 100
true_logits = np.random.uniform(-2, 1, (n_samples, n_classes))
true_labels = np.random.randint(0, n_classes, n_samples)
for i in range(n_samples):
    true_logits[i, true_labels[i]] = np.random.uniform(3.0, 5.0)

T = 3.0
n_mc = 10

def softmax_np(logits, temp=1.0):
    e = np.exp((logits - logits.max(axis=1, keepdims=True)) / temp)
    return e / e.sum(axis=1, keepdims=True)

def run_scenario(C, lambda_mult, use_blue=False):
    """Run one (C, lambda_mult) scenario, return results per gamma."""
    # Scale lambda values
    gen2 = HeterogeneityGenerator(
        n_devices=50,
        config_path='config/devices/heterogeneity.yaml',
        config_override={'privacy_sensitivity': {'lambda_mult': lambda_mult}},
        seed=42
    )
    devs = gen2.generate()
    
    clipped = np.clip(true_logits, -C, C)
    clean_probs = softmax_np(clipped, T)
    
    results = {}
    for gamma in gamma_values:
        solver = StackelbergSolver(gamma=gamma)
        r = solver.solve(devs)
        parts = [d for d in r['decisions'] if d.participates]
        N = len(parts)
        
        if N == 0:
            results[gamma] = {'N': 0, 'kl': 99, 'top5': 0, 'argmax': 0, 'snr': 0}
            continue
        
        eps_vals = [d.eps_star for d in parts]
        
        # Monte Carlo simulation
        acc_list, top5_list, kl_list = [], [], []
        for _ in range(n_mc):
            if use_blue:
                # ε²-weighted (BLUE) aggregation
                w = np.array([e**2 for e in eps_vals])
                w = w / w.sum()
                weighted_sum = np.zeros_like(clipped)
                for j, d in enumerate(parts):
                    scale = 2.0 * C / d.eps_star
                    noise = np.random.laplace(0, scale, clipped.shape).astype(np.float32)
                    weighted_sum += w[j] * (clipped + noise)
                avg_logits = weighted_sum  # already weighted to sum to 1
            else:
                # Uniform average
                noisy_sum = np.zeros_like(clipped)
                for d in parts:
                    scale = 2.0 * C / d.eps_star
                    noise = np.random.laplace(0, scale, clipped.shape).astype(np.float32)
                    noisy_sum += (clipped + noise)
                avg_logits = noisy_sum / N
            
            acc_list.append(np.mean(np.argmax(avg_logits, axis=1) == true_labels))
            top5 = np.argsort(avg_logits, axis=1)[:, -5:]
            top5_list.append(np.mean([true_labels[i] in top5[i] for i in range(n_samples)]))
            noisy_probs = softmax_np(avg_logits, T)
            kl = np.mean(np.sum(clean_probs * np.log((clean_probs + 1e-10) / (noisy_probs + 1e-10)), axis=1))
            kl_list.append(kl)
        
        avg_eps = np.mean(eps_vals)
        per_dev_std = 2*C/avg_eps * np.sqrt(2)
        agg_std = per_dev_std / np.sqrt(N)
        snr = C / agg_std
        
        results[gamma] = {
            'N': N, 'avg_eps': avg_eps, 'snr': snr,
            'argmax': np.mean(acc_list), 'top5': np.mean(top5_list), 'kl': np.mean(kl_list)
        }
    return results

# ─── Sweep ───
print("=" * 100)
print("PARAMETER SWEEP: (C, lambda_mult, weighting) → γ differentiation")
print("  Metric: KL ratio (γ=3/γ=10) and top5 spread")
print("=" * 100)

best_combo = None
best_metric = 0

for C in [1, 2, 3, 5]:
    for lm in [0.1, 0.2, 0.3, 0.5, 1.0]:
        for blue in [False, True]:
            res = run_scenario(C, lm, blue)
            
            # Skip if any gamma has N=0
            if any(res[g]['N'] == 0 for g in gamma_values):
                continue
            
            kl3 = res[3]['kl']
            kl10 = res[10]['kl']
            kl_ratio = kl10 / max(kl3, 0.001)
            
            top5_spread = res[3]['top5'] - res[10]['top5']
            argmax_spread = res[3]['argmax'] - res[10]['argmax']
            
            # We want: meaningful SNR (>1 for γ=3), kl_ratio > 2, top5 spread > 10%
            snr3 = res[3]['snr']
            snr10 = res[10]['snr']
            
            tag = "BLUE" if blue else "UNIF"
            
            # Composite metric: want high snr3, high kl_ratio, high top5_spread
            # But need snr3 > 1 (otherwise all γ are random)
            metric = 0
            if snr3 > 0.8:
                metric = kl_ratio * top5_spread * 100  # higher is better
            
            if metric > best_metric:
                best_metric = metric
                best_combo = (C, lm, blue)
            
            if C in [2, 5] or lm in [0.3, 1.0] or blue:
                # Print selected rows
                print(f"C={C} λm={lm:.1f} {tag:4s} | ", end="")
                for g in gamma_values:
                    r = res[g]
                    print(f"γ{g}:N={r['N']:>2} ε={r['avg_eps']:.2f} snr={r['snr']:.2f} "
                          f"top5={r['top5']*100:.0f}% kl={r['kl']:.2f} | ", end="")
                print(f"KL_ratio={kl_ratio:.2f} top5_Δ={top5_spread*100:.1f}%")

print()
print(f"Best combo: C={best_combo[0]}, lambda_mult={best_combo[1]}, BLUE={best_combo[2]}")
print(f"Metric: {best_metric:.2f}")

# ─── Detail of best combo ───
C_best, lm_best, blue_best = best_combo
print(f"\n{'='*80}")
print(f"DETAILED: C={C_best}, lambda_mult={lm_best}, {'BLUE' if blue_best else 'UNIFORM'}")
print(f"{'='*80}")

res = run_scenario(C_best, lm_best, blue_best)
print(f"{'gamma':>6} {'N':>4} {'avg_ε':>7} {'SNR':>6} {'argmax%':>9} {'top5%':>7} {'KL_div':>8}")
print("-" * 55)
for g in gamma_values:
    r = res[g]
    print(f"{g:>6} {r['N']:>4} {r['avg_eps']:>7.3f} {r['snr']:>6.2f} "
          f"{r['argmax']*100:>8.1f}% {r['top5']*100:>6.1f}% {r['kl']:>8.3f}")
