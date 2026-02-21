#!/usr/bin/env python3
"""
Diagnose: Why do different gamma values produce nearly identical accuracy?

Traces the FULL pipeline:
  gamma → price → participation → ε_i → noise_scale → pseudo_acc → Δacc

Goal: find which link in the chain "absorbs" the gamma difference,
and determine parameter adjustments WITHIN the existing theory to fix it.
"""
import numpy as np
import sys, os
sys.path.insert(0, '.')

from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import StackelbergSolver

# ============================================================
# 1. GAME LAYER: gamma → price → participation → epsilon
# ============================================================
print("=" * 70)
print("STEP 1: Stackelberg Game Outputs (no lambda_mult override)")
print("=" * 70)

gen = HeterogeneityGenerator(
    n_devices=50,
    config_path='config/devices/heterogeneity.yaml',
    seed=42
)
devices = gen.generate()

lambdas = [d.lambda_i for d in devices]
c_totals = [d.c_total for d in devices]
print(f"Device λ_i:  min={min(lambdas):.4f}, max={max(lambdas):.4f}, mean={np.mean(lambdas):.4f}")
print(f"Device c_i:  min={min(c_totals):.4f}, max={max(c_totals):.4f}, mean={np.mean(c_totals):.4f}")
print(f"λ sorted: {sorted([round(l, 3) for l in lambdas])}")
print()

gamma_values = [3, 5, 7, 10]
game_results = {}

print(f"{'gamma':>6} {'price':>7} {'part%':>6} {'N_part':>6} "
      f"{'avg_ε':>7} {'min_ε':>7} {'max_ε':>7} "
      f"{'avg_s':>7} {'Q_total':>8} {'U_server':>9}")
print("-" * 90)

for gamma in gamma_values:
    solver = StackelbergSolver(gamma=gamma)
    result = solver.solve(devices)
    
    parts = [d for d in result['decisions'] if d.participates]
    N = len(parts)
    
    if N == 0:
        print(f"{gamma:>6}  NO PARTICIPANTS")
        game_results[gamma] = {'N': 0, 'avg_eps': 0, 'noise_scale': float('inf')}
        continue
    
    eps_vals = [d.eps_star for d in parts]
    s_vals = [d.s_star for d in parts]
    
    game_results[gamma] = {
        'N': N,
        'price': result['price'],
        'avg_eps': np.mean(eps_vals),
        'min_eps': min(eps_vals),
        'max_eps': max(eps_vals),
        'eps_vals': eps_vals,  # per-device ε for BLUE simulation
        'avg_s': np.mean(s_vals),
        'Q': result['total_quality'],
        'U': result['server_utility'],
        'part_rate': N / 50,
    }
    
    r = game_results[gamma]
    print(f"{gamma:>6} {r['price']:>7.3f} {r['part_rate']*100:>5.0f}% {N:>6} "
          f"{r['avg_eps']:>7.3f} {r['min_eps']:>7.3f} {r['max_eps']:>7.3f} "
          f"{r['avg_s']:>7.1f} {r['Q']:>8.2f} {r['U']:>9.2f}")

# ============================================================
# 2. NOISE LAYER: Per-Device LDP → average → noise on pseudo-labels
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Per-Device LDP Noise Analysis (BLUE vs Uniform)")
print("  Each device adds Lap(0, 2C/ε_i) locally → server aggregates")
print("  BLUE: w_i ∝ ε_i² (inverse-variance optimal weighting)")
print("=" * 70)

C = 2.0  # clip_bound (C=2: reduced sensitivity, better SNR)
print(f"Clip bound C = {C}")
print()

print(f"{'gamma':>6} {'N':>4} {'avg_ε':>7} {'per_dev_σ':>10} "
      f"{'unif_σ':>8} {'BLUE_σ':>8} {'SNR_unif':>9} {'SNR_BLUE':>9}")
print("-" * 70)

for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    
    N = r['N']
    eps_vals = r['eps_vals']
    avg_eps = r['avg_eps']
    
    # Per-device: Lap(0, 2C/ε_i), Var = 2(2C/ε_i)²
    per_device_vars = [2 * (2*C/e)**2 for e in eps_vals]
    per_device_stds = [np.sqrt(v) for v in per_device_vars]
    avg_per_dev_std = np.mean(per_device_stds)
    
    # Uniform averaging: Var_unif = (1/N²) Σ Var_i
    var_unif = sum(per_device_vars) / (N**2)
    std_unif = np.sqrt(var_unif)
    
    # BLUE: w_i = ε_i² / Σ ε_j², Var_BLUE = 1 / Σ(1/Var_i)
    # Actually: Var_BLUE = 1 / Σ(ε_i²/(8C²)) = 8C² / Σ ε_i²
    sum_eps_sq = sum(e**2 for e in eps_vals)
    var_blue = 8 * C**2 / sum_eps_sq
    std_blue = np.sqrt(var_blue)
    
    snr_unif = C / std_unif
    snr_blue = C / std_blue
    
    r['per_device_scale'] = 2.0 * C / avg_eps  # for MC simulation
    r['per_device_std'] = avg_per_dev_std
    r['noise_std'] = std_unif
    r['noise_std_blue'] = std_blue
    r['snr'] = snr_unif
    r['snr_blue'] = snr_blue
    
    print(f"{gamma:>6} {N:>4} {avg_eps:>7.3f} {avg_per_dev_std:>10.4f} "
          f"{std_unif:>8.4f} {std_blue:>8.4f} "
          f"{snr_unif:>9.2f} {snr_blue:>9.2f}")

# ============================================================
# 3. SIMULATE pseudo-label + soft-label quality (BLUE aggregation)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Soft-Label Quality with BLUE Aggregation")
print("  Per-device LDP → ε²-weighted avg → softmax(T=3) → quality metrics")
print("=" * 70)

np.random.seed(42)
n_samples = 5000
n_classes = 100

true_logits = np.random.uniform(-2, 1, (n_samples, n_classes))
true_labels = np.random.randint(0, n_classes, n_samples)
for i in range(n_samples):
    true_logits[i, true_labels[i]] = np.random.uniform(3.0, 5.0)
true_logits = np.clip(true_logits, -C, C)

print(f"\nTrue logit stats: correct_class mean="
      f"{np.mean([true_logits[i, true_labels[i]] for i in range(n_samples)]):.2f}")
print(f"True argmax accuracy: "
      f"{np.mean(np.argmax(true_logits, axis=1) == true_labels) * 100:.1f}%")
print()

# Soft-label quality: use temperature T=3 (matches PAID-FD)
T = 3.0

def softmax_np(logits, temp=1.0):
    e = np.exp((logits - logits.max(axis=1, keepdims=True)) / temp)
    return e / e.sum(axis=1, keepdims=True)

clean_probs = softmax_np(true_logits, T)

n_mc = 20

print(f"{'gamma':>6} {'N':>4} {'BLUE_σ':>8} {'argmax%':>9} {'top5%':>7} "
      f"{'KL_div':>8} {'soft_CE':>8}")
print("-" * 60)

for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    
    N = r['N']
    eps_vals = r['eps_vals']
    
    # BLUE weights: w_i ∝ ε_i²
    eps_sq = np.array([e**2 for e in eps_vals])
    w_blue = eps_sq / eps_sq.sum()
    
    acc_list, top5_list, kl_list, ce_list = [], [], [], []
    for _ in range(n_mc):
        # ε²-weighted (BLUE) aggregation with per-device noise
        weighted_sum = np.zeros_like(true_logits)
        for j, eps_j in enumerate(eps_vals):
            scale_j = 2.0 * C / eps_j
            noise_j = np.random.laplace(0, scale_j, true_logits.shape)
            weighted_sum += w_blue[j] * (true_logits + noise_j)
        avg_logits = weighted_sum  # already sums to 1
        
        # Argmax accuracy (hard-label quality)
        acc_list.append(np.mean(np.argmax(avg_logits, axis=1) == true_labels))
        
        # Top-5 accuracy
        top5 = np.argsort(avg_logits, axis=1)[:, -5:]
        top5_acc = np.mean([true_labels[i] in top5[i] for i in range(n_samples)])
        top5_list.append(top5_acc)
        
        # Soft-label quality: KL(clean || noisy)
        noisy_probs = softmax_np(avg_logits, T)
        kl = np.mean(np.sum(clean_probs * np.log((clean_probs + 1e-10) / (noisy_probs + 1e-10)), axis=1))
        kl_list.append(kl)
        
        # Cross-entropy of noisy probs with true labels
        ce = -np.mean(np.log(noisy_probs[np.arange(n_samples), true_labels] + 1e-10))
        ce_list.append(ce)
    
    r['pseudo_acc'] = np.mean(acc_list)
    r['top5_acc'] = np.mean(top5_list)
    r['kl_div'] = np.mean(kl_list)
    r['soft_ce'] = np.mean(ce_list)
    
    print(f"{gamma:>6} {N:>4} {r['noise_std']:>8.4f} "
          f"{r['pseudo_acc']*100:>8.1f}% {r['top5_acc']*100:>6.1f}% "
          f"{r['kl_div']:>8.3f} {r['soft_ce']:>8.3f}")

# ============================================================
# 4. SOFT-LABEL QUALITY SPREAD
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Soft-Label Quality Spread Across Gamma")
print("  Pure KL distillation (α=1.0): FL signal = soft-label quality directly")
print("=" * 70)
print()

print(f"{'gamma':>6} {'N':>4} {'part%':>6} {'argmax%':>9} {'top5%':>7} "
      f"{'KL_div':>8} {'ΔKL vs γ=3':>12}")
print("-" * 60)

base_kl = None
for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    kl = r['kl_div']
    if base_kl is None:
        base_kl = kl
    delta_kl = kl - base_kl
    
    print(f"{gamma:>6} {r['N']:>4} {r['part_rate']*100:>5.0f}% "
          f"{r['pseudo_acc']*100:>8.1f}% {r['top5_acc']*100:>6.1f}% "
          f"{kl:>8.3f} {delta_kl:>+11.3f}")

# ============================================================
# 5. DIAGNOSIS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY (Per-Device LDP + BLUE Aggregation + Soft-Label KL)")
print("=" * 70)

if len(game_results) >= 2:
    gammas_ok = [g for g in gamma_values if game_results[g]['N'] > 0]
    
    part_rates = [game_results[g]['part_rate'] for g in gammas_ok]
    avg_eps_list = [game_results[g]['avg_eps'] for g in gammas_ok]
    per_dev_stds = [game_results[g]['per_device_std'] for g in gammas_ok]
    noise_stds = [game_results[g]['noise_std'] for g in gammas_ok]
    noise_stds_blue = [game_results[g]['noise_std_blue'] for g in gammas_ok]
    pseudo_accs = [game_results[g]['pseudo_acc'] for g in gammas_ok]
    top5_accs = [game_results[g]['top5_acc'] for g in gammas_ok]
    kl_divs = [game_results[g]['kl_div'] for g in gammas_ok]
    
    print(f"\nSignal propagation (γ={gammas_ok[0]} → γ={gammas_ok[-1]}), C={C}:")
    print(f"  Participation rate:    {part_rates[0]*100:.0f}% → {part_rates[-1]*100:.0f}%  "
          f"(spread: {(max(part_rates)-min(part_rates))*100:.0f}%)")
    print(f"  Average ε:             {avg_eps_list[0]:.3f} → {avg_eps_list[-1]:.3f}  "
          f"(spread: {max(avg_eps_list)-min(avg_eps_list):.3f})")
    print(f"  Per-device noise σ:    {per_dev_stds[0]:.4f} → {per_dev_stds[-1]:.4f}  "
          f"(ratio: {max(per_dev_stds)/max(min(per_dev_stds),1e-9):.2f}x)")
    print(f"  BLUE-aggregated σ:     {noise_stds_blue[0]:.4f} → {noise_stds_blue[-1]:.4f}  "
          f"(ratio: {max(noise_stds_blue)/max(min(noise_stds_blue),1e-9):.2f}x)")
    print(f"  BLUE SNR:              {game_results[gammas_ok[0]]['snr_blue']:.2f} → "
          f"{game_results[gammas_ok[-1]]['snr_blue']:.2f}")
    print(f"  Argmax accuracy:       {pseudo_accs[0]*100:.1f}% → {pseudo_accs[-1]*100:.1f}%  "
          f"(spread: {(max(pseudo_accs)-min(pseudo_accs))*100:.1f}%)")
    print(f"  Top-5 accuracy:        {top5_accs[0]*100:.1f}% → {top5_accs[-1]*100:.1f}%  "
          f"(spread: {(max(top5_accs)-min(top5_accs))*100:.1f}%)")
    print(f"  KL divergence:         {kl_divs[0]:.3f} → {kl_divs[-1]:.3f}  "
          f"(ratio: {max(kl_divs)/max(min(kl_divs),1e-9):.2f}x)")
    
    print(f"\n--- Key changes from Fix 15 ---")
    print(f"  C = {C} (was 5): sensitivity 2C = {2*C} (was 10) → noise reduced {5/C:.1f}×")
    print(f"  BLUE weighting: w_i ∝ ε_i² → noisy marginal devices down-weighted")
    print(f"  Combined effect: better SNR, wider quality spread across γ")

print("\nDone.")
