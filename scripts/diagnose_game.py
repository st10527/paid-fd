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
print("STEP 2: Per-Device LDP Noise Analysis")
print("  Each device adds Lap(0, 2C/ε_i) locally → server averages N noisy logits")
print("  After averaging: Var[avg] = (1/N²)·Σ Var[noise_i] ∝ 1/N")
print("=" * 70)

C = 5.0  # clip_bound
print(f"Clip bound C = {C}")
print()

print(f"{'gamma':>6} {'N':>4} {'avg_ε':>7} {'per_dev_σ':>10} {'agg_σ':>8} "
      f"{'SNR':>6}")
print("-" * 50)

for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    
    N = r['N']
    avg_eps = r['avg_eps']
    
    # Per-device LDP: each device adds Lap(0, 2C/ε_i)
    # Per-device Laplace scale = 2C / avg_ε
    per_device_scale = 2.0 * C / avg_eps
    per_device_std = per_device_scale * np.sqrt(2)  # Laplace std = scale * sqrt(2)
    
    # After averaging N devices: Var[avg] = Var[per_device] / N
    # So std[avg] = std[per_device] / sqrt(N)
    aggregated_noise_std = per_device_std / np.sqrt(N)
    aggregated_noise_scale = per_device_scale / np.sqrt(N)  # effective Laplace scale
    
    # Signal magnitude: logits in [-C, C], typical spread ~C
    signal_mag = C
    snr = signal_mag / aggregated_noise_std
    
    r['per_device_scale'] = per_device_scale
    r['per_device_std'] = per_device_std
    r['noise_scale'] = aggregated_noise_scale  # used in Step 3 Monte Carlo
    r['noise_std'] = aggregated_noise_std
    r['snr'] = snr
    
    print(f"{gamma:>6} {N:>4} {avg_eps:>7.3f} {per_device_std:>10.4f} "
          f"{aggregated_noise_std:>8.4f} "
          f"{snr:>6.2f}")

# ============================================================
# 3. SIMULATE pseudo-label + soft-label quality (Per-Device LDP)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Pseudo-Label & Soft-Label Quality (1-round, no EMA)")
print("  Per-device LDP noise → average → softmax(T=3) → quality metrics")
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

print(f"{'gamma':>6} {'N':>4} {'agg_σ':>8} {'argmax%':>9} {'top5%':>7} "
      f"{'KL_div':>8} {'soft_CE':>8}")
print("-" * 60)

for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    
    N = r['N']
    per_device_scale = r['per_device_scale']
    
    acc_list, top5_list, kl_list, ce_list = [], [], [], []
    for _ in range(n_mc):
        noisy_sum = np.zeros_like(true_logits)
        for dev in range(N):
            device_noise = np.random.laplace(0, per_device_scale, true_logits.shape)
            noisy_sum += (true_logits + device_noise)
        avg_logits = noisy_sum / N
        
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
print("DIAGNOSIS SUMMARY (Per-Device LDP + Soft-Label KL)")
print("=" * 70)

if len(game_results) >= 2:
    gammas_ok = [g for g in gamma_values if game_results[g]['N'] > 0]
    
    part_rates = [game_results[g]['part_rate'] for g in gammas_ok]
    avg_eps_list = [game_results[g]['avg_eps'] for g in gammas_ok]
    per_dev_stds = [game_results[g]['per_device_std'] for g in gammas_ok]
    noise_stds = [game_results[g]['noise_std'] for g in gammas_ok]
    pseudo_accs = [game_results[g]['pseudo_acc'] for g in gammas_ok]
    top5_accs = [game_results[g]['top5_acc'] for g in gammas_ok]
    kl_divs = [game_results[g]['kl_div'] for g in gammas_ok]
    
    print(f"\nSignal propagation (γ={gammas_ok[0]} → γ={gammas_ok[-1]}):")
    print(f"  Participation rate:    {part_rates[0]*100:.0f}% → {part_rates[-1]*100:.0f}%  "
          f"(spread: {(max(part_rates)-min(part_rates))*100:.0f}%)")
    print(f"  Average ε:             {avg_eps_list[0]:.3f} → {avg_eps_list[-1]:.3f}  "
          f"(spread: {max(avg_eps_list)-min(avg_eps_list):.3f})")
    print(f"  Per-device noise σ:    {per_dev_stds[0]:.4f} → {per_dev_stds[-1]:.4f}  "
          f"(ratio: {max(per_dev_stds)/max(min(per_dev_stds),1e-9):.2f}x)")
    print(f"  Aggregated noise σ:    {noise_stds[0]:.4f} → {noise_stds[-1]:.4f}  "
          f"(ratio: {max(noise_stds)/max(min(noise_stds),1e-9):.2f}x)")
    print(f"  Argmax accuracy:       {pseudo_accs[0]*100:.1f}% → {pseudo_accs[-1]*100:.1f}%  "
          f"(spread: {(max(pseudo_accs)-min(pseudo_accs))*100:.1f}%)")
    print(f"  Top-5 accuracy:        {top5_accs[0]*100:.1f}% → {top5_accs[-1]*100:.1f}%  "
          f"(spread: {(max(top5_accs)-min(top5_accs))*100:.1f}%)")
    print(f"  KL divergence:         {kl_divs[0]:.3f} → {kl_divs[-1]:.3f}  "
          f"(ratio: {max(kl_divs)/max(min(kl_divs),1e-9):.2f}x)")
    
    print(f"\n--- Key insight: Per-device LDP + Soft-label KL ---")
    print(f"  Per-device noise ∝ 2C/ε, aggregated noise ∝ (2C/ε)/√N")
    print(f"  Higher γ → more participants (N↑) but smaller ε (ε↓)")
    print(f"  Net effect on noise: ε decrease dominates √N averaging")
    print(f"  BUT: soft-label KL preserves distribution info even under noise")
    print(f"  AND: more diverse ensemble (N models) provides richer knowledge")
    print(f"  → Empirical result will determine optimal γ tradeoff")

print("\nDone.")
