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

gamma_values = [3, 5, 7, 10, 15]
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
      f"{'SNR':>6} {'EMA_σ':>8} {'EMA_SNR':>8}")
print("-" * 80)

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
    
    # After EMA with β=0.3 over many rounds, noise variance reduces further
    # For stationary EMA: var_ema = var_single * (1-β)/(1+β)
    beta = 0.3
    ema_reduction = np.sqrt((1 - beta) / (1 + beta))
    ema_noise_std = aggregated_noise_std * ema_reduction
    ema_snr = signal_mag / ema_noise_std
    
    r['per_device_scale'] = per_device_scale
    r['per_device_std'] = per_device_std
    r['noise_scale'] = aggregated_noise_scale  # used in Step 3 Monte Carlo
    r['noise_std'] = aggregated_noise_std
    r['snr'] = snr
    r['ema_noise_std'] = ema_noise_std
    r['ema_snr'] = ema_snr
    
    print(f"{gamma:>6} {N:>4} {avg_eps:>7.3f} {per_device_std:>10.4f} "
          f"{aggregated_noise_std:>8.4f} "
          f"{snr:>6.2f} {ema_noise_std:>8.4f} {ema_snr:>8.2f}")

# ============================================================
# 3. SIMULATE pseudo-label accuracy at each noise level (Per-Device LDP)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Simulated Pseudo-Label Accuracy (Per-Device LDP Monte Carlo)")
print("  Each device independently adds noise, then server averages")
print("=" * 70)

# Generate "true" logits: typical ResNet output on CIFAR-100
# True logits have one dominant class with margin ~2-5 over others
np.random.seed(42)
n_samples = 5000
n_classes = 100

# Simulate realistic logit distribution:
# - Correct class: ~3.0 to 5.0
# - Wrong classes: ~-2.0 to 1.0
true_logits = np.random.uniform(-2, 1, (n_samples, n_classes))
true_labels = np.random.randint(0, n_classes, n_samples)
for i in range(n_samples):
    true_logits[i, true_labels[i]] = np.random.uniform(3.0, 5.0)

# Clip to [-C, C]
true_logits = np.clip(true_logits, -C, C)

print(f"\nTrue logit stats: correct_class mean="
      f"{np.mean([true_logits[i, true_labels[i]] for i in range(n_samples)]):.2f}")
print(f"True argmax accuracy: "
      f"{np.mean(np.argmax(true_logits, axis=1) == true_labels) * 100:.1f}%")
print()

n_mc = 20  # Monte Carlo trials

print(f"{'gamma':>6} {'N':>4} {'per_σ':>8} {'agg_σ':>8} {'1-round':>10} "
      f"{'EMA(β=0.3)':>12} {'no_EMA':>10}")
print("-" * 75)

for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    
    N = r['N']
    per_device_scale = r['per_device_scale']
    
    # 1-round pseudo accuracy: simulate N devices each adding noise, then average
    acc_1round = []
    for _ in range(n_mc):
        # Each of N devices adds independent Laplace noise
        noisy_sum = np.zeros_like(true_logits)
        for dev in range(N):
            device_noise = np.random.laplace(0, per_device_scale, true_logits.shape)
            noisy_sum += (true_logits + device_noise)
        avg_logits = noisy_sum / N
        acc_1round.append(np.mean(np.argmax(avg_logits, axis=1) == true_labels))
    
    # EMA simulation: run 100 rounds, check final buffer accuracy
    ema_results = {}
    for beta in [0.3, 0.0]:  # 0.0 = no EMA (just last round)
        acc_ema = []
        for _ in range(n_mc):
            buf = None
            for round_idx in range(100):
                # Per-device LDP: each device adds noise independently
                noisy_sum = np.zeros_like(true_logits)
                for dev in range(N):
                    device_noise = np.random.laplace(0, per_device_scale, true_logits.shape)
                    noisy_sum += (true_logits + device_noise)
                avg_logits = noisy_sum / N
                
                if buf is None:
                    buf = avg_logits.copy()
                else:
                    if beta > 0:
                        buf = beta * buf + (1 - beta) * avg_logits
                    else:
                        buf = avg_logits  # no smoothing
            acc = np.mean(np.argmax(buf, axis=1) == true_labels)
            acc_ema.append(acc)
        ema_results[beta] = np.mean(acc_ema)
    
    r['pseudo_acc_1round'] = np.mean(acc_1round)
    r['pseudo_acc_ema03'] = ema_results[0.3]
    r['pseudo_acc_ema03'] = ema_results[0.3]
    r['pseudo_acc_no_ema'] = ema_results[0.0]
    
    print(f"{gamma:>6} {N:>4} {r['per_device_std']:>8.4f} {r['noise_std']:>8.4f} "
          f"{np.mean(acc_1round)*100:>9.1f}% "
          f"{ema_results[0.3]*100:>11.1f}% "
          f"{ema_results[0.0]*100:>9.1f}%")

# ============================================================
# 4. EFFECTIVE FL SIGNAL: alpha × pseudo_acc
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Effective FL Signal = α × pseudo_accuracy")
print("=" * 70)

alpha = 0.5
print(f"Current α = {alpha}")
print(f"loss = {alpha}*CE(pseudo) + {1-alpha}*CE(true)")
print()

print(f"{'gamma':>6} {'N':>4} {'part%':>6} {'pseudo_acc':>11} {'eff_signal':>12} {'Δ from γ=3':>11}")
print("-" * 60)

base_signal = None
for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    pseudo_acc = r['pseudo_acc_ema03']
    eff = alpha * pseudo_acc
    if base_signal is None:
        base_signal = eff
    delta = eff - base_signal
    
    print(f"{gamma:>6} {r['N']:>4} {r['part_rate']*100:>5.0f}% {pseudo_acc*100:>10.1f}% "
          f"{eff*100:>11.1f}% {delta*100:>+10.2f}%")

# ============================================================
# 5. DIAGNOSIS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY (Per-Device LDP)")
print("=" * 70)

if len(game_results) >= 2:
    gammas_ok = [g for g in gamma_values if game_results[g]['N'] > 0]
    
    part_rates = [game_results[g]['part_rate'] for g in gammas_ok]
    avg_eps_list = [game_results[g]['avg_eps'] for g in gammas_ok]
    per_dev_stds = [game_results[g]['per_device_std'] for g in gammas_ok]
    noise_stds = [game_results[g]['noise_std'] for g in gammas_ok]
    pseudo_accs_03 = [game_results[g]['pseudo_acc_ema03'] for g in gammas_ok]
    pseudo_accs_no = [game_results[g]['pseudo_acc_no_ema'] for g in gammas_ok]
    eff_signals = [alpha * p for p in pseudo_accs_03]
    
    print(f"\nSignal propagation (γ={gammas_ok[0]} → γ={gammas_ok[-1]}):")
    print(f"  Participation rate:    {part_rates[0]*100:.0f}% → {part_rates[-1]*100:.0f}%  "
          f"(spread: {(max(part_rates)-min(part_rates))*100:.0f}%)")
    print(f"  Average ε:             {avg_eps_list[0]:.3f} → {avg_eps_list[-1]:.3f}  "
          f"(spread: {max(avg_eps_list)-min(avg_eps_list):.3f})")
    print(f"  Per-device noise σ:    {per_dev_stds[0]:.4f} → {per_dev_stds[-1]:.4f}  "
          f"(ratio: {max(per_dev_stds)/max(min(per_dev_stds),1e-9):.2f}x)")
    print(f"  Aggregated noise σ:    {noise_stds[0]:.4f} → {noise_stds[-1]:.4f}  "
          f"(ratio: {max(noise_stds)/max(min(noise_stds),1e-9):.2f}x)")
    print(f"  Pseudo acc (EMA=0.3):  {pseudo_accs_03[0]*100:.1f}% → {pseudo_accs_03[-1]*100:.1f}%  "
          f"(spread: {(max(pseudo_accs_03)-min(pseudo_accs_03))*100:.1f}%)")
    print(f"  Pseudo acc (no EMA):   {pseudo_accs_no[0]*100:.1f}% → {pseudo_accs_no[-1]*100:.1f}%  "
          f"(spread: {(max(pseudo_accs_no)-min(pseudo_accs_no))*100:.1f}%)")
    print(f"  Eff FL signal (α={alpha}): {min(eff_signals)*100:.1f}% → {max(eff_signals)*100:.1f}%  "
          f"(spread: {(max(eff_signals)-min(eff_signals))*100:.1f}%)")
    
    print(f"\n--- Key insight: Per-device LDP ---")
    print(f"  Old (global): noise ∝ 2C/(N×ε) → N×ε ≈ const → same noise for all γ")
    print(f"  New (per-dev): noise ∝ (2C/ε)/√N → both ε AND N matter independently")
    print(f"  N spread: {min([game_results[g]['N'] for g in gammas_ok])} → "
          f"{max([game_results[g]['N'] for g in gammas_ok])}")
    print(f"  √N ratio: {np.sqrt(max([game_results[g]['N'] for g in gammas_ok]))/np.sqrt(min([game_results[g]['N'] for g in gammas_ok])):.2f}x")

print("\nDone.")
