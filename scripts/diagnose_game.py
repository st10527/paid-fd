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

gamma_values = [2, 3, 5, 7, 10]
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
# 2. NOISE LAYER: participation + ε → noise_scale → pseudo_acc
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Noise Analysis (DP Laplace on aggregated logits)")
print("=" * 70)

C = 5.0  # clip_bound
print(f"Clip bound C = {C}")
print()

print(f"{'gamma':>6} {'N':>4} {'avg_ε':>7} {'sens':>8} {'noise_σ':>8} "
      f"{'SNR':>6} {'EMA_σ':>8} {'EMA_SNR':>8}")
print("-" * 75)

for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    
    N = r['N']
    avg_eps = r['avg_eps']
    
    # Sensitivity per element: 2C / N  (each device contributes weight 1/N)
    sensitivity = 2.0 * C / N
    noise_scale = sensitivity / avg_eps  # Laplace scale parameter
    noise_std = noise_scale * np.sqrt(2)  # Laplace std = scale * sqrt(2)
    
    # Signal magnitude: logits in [-C, C], typical spread ~C
    signal_mag = C
    snr = signal_mag / noise_std
    
    # After EMA with β=0.7 over many rounds, noise variance reduces
    # For stationary EMA: var_ema = var_single * (1-β)/(1+β)
    beta = 0.7
    ema_reduction = np.sqrt((1 - beta) / (1 + beta))
    ema_noise_std = noise_std * ema_reduction
    ema_snr = signal_mag / ema_noise_std
    
    r['noise_scale'] = noise_scale
    r['noise_std'] = noise_std
    r['snr'] = snr
    r['ema_noise_std'] = ema_noise_std
    r['ema_snr'] = ema_snr
    
    print(f"{gamma:>6} {N:>4} {avg_eps:>7.3f} {sensitivity:>8.4f} {noise_std:>8.4f} "
          f"{snr:>6.2f} {ema_noise_std:>8.4f} {ema_snr:>8.2f}")

# ============================================================
# 3. SIMULATE pseudo-label accuracy at each noise level
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Simulated Pseudo-Label Accuracy (Monte Carlo)")
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

print(f"{'gamma':>6} {'noise_σ':>8} {'1-round':>10} {'EMA(β=0.7)':>12} "
      f"{'EMA(β=0.5)':>12} {'EMA(β=0.3)':>12} {'no_EMA':>10}")
print("-" * 80)

for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    
    noise_std = r['noise_std']
    noise_scale = r['noise_scale']  # Laplace scale
    
    # 1-round pseudo accuracy (no EMA)
    acc_1round = []
    for _ in range(n_mc):
        noise = np.random.laplace(0, noise_scale, true_logits.shape)
        noisy = true_logits + noise
        acc_1round.append(np.mean(np.argmax(noisy, axis=1) == true_labels))
    
    # EMA simulation: run 100 rounds, check final buffer accuracy
    ema_results = {}
    for beta in [0.7, 0.5, 0.3, 0.0]:  # 0.0 = no EMA (just last round)
        acc_ema = []
        for _ in range(n_mc):
            buf = None
            for round_idx in range(100):
                noise = np.random.laplace(0, noise_scale, true_logits.shape)
                noisy = true_logits + noise
                if buf is None:
                    buf = noisy.copy()
                else:
                    if beta > 0:
                        buf = beta * buf + (1 - beta) * noisy
                    else:
                        buf = noisy  # no smoothing
            acc = np.mean(np.argmax(buf, axis=1) == true_labels)
            acc_ema.append(acc)
        ema_results[beta] = np.mean(acc_ema)
    
    r['pseudo_acc_1round'] = np.mean(acc_1round)
    r['pseudo_acc_ema07'] = ema_results[0.7]
    r['pseudo_acc_ema05'] = ema_results[0.5]
    r['pseudo_acc_ema03'] = ema_results[0.3]
    r['pseudo_acc_no_ema'] = ema_results[0.0]
    
    print(f"{gamma:>6} {noise_std:>8.4f} "
          f"{np.mean(acc_1round)*100:>9.1f}% "
          f"{ema_results[0.7]*100:>11.1f}% "
          f"{ema_results[0.5]*100:>11.1f}% "
          f"{ema_results[0.3]*100:>11.1f}% "
          f"{ema_results[0.0]*100:>9.1f}%")

# ============================================================
# 4. EFFECTIVE FL SIGNAL: alpha × pseudo_acc
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Effective FL Signal = α × pseudo_accuracy")
print("=" * 70)

alpha = 0.3
print(f"Current α = {alpha}")
print(f"loss = {alpha}*CE(pseudo) + {1-alpha}*CE(true)")
print()

print(f"{'gamma':>6} {'part%':>6} {'pseudo_acc':>11} {'eff_signal':>12} {'Δ from γ=3':>11}")
print("-" * 55)

base_signal = None
for gamma in gamma_values:
    r = game_results[gamma]
    if r['N'] == 0:
        continue
    pseudo_acc = r['pseudo_acc_ema07']
    eff = alpha * pseudo_acc
    if base_signal is None:
        base_signal = eff
    delta = eff - base_signal
    
    print(f"{gamma:>6} {r['part_rate']*100:>5.0f}% {pseudo_acc*100:>10.1f}% "
          f"{eff*100:>11.1f}% {delta*100:>+10.2f}%")

# ============================================================
# 5. DIAGNOSIS SUMMARY & WHAT-IF ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

if len(game_results) >= 2:
    gammas_ok = [g for g in gamma_values if game_results[g]['N'] > 0]
    
    part_rates = [game_results[g]['part_rate'] for g in gammas_ok]
    avg_eps_list = [game_results[g]['avg_eps'] for g in gammas_ok]
    noise_stds = [game_results[g]['noise_std'] for g in gammas_ok]
    pseudo_accs_07 = [game_results[g]['pseudo_acc_ema07'] for g in gammas_ok]
    pseudo_accs_no = [game_results[g]['pseudo_acc_no_ema'] for g in gammas_ok]
    eff_signals = [alpha * p for p in pseudo_accs_07]
    
    print(f"\nSignal propagation (γ={gammas_ok[0]} → γ={gammas_ok[-1]}):")
    print(f"  Participation rate:  {part_rates[0]*100:.0f}% → {part_rates[-1]*100:.0f}%  "
          f"(spread: {(max(part_rates)-min(part_rates))*100:.0f}%)")
    print(f"  Average ε:           {avg_eps_list[0]:.3f} → {avg_eps_list[-1]:.3f}  "
          f"(spread: {max(avg_eps_list)-min(avg_eps_list):.3f})")
    print(f"  Noise σ (raw):       {noise_stds[0]:.4f} → {noise_stds[-1]:.4f}  "
          f"(ratio: {max(noise_stds)/max(min(noise_stds),1e-9):.1f}x)")
    print(f"  Pseudo acc (EMA=0.7):{pseudo_accs_07[0]*100:.1f}% → {pseudo_accs_07[-1]*100:.1f}%  "
          f"(spread: {(max(pseudo_accs_07)-min(pseudo_accs_07))*100:.1f}%)")
    print(f"  Pseudo acc (no EMA): {pseudo_accs_no[0]*100:.1f}% → {pseudo_accs_no[-1]*100:.1f}%  "
          f"(spread: {(max(pseudo_accs_no)-min(pseudo_accs_no))*100:.1f}%)")
    print(f"  Eff FL signal:       {min(eff_signals)*100:.1f}% → {max(eff_signals)*100:.1f}%  "
          f"(spread: {(max(eff_signals)-min(eff_signals))*100:.1f}%)")
    
    # What-if: different α
    print(f"\n--- What-if: different α (with EMA β=0.7) ---")
    for test_alpha in [0.3, 0.5, 0.7, 1.0]:
        signals = [test_alpha * p for p in pseudo_accs_07]
        spread = (max(signals) - min(signals)) * 100
        print(f"  α={test_alpha}: spread = {spread:.1f}%  "
              f"({min(signals)*100:.1f}% → {max(signals)*100:.1f}%)")
    
    # What-if: different β
    print(f"\n--- What-if: different EMA β (with α=0.3) ---")
    for beta_test in [0.7, 0.5, 0.3, 0.0]:
        key = {0.7: 'pseudo_acc_ema07', 0.5: 'pseudo_acc_ema05', 
               0.3: 'pseudo_acc_ema03', 0.0: 'pseudo_acc_no_ema'}[beta_test]
        accs = [game_results[g][key] for g in gammas_ok]
        spread = (max(accs) - min(accs)) * 100
        label = f"β={beta_test}" if beta_test > 0 else "no EMA"
        print(f"  {label:>8}: pseudo_acc spread = {spread:.1f}%  "
              f"({min(accs)*100:.1f}% → {max(accs)*100:.1f}%)")
    
    # What-if: different pretrain
    print(f"\n--- What-if: different pre-train epochs ---")
    print(f"  Current: 50 epochs → ~52% test start, FL adds ~8% → ~60%")
    print(f"  Less pre-train = more room for FL differentiation")

print("\nDone.")
