#!/usr/bin/env python3
"""
v9.0 Result Diagnosis: Why does distillation still degrade despite SNR >> 1?

Key observations from v9.0:
  - Pretrained model: 45.4%
  - After 1st round distillation: 39.2% (6% DROP in one round!)
  - After 100 rounds: 35.2% (slow monotonic decline)
  - No gamma differentiation (0.3% gap)

Hypothesis: The teacher signal is nearly UNIFORM after clip(C=2) + softmax(T=3)
on 100 classes, so KL distillation teaches the model "everything is equally likely."

This script quantifies the teacher signal strength.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

print("=" * 70)
print("  v9.0 DIAGNOSIS: Teacher Signal Strength")
print("=" * 70)

# Simulate what the teacher signal looks like
C = 2.0       # clip bound
T = 3.0       # temperature
K = 100       # classes
n_devices = 50
eps = 3.0     # typical eps from solver fix

print("\n[1] Per-sample teacher signal analysis")
print("    C=%.1f, T=%.1f, K=%d, n_devices=%d, eps=%.1f" % (C, T, K, n_devices, eps))

# Simulate: a local model trained on small private data
# Correct class logit ~ 1.5 (clipped from ~3-5), wrong classes ~ 0 (clipped from ~-1 to 1)
# This is OPTIMISTIC - real local models on 600 samples/device are much worse

for correct_logit in [0.5, 1.0, 1.5, 2.0]:
    print("\n  Correct class logit = %.1f (after clipping):" % correct_logit)
    
    # Single device logit vector (1 sample)
    logits = torch.zeros(K)
    logits[0] = correct_logit  # class 0 is correct
    
    # After Laplace noise (per device)
    noise_scale = 2 * C / eps
    print("    Laplace noise scale (b=2C/eps): %.3f" % noise_scale)
    print("    Laplace noise std: %.3f" % (noise_scale * np.sqrt(2)))
    
    # Monte Carlo: average over many noise realizations
    n_mc = 10000
    teacher_correct_probs = []
    kl_from_uniform = []
    
    for _ in range(n_mc):
        # n_devices noisy logit vectors
        noisy_logits = []
        for d in range(n_devices):
            noise = np.random.laplace(0, noise_scale, K).astype(np.float32)
            noisy_logits.append(logits + torch.from_numpy(noise))
        
        # BLUE (equal eps -> equal weights)
        aggregated = torch.stack(noisy_logits).mean(dim=0)
        
        # Teacher probs
        teacher = F.softmax(aggregated / T, dim=0)
        teacher_correct_probs.append(teacher[0].item())
        
        # KL from uniform
        uniform = torch.ones(K) / K
        kl = F.kl_div(uniform.log(), teacher, reduction='sum').item()
        kl_from_uniform.append(kl)
    
    avg_correct_prob = np.mean(teacher_correct_probs)
    std_correct_prob = np.std(teacher_correct_probs)
    avg_kl = np.mean(kl_from_uniform)
    
    uniform_prob = 1.0 / K
    signal_above_uniform = avg_correct_prob - uniform_prob
    
    print("    Teacher prob for correct class: %.4f +/- %.4f" % (avg_correct_prob, std_correct_prob))
    print("    Uniform prob: %.4f" % uniform_prob)
    print("    Signal above uniform: %.4f (%.1f%% improvement)" % (
        signal_above_uniform, signal_above_uniform / uniform_prob * 100))
    print("    KL(teacher || uniform): %.6f" % avg_kl)
    print("    KL * T^2 (actual loss scale): %.6f" % (avg_kl * T * T))

# Now test different temperatures
print("\n" + "=" * 70)
print("[2] Temperature Impact on Teacher Signal")
print("    (correct class logit = 1.5 after clip, C=2.0, %d devices, eps=3.0)" % n_devices)
print("=" * 70)

correct_logit = 1.5
logits = torch.zeros(K)
logits[0] = correct_logit

for T_test in [1.0, 1.5, 2.0, 3.0, 5.0]:
    noise_scale = 2 * C / eps
    n_mc = 5000
    correct_probs = []
    kls = []
    
    for _ in range(n_mc):
        noisy = []
        for d in range(n_devices):
            noise = np.random.laplace(0, noise_scale, K).astype(np.float32)
            noisy.append(logits + torch.from_numpy(noise))
        agg = torch.stack(noisy).mean(dim=0)
        teacher = F.softmax(agg / T_test, dim=0)
        correct_probs.append(teacher[0].item())
        kl = F.kl_div((torch.ones(K)/K).log(), teacher, reduction='sum').item()
        kls.append(kl)
    
    avg_p = np.mean(correct_probs)
    avg_kl = np.mean(kls)
    signal = avg_p - 1/K
    print("  T=%.1f: correct_prob=%.4f  signal=%.4f (%.1f%% above uniform)  KL=%.4f  KL*T^2=%.4f" % (
        T_test, avg_p, signal, signal*K*100, avg_kl, avg_kl * T_test**2))

# Test different clip bounds
print("\n" + "=" * 70)
print("[3] Clip Bound Impact on Teacher Signal")
print("    (T=3.0, %d devices, eps=3.0, correct class unclipped logit = 4.0)" % n_devices)
print("=" * 70)

unclipped_correct = 4.0
for C_test in [1.0, 2.0, 3.0, 5.0, 8.0]:
    clipped_correct = min(unclipped_correct, C_test)
    logits = torch.zeros(K)
    logits[0] = clipped_correct
    
    noise_scale = 2 * C_test / eps
    n_mc = 5000
    correct_probs = []
    
    for _ in range(n_mc):
        noisy = []
        for d in range(n_devices):
            noise = np.random.laplace(0, noise_scale, K).astype(np.float32)
            noisy.append(logits + torch.from_numpy(noise))
        agg = torch.stack(noisy).mean(dim=0)
        teacher = F.softmax(agg / 3.0, dim=0)
        correct_probs.append(teacher[0].item())
    
    avg_p = np.mean(correct_probs)
    signal = avg_p - 1/K
    # SNR = eps^2 * n / 8 (independent of C, for per-class)
    per_device_nvar = 2 * (2*C_test/eps)**2
    agg_nvar = per_device_nvar / n_devices
    print("  C=%.1f: clipped_logit=%.1f  noise_b=%.2f  agg_nvar=%.3f  correct_prob=%.4f  signal=%.4f" % (
        C_test, clipped_correct, noise_scale, agg_nvar, avg_p, signal))

# Compare with T=1
print("\n" + "=" * 70)
print("[4] Combined: C x T grid (best combos)")
print("    (%d devices, eps=3.0, unclipped correct logit=4.0)" % n_devices)
print("=" * 70)
print("  {:>4s} {:>4s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
    "C", "T", "clip_log", "noise_b", "corr_prob", "signal"))
print("  " + "-" * 50)

for C_test in [2.0, 5.0, 8.0]:
    for T_test in [1.0, 2.0, 3.0]:
        clipped = min(4.0, C_test)
        logits = torch.zeros(K)
        logits[0] = clipped
        noise_scale = 2 * C_test / eps
        
        n_mc = 3000
        correct_probs = []
        for _ in range(n_mc):
            noisy = []
            for d in range(n_devices):
                noise = np.random.laplace(0, noise_scale, K).astype(np.float32)
                noisy.append(logits + torch.from_numpy(noise))
            agg = torch.stack(noisy).mean(dim=0)
            teacher = F.softmax(agg / T_test, dim=0)
            correct_probs.append(teacher[0].item())
        
        avg_p = np.mean(correct_probs)
        signal = avg_p - 1/K
        print("  {:4.1f} {:4.1f} {:8.1f} {:10.2f} {:10.4f} {:10.4f}".format(
            C_test, T_test, clipped, noise_scale, avg_p, signal))

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
With C=2.0, T=3.0, K=100 classes:
  - Teacher signal is NEARLY UNIFORM
  - Correct class gets ~1.4% probability vs 1.0% uniform
  - KL divergence from uniform is tiny
  - Each distillation round slightly damages the pretrained model
  - Over 100 rounds, 45.4% -> 35.2%

The solver fix is correct (SNR >> 1 at aggregate level).
But the FD pipeline destroys the signal via:
  1. Clip(C=2) truncates logits (information loss)
  2. softmax(T=3) on 100 classes -> nearly uniform
  3. Pure KL from near-uniform teacher teaches nothing useful

Potential fixes:
  A. Lower T (T=1 or T=2) -> stronger signal
  B. Higher C (C=5) with same eps -> preserves logit structure
  C. CE anchor (alpha=0.3-0.5) -> ground truth anchors model
  D. MSE on raw logits instead of KL on softmax
  E. Adaptive T: lower T for classes with strong signal
""")
