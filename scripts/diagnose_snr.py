#!/usr/bin/env python3
"""Diagnose SNR (signal-to-noise ratio) in PAID-FD aggregated logits."""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import ServerPricing

print("=" * 70)
print("PAID-FD Noise Analysis: Signal-to-Noise Ratio in Aggregated Logits")
print("=" * 70)
print()

for gamma in [3, 5, 7, 10, 15, 20]:
    gen = HeterogeneityGenerator(50, 'config/devices/heterogeneity.yaml', seed=42)
    devices = gen.generate()
    sp = ServerPricing(gamma=gamma, delta=0.001)
    p, decs = sp.solve(devices)
    parts = [d for d in decs if d.participates]
    N = len(parts)
    if N == 0:
        print("g=%2d: NO PARTICIPANTS" % gamma)
        continue
    avg_eps = np.mean([d.eps_star for d in parts])
    min_eps = min(d.eps_star for d in parts)
    max_eps = max(d.eps_star for d in parts)

    # Noise analysis
    C = 5.0  # clip_bound
    sens = 2.0 * C / N  # sensitivity per element after averaging
    noise_scale = sens / avg_eps  # Laplace scale parameter

    # Typical logit magnitude for 100-class: signal ~ 2-5
    signal = 3.0
    snr = signal / noise_scale
    noise_std = noise_scale * np.sqrt(2)  # Laplace std = b*sqrt(2)

    flag = " <-- NOISY!" if snr < 1.0 else (" OK" if snr > 3.0 else " marginal")
    print("g=%2d: N=%2d, eps=[%.2f, %.2f] avg=%.3f, noise_scale=%.3f, noise_std=%.3f, SNR=%.2f%s" %
          (gamma, N, min_eps, max_eps, avg_eps, noise_scale, noise_std, snr, flag))

print()
print("=== KEY INSIGHT ===")
print("clip_bound C=5.0, sensitivity per elem = 2C/N")
print("noise_scale = sensitivity / avg_eps = (2C/N) / avg_eps")
print()
print("SNR < 1  => noise DOMINATES signal => teacher is garbage")
print("SNR 1-3  => marginal => some learning but very slow")
print("SNR > 3  => signal dominates => teacher is useful")
print()
print("Problem: avg_eps is very small (strong privacy) => huge noise!")
print()

# What if we reduce clip_bound?
print("=== CLIP BOUND SENSITIVITY ===")
for C_test in [5.0, 3.0, 2.0, 1.0]:
    for gamma in [5, 10]:
        gen = HeterogeneityGenerator(50, 'config/devices/heterogeneity.yaml', seed=42)
        devices = gen.generate()
        sp = ServerPricing(gamma=gamma, delta=0.001)
        p, decs = sp.solve(devices)
        parts = [d for d in decs if d.participates]
        N = len(parts)
        if N == 0:
            continue
        avg_eps = np.mean([d.eps_star for d in parts])
        sens = 2.0 * C_test / N
        noise_scale = sens / avg_eps
        snr = 3.0 / noise_scale
        print("  C=%.1f, g=%2d: N=%2d, noise_scale=%.3f, SNR=%.2f" % (C_test, gamma, N, noise_scale, snr))
