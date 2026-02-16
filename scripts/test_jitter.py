#!/usr/bin/env python3
"""Test what adding lambda jitter does to participation curves."""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import ServerPricing

# Test with JITTERED lambdas
gen = HeterogeneityGenerator(50, 'config/devices/heterogeneity.yaml', seed=42)
devices = gen.generate()

# Add ±30% jitter
rng = np.random.RandomState(42)
for d in devices:
    jitter = 1.0 + rng.uniform(-0.3, 0.3)
    d.lambda_i = d.lambda_i * jitter

lambdas = [d.lambda_i for d in devices]
print(f"Jittered lambdas: {len(set([round(l,3) for l in lambdas]))} unique values")
print(f"  range: [{min(lambdas):.3f}, {max(lambdas):.3f}]")
print(f"  c_total range: [{min(d.c_total for d in devices):.3f}, {max(d.c_total for d in devices):.3f}]")

print("\n=== Participation curve with jittered lambda ===")
for gamma in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]:
    sp = ServerPricing(gamma=gamma, delta=0.001)
    p, decisions = sp.solve(devices)
    parts = [d for d in decisions if d.participates]
    n = len(parts)
    if n > 0:
        eps_avg = np.mean([d.eps_star for d in parts])
        s_avg = np.mean([d.s_star for d in parts])
        print(f"  gamma={gamma:3d}: price={p:.3f}, part={n}/50 ({n*2}%), eps_avg={eps_avg:.3f}, s_avg={s_avg:.1f}")
    else:
        print(f"  gamma={gamma:3d}: NO PARTICIPANTS")

# Also test WITHOUT jitter for comparison
print("\n=== Without jitter (original discrete levels) ===")
gen2 = HeterogeneityGenerator(50, 'config/devices/heterogeneity.yaml', seed=42)
devices2 = gen2.generate()
for gamma in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]:
    sp = ServerPricing(gamma=gamma, delta=0.001)
    p, decisions = sp.solve(devices2)
    parts = [d for d in decisions if d.participates]
    n = len(parts)
    if n > 0:
        eps_avg = np.mean([d.eps_star for d in parts])
        print(f"  gamma={gamma:3d}: price={p:.3f}, part={n}/50 ({n*2}%), eps_avg={eps_avg:.3f}")
    else:
        print(f"  gamma={gamma:3d}: NO PARTICIPANTS")
