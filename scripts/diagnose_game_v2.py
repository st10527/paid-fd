#!/usr/bin/env python3
"""
Deeper diagnosis of Stackelberg game — understand the regime where gamma matters.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import DeviceBestResponse, ServerPricing
from src.game.utility import QualityFunction

# Generate devices with the RECALIBRATED parameters (no lambda_mult override)
gen = HeterogeneityGenerator(
    n_devices=50,
    config_path='config/devices/heterogeneity.yaml',
    seed=42
)
devices = gen.generate()

lambdas = [d.lambda_i for d in devices]
c_totals = [d.c_total for d in devices]
print("=== Device Parameters (NEW calibration, no lambda_mult override) ===")
print(f"  lambda_i: min={min(lambdas):.3f}, max={max(lambdas):.3f}, mean={np.mean(lambdas):.3f}")
print(f"  c_total:  min={min(c_totals):.3f}, max={max(c_totals):.3f}, mean={np.mean(c_totals):.3f}")
print(f"  unique lambdas: {sorted(set([round(l,3) for l in lambdas]))}")
print()

# For a single device, find the minimum price that induces participation
print("=== Per-device participation thresholds ===")
br = DeviceBestResponse()
qf = QualityFunction()

thresholds = []
for d in devices:
    c = d.c_total
    lam = d.lambda_i
    # Binary search for minimum price that induces participation
    p_lo, p_hi = 1.0, 100.0
    for _ in range(50):
        p_mid = (p_lo + p_hi) / 2
        dec = br.solve(p_mid, c, lam)
        if dec.participates:
            p_hi = p_mid
        else:
            p_lo = p_mid
    thresholds.append(p_hi)

thresholds_sorted = sorted(thresholds)
print(f"  Min participation price: min={min(thresholds):.2f}, max={max(thresholds):.2f}")
print(f"  Percentiles: 10%={np.percentile(thresholds, 10):.2f}, "
      f"50%={np.percentile(thresholds, 50):.2f}, "
      f"90%={np.percentile(thresholds, 90):.2f}")
print(f"  For 50% participation, price >= {thresholds_sorted[24]:.2f}")
print(f"  For 80% participation, price >= {thresholds_sorted[39]:.2f}")
print(f"  For 100% participation, price >= {thresholds_sorted[49]:.2f}")
print()

# Now test with finer gamma range — especially around the participation thresholds
print("=== Game outcomes for various gamma values ===")
# The key: server chooses p* to maximize (γ-p)*Q(p)
# When γ is barely above the max threshold, the server can barely afford to set p high enough
# When γ is much larger, price stabilizes and all participate

gamma_values = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500]
for gamma in gamma_values:
    sp = ServerPricing(gamma=gamma, delta=0.001)
    p_star, decisions = sp.solve(devices)
    
    participants = [d for d in decisions if d.participates]
    n_part = len(participants)
    
    if n_part == 0:
        print(f"  gamma={gamma:5d}: NO PARTICIPANTS")
        continue
    
    eps_vals = [d.eps_star for d in participants]
    s_vals = [d.s_star for d in participants]
    Q = sum(d.quality for d in participants)
    U_server = (gamma - p_star) * Q
    
    print(f"  gamma={gamma:5d}: price={p_star:.3f}, "
          f"part={n_part}/50 ({n_part/50:.0%}), "
          f"eps=[{min(eps_vals):.3f}-{max(eps_vals):.3f}] avg={np.mean(eps_vals):.3f}, "
          f"s_avg={np.mean(s_vals):.1f}, Q={Q:.2f}, U_server={U_server:.2f}")

# Also test with lambda_mult=0.1 (reduced privacy sensitivity)
print()
print("=== With lambda_mult=0.1 ===")
gen2 = HeterogeneityGenerator(
    n_devices=50,
    config_path='config/devices/heterogeneity.yaml',
    config_override={'privacy_sensitivity': {'lambda_mult': 0.1}},
    seed=42
)
devices2 = gen2.generate()
lambdas2 = [d.lambda_i for d in devices2]
print(f"  lambda_i: {sorted(set([round(l,4) for l in lambdas2]))}")

for gamma in gamma_values:
    sp = ServerPricing(gamma=gamma, delta=0.001)
    p_star, decisions = sp.solve(devices2)
    
    participants = [d for d in decisions if d.participates]
    n_part = len(participants)
    
    if n_part == 0:
        print(f"  gamma={gamma:5d}: NO PARTICIPANTS")
        continue
    
    eps_vals = [d.eps_star for d in participants]
    s_vals = [d.s_star for d in participants]
    Q = sum(d.quality for d in participants)
    U_server = (gamma - p_star) * Q
    
    print(f"  gamma={gamma:5d}: price={p_star:.3f}, "
          f"part={n_part}/50, "
          f"eps_avg={np.mean(eps_vals):.3f}, "
          f"s_avg={np.mean(s_vals):.1f}, Q={Q:.2f}, U_server={U_server:.2f}")
