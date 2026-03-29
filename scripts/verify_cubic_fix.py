#!/usr/bin/env python3
"""Quick verification of the cubic solver fix."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game.stackelberg import StackelbergSolver
from src.game.utility import verify_foc_conditions, QualityFunction
from src.devices.heterogeneity import HeterogeneityGenerator
import numpy as np

# Generate test devices
gen = HeterogeneityGenerator(n_devices=20, seed=42)
devices = gen.generate()

# Print typical cost values
print("=== Device Cost Parameters ===")
for d in devices[:5]:
    print(f"  Device {d.device_id}: c_inf={d.c_inf:.3f}, c_comm={d.c_comm:.3f}, "
          f"c_total={d.c_total:.3f}, lambda={d.lambda_i:.3f}")

# Test solver with gamma=5
solver = StackelbergSolver(gamma=5.0)
result = solver.solve(devices)

print(f"\n=== Game Result (gamma=5) ===")
print(f"  Price p*: {result['price']:.4f}")
print(f"  Server utility: {result['server_utility']:.4f}")
print(f"  Participation: {result['participation_rate']:.2%}")
print(f"  Avg eps*: {result['avg_eps']:.4f}")
print(f"  Avg s*: {result['avg_s']:.2f}")

# Verify FOC for participating devices
print(f"\n=== FOC Verification ===")
qf = QualityFunction()
n_foc_ok = 0
n_foc_total = 0
for dec in result['decisions']:
    if not dec.participates:
        continue
    n_foc_total += 1
    d = devices[dec.device_id]
    foc = verify_foc_conditions(result['price'], dec.s_star, dec.eps_star, d.c_total, d.lambda_i)
    ok = foc["both_satisfied"]
    if ok:
        n_foc_ok += 1
    print(f"  Device {dec.device_id}: s*={dec.s_star:.2f}, eps*={dec.eps_star:.4f}, "
          f"dU/ds={foc['dU_ds']:.6f}, dU/deps={foc['dU_deps']:.6f}, FOC_ok={ok}")

print(f"\n  FOC satisfied: {n_foc_ok}/{n_foc_total}")

# Gamma sweep
print(f"\n=== Gamma Sweep ===")
for gamma in [2, 3, 5, 7, 10, 15, 20]:
    solver = StackelbergSolver(gamma=gamma)
    result = solver.solve(devices)
    parts = [d for d in result['decisions'] if d.participates]
    avg_eps = np.mean([d.eps_star for d in parts]) if parts else 0
    avg_s = np.mean([d.s_star for d in parts]) if parts else 0
    n_part = len(parts)
    print(f"  gamma={gamma:2d}: p*={result['price']:.3f}, "
          f"part={n_part}/{len(devices)} ({result['participation_rate']:.0%}), "
          f"avg_eps={avg_eps:.3f}, avg_s={avg_s:.1f}, "
          f"U_ES={result['server_utility']:.2f}")
