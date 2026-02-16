#!/usr/bin/env python3
"""Diagnose Stackelberg game behavior with actual device parameters."""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import StackelbergSolver

# Reproduce same settings as phase 1.1
gen = HeterogeneityGenerator(
    n_devices=50,
    config_path='config/devices/heterogeneity.yaml',
    config_override={'privacy_sensitivity': {'lambda_mult': 0.1}},
    seed=42
)
devices = gen.generate()

# Check device parameters
lambdas = [d.lambda_i for d in devices]
c_totals = [d.c_total for d in devices]
print('Device parameters:')
print(f'  lambda_i: min={min(lambdas):.4f}, max={max(lambdas):.4f}, mean={np.mean(lambdas):.4f}')
print(f'  c_total:  min={min(c_totals):.4f}, max={max(c_totals):.4f}, mean={np.mean(c_totals):.4f}')
print(f'  unique lambdas: {sorted(set(lambdas))}')
print()

# Test with different gamma values
for gamma in [100, 300, 500, 700, 1000]:
    solver = StackelbergSolver(gamma=gamma)
    result = solver.solve(devices)
    
    participants = [d for d in result['decisions'] if d.participates]
    if not participants:
        print(f'gamma={gamma:5d}: NO PARTICIPANTS')
        continue
    eps_vals = [d.eps_star for d in participants]
    s_vals = [d.s_star for d in participants]
    
    price = result['price']
    part_rate = result['participation_rate']
    avg_eps = np.mean(eps_vals)
    avg_s = np.mean(s_vals)
    Q = result['total_quality']
    U = result['server_utility']
    
    print(f'gamma={gamma:5d}: price={price:.4f}, '
          f'part={part_rate:.2f}, '
          f'eps=[{min(eps_vals):.3f}-{max(eps_vals):.3f}] avg={avg_eps:.3f}, '
          f's=[{min(s_vals):.1f}-{max(s_vals):.1f}] avg={avg_s:.1f}, '
          f'Q={Q:.2f}, U_server={U:.2f}')

# Now check: what's the noise level?
print('\n--- Noise analysis ---')
C = 5.0  # clip_bound
N = 50   # all participate
avg_eps_all = np.mean([d.eps_star for d in result['decisions'] if d.participates])
sensitivity = 2.0 * C / N
noise_scale = sensitivity / avg_eps_all
print(f'C={C}, N={N}, avg_eps={avg_eps_all:.3f}')
print(f'sensitivity_per_elem = 2C/N = {sensitivity:.4f}')
print(f'noise_scale = sensitivity/eps = {noise_scale:.4f}')
print(f'Logit range: [-{C}, +{C}], signal magnitude: ~{C:.1f}')
print(f'SNR (approx): {C / noise_scale:.2f}')

# Also check: what if lambda_mult was NOT applied (i.e. raw yaml values)?
print('\n--- Without lambda_mult override (raw yaml values) ---')
gen2 = HeterogeneityGenerator(
    n_devices=50,
    config_path='config/devices/heterogeneity.yaml',
    seed=42
)
devices2 = gen2.generate()
lambdas2 = [d.lambda_i for d in devices2]
print(f'  lambda_i: {sorted(set(lambdas2))}')

for gamma in [100, 500, 1000]:
    solver = StackelbergSolver(gamma=gamma)
    result = solver.solve(devices2)
    participants = [d for d in result['decisions'] if d.participates]
    if not participants:
        print(f'  gamma={gamma}: NO PARTICIPANTS')
        continue
    eps_vals = [d.eps_star for d in participants]
    s_vals = [d.s_star for d in participants]
    print(f'  gamma={gamma}: price={result["price"]:.4f}, '
          f'part={result["participation_rate"]:.2f}, '
          f'eps avg={np.mean(eps_vals):.3f}, s avg={np.mean(s_vals):.1f}')
