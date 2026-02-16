#!/usr/bin/env python3
"""Test lambda_mult sweep behavior with various gamma values."""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import ServerPricing

def test_sweep(gamma, lm_values):
    print(f'=== Lambda mult sweep at gamma={gamma} ===')
    for lm in lm_values:
        gen = HeterogeneityGenerator(50, 'config/devices/heterogeneity.yaml',
                                      config_override={'privacy_sensitivity': {'lambda_mult': lm}}, seed=42)
        devices = gen.generate()
        lambdas = [d.lambda_i for d in devices]
        sp = ServerPricing(gamma=gamma, delta=0.001)
        p, decisions = sp.solve(devices)
        parts = [d for d in decisions if d.participates]
        n = len(parts)
        lam_set = sorted(set([round(l,3) for l in lambdas]))
        if n > 0:
            eps_avg = np.mean([d.eps_star for d in parts])
            s_avg = np.mean([d.s_star for d in parts])
            msg = f'  lm={lm:.1f}: lam={lam_set}, part={n}/50, price={p:.3f}, eps_avg={eps_avg:.3f}, s_avg={s_avg:.1f}'
            print(msg)
        else:
            print(f'  lm={lm:.1f}: lam={lam_set}, NO PARTICIPANTS')
    print()

test_sweep(5, [0.1, 0.5, 1.0, 2.0, 5.0])
test_sweep(7, [0.1, 0.5, 1.0, 2.0, 5.0])
test_sweep(10, [0.1, 0.5, 1.0, 2.0, 5.0])
test_sweep(15, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
test_sweep(20, [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
