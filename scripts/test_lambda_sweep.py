#!/usr/bin/env python3
"""Quick test: what lambda_mult gives SNR > 2?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import ServerPricing

C = 2.0
Delta = 2 * C

for lm in [1.0, 0.1, 0.01, 0.001]:
    override = {'privacy_sensitivity': {'lambda_mult': lm}} if lm != 1.0 else None
    gen = HeterogeneityGenerator(50, 'config/devices/heterogeneity.yaml', override, seed=42)
    devices = gen.generate()
    lam_min = min(d.lambda_i for d in devices)
    lam_max = max(d.lambda_i for d in devices)
    print("lambda_mult=%.3f  lambda_range=[%.5f, %.5f]" % (lm, lam_min, lam_max))
    for g in [3, 5, 7, 10]:
        sp = ServerPricing(gamma=g, delta=0.01)
        price, decs = sp.solve(devices)
        parts = [d for d in decs if d.participates]
        n = len(parts)
        if n == 0:
            print("  g=%2d: 0 participants" % g)
            continue
        eps_list = [d.eps_star for d in parts]
        sum_eps2 = sum(e**2 for e in eps_list)
        var_blue = 2 * Delta**2 / sum_eps2
        snr = 0.5 / var_blue
        print("  g=%2d: n=%2d  price=%.3f  avg_eps=%.3f  sum_eps2=%7.1f  Var=%.4f  SNR=%.2f" % (
            g, n, price, np.mean(eps_list), sum_eps2, var_blue, snr))
    print()
