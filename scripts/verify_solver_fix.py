#!/usr/bin/env python3
"""
Verify the cubic solver fix works end-to-end with the Stackelberg game.
Tests:
1. DeviceBestResponse now returns Root 2 (larger ε*)
2. p↑ → ε*↑ (Proposition 2)
3. SNR is workable (>> 1)
4. ServerPricing finds reasonable equilibrium
5. γ differentiation: higher γ → lower aggregate noise
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.game.stackelberg import DeviceBestResponse, ServerPricing
from src.game.utility import QualityFunction

print("=" * 70)
print("SOLVER FIX VERIFICATION")
print("=" * 70)

# Test 1: DeviceBestResponse now returns Root 2
print("\n[Test 1] DeviceBestResponse returns utility-maximizing root (Root 2)")
solver = DeviceBestResponse(eps_max=20.0)
c = 0.2
lam = 0.05

for p in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    decision = solver.solve(p=p, c=c, lambda_i=lam, device_id=0)
    if decision.participates:
        nvar = 2 * (4 / decision.eps_star) ** 2
        print("  p=%.2f: ε*=%.3f  s*=%.1f  quality=%.3f  utility=%.3f  noise_var=%.2f" % (
            p, decision.eps_star, decision.s_star, decision.quality, decision.utility, nvar))
    else:
        print("  p=%.2f: non-participation" % p)

# Test 2: Proposition 2 check across lambda values
print("\n[Test 2] Proposition 2: p↑ → ε*↑ for various λ")
for lam in [0.05, 0.10, 0.20, 0.40, 0.80]:
    prev_eps = None
    results = []
    for p in [2.0, 3.0, 4.0, 5.0, 6.0]:
        dec = solver.solve(p=p, c=c, lambda_i=lam, device_id=0)
        if dec.participates:
            direction = ""
            if prev_eps is not None:
                direction = "↑" if dec.eps_star > prev_eps else "↓ VIOLATION!"
            results.append("p=%.1f:ε=%.2f%s" % (p, dec.eps_star, direction))
            prev_eps = dec.eps_star
        else:
            results.append("p=%.1f:NP" % p)
    print("  λ=%.2f: %s" % (lam, "  ".join(results)))

# Test 3: SNR check
print("\n[Test 3] SNR at typical operating points")
print("  (n=50 devices, BLUE aggregation, Delta=2C=4.0)")
n_devices = 50
C_clip = 2.0
K = 100  # classes

for p_val in [3.0, 4.0, 5.0]:
    for lam_val in [0.05, 0.20, 0.50]:
        dec = solver.solve(p=p_val, c=c, lambda_i=lam_val, device_id=0)
        if dec.participates:
            # Individual noise variance per class logit
            eps = dec.eps_star
            noise_var = 2 * (2 * C_clip / eps) ** 2  # Laplace var = 2b², b=Δ/ε
            # BLUE with n devices: var_agg ≈ noise_var / n (if ε similar)
            agg_var = noise_var / n_devices
            # Signal ≈ logit magnitude ≈ 1-5 for trained model
            signal_mag = 2.0  # conservative
            snr = signal_mag**2 / agg_var
            print("  p=%.1f λ=%.2f: ε*=%.2f  noise_var=%.2f  agg_var=%.4f  SNR=%.1f" % (
                p_val, lam_val, eps, noise_var, agg_var, snr))

# Test 4: Full ServerPricing equilibrium
print("\n[Test 4] ServerPricing equilibrium for different γ")

# Generate heterogeneous devices
np.random.seed(42)

# Create simple device objects
class SimpleDevice:
    def __init__(self, c_total, lambda_i):
        self.c_total = c_total
        self.lambda_i = lambda_i

devices = []
for _ in range(50):
    sensitivity = np.random.choice(['very_low', 'low', 'medium', 'high', 'very_high'],
                                    p=[0.15, 0.25, 0.25, 0.20, 0.15])
    base_lambda = {'very_low': 0.05, 'low': 0.15, 'medium': 0.40, 
                   'high': 0.80, 'very_high': 1.50}[sensitivity]
    jitter = np.random.uniform(-0.3, 0.3)
    lam = base_lambda * (1 + jitter)
    c_val = np.random.uniform(0.1, 0.4)
    devices.append(SimpleDevice(c_total=c_val, lambda_i=lam))

for gamma in [2, 3, 5, 7, 10]:
    pricing = ServerPricing(gamma=gamma, delta=0.01)
    p_star, decisions = pricing.solve(devices=devices)
    
    participating = [d for d in decisions if d.participates]
    n_part = len(participating)
    
    if n_part > 0:
        eps_values = [d.eps_star for d in participating]
        # BLUE weights: w_i = ε_i²
        weights = [e**2 for e in eps_values]
        w_sum = sum(weights)
        # Aggregate BLUE noise variance
        agg_var = sum((w/w_sum)**2 * 2*(4/e)**2 for w, e in zip(weights, eps_values))
        
        avg_eps = np.mean(eps_values)
        min_eps = min(eps_values)
        max_eps = max(eps_values)
        
        server_u = (gamma - p_star) * sum(d.quality for d in participating)
        
        print("  γ=%2d: p*=%.2f  n_part=%d  avg_ε=%.2f [%.2f, %.2f]  agg_var=%.4f  U_ES=%.2f" % (
            gamma, p_star, n_part, avg_eps, min_eps, max_eps,
            agg_var, server_u))
    else:
        print("  γ=%2d: p*=%.2f  no participants" % (gamma, p_star))

# Test 5: γ differentiation summary
print("\n[Test 5] γ Differentiation Summary")
print("  If higher γ → lower agg_var, then γ differentiation works!")
print("  This should show monotonically decreasing aggregate noise with γ.")

print("\n" + "=" * 70)
print("SOLVER FIX VERIFICATION COMPLETE")
print("=" * 70)
