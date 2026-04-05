#!/usr/bin/env python3
"""Verify the cubic solver bug: does f(eps) have two positive roots?"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.game.utility import QualityFunction

qf = QualityFunction()

def f(eps, lam, p):
    """The cubic: f(eps) = lam*eps^3 + lam*eps^2 + (1-p)*eps + 1"""
    return lam * eps**3 + lam * eps**2 + (1 - p) * eps + 1

def device_utility(p, s, eps, c, lam):
    """U_i = p * q(s,eps) - c*s - lam*eps"""
    q = qf.q(s, eps)
    return p * q - c * s - lam * eps

def s_star(p, c, eps):
    """s* = p/c - (1+eps)/eps"""
    return p / c - (1 + eps) / eps

# Test case: p=3.16, lambda=0.05, c=0.2
p = 3.16
lam = 0.05
c = 0.2

print("=" * 70)
print("CUBIC SOLVER BUG VERIFICATION")
print("f(eps) = lam*eps^3 + lam*eps^2 + (1-p)*eps + 1 = 0")
print("p=%.2f, lambda=%.2f, c=%.2f" % (p, lam, c))
print("=" * 70)

# Plot f(eps) to find all roots
print("\nf(eps) sampled:")
eps_range = np.linspace(0.01, 15, 1000)
f_vals = [f(e, lam, p) for e in eps_range]

# Find sign changes
roots = []
for i in range(len(f_vals) - 1):
    if f_vals[i] * f_vals[i+1] < 0:
        # Bisect to find root
        lo, hi = eps_range[i], eps_range[i+1]
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid, lam, p) * f(lo, lam, p) > 0:
                lo = mid
            else:
                hi = mid
        roots.append((lo + hi) / 2)

print("  Roots found: %d" % len(roots))
for i, r in enumerate(roots):
    print("  Root %d: eps=%.6f, f(eps)=%.2e" % (i+1, r, f(r, lam, p)))

# For each root, compute s*, utility
print("\nComparison of roots:")
print("  %-8s %8s %8s %10s %10s %10s" % ("Root", "eps*", "s*", "quality", "utility", "noise_var"))
print("  " + "-" * 58)
for i, eps in enumerate(roots):
    s = s_star(p, c, eps)
    if s > 0:
        q = qf.q(s, eps)
        u = device_utility(p, s, eps, c, lam)
        nvar = 2 * (4 / eps) ** 2  # Laplace noise var, Delta=2*C=4
        print("  Root %-2d %8.4f %8.1f %10.4f %10.4f %10.2f" % (i+1, eps, s, q, u, nvar))
    else:
        print("  Root %-2d %8.4f   s*<0 (infeasible)" % (i+1, eps))

print()
print("=" * 70)
print("TESTING ACROSS DIFFERENT (p, lambda) COMBINATIONS")
print("=" * 70)

for p_val in [2.37, 3.16, 3.68, 4.41]:
    for lam_val in [0.05, 0.10, 0.20, 0.40, 0.80, 1.50]:
        eps_range2 = np.linspace(0.001, 30, 5000)
        f_vals2 = [f(e, lam_val, p_val) for e in eps_range2]
        roots2 = []
        for i in range(len(f_vals2) - 1):
            if f_vals2[i] * f_vals2[i+1] < 0:
                lo, hi = eps_range2[i], eps_range2[i+1]
                for _ in range(100):
                    mid = (lo + hi) / 2
                    if f(mid, lam_val, p_val) * f(lo, lam_val, p_val) > 0:
                        lo = mid
                    else:
                        hi = mid
                roots2.append((lo + hi) / 2)

        if len(roots2) >= 2:
            # Compare utilities
            best_u = -1e9
            best_root = -1
            for j, eps in enumerate(roots2):
                s = s_star(p_val, c, eps)
                if s > 0:
                    u = device_utility(p_val, s, eps, c, lam_val)
                    if u > best_u:
                        best_u = u
                        best_root = j
            
            r1, r2 = roots2[0], roots2[-1]
            s1, s2 = s_star(p_val, c, r1), s_star(p_val, c, r2)
            u1 = device_utility(p_val, s1, r1, c, lam_val) if s1 > 0 else -999
            u2 = device_utility(p_val, s2, r2, c, lam_val) if s2 > 0 else -999
            nv1 = 2 * (4/r1)**2
            nv2 = 2 * (4/r2)**2
            winner = "Root%d" % (best_root + 1)
            print("  p=%.2f lam=%.2f: %d roots  R1=%.3f(U=%.2f,nv=%.1f) R2=%.3f(U=%.2f,nv=%.1f) -> %s wins" % (
                p_val, lam_val, len(roots2), r1, u1, nv1, r2, u2, nv2, winner))
        elif len(roots2) == 1:
            print("  p=%.2f lam=%.2f: 1 root  eps=%.3f" % (p_val, lam_val, roots2[0]))
        else:
            print("  p=%.2f lam=%.2f: 0 roots" % (p_val, lam_val))

print()
print("=" * 70)
print("PROPOSITION 2 CHECK: does eps* increase with p? (using correct root)")
print("=" * 70)

for lam_val in [0.05, 0.20, 0.50, 1.00]:
    print("\n  lambda=%.2f:" % lam_val)
    prev_eps = None
    for p_val in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        eps_range3 = np.linspace(0.001, 50, 10000)
        f_vals3 = [f(e, lam_val, p_val) for e in eps_range3]
        roots3 = []
        for i in range(len(f_vals3) - 1):
            if f_vals3[i] * f_vals3[i+1] < 0:
                lo, hi = eps_range3[i], eps_range3[i+1]
                for _ in range(100):
                    mid = (lo + hi) / 2
                    if f(mid, lam_val, p_val) * f(lo, lam_val, p_val) > 0:
                        lo = mid
                    else:
                        hi = mid
                roots3.append((lo + hi) / 2)

        if roots3:
            # Pick utility-maximizing root
            best_eps = roots3[0]
            best_u = -1e9
            for eps in roots3:
                s = s_star(p_val, c, eps)
                if s > 0:
                    u = device_utility(p_val, s, eps, c, lam_val)
                    if u > best_u:
                        best_u = u
                        best_eps = eps
            
            direction = ""
            if prev_eps is not None:
                direction = "UP" if best_eps > prev_eps else "DOWN"
            nvar = 2 * (4/best_eps)**2
            print("    p=%.2f: eps*=%.4f (n_roots=%d) nvar=%.2f  %s" % (
                p_val, best_eps, len(roots3), nvar, direction))
            prev_eps = best_eps
        else:
            print("    p=%.2f: no root" % p_val)
            prev_eps = None
