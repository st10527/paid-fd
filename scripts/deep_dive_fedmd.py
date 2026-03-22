#!/usr/bin/env python3
"""Deep dive into FedMD and PAID-FD vs Fixed-eps analysis."""
import json
import numpy as np

with open('results/experiments/routeB_exp1_merged_3seeds.json') as f:
    d = json.load(f)

# FedMD: why is it DECREASING over rounds?
print("=== FedMD Trajectory Analysis ===")
for run in d['runs']['FedMD']:
    seed = run.get('seed', '?')
    accs = run['accuracies']
    q1 = np.mean(accs[:25])
    q2 = np.mean(accs[25:50])
    q3 = np.mean(accs[50:75])
    q4 = np.mean(accs[75:100])
    peak_r = np.argmax(accs)
    peak_v = max(accs)
    drift = q4 - q1
    print(f"  seed={seed}: Q1={q1:.4f} Q2={q2:.4f} Q3={q3:.4f} Q4={q4:.4f}")
    print(f"    peak at R{peak_r} ({peak_v:.4f}), final={accs[-1]:.4f}")
    print(f"    drift = {drift:+.4f} (Q4 - Q1)")

# PAID-FD game epsilon
print("\n=== PAID-FD Game Epsilon vs Fixed-eps ===")
for run in d['runs']['PAID-FD']:
    seed = run.get('seed', '?')
    avg_eps = run['extras'][0]['avg_eps']
    price = run['extras'][0]['price']
    print("  seed=%s: game eps=%.4f, price=%.4f" % (seed, avg_eps, price))

print("  Fixed-eps-0.5: epsilon = 0.5")
print("  Fixed-eps-1.0: epsilon = 1.0")
print("  => Game epsilon (~0.52-0.57) closest to Fixed-eps-0.5")

# Accuracy comparison
paid_finals = [r['accuracies'][-1] for r in d['runs']['PAID-FD']]
fe05_finals = [r['accuracies'][-1] for r in d['runs']['Fixed-eps-0.5']]
fe10_finals = [r['accuracies'][-1] for r in d['runs']['Fixed-eps-1.0']]
print("\n  PAID-FD:   %.2f +/- %.2f%%" % (np.mean(paid_finals)*100, np.std(paid_finals)*100))
print("  Fixed-0.5: %.2f +/- %.2f%%" % (np.mean(fe05_finals)*100, np.std(fe05_finals)*100))
print("  Fixed-1.0: %.2f +/- %.2f%%" % (np.mean(fe10_finals)*100, np.std(fe10_finals)*100))
print("  => All within ~0.5%, effectively equivalent accuracy")
