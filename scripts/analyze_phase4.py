#!/usr/bin/env python3
"""Analyze Phase 4 results."""
import json, glob, os, numpy as np

base = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments', 'tmc')

print("=" * 85)
print("PHASE 4 RESULTS - Reviewer-Defense Experiments")
print("=" * 85)

# --- Exp F: Fair Fixed-e ---
print("\n-- Exp F: Fair Fixed-e (PAID-FD pipeline, fixed epsilon) --")
print(f"{'Label':<28} {'Best':>7} {'Final':>7} {'e':>5} {'Part':>6}")
print("-" * 60)
for f in sorted(glob.glob(os.path.join(base, 'expF_*.json'))):
    d = json.load(open(f))
    s = d['summary']
    eps = d['avg_eps'][0]
    print(f"{os.path.basename(f):<28} {s['best_acc']*100:6.2f}% {s['final_acc']*100:6.2f}% {eps:5.1f} {s['avg_participation']*100:5.0f}%")

# --- Exp G: e-sweep ---
print("\n-- Exp G: Privacy-Utility Curve (e sweep) --")
print(f"{'e':>6} {'Best':>8} {'Final':>8} {'R0':>7} {'R50':>7} {'R99':>7}")
print("-" * 50)
g_files = sorted(glob.glob(os.path.join(base, 'expG_*.json')),
                 key=lambda x: json.load(open(x))['avg_eps'][0])
for f in g_files:
    d = json.load(open(f))
    s = d['summary']
    accs = d['accuracies']
    eps = d['avg_eps'][0]
    print(f"{eps:6.1f} {s['best_acc']*100:7.2f}% {s['final_acc']*100:7.2f}% "
          f"{accs[0]*100:6.2f}% {accs[50]*100:6.2f}% {accs[99]*100:6.2f}%")

# --- Exp H: Hetero-l BLUE ---
print("\n-- Exp H: Heterogeneous-lambda BLUE Validation --")
print(f"{'Label':<35} {'Best':>7} {'Final':>7} {'AvgEps':>7} {'Part':>6}")
print("-" * 65)
for f in sorted(glob.glob(os.path.join(base, 'expH_*.json'))):
    d = json.load(open(f))
    s = d['summary']
    eps_list = d.get('avg_eps', [])
    avg_eps = np.mean(eps_list) if eps_list else 0
    print(f"{os.path.basename(f):<35} {s['best_acc']*100:6.2f}% "
          f"{s['final_acc']*100:6.2f}% {avg_eps:6.2f} "
          f"{s['avg_participation']*100:5.0f}%")

# --- Cross-comparison ---
print("\n-- Cross-Comparison: Game vs No-Game --")
print("PAID-FD (g=5, game):    best~61.4%, final~60.8%, part~79%, e*~2.84")
for f in sorted(glob.glob(os.path.join(base, 'expF_*.json'))):
    d = json.load(open(f))
    s = d['summary']
    eps = d['avg_eps'][0]
    print(f"Fair Fixed-e={eps:.0f} (no game): best={s['best_acc']*100:.2f}%, "
          f"final={s['final_acc']*100:.2f}%, part=100%")

# --- Privacy-utility curve data ---
print("\n-- Privacy-Utility Curve Data (for Fig.) --")
print(f"{'e':>6} {'Best Acc':>10}")
print("-" * 20)
for f in g_files:
    d = json.load(open(f))
    eps = d['avg_eps'][0]
    print(f"{eps:6.2f} {d['summary']['best_acc']*100:9.2f}%")
print(f"  2.84    61.43%  <- PAID-FD (game-selected)")
