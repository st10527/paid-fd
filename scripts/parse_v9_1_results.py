#!/usr/bin/env python3
"""Parse and display v9.1 distillation fix results."""
import json
import sys

with open('results/experiments/v9_1_distill_fix.json') as f:
    data = json.load(f)

configs = data['configs']
runs = data['runs']

# Config format: [C, T, CE_alpha, gamma, description]
print("=" * 95)
hdr = f"{'Config':<20} {'C':>3} {'T':>3} {'CE':>4} {'g':>3}"
hdr += f" {'Part%':>6} {'R1':>7} {'R10':>7} {'R25':>7}"
hdr += f" {'Final':>7} {'Best':>7} {'D':>8}"
print(hdr)
print("=" * 95)

for name in runs:
    run = runs[name]
    cfg = configs[name]
    accs = run['accuracies']
    parts = run['participation_rates']

    C, T, ce, gamma = cfg[0], cfg[1], cfg[2], cfg[3]

    # Detect if values are fractions (0-1) or percentages (0-100)
    scale = 100 if max(accs) <= 1.0 else 1
    r1 = accs[0] * scale
    r10 = (accs[9] if len(accs) > 9 else accs[-1]) * scale
    r25 = (accs[24] if len(accs) > 24 else accs[-1]) * scale
    final = accs[-1] * scale
    best = max(accs) * scale
    delta = final - r1
    avg_part = sum(parts) / len(parts) if parts else 0

    line = f"{name:<20} {C:>3.0f} {T:>3.0f} {ce:>4.1f} {gamma:>3}"
    line += f" {avg_part*100:>5.0f}%"
    line += f" {r1:>6.1f}% {r10:>6.1f}% {r25:>6.1f}%"
    line += f" {final:>6.1f}% {best:>6.1f}% {delta:>+7.1f}%"
    print(line)

print("=" * 95)

# Q1/Q2/Q3
d2_accs = runs['D2_C5_T1_g5']['accuracies']
d3_accs = runs['D3_C5_T2_g5']['accuracies']
d7_accs = runs['D7_C5_T1_g10']['accuracies']
d8_accs = runs['D8_C5_T1_g3']['accuracies']

# Scale if stored as fractions
sc = 100 if max(d2_accs) <= 1.0 else 1
d2_r1, d2_final = d2_accs[0]*sc, d2_accs[-1]*sc
d3_r1, d3_final = d3_accs[0]*sc, d3_accs[-1]*sc
d7_r1, d7_final = d7_accs[0]*sc, d7_accs[-1]*sc
d8_r1, d8_final = d8_accs[0]*sc, d8_accs[-1]*sc

print()
print("=== DECISION FRAMEWORK ===")
print()

# Q1
thresh = d2_r1 - 2
q1 = d2_final >= thresh
print(f"Q1: D2 final={d2_final:.1f}% >= R1-2%={thresh:.1f}%?")
print(f"    -> {'YES' if q1 else 'NO'}: Distillation {'works' if q1 else 'STILL DEGRADES'}")
print()

# Q2
q2_mono = d7_final > d2_final > d8_final
q2_gap = d7_final - d8_final
q2 = q2_mono and q2_gap > 3
print(f"Q2: D7(g=10)={d7_final:.1f}% > D2(g=5)={d2_final:.1f}% > D8(g=3)={d8_final:.1f}%?")
print(f"    Monotonic: {'YES' if q2_mono else 'NO'}")
print(f"    Gap (g10-g3): {q2_gap:.1f}%")
print(f"    -> {'YES' if q2 else 'NO'}: gamma differentiation {'VALIDATED' if q2 else 'NOT sufficient'}")
print()

# Q3
thresh3 = d3_r1 - 2
q3 = d3_final >= thresh3
print(f"Q3: D3(C=5,T=2) final={d3_final:.1f}% >= R1-2%={thresh3:.1f}%?")
print(f"    -> {'YES' if q3 else 'NO'}: Soft labels {'preserved' if q3 else 'NOT preserved'}")
print()

# Outcome
print("=== OUTCOME ===")
if q1 and q2 and q3:
    print("Q1=YES Q2=YES Q3=YES")
    print("-> Use C=5, T=3 (or T=2), proceed to formal experiments!")
elif q1 and q2:
    print("Q1=YES Q2=YES Q3=NO")
    print("-> Use C=5, T=1 (hard labels), proceed to formal experiments")
elif q1:
    print("Q1=YES Q2=NO")
    print("-> gamma no differentiation, consider efficiency story (Route B)")
else:
    print("Q1=NO")
    print("-> Fundamental rethink needed")

# Trajectories
print()
print("=== KEY TRAJECTORIES (every 5 rounds) ===")
for name in ['D1_C2_T3_g5', 'D2_C5_T1_g5', 'D3_C5_T2_g5',
             'D7_C5_T1_g10', 'D8_C5_T1_g3']:
    accs = runs[name]['accuracies']
    sc2 = 100 if max(accs) <= 1.0 else 1
    pts = [f"{accs[i]*sc2:.1f}" for i in range(0, len(accs), 5)]
    print(f"  {name}:")
    print(f"    {' -> '.join(pts)}")

# Ablation comparisons
print()
print("=== ABLATION INSIGHTS ===")
d1_f = runs['D1_C2_T3_g5']['accuracies'][-1] * sc
d2_f = d2_final
d3_f = d3_final
d4_f = runs['D4_C5_T1_CE_g5']['accuracies'][-1] * sc
d5_f = runs['D5_C2_T1_g5']['accuracies'][-1] * sc
d6_f = runs['D6_C5_T3_g5']['accuracies'][-1] * sc

print(f"T effect (C=5):  T=1({d2_f:.1f}%) vs T=2({d3_f:.1f}%) vs T=3({d6_f:.1f}%)")
print(f"C effect (T=1):  C=5({d2_f:.1f}%) vs C=2({d5_f:.1f}%)")
print(f"CE effect:       no-CE({d2_f:.1f}%) vs CE=0.3({d4_f:.1f}%)")
print(f"Baseline v9.0:   C=2,T=3({d1_f:.1f}%) vs C=5,T=1({d2_f:.1f}%)")
print(f"gamma effect:    g=3({d8_final:.1f}%) vs g=5({d2_f:.1f}%) vs g=10({d7_final:.1f}%)")
