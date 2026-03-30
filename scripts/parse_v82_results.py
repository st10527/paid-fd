#!/usr/bin/env python3
"""Parse v8.2 Phase 0.2 verification results."""
import json, sys, os

fpath = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments', 'v8_2_phase0_verification.json')
data = json.load(open(fpath))
runs = data['runs']

print("=" * 80)
print("V8.2 PHASE 0.2 VERIFICATION RESULTS  (20 rounds, seed=42)")
print("=" * 80)

summaries = []

for name in runs:
    r = runs[name]
    accs = r['accuracies']
    cfg = r.get('config', {})
    mc = cfg.get('method_config', {})
    elapsed = r.get('elapsed_sec', 0)
    parts = r.get('participation_rates', [])
    avg_part = sum(parts) / len(parts) if parts else 0
    eps_list = r.get('avg_eps', [])
    avg_eps = sum(eps_list) / len(eps_list) if eps_list else 0

    r1 = accs[0]
    final = accs[-1]
    best = max(accs)
    worst = min(accs)
    delta = final - r1

    print()
    print("-" * 70)
    print("  %s" % name)
    print("-" * 70)
    dn = mc.get('use_denoising', '?')
    al = mc.get('ce_anchor_alpha', '?')
    gm = mc.get('gamma', cfg.get('gamma', '?'))
    cb = mc.get('clip_bound', '?')
    print("  denoise=%s  alpha=%s  gamma=%s  C=%s" % (dn, al, gm, cb))
    print("  R1=%.4f  Final=%.4f  Best=%.4f  Worst=%.4f" % (r1, final, best, worst))
    tag = "IMPROVED" if delta > 0 else "DEGRADED"
    print("  Delta=%+.4f (%s)" % (delta, tag))
    print("  Part=%.0f%%  AvgEps=%.3f  Time=%.0fs" % (avg_part * 100, avg_eps, elapsed))
    traj = [round(a, 4) for a in accs]
    print("  Trajectory: %s" % traj)

    summaries.append({
        'name': name, 'r1': r1, 'final': final, 'best': best,
        'delta': delta, 'avg_part': avg_part, 'avg_eps': avg_eps,
    })

# === COMPARISON TABLE ===
print()
print("=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
header = "%-52s %6s %6s %6s %7s %5s" % ("Config", "R1", "Final", "Best", "Delta", "Part")
print(header)
print("-" * 82)
for s in summaries:
    tag = "OK" if s['delta'] > 0.005 else ("~" if s['delta'] > -0.02 else "X")
    n = s['name'][:52]
    print("%-52s %5.1f%% %5.1f%% %5.1f%% %+5.1f%% %4.0f%% %s" % (
        n, s['r1']*100, s['final']*100, s['best']*100,
        s['delta']*100, s['avg_part']*100, tag))

# === KEY COMPARISONS ===
print()
print("=" * 80)
print("KEY COMPARISONS")
print("=" * 80)

lookup = {s['name']: s for s in summaries}

def get(prefix):
    for k, v in lookup.items():
        if k.startswith(prefix):
            return v
    return None

d1 = get("D1:")
d2 = get("D2:")
d3 = get("D3:")
d4 = get("D4:")
d5 = get("D5:")
d6 = get("D6:")
d7 = get("D7:")
d8 = get("D8:")

if d1 and d4:
    print()
    print("[1] DENOISING EFFECT (D1 vs D4, both gamma=5, pure KL):")
    print("    D1 (denoise):    R1=%.1f%% Final=%.1f%% (delta=%+.1f%%)" % (
        d1['r1']*100, d1['final']*100, d1['delta']*100))
    print("    D4 (no denoise): R1=%.1f%% Final=%.1f%% (delta=%+.1f%%)" % (
        d4['r1']*100, d4['final']*100, d4['delta']*100))
    gap = d1['final'] - d4['final']
    label = "DENOISING HELPS!" if gap > 0.02 else "Minimal difference"
    print("    Gap: %+.1f%% -> %s" % (gap*100, label))

if d1 and d5:
    print()
    print("[2] DENOISE vs CE ANCHOR (D1 vs D5, both gamma=5):")
    print("    D1 (denoise):   Final=%.1f%% (delta=%+.1f%%)" % (
        d1['final']*100, d1['delta']*100))
    print("    D5 (CE anchor): Final=%.1f%% (delta=%+.1f%%)" % (
        d5['final']*100, d5['delta']*100))
    gap = d1['final'] - d5['final']
    if gap > 0.01:
        label = "DENOISE > CE ANCHOR"
    elif gap < -0.01:
        label = "CE ANCHOR > DENOISE"
    else:
        label = "Similar"
    print("    Gap: %+.1f%% -> %s" % (gap*100, label))

if d2 and d1 and d3:
    print()
    print("[3] GAMMA DIFFERENTIATION (D2=g3, D1=g5, D3=g10):")
    print("    D2 (gamma=3):  Final=%.1f%% Part=%.0f%% Eps=%.3f" % (
        d2['final']*100, d2['avg_part']*100, d2['avg_eps']))
    print("    D1 (gamma=5):  Final=%.1f%% Part=%.0f%% Eps=%.3f" % (
        d1['final']*100, d1['avg_part']*100, d1['avg_eps']))
    print("    D3 (gamma=10): Final=%.1f%% Part=%.0f%% Eps=%.3f" % (
        d3['final']*100, d3['avg_part']*100, d3['avg_eps']))
    if d2['final'] < d1['final'] < d3['final']:
        gap = d3['final'] - d2['final']
        print("    D2 < D1 < D3 -> GAMMA WORKS! Gap=%.1f%%" % (gap * 100))
    elif d2['final'] < d1['final']:
        print("    D2 < D1 but D3 not highest -> Partial gamma effect")
    else:
        print("    Gamma ordering broken -> Need investigation")

if d7 and d8:
    print()
    print("[4] ORACLE SANITY (D7 vs D8, no noise):")
    print("    D7 (no denoise): Final=%.1f%% (delta=%+.1f%%)" % (
        d7['final']*100, d7['delta']*100))
    print("    D8 (denoise):    Final=%.1f%% (delta=%+.1f%%)" % (
        d8['final']*100, d8['delta']*100))
    gap = abs(d8['final'] - d7['final'])
    label = "Similar (expected)" if gap < 0.03 else "UNEXPECTED DIFFERENCE"
    print("    Gap: %.1f%% -> %s" % (gap * 100, label))

if d6:
    print()
    print("[5] BELT+SUSPENDERS (D6: denoise + CE alpha=0.3):")
    print("    D6: Final=%.1f%% (delta=%+.1f%%)" % (d6['final']*100, d6['delta']*100))
    if d1:
        gap = d6['final'] - d1['final']
        print("    vs D1 (denoise only): %+.1f%%" % (gap * 100))

# === VERDICT ===
print()
print("=" * 80)
print("VERDICT")
print("=" * 80)

if d1 and d1['delta'] > 0.005:
    print("  D1 IMPROVED: v8.2 denoising + pure KL WORKS!")
elif d1 and d1['delta'] > -0.02:
    print("  D1 STABLE: denoising prevents degradation but doesn't improve")
elif d1:
    print("  D1 STILL DEGRADES: denoising insufficient, need deeper fix")

gamma_ok = (d2 and d1 and d3 and d2['final'] < d1['final'] < d3['final'])
if gamma_ok:
    print("  GAMMA DIFFERENTIATION: CONFIRMED (D2 < D1 < D3)")
else:
    print("  GAMMA DIFFERENTIATION: NOT confirmed yet")

# Best config
best_s = max(summaries, key=lambda x: x['final'])
print()
print("  BEST CONFIG: %s -> Final=%.1f%%" % (best_s['name'], best_s['final']*100))
