#!/usr/bin/env python3
"""Parse v8.3 Phase 0.3 gamma sweep results."""
import json, os

fpath = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiments', 'v8_3_phase0_gamma_sweep.json')
data = json.load(open(fpath))
runs = data['runs']

print("=" * 80)
print("V8.3 PHASE 0.3: PURE BLUE GAMMA SWEEP (100 rounds, seed=42)")
print("Config: pure KL, no denoise, no CE anchor, fresh SGD")
print("=" * 80)

summaries = []

for label in runs:
    r = runs[label]
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
    best_r = accs.index(best)
    worst = min(accs)

    print()
    print("-" * 70)
    print("  %s" % label)
    print("-" * 70)
    print("  R1=%.4f  Final=%.4f  Best=%.4f (R%d)  Worst=%.4f" % (r1, final, best, best_r, worst))
    print("  Delta(final-R1)=%+.4f  Delta(best-R1)=%+.4f" % (final - r1, best - r1))
    print("  Part=%.0f%%  AvgEps=%.3f  Time=%.0fs" % (avg_part * 100, avg_eps, elapsed))

    # Trajectory every 10 rounds
    print("  Trajectory (every 10 rounds):")
    for i in range(0, len(accs), 10):
        print("    R%3d: %.4f" % (i, accs[i]), end="")
        if i == 0:
            print(" <- start", end="")
        if accs[i] == best:
            print(" <- BEST", end="")
        print()
    if (len(accs) - 1) % 10 != 0:
        print("    R%3d: %.4f <- final" % (len(accs) - 1, accs[-1]))

    summaries.append({
        'label': label, 'gamma': float(label.split('=')[1]),
        'r1': r1, 'final': final, 'best': best, 'best_r': best_r,
        'delta_final': final - r1, 'delta_best': best - r1,
        'avg_part': avg_part, 'avg_eps': avg_eps, 'elapsed': elapsed,
        'accs': accs,
    })

# Sort by gamma
summaries.sort(key=lambda x: x['gamma'])

# ====== SUMMARY TABLE ======
print()
print("=" * 80)
print("SUMMARY TABLE (sorted by gamma)")
print("=" * 80)
print("  %-10s %5s %6s %6s %6s %7s %7s %5s %5s" % (
    "Config", "Part", "R1", "Final", "Best", "dFinal", "dBest", "Eps", "BestR"))
print("  " + "-" * 72)
for s in summaries:
    print("  %-10s %4.0f%% %5.1f%% %5.1f%% %5.1f%% %+5.1f%% %+5.1f%% %5.3f  R%d" % (
        s['label'], s['avg_part'] * 100, s['r1'] * 100, s['final'] * 100,
        s['best'] * 100, s['delta_final'] * 100, s['delta_best'] * 100,
        s['avg_eps'], s['best_r']))

# ====== GAP ANALYSIS ======
print()
print("=" * 80)
print("GAP ANALYSIS")
print("=" * 80)

# Active configs (participation > 5%)
active = [s for s in summaries if s['avg_part'] > 0.05]
if len(active) < 2:
    print("  Too few active configs!")
else:
    # Final accuracy gap
    best_s = max(active, key=lambda x: x['final'])
    worst_s = min(active, key=lambda x: x['final'])
    gap_final = best_s['final'] - worst_s['final']

    # Best accuracy gap
    best_b = max(active, key=lambda x: x['best'])
    worst_b = min(active, key=lambda x: x['best'])
    gap_best = best_b['best'] - worst_b['best']

    print()
    print("  Final accuracy gap: %.1f%% (%s=%.1f%% vs %s=%.1f%%)" % (
        gap_final * 100, best_s['label'], best_s['final'] * 100,
        worst_s['label'], worst_s['final'] * 100))
    print("  Best accuracy gap:  %.1f%% (%s=%.1f%% vs %s=%.1f%%)" % (
        gap_best * 100, best_b['label'], best_b['best'] * 100,
        worst_b['label'], worst_b['best'] * 100))

    # Monotonicity check
    print()
    print("  Monotonicity check (higher gamma -> higher accuracy?):")
    for s in active:
        print("    %s: final=%.2f%%, best=%.2f%%, part=%.0f%%, eps=%.3f" % (
            s['label'], s['final'] * 100, s['best'] * 100,
            s['avg_part'] * 100, s['avg_eps']))

    finals_ordered = [s['final'] for s in active]
    is_mono = all(finals_ordered[i] <= finals_ordered[i+1] for i in range(len(finals_ordered)-1))
    print("  Final monotone increasing: %s" % ("YES" if is_mono else "NO"))

    bests_ordered = [s['best'] for s in active]
    is_mono_best = all(bests_ordered[i] <= bests_ordered[i+1] for i in range(len(bests_ordered)-1))
    print("  Best monotone increasing: %s" % ("YES" if is_mono_best else "NO"))

# ====== TREND ANALYSIS ======
print()
print("=" * 80)
print("TREND ANALYSIS (do accuracies improve, degrade, or plateau?)")
print("=" * 80)
for s in summaries:
    if s['avg_part'] < 0.05:
        print("  %s: 0%% participation (skipped)" % s['label'])
        continue
    accs = s['accs']
    # Split into quarters
    q1 = accs[:25]
    q2 = accs[25:50]
    q3 = accs[50:75]
    q4 = accs[75:]
    avg_q1 = sum(q1) / len(q1)
    avg_q2 = sum(q2) / len(q2)
    avg_q3 = sum(q3) / len(q3)
    avg_q4 = sum(q4) / len(q4)

    trend = "IMPROVING" if avg_q4 > avg_q1 + 0.005 else ("DEGRADING" if avg_q4 < avg_q1 - 0.005 else "STABLE")

    print("  %s: Q1=%.2f%% Q2=%.2f%% Q3=%.2f%% Q4=%.2f%% -> %s" % (
        s['label'], avg_q1 * 100, avg_q2 * 100, avg_q3 * 100, avg_q4 * 100, trend))

# ====== PAIRWISE COMPARISONS ======
print()
print("=" * 80)
print("PAIRWISE: gamma=3 vs gamma=5 vs gamma=7 vs gamma=10")
print("=" * 80)
by_g = {s['gamma']: s for s in summaries}
pairs = [(3, 5), (5, 7), (5, 10), (3, 10)]
for g1, g2 in pairs:
    if g1 in by_g and g2 in by_g:
        s1, s2 = by_g[g1], by_g[g2]
        gap_f = s2['final'] - s1['final']
        gap_b = s2['best'] - s1['best']
        print("  g=%s vs g=%s: final gap=%+.1f%%, best gap=%+.1f%%" % (
            g1, g2, gap_f * 100, gap_b * 100))

# ====== VERDICT ======
print()
print("=" * 80)
print("VERDICT")
print("=" * 80)

if len(active) >= 2:
    if gap_final > 0.03:
        print("  GAP > 3%%: BLUE creates gamma differentiation!")
        print("  -> Paper story: Stackelberg + BLUE = accuracy gap. Pure FD.")
        print("  -> Next: 3-seed full experiments")
    elif gap_final > 0.01:
        print("  GAP 1-3%%: Marginal gamma effect")
        print("  -> May need multi-seed to confirm, or accept CE anchor (Option A)")
    else:
        print("  GAP < 1%%: BLUE alone insufficient")
        print("  -> Proceed to Option A: CE anchor + game for efficiency story")

    # Check gamma=2 special case
    g2 = by_g.get(2.0)
    if g2:
        if g2['avg_part'] < 0.05:
            print()
            print("  gamma=2: 0%% participation (below threshold) - as expected")
        else:
            print()
            print("  gamma=2: %.0f%% participation, final=%.1f%% (unexpected!)" % (
                g2['avg_part'] * 100, g2['final'] * 100))
