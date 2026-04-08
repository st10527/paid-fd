#!/usr/bin/env python3
"""Parse v9.2 self-anchor sweep results with full diagnostics."""

import json
import sys

RESULTS_PATH = "results/experiments/v9_2_self_anchor.json"

with open(RESULTS_PATH) as f:
    data = json.load(f)

print("=" * 90)
print("  v9.2 SELF-ANCHOR RESULTS  (%d rounds, seed=%d)" % (data["n_rounds"], data["seed"]))
print("=" * 90)

# Detect scale (fraction vs percentage)
sample_run = list(data["runs"].values())[0]
sample_accs = sample_run["accuracies"]
scale = 100 if max(sample_accs) <= 1.0 else 1

configs = data["configs"]
runs = data["runs"]

# ========== MAIN TABLE ==========
print("\n  MAIN TABLE")
print("  {:<20s} {:>4s} {:>4s} {:>5s} {:>5s} {:>6s} {:>6s} {:>6s} {:>7s} {:>5s} {:>5s}".format(
    "Config", "C", "T", "αce", "αsa", "R1", "Final", "Best", "Delta", "Part", "Eps"))
print("  " + "-" * 88)

for key in configs:
    if key not in runs:
        continue
    cfg = configs[key]
    # Format: (C, T, ce_alpha, self_anchor_alpha, gamma, label)
    C, T, ce_a, sa_a, gamma, desc = cfg
    run = runs[key]
    accs = [a * scale for a in run["accuracies"]]
    r1 = accs[0]
    final = accs[-1]
    best = max(accs)
    delta = final - r1
    parts = run.get("participation_rates", [])
    avg_part = sum(parts) / len(parts) * 100 if parts else 0
    eps_list = run.get("avg_eps", [])
    avg_eps = sum(eps_list) / len(eps_list) if eps_list else 0

    tag = "+" if delta > 0.5 else ("~" if delta > -2.0 else "-")
    print("  {:<20s} {:4.0f} {:4.0f} {:5.1f} {:5.1f} {:5.1f}% {:5.1f}% {:5.1f}% {:+5.1f}% {:4.0f}% {:5.2f} {} (g={})".format(
        key, C, T, ce_a, sa_a, r1, final, best, delta, avg_part, avg_eps, tag, gamma))

# ========== TRAJECTORY TABLE ==========
print("\n  TRAJECTORY (every 5 rounds)")
header = "  {:<20s}".format("Config")
for r in range(0, 50, 5):
    header += " {:>5s}".format("R%d" % r)
header += " {:>5s}".format("R49")
print(header)
print("  " + "-" * (20 + 11 * 6))

for key in configs:
    if key not in runs:
        continue
    accs = [a * scale for a in runs[key]["accuracies"]]
    line = "  {:<20s}".format(key)
    for r in range(0, min(50, len(accs)), 5):
        line += " {:5.1f}".format(accs[r])
    if len(accs) > 0:
        line += " {:5.1f}".format(accs[-1])
    print(line)

# ========== DIAGNOSTICS ==========
print("\n  DIAGNOSTICS (from extras)")
has_diag = False
for key in configs:
    if key not in runs:
        continue
    extras = runs[key].get("extras", [])
    if not extras:
        continue
    
    # Check if diagnostics exist
    sample = extras[0] if extras else {}
    diag_keys = ["pre_distill_acc", "post_distill_acc", "distill_delta",
                 "kl_teacher_self", "mean_loss_teacher", "mean_loss_self"]
    available = [k for k in diag_keys if k in sample]
    
    if not available and not has_diag:
        print("  (No diagnostic signals found in extras)")
        break
    
    if not has_diag:
        has_diag = True
        print()
        print("  Pre/Post distillation accuracy per round:")
        print("  {:<20s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
            "Config", "PreAvg", "PostAvg", "DistΔ", "KL(t||s)", "LossT", "LossS"))
        print("  " + "-" * 80)

    pre_accs = [e.get("pre_distill_acc", 0) for e in extras if "pre_distill_acc" in e]
    post_accs = [e.get("post_distill_acc", 0) for e in extras if "post_distill_acc" in e]
    dist_deltas = [e.get("distill_delta", 0) for e in extras if "distill_delta" in e]
    kl_ts = [e.get("kl_teacher_self", 0) for e in extras if "kl_teacher_self" in e]
    loss_t = [e.get("mean_loss_teacher", 0) for e in extras if "mean_loss_teacher" in e]
    loss_s = [e.get("mean_loss_self", 0) for e in extras if "mean_loss_self" in e]

    avg_pre = sum(pre_accs) / len(pre_accs) * scale if pre_accs else 0
    avg_post = sum(post_accs) / len(post_accs) * scale if post_accs else 0
    avg_dd = sum(dist_deltas) / len(dist_deltas) * scale if dist_deltas else 0
    avg_kl = sum(kl_ts) / len(kl_ts) if kl_ts else 0
    avg_lt = sum(loss_t) / len(loss_t) if loss_t else 0
    avg_ls = sum(loss_s) / len(loss_s) if loss_s else 0

    print("  {:<20s} {:7.2f}% {:7.2f}% {:+7.2f}% {:10.4f} {:10.4f} {:10.4f}".format(
        key, avg_pre, avg_post, avg_dd,
        avg_kl, avg_lt, avg_ls))

# Per-round distill delta trajectory for key configs
if has_diag:
    print("\n  DISTILL DELTA TRAJECTORY (per round, showing every 10 rounds)")
    print("  {:<20s}".format("Config"), end="")
    for r in range(0, 50, 10):
        print(" {:>7s}".format("R%d" % r), end="")
    print(" {:>7s}".format("R49"))
    print("  " + "-" * 60)
    for key in configs:
        if key not in runs:
            continue
        extras = runs[key].get("extras", [])
        deltas = [e.get("distill_delta", None) for e in extras]
        if not any(d is not None for d in deltas):
            continue
        line = "  {:<20s}".format(key)
        for r in range(0, min(50, len(deltas)), 10):
            if deltas[r] is not None:
                line += " {:+6.2f}%".format(deltas[r] * scale)
            else:
                line += "     N/A"
        if len(deltas) > 0 and deltas[-1] is not None:
            line += " {:+6.2f}%".format(deltas[-1] * scale)
        print(line)

# ========== KEY COMPARISONS ==========
print("\n" + "=" * 90)
print("  KEY COMPARISONS")
print("=" * 90)

# Q1: Does self-anchor stop degradation?
print("\n  Q1: Does self-anchor stop degradation? (compare A0 vs A1/A2/A3)")
for key in ["A0_baseline", "A1_sa03", "A2_sa05", "A3_sa07"]:
    if key not in runs:
        continue
    accs = [a * scale for a in runs[key]["accuracies"]]
    sa_a = configs[key][3]
    print("    αsa=%.1f: R1=%.1f%% → Final=%.1f%% (Δ=%+.1f%%, Best=%.1f%%)" % (
        sa_a, accs[0], accs[-1], accs[-1] - accs[0], max(accs)))

# Q2: Best α_sa?
print("\n  Q2: Which α_sa is best?")
alpha_finals = []
for key in ["A0_baseline", "A1_sa03", "A2_sa05", "A3_sa07"]:
    if key not in runs:
        continue
    accs = [a * scale for a in runs[key]["accuracies"]]
    alpha_finals.append((key, configs[key][3], accs[-1], accs[-1] - accs[0]))
if alpha_finals:
    best = max(alpha_finals, key=lambda x: x[2])
    least_degrade = max(alpha_finals, key=lambda x: x[3])
    print("    Highest final:    %s (αsa=%.1f, Final=%.1f%%)" % (best[0], best[1], best[2]))
    print("    Least degradation: %s (αsa=%.1f, Δ=%+.1f%%)" % (least_degrade[0], least_degrade[1], least_degrade[3]))

# Q3: Self-anchor vs CE anchor
print("\n  Q3: Self-anchor vs CE anchor?")
for key in ["A2_sa05", "A4_ce03"]:
    if key not in runs:
        continue
    accs = [a * scale for a in runs[key]["accuracies"]]
    print("    %s: R1=%.1f%% → Final=%.1f%% (Δ=%+.1f%%)" % (
        key, accs[0], accs[-1], accs[-1] - accs[0]))

# Q4: Gamma differentiation
print("\n  Q4: γ differentiation with self-anchor?")
print("    With self-anchor (αsa=0.5):")
gamma_sa = []
for key in ["B1_g3_sa05", "B2_g5_sa05", "B3_g10_sa05"]:
    if key not in runs:
        continue
    accs = [a * scale for a in runs[key]["accuracies"]]
    gamma = configs[key][4]
    gamma_sa.append((gamma, accs[-1], accs[-1] - accs[0]))
    print("      γ=%d: Final=%.1f%% (Δ=%+.1f%%)" % (gamma, accs[-1], accs[-1] - accs[0]))

print("    Without anchor:")
gamma_no = []
for key in ["B4_g3_noanchor", "A0_baseline", "B5_g10_noanchor"]:
    if key not in runs:
        continue
    accs = [a * scale for a in runs[key]["accuracies"]]
    gamma = configs[key][4]
    gamma_no.append((gamma, accs[-1], accs[-1] - accs[0]))
    print("      γ=%d: Final=%.1f%% (Δ=%+.1f%%)" % (gamma, accs[-1], accs[-1] - accs[0]))

if len(gamma_sa) >= 2:
    gap_sa = max(f for _, f, _ in gamma_sa) - min(f for _, f, _ in gamma_sa)
    print("    → Self-anchor γ gap: %.1f%%" % gap_sa)
if len(gamma_no) >= 2:
    gap_no = max(f for _, f, _ in gamma_no) - min(f for _, f, _ in gamma_no)
    print("    → No-anchor γ gap:   %.1f%% (v9.1 was 0.9%%)" % gap_no)

# ========== VERDICT ==========
print("\n" + "=" * 90)
print("  VERDICT")
print("=" * 90)

# Auto-classify
if alpha_finals:
    baseline_delta = [d for k, _, _, d in alpha_finals if k == "A0_baseline"]
    best_sa_delta = max((d for k, _, _, d in alpha_finals if "sa" in k), default=-999)
    
    if baseline_delta and best_sa_delta > -1.0:
        if best_sa_delta > 0.5:
            print("  🎉 IDEAL: Self-anchor stops degradation AND improves accuracy")
        else:
            print("  ✅ GOOD: Self-anchor stops degradation (stable)")
    elif baseline_delta and best_sa_delta > baseline_delta[0] + 1.0:
        print("  📊 PARTIAL: Self-anchor reduces degradation but doesn't eliminate it")
    else:
        print("  ❌ FAILED: Self-anchor doesn't help")

    if len(gamma_sa) >= 2:
        gap = max(f for _, f, _ in gamma_sa) - min(f for _, f, _ in gamma_sa)
        if gap > 3.0:
            print("  🎯 γ differentiation: STRONG (%.1f%%)" % gap)
        elif gap > 1.5:
            print("  📈 γ differentiation: MODERATE (%.1f%%)" % gap)
        else:
            print("  ⚠️  γ differentiation: WEAK (%.1f%%)" % gap)

print()
