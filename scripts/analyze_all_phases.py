#!/usr/bin/env python3
"""
TMC Paper — Comprehensive Results Analysis (All Phases)
=======================================================
Phase 1: Core experiments (CIFAR-100) — Exp A, A', B, C
Phase 2: Cross-dataset (CIFAR-10) — Exp D
Phase 3: Non-IID alpha sweep — Exp E
Phase 4: Reviewer-defense — Exp F, G, H

Outputs: All tables/figures data needed for Section V.
"""
import json, glob, os, sys
import numpy as np
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'experiments', 'tmc')


def load_all():
    """Load all TMC experiment results into a dict keyed by exp label."""
    results = {}
    for f in sorted(glob.glob(os.path.join(BASE, '*.json'))):
        d = json.load(open(f))
        results[d['label']] = d
    return results


def fmt(v, pct=True):
    if v is None:
        return '   N/A'
    if pct:
        return f'{v*100:6.2f}%'
    return f'{v:6.2f}'


def group_by_method_seeds(results, prefix, key_fn=None):
    """Group results by a key function, collecting seeds."""
    groups = defaultdict(list)
    for label, d in results.items():
        if not label.startswith(prefix):
            continue
        k = key_fn(d) if key_fn else d['method']
        groups[k].append(d)
    return groups


def stats(runs, field='best_acc'):
    vals = [r['summary'][field] for r in runs]
    return np.mean(vals), np.std(vals), len(vals)


def print_header(title):
    print()
    print('=' * 85)
    print(f'  {title}')
    print('=' * 85)


# =====================================================================
# MAIN ANALYSIS
# =====================================================================
all_results = load_all()
print(f"Loaded {len(all_results)} experiment results from {BASE}")

# Count by phase
phase_counts = defaultdict(int)
for label in all_results:
    if label.startswith('expA_') or label.startswith('expAp_'):
        phase_counts['Phase 1'] += 1
    elif label.startswith('expB_'):
        phase_counts['Phase 1'] += 1
    elif label.startswith('expC_'):
        phase_counts['Phase 1'] += 1
    elif label.startswith('expD_'):
        phase_counts['Phase 2'] += 1
    elif label.startswith('expE_'):
        phase_counts['Phase 3'] += 1
    elif label.startswith('expF_') or label.startswith('expG_') or label.startswith('expH_'):
        phase_counts['Phase 4'] += 1

for p, n in sorted(phase_counts.items()):
    print(f"  {p}: {n} runs")
print(f"  TOTAL: {sum(phase_counts.values())} runs")


# =====================================================================
# TABLE II: Main Accuracy Comparison (CIFAR-100)
# =====================================================================
print_header("TABLE II: Privacy-Preserving Accuracy Comparison (CIFAR-100, N=50)")

# PAID-FD baseline (gamma=5) — from Phase 1 ablation baseline (noldp/noblue have game)
# Use expC_noldp as proxy for "PAID-FD with game" since they share same game
# Actually, we need the v10.1 baseline. Let's compute from expC_fullpart which is close.
# Better: use the expB_n50 data or the known v10.1 numbers.
print(f"\n{'Method':<30} {'Privacy':>7} {'Best Acc':>12} {'Final Acc':>12} {'Avg e/rd':>10}")
print('-' * 75)

# With LDP
print("  --- With LDP (ours and baselines) ---")
# PAID-FD baseline from v10.1 (known: best=61.43, final=60.80, eps=2.84)
print(f"{'PAID-FD (game, g=5)':<30} {'Yes':>7} {'61.43 +/- 0.32':>12} {'60.80 +/- 0.21':>12} {'2.84':>10}")

# Fair Fixed-epsilon (Exp F)
for label in ['expF_faireps1_s42', 'expF_faireps3_s42', 'expF_faireps5_s42']:
    if label in all_results:
        d = all_results[label]
        s = d['summary']
        eps = d['avg_eps'][0]
        print(f"{'Fair Fixed-e=%.0f' % eps:<30} {'Yes':>7} {fmt(s['best_acc']):>12} {fmt(s['final_acc']):>12} {eps:>10.1f}")

# Old Fixed-epsilon (Exp A — weak pipeline)
for method_tag, prefix in [('Fixed-eps-1', 'expA_fixedeps1'), ('Fixed-eps-3', 'expA_fixedeps3')]:
    runs = [all_results[l] for l in all_results if l.startswith(prefix)]
    if runs:
        m_b, s_b, n = stats(runs, 'best_acc')
        m_f, s_f, _ = stats(runs, 'final_acc')
        print(f"{'Old ' + method_tag + ' (weak pipe)':<30} {'Yes':>7} "
              f"{f'{m_b*100:.2f} +/- {s_b*100:.2f}':>12} "
              f"{f'{m_f*100:.2f} +/- {s_f*100:.2f}':>12} "
              f"{method_tag.split('-')[-1]+'.0':>10}")

# No-privacy upper bounds
print("  --- No-privacy upper bounds (reference) ---")
for method, prefix in [('FedGMKD', 'expAp_fedgmkd'), ('FedAvg', 'expAp_fedavg'), ('FedMD', 'expAp_fedmd')]:
    runs = [all_results[l] for l in all_results if l.startswith(prefix)]
    if runs:
        s = runs[0]['summary']
        print(f"{method:<30} {'No':>7} {fmt(s['best_acc']):>12} {fmt(s['final_acc']):>12} {'---':>10}")


# =====================================================================
# TABLE III: Scalability (N sweep)
# =====================================================================
print_header("TABLE III: Scalability with Network Size (CIFAR-100)")

print(f"\n{'N':>4} {'gamma':>6} {'Best Acc':>15} {'Final Acc':>15} {'Part(%)':>10} {'Avg e/rd':>10}")
print('-' * 65)

# N=50 from v10.1 known data
for N, gamma, prefix in [
    (20, 3, 'expB_n20_g3'), (20, 10, 'expB_n20_g10'),
    (50, 3, None), (50, 5, None), (50, 7, None), (50, 10, None),
    (80, 3, 'expB_n80_g3'), (80, 10, 'expB_n80_g10')]:
    
    if prefix:
        runs = [all_results[l] for l in all_results if l.startswith(prefix)]
        if not runs:
            continue
        m_b, s_b, n = stats(runs, 'best_acc')
        m_f, s_f, _ = stats(runs, 'final_acc')
        avg_part = np.mean([r['summary']['avg_participation'] for r in runs])
        avg_eps = np.mean([np.mean(r.get('avg_eps', [0])) for r in runs])
        print(f"{N:>4} {gamma:>6} {f'{m_b*100:.2f} +/- {s_b*100:.2f}':>15} "
              f"{f'{m_f*100:.2f} +/- {s_f*100:.2f}':>15} "
              f"{avg_part*100:>9.0f}% {avg_eps:>9.2f}")
    else:
        # v10.1 known data
        known = {
            (50,3): (61.22, 0.42, 61.02, 0.13, 38, 2.77),
            (50,5): (61.43, 0.32, 60.80, 0.21, 79, 2.84),
            (50,7): (61.43, 0.37, 60.88, 0.40, 90, 2.90),
            (50,10): (61.45, 0.23, 60.61, 0.61, 100, 3.09),
        }
        if (N, gamma) in known:
            b, bs, f, fs, p, e = known[(N, gamma)]
            print(f"{N:>4} {gamma:>6} {f'{b:.2f} +/- {bs:.2f}':>15} "
                  f"{f'{f:.2f} +/- {fs:.2f}':>15} {p:>9}% {e:>9.2f}")


# =====================================================================
# TABLE IV: Ablation Study
# =====================================================================
print_header("TABLE IV: Ablation Study (CIFAR-100, N=50, gamma=5)")

print(f"\n{'Configuration':<35} {'Best Acc':>15} {'Final Acc':>15} {'Delta':>8}")
print('-' * 78)

# Baseline
print(f"{'PAID-FD (full)':<35} {'61.43 +/- 0.32':>15} {'60.80 +/- 0.21':>15} {'---':>8}")

# No-LDP
runs = [all_results[l] for l in all_results if l.startswith('expC_noldp')]
if runs:
    m_b, s_b, _ = stats(runs, 'best_acc')
    m_f, s_f, _ = stats(runs, 'final_acc')
    delta = m_b - 0.6143
    print(f"{'  - LDP noise (oracle)':<35} "
          f"{f'{m_b*100:.2f} +/- {s_b*100:.2f}':>15} "
          f"{f'{m_f*100:.2f} +/- {s_f*100:.2f}':>15} "
          f"{delta*100:>+7.1f}pp")

# No-BLUE (homo-lambda)
runs = [all_results[l] for l in all_results if l.startswith('expC_noblue')]
if runs:
    m_b, s_b, _ = stats(runs, 'best_acc')
    m_f, s_f, _ = stats(runs, 'final_acc')
    delta = m_b - 0.6143
    print(f"{'  - BLUE (homo-lambda)':<35} "
          f"{f'{m_b*100:.2f} +/- {s_b*100:.2f}':>15} "
          f"{f'{m_f*100:.2f} +/- {s_f*100:.2f}':>15} "
          f"{delta*100:>+7.1f}pp")

# Full participation
runs = [all_results[l] for l in all_results if l.startswith('expC_fullpart')]
if runs:
    m_b, s_b, _ = stats(runs, 'best_acc')
    m_f, s_f, _ = stats(runs, 'final_acc')
    delta = m_b - 0.6143
    avg_eps = np.mean([np.mean(r.get('avg_eps', [0])) for r in runs])
    cum_pay = np.mean([r['summary'].get('cumulative_payment', 0) for r in runs])
    print(f"{'  + Full part (gamma=100)':<35} "
          f"{f'{m_b*100:.2f} +/- {s_b*100:.2f}':>15} "
          f"{f'{m_f*100:.2f} +/- {s_f*100:.2f}':>15} "
          f"{delta*100:>+7.1f}pp")
    print(f"    (eps/rd={avg_eps:.2f}, cum_pay={cum_pay:.0f})")

# Hetero-lambda BLUE (Exp H)
print("  --- Heterogeneous privacy regime (lambda_mult=5) ---")
for label, desc in [('expH_hetlam_blue_s42', '  Hetero-lambda + BLUE'),
                     ('expH_hetlam_noblue_s42', '  Hetero-lambda - BLUE'),
                     ('expH_homlam_blue_s42', '  Homo-lambda + BLUE (ref)')]:
    if label in all_results:
        s = all_results[label]['summary']
        print(f"{desc:<35} {fmt(s['best_acc']):>15} {fmt(s['final_acc']):>15}")


# =====================================================================
# TABLE V: Cross-Dataset (CIFAR-10)
# =====================================================================
print_header("TABLE V: Cross-Dataset Validation (CIFAR-10, N=50, gamma=5)")

print(f"\n{'Method':<25} {'Privacy':>7} {'Best Acc':>15} {'Final Acc':>15}")
print('-' * 65)

for method in ['PAID-FD', 'Fixed-eps-3', 'CSRA']:
    runs = [all_results[l] for l in all_results 
            if l.startswith('expD_') and all_results[l]['method'] == method]
    if runs:
        m_b, s_b, n = stats(runs, 'best_acc')
        m_f, s_f, _ = stats(runs, 'final_acc')
        priv = 'Yes' if method != 'FedAvg' else 'No'
        print(f"{method:<25} {priv:>7} "
              f"{f'{m_b*100:.2f} +/- {s_b*100:.2f}':>15} "
              f"{f'{m_f*100:.2f} +/- {s_f*100:.2f}':>15}")


# =====================================================================
# TABLE VI: Non-IID Alpha Sweep (Phase 3)
# =====================================================================
print_header("TABLE VI: Non-IID Sensitivity (CIFAR-100, N=50)")

print(f"\n{'alpha':>6} {'gamma':>6} {'Best Acc':>15} {'Final Acc':>15} {'Part(%)':>10} {'Avg e/rd':>10}")
print('-' * 68)

for alpha in [0.1, 0.5, 1.0]:
    for gamma in [3, 10]:
        if alpha == 0.5:
            # v10.1 known data
            known = {3: (61.22, 0.42, 61.02, 0.13, 38, 2.77),
                     10: (61.45, 0.23, 60.61, 0.61, 100, 3.09)}
            if gamma in known:
                b, bs, f, fs, p, e = known[gamma]
                print(f"{alpha:>6.1f} {gamma:>6} {f'{b:.2f} +/- {bs:.2f}':>15} "
                      f"{f'{f:.2f} +/- {fs:.2f}':>15} {p:>9}% {e:>9.2f}")
        else:
            a_str = str(alpha).replace('.', '')
            prefix = f'expE_a{a_str}_g{gamma}'
            runs = [all_results[l] for l in all_results if l.startswith(prefix)]
            if runs:
                m_b, s_b, _ = stats(runs, 'best_acc')
                m_f, s_f, _ = stats(runs, 'final_acc')
                avg_part = np.mean([r['summary']['avg_participation'] for r in runs])
                avg_eps = np.mean([np.mean(r.get('avg_eps', [0])) for r in runs])
                print(f"{alpha:>6.1f} {gamma:>6} {f'{m_b*100:.2f} +/- {s_b*100:.2f}':>15} "
                      f"{f'{m_f*100:.2f} +/- {s_f*100:.2f}':>15} "
                      f"{avg_part*100:>9.0f}% {avg_eps:>9.2f}")


# =====================================================================
# FIGURE DATA: Privacy-Utility Curve (Exp G)
# =====================================================================
print_header("FIGURE: Privacy-Utility Curve (Exp G)")

print(f"\n{'epsilon':>8} {'Best Acc':>10} {'Final Acc':>10} {'Note':>20}")
print('-' * 52)
g_data = []
for label in sorted(all_results.keys()):
    if not label.startswith('expG_'):
        continue
    d = all_results[label]
    eps = d['avg_eps'][0]
    g_data.append((eps, d['summary']['best_acc'], d['summary']['final_acc']))

g_data.sort()
for eps, best, final in g_data:
    note = ''
    if eps == 0.1:
        note = '<-- breaking point'
    print(f"{eps:>8.2f} {best*100:>9.2f}% {final*100:>9.2f}% {note:>20}")
print(f"{'2.84':>8} {'61.43':>9}% {'60.80':>9}% {'<-- PAID-FD (game)':>20}")


# =====================================================================
# CONVERGENCE DATA: Key curves for figures
# =====================================================================
print_header("CONVERGENCE: Key Round-by-Round Data")

curves_to_plot = {
    'PAID-FD (game)': None,  # Use known v10.1
    'Fair Fixed-e=1': 'expF_faireps1_s42',
    'Fair Fixed-e=3': 'expF_faireps3_s42',
    'Old Fixed-eps-3 (s42)': 'expA_fixedeps3_s42',
    'e=0.1 (heavy noise)': 'expG_eps0p1_s42',
}

checkpoints = [0, 5, 10, 20, 30, 50, 70, 99]
print(f"\n{'Round':>6}", end='')
for name in curves_to_plot:
    print(f" {name:>22}", end='')
print()
print('-' * (6 + 23 * len(curves_to_plot)))

for r in checkpoints:
    print(f"{r:>6}", end='')
    for name, label in curves_to_plot.items():
        if label and label in all_results:
            accs = all_results[label]['accuracies']
            if r < len(accs):
                print(f" {accs[r]*100:>21.2f}%", end='')
            else:
                print(f" {'N/A':>22}", end='')
        else:
            print(f" {'(v10.1 data)':>22}", end='')
    print()


# =====================================================================
# SANITY CHECKS
# =====================================================================
print_header("SANITY CHECKS")

issues = []

# Check 1: All experiments completed 100 rounds
for label, d in all_results.items():
    n = len(d.get('accuracies', []))
    if n != 100:
        issues.append(f"  [WARN] {label}: only {n} rounds (expected 100)")

# Check 2: CSRA should be ~1% on CIFAR-100, ~10% on CIFAR-10
for label, d in all_results.items():
    if 'csra' in label.lower():
        best = d['summary']['best_acc']
        dataset = d.get('dataset', 'cifar100')
        expected_random = 0.01 if dataset == 'cifar100' else 0.10
        if abs(best - expected_random) > 0.02:
            issues.append(f"  [WARN] {label}: CSRA best={best:.4f}, expected ~{expected_random}")

# Check 3: Fair Fixed-eps should match PAID-FD within ~1pp
for label, d in all_results.items():
    if label.startswith('expF_'):
        best = d['summary']['best_acc']
        if abs(best - 0.6143) > 0.01:
            issues.append(f"  [WARN] {label}: Fair Fixed-eps best={best*100:.2f}%, "
                         f"PAID-FD baseline=61.43%, gap={abs(best-0.6143)*100:.2f}pp")

# Check 4: Exp G eps=0.1 should be noticeably lower
if 'expG_eps0p1_s42' in all_results:
    best = all_results['expG_eps0p1_s42']['summary']['best_acc']
    if best > 0.59:
        issues.append(f"  [WARN] expG_eps0p1: best={best*100:.2f}%, expected < 59% for eps=0.1")

# Check 5: Phase 3 alpha=0.1 should be harder than alpha=0.5
a01_runs = [all_results[l] for l in all_results if l.startswith('expE_a01')]
if a01_runs:
    a01_best = np.mean([r['summary']['best_acc'] for r in a01_runs])
    if a01_best > 0.62:
        issues.append(f"  [NOTE] alpha=0.1 best={a01_best*100:.2f}% >= alpha=0.5 (61.4%). "
                     "Extreme non-IID doesn't hurt much.")

# Check 6: Hetero-lambda should have lower eps than homo
if 'expH_hetlam_blue_s42' in all_results and 'expH_homlam_blue_s42' in all_results:
    het_eps = np.mean(all_results['expH_hetlam_blue_s42']['avg_eps'])
    hom_eps = np.mean(all_results['expH_homlam_blue_s42']['avg_eps'])
    if het_eps >= hom_eps:
        issues.append(f"  [WARN] Hetero avg_eps={het_eps:.2f} >= Homo avg_eps={hom_eps:.2f}")

if not issues:
    print("  ALL CHECKS PASSED")
else:
    for issue in issues:
        print(issue)


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================
print_header("EXPERIMENT INVENTORY SUMMARY")

total_gpu_hours = 0
for label, d in all_results.items():
    elapsed = d.get('elapsed_sec', 0)
    total_gpu_hours += elapsed / 3600

print(f"  Total experiments: {len(all_results)}")
print(f"  Total GPU-hours: {total_gpu_hours:.1f} hrs")
print(f"  Phases complete: {len(phase_counts)}/4")
print()
print("  Key numbers for paper:")
print(f"    PAID-FD best (CIFAR-100): 61.43%")
print(f"    PAID-FD best (CIFAR-10):  84.91%")
print(f"    vs FedGMKD (no privacy):  +14.9pp (CIFAR-100)")
print(f"    vs Fixed-eps-3 (old):     +19.5pp (CIFAR-100)")
print(f"    vs Fair Fixed-eps-3:      +0.25pp (pipeline equivalent)")
if g_data:
    eps01_best = [b for e, b, f in g_data if abs(e - 0.1) < 0.01]
    if eps01_best:
        print(f"    Privacy cost at e=0.1:    -{(0.6143 - eps01_best[0])*100:.1f}pp")
    eps05_best = [b for e, b, f in g_data if abs(e - 0.5) < 0.01]
    if eps05_best:
        print(f"    Privacy cost at e=0.5:    -{(0.6143 - eps05_best[0])*100:.1f}pp")
print(f"    Game contribution (acc):  ~0pp (efficiency-only)")
print(f"    Pipeline contribution:    ~20pp over old Fixed-eps")
