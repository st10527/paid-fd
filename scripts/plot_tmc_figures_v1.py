#!/usr/bin/env python3
"""
TMC Paper — Publication-Quality Figure Generator
=================================================
Generates all 10 data figures (Fig 3–10) for the TMC paper.
Fig 1–2 are conceptual diagrams (drawn manually).

Output: results/figures/tmc_fig{3..10}_{name}.pdf

Style: IEEE TMC standard — single column width (3.5 in),
       8pt fonts, no legend overlap, clean gridlines.
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

# ============================================================
# Global Style — IEEE TMC Publication Quality
# ============================================================
plt.rcParams.update({
    'figure.figsize': (3.5, 2.6),        # single-column width
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,             # Set True if LaTeX available
})

# Color palette — colorblind-friendly
COLORS = {
    'paid_fd':     '#2171B5',  # blue
    'fair_fixed':  '#6BAED6',  # light blue
    'fixed_eps':   '#D94801',  # orange
    'csra':        '#A63603',  # dark orange
    'fedavg':      '#74C476',  # green
    'fedmd':       '#31A354',  # dark green
    'fedgmkd':     '#006D2C',  # darker green
    'no_ema':      '#9E9AC8',  # purple
    'no_mixed':    '#756BB1',  # dark purple
    'no_persist':  '#54278F',  # darker purple
    'accent':      '#E31A1C',  # red (for highlights)
}

MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

OUTDIR = Path('results/figures')
OUTDIR.mkdir(parents=True, exist_ok=True)

TMC_DIR = Path('results/experiments/tmc')
V101_3SEEDS = Path('results/experiments/v10_1_3seeds_20260409_0922.json')
V101_COMBINED = Path('results/experiments/v10_1_combined_20260409_2304.json')


# ============================================================
# Data Loading Helpers
# ============================================================
def load_tmc(label):
    """Load a TMC experiment result by label."""
    p = TMC_DIR / f'{label}.json'
    if p.exists():
        return json.load(open(p))
    return None

def load_tmc_group(prefix):
    """Load all TMC results matching a prefix, return dict of label->data."""
    results = {}
    for f in sorted(TMC_DIR.glob(f'{prefix}*.json')):
        d = json.load(open(f))
        results[d['label']] = d
    return results

def load_v101_combined():
    """Load v10.1 combined results (summaries only)."""
    return json.load(open(V101_COMBINED))

def load_v101_3seeds():
    """Load v10.1 3-seeds data (with trajectories)."""
    return json.load(open(V101_3SEEDS))

def mean_std(values):
    return np.mean(values), np.std(values)

def group_by_seeds(data_dict, key_fn, val_fn):
    """Group experiment results by a key function, aggregate by seeds."""
    groups = defaultdict(list)
    for label, d in data_dict.items():
        k = key_fn(d)
        groups[k].append(val_fn(d))
    result = {}
    for k, vals in sorted(groups.items()):
        vals = np.array(vals)
        result[k] = (np.mean(vals), np.std(vals), len(vals))
    return result


# ============================================================
# Fig 3: Proposition 2 Verification (ε* vs γ)
# ============================================================
def plot_fig3():
    """ε* monotonicity with γ — verifies Proposition 2."""
    combined = load_v101_combined()
    sums = combined['summaries']

    # Group by gamma, collect avg_eps across seeds (lm1.0 = default lambda)
    gamma_eps = defaultdict(list)
    for k, v in sums.items():
        if k.startswith('g') and '_s' in k:
            gamma = int(k.split('_')[0][1:])
            eps = v.get('avg_eps', v.get('avg_eps_per_round', 0))
            gamma_eps[gamma].append(eps)

    gammas = sorted(gamma_eps.keys())
    means = [np.mean(gamma_eps[g]) for g in gammas]
    stds = [np.std(gamma_eps[g]) for g in gammas]

    fig, ax = plt.subplots()
    ax.errorbar(gammas, means, yerr=stds, fmt='o-',
                color=COLORS['paid_fd'], capsize=3, capthick=1,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    ax.set_xlabel(r'Server valuation $\gamma$')
    ax.set_ylabel(r'Equilibrium $\varepsilon^*$ (avg per round)')
    ax.set_xticks(gammas)
    ax.set_ylim(2.5, 3.4)

    # Annotate monotonicity
    ax.annotate('Proposition 2:\nmonotone increasing',
                xy=(7, means[2]), xytext=(4.5, 3.25),
                fontsize=6.5, fontstyle='italic', color='0.4',
                arrowprops=dict(arrowstyle='->', color='0.5', lw=0.8))

    fig.savefig(OUTDIR / 'tmc_fig3_prop2_monotonicity.pdf')
    fig.savefig(OUTDIR / 'tmc_fig3_prop2_monotonicity.png', dpi=300)
    plt.close(fig)
    print('[Fig 3] Prop 2 monotonicity — done')


# ============================================================
# Fig 4: Efficiency Frontier (Cumulative Payment vs Accuracy)
# ============================================================
def plot_fig4():
    """Efficiency frontier: cumulative payment vs best accuracy."""
    combined = load_v101_combined()
    sums = combined['summaries']

    gamma_data = defaultdict(lambda: {'acc': [], 'pay': [], 'part': []})
    for k, v in sums.items():
        if k.startswith('g') and '_s' in k:
            gamma = int(k.split('_')[0][1:])
            gamma_data[gamma]['acc'].append(v['best_acc'] * 100)
            gamma_data[gamma]['pay'].append(v.get('cumulative_payment', 0))
            gamma_data[gamma]['part'].append(v.get('avg_participation', 0) * 100)

    gammas = sorted(gamma_data.keys())
    mean_acc = [np.mean(gamma_data[g]['acc']) for g in gammas]
    std_acc = [np.std(gamma_data[g]['acc']) for g in gammas]
    mean_pay = [np.mean(gamma_data[g]['pay']) for g in gammas]

    fig, ax = plt.subplots()
    ax.errorbar(np.array(mean_pay) / 1000, mean_acc, yerr=std_acc,
                fmt='o-', color=COLORS['paid_fd'], capsize=3, capthick=1,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    # Label each point with γ value
    for g, pay, acc in zip(gammas, mean_pay, mean_acc):
        part = np.mean(gamma_data[g]['part'])
        ax.annotate(f'$\\gamma$={g}\n({part:.0f}%)',
                    xy=(pay / 1000, acc), fontsize=6,
                    textcoords='offset points', xytext=(8, -3),
                    color='0.4')

    ax.set_xlabel('Cumulative Payment (×$10^3$)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_ylim(60.5, 62.5)

    # Highlight diminishing returns
    ax.axhline(y=mean_acc[-1], color='0.7', linestyle='--', linewidth=0.6, zorder=1)
    ax.text(10, mean_acc[-1] + 0.15, 'accuracy plateau',
            fontsize=6, color='0.5', fontstyle='italic')

    fig.savefig(OUTDIR / 'tmc_fig4_efficiency_frontier.pdf')
    fig.savefig(OUTDIR / 'tmc_fig4_efficiency_frontier.png', dpi=300)
    plt.close(fig)
    print('[Fig 4] Efficiency frontier — done')


# ============================================================
# Fig 5: λ Sensitivity
# ============================================================
def plot_fig5():
    """λ sensitivity: accuracy vs γ for different λ_mult values."""
    combined = load_v101_combined()
    sums = combined['summaries']

    # Parse lm{x}_g{y} entries
    lm_data = defaultdict(lambda: defaultdict(list))
    for k, v in sums.items():
        if k.startswith('lm'):
            parts = k.split('_')
            lm = float(parts[0][2:])
            gamma = int(parts[1][1:])
            lm_data[lm][gamma].append(v['best_acc'] * 100)

    fig, ax = plt.subplots()
    lm_labels = {0.5: r'$\lambda_{\mathrm{mult}}=0.5$ (low privacy)',
                 1.0: r'$\lambda_{\mathrm{mult}}=1.0$ (default)',
                 2.0: r'$\lambda_{\mathrm{mult}}=2.0$ (high privacy)'}
    lm_colors = {0.5: '#6BAED6', 1.0: '#2171B5', 2.0: '#08519C'}

    for i, lm in enumerate(sorted(lm_data.keys())):
        gammas = sorted(lm_data[lm].keys())
        means = [np.mean(lm_data[lm][g]) for g in gammas]
        ax.plot(gammas, means, marker=MARKERS[i], label=lm_labels.get(lm, f'λ×{lm}'),
                color=lm_colors.get(lm, COLORS['paid_fd']), zorder=5)

    ax.set_xlabel(r'Server valuation $\gamma$')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks([3, 5, 7, 10])
    ax.set_ylim(60.0, 62.5)
    ax.legend(loc='lower right')

    fig.savefig(OUTDIR / 'tmc_fig5_lambda_sensitivity.pdf')
    fig.savefig(OUTDIR / 'tmc_fig5_lambda_sensitivity.png', dpi=300)
    plt.close(fig)
    print('[Fig 5] λ sensitivity — done')


# ============================================================
# Fig 6: Convergence Curves (CIFAR-100)
# ============================================================
def plot_fig6():
    """Convergence curves: PAID-FD vs baselines on CIFAR-100."""
    # PAID-FD baseline (v10.1 γ=5 seed=42)
    v101 = load_v101_3seeds()

    # Baselines from TMC
    methods = {
        'PAID-FD': ('paid_fd', v101['runs'].get('g5_s42', v101['runs'].get('g3_s42', {}))),
        'Fixed-ε=3 (old)': ('fixed_eps', load_tmc('expA_fixedeps3_s42')),
        'Fair Fixed-ε=1': ('fair_fixed', load_tmc('expF_faireps1_s42')),
        'FedGMKD': ('fedgmkd', load_tmc('expAp_fedgmkd_s42')),
        'FedAvg': ('fedavg', load_tmc('expAp_fedavg_s42')),
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    rounds = np.arange(100)

    for name, (ckey, data) in methods.items():
        if data is None:
            continue
        accs = np.array(data['accuracies'][:100]) * 100
        ls = '-' if 'PAID' in name or 'Fair' in name else '--'
        lw = 1.5 if 'PAID' in name else 1.0
        ax.plot(rounds, accs, label=name, color=COLORS[ckey],
                linestyle=ls, linewidth=lw, zorder=5 if 'PAID' in name else 3)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 68)

    # Legend outside overlap zone — upper left
    ax.legend(loc='lower right', ncol=1, frameon=True)

    fig.savefig(OUTDIR / 'tmc_fig6_convergence_cifar100.pdf')
    fig.savefig(OUTDIR / 'tmc_fig6_convergence_cifar100.png', dpi=300)
    plt.close(fig)
    print('[Fig 6] Convergence CIFAR-100 — done')


# ============================================================
# Fig 7: Privacy-Utility Curve (武器圖)
# ============================================================
def plot_fig7():
    """Privacy-utility curve: accuracy vs ε (Exp G + PAID-FD star)."""
    eps_data = []
    for f in sorted(TMC_DIR.glob('expG_*.json')):
        d = json.load(open(f))
        label = d['label']
        # Parse epsilon from label: expG_eps{X}_s42
        eps_str = label.split('_')[1].replace('eps', '').replace('p', '.')
        eps = float(eps_str)
        best = d['summary']['best_acc'] * 100
        final = d['summary']['final_acc'] * 100
        eps_data.append((eps, best, final))

    eps_data.sort()
    epsilons = [e[0] for e in eps_data]
    bests = [e[1] for e in eps_data]

    fig, ax = plt.subplots()

    # Main curve
    ax.plot(epsilons, bests, 'o-', color=COLORS['fair_fixed'],
            markeredgecolor='white', markeredgewidth=0.5,
            label='Fixed $\\varepsilon$ (identical pipeline)', zorder=5)

    # PAID-FD star
    ax.plot(2.84, 61.4, '*', markersize=10, color=COLORS['accent'],
            markeredgecolor='white', markeredgewidth=0.5,
            label='PAID-FD ($\\varepsilon^*\\approx 2.84$)', zorder=10)

    # Breaking point annotation
    ax.annotate('Breaking point\n$\\varepsilon=0.1$: 56.2%',
                xy=(0.1, 56.2), xytext=(0.8, 54.5),
                fontsize=6.5, color=COLORS['accent'],
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=0.8))

    # Flat region annotation
    ax.axhspan(60.5, 62.0, alpha=0.08, color=COLORS['paid_fd'], zorder=1)
    ax.text(4, 62.3, 'noise-resilient plateau', fontsize=6,
            color='0.5', fontstyle='italic', ha='center')

    ax.set_xlabel(r'Privacy budget $\varepsilon$ per round')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xscale('log')
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(0.07, 15)
    ax.set_ylim(53, 64)

    ax.legend(loc='lower right')

    fig.savefig(OUTDIR / 'tmc_fig7_privacy_utility.pdf')
    fig.savefig(OUTDIR / 'tmc_fig7_privacy_utility.png', dpi=300)
    plt.close(fig)
    print('[Fig 7] Privacy-utility curve — done')


# ============================================================
# Fig 8: N Scalability
# ============================================================
def plot_fig8():
    """Scalability: accuracy vs N for γ=3 and γ=10."""
    # Phase 1 Exp B: N=20,80 × γ=3,10 × 3 seeds
    # Plus N=50 from v10.1 combined
    expB = load_tmc_group('expB_')
    combined = load_v101_combined()['summaries']

    # Build: (N, γ) -> [best_accs]
    data = defaultdict(list)
    for label, d in expB.items():
        s = d['summary']
        cfg = d['config']
        n = cfg['n_devices']
        gamma = cfg['gamma']
        data[(n, gamma)].append(s['best_acc'] * 100)

    # Add N=50 from combined
    for k, v in combined.items():
        if k.startswith('g') and '_s' in k:
            gamma = int(k.split('_')[0][1:])
            if gamma in [3, 10]:
                data[(50, gamma)].append(v['best_acc'] * 100)

    fig, ax = plt.subplots()

    for gi, gamma in enumerate([3, 10]):
        ns = sorted(set(n for (n, g) in data if g == gamma))
        means = [np.mean(data[(n, gamma)]) for n in ns]
        stds = [np.std(data[(n, gamma)]) for n in ns]
        color = COLORS['paid_fd'] if gamma == 3 else COLORS['fair_fixed']
        ax.errorbar(ns, means, yerr=stds, fmt=f'{MARKERS[gi]}-',
                    color=color, capsize=3, capthick=1,
                    label=f'$\\gamma$={gamma}',
                    markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    ax.set_xlabel('Number of Devices ($N$)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks([20, 50, 80])
    ax.set_ylim(60.0, 63.0)
    ax.legend(loc='lower right')

    fig.savefig(OUTDIR / 'tmc_fig8_scalability.pdf')
    fig.savefig(OUTDIR / 'tmc_fig8_scalability.png', dpi=300)
    plt.close(fig)
    print('[Fig 8] Scalability — done')


# ============================================================
# Fig 9: Non-IID Robustness (α sweep)
# ============================================================
def plot_fig9():
    """Non-IID robustness: grouped bar chart of accuracy vs α."""
    expE = load_tmc_group('expE_')

    # Also need α=0.5 baseline from Phase 1 (v10.1 combined or Exp B N=50)
    combined = load_v101_combined()['summaries']

    # Build: (α, γ) -> [best_accs]
    data = defaultdict(list)
    for label, d in expE.items():
        s = d['summary']
        cfg = d['config']
        alpha = cfg.get('alpha', 0.5)
        gamma = cfg['gamma']
        data[(alpha, gamma)].append(s['best_acc'] * 100)

    # Add α=0.5 from combined
    for k, v in combined.items():
        if k.startswith('g') and '_s' in k:
            gamma = int(k.split('_')[0][1:])
            if gamma in [3, 10]:
                data[(0.5, gamma)].append(v['best_acc'] * 100)

    fig, ax = plt.subplots()
    alphas = [0.1, 0.5, 1.0]
    x = np.arange(len(alphas))
    width = 0.3

    for gi, gamma in enumerate([3, 10]):
        means = [np.mean(data[(a, gamma)]) for a in alphas]
        stds = [np.std(data[(a, gamma)]) for a in alphas]
        color = COLORS['paid_fd'] if gamma == 3 else COLORS['fair_fixed']
        offset = (gi - 0.5) * width
        bars = ax.bar(x + offset, means, width * 0.9, yerr=stds,
                      label=f'$\\gamma$={gamma}', color=color,
                      edgecolor='white', linewidth=0.5,
                      capsize=2, error_kw={'linewidth': 0.8}, zorder=5)

    ax.set_xlabel(r'Dirichlet concentration $\alpha$ (non-IID severity)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([r'$\alpha$=0.1' + '\n(extreme)', r'$\alpha$=0.5' + '\n(moderate)', r'$\alpha$=1.0' + '\n(mild)'])
    ax.set_ylim(59.5, 63.0)
    ax.legend(loc='upper right')

    # Annotate Δ
    ax.annotate(f'$\\Delta < 0.5$ pp',
                xy=(1, 62.5), fontsize=7, ha='center',
                color='0.4', fontstyle='italic')

    fig.savefig(OUTDIR / 'tmc_fig9_noniid_alpha.pdf')
    fig.savefig(OUTDIR / 'tmc_fig9_noniid_alpha.png', dpi=300)
    plt.close(fig)
    print('[Fig 9] Non-IID robustness — done')


# ============================================================
# Fig 10: Privacy Composition Over Rounds
# ============================================================
def plot_fig10():
    """Cumulative privacy budget over rounds."""
    # Use v10.1 γ=5 seed=42 trajectory
    v101 = load_v101_3seeds()
    key = 'g5_s42' if 'g5_s42' in v101['runs'] else list(v101['runs'].keys())[0]
    run = v101['runs'][key]
    avg_eps = np.array(run['avg_eps'][:100])
    part_rates = np.array(run['participation_rates'][:100])
    rounds = np.arange(1, 101)

    # Basic composition: cumulative sum of per-round ε
    basic_cum = np.cumsum(avg_eps)

    # Advanced composition (Theorem 7):
    # ε_adv(R) = ε* × sqrt(2R × ln(1/δ)) + R × ε* × (e^ε* - 1)
    # For partial participation, effective rounds = R × participation_rate
    delta = 1e-5
    eps_star = np.mean(avg_eps)
    effective_rounds = np.cumsum(part_rates)  # fractional rounds participated

    adv_cum = eps_star * np.sqrt(2 * effective_rounds * np.log(1 / delta)) + \
              effective_rounds * eps_star * (np.exp(eps_star) - 1)

    # Selective participation: only count rounds participated
    mean_part = np.mean(part_rates)
    basic_selective = basic_cum * mean_part
    selective_rounds = rounds * mean_part
    adv_selective = eps_star * np.sqrt(2 * selective_rounds * np.log(1 / delta)) + \
                    selective_rounds * eps_star * (np.exp(eps_star) - 1)

    fig, ax = plt.subplots()
    ax.plot(rounds, basic_cum, '-', color=COLORS['fixed_eps'],
            label=f'Basic composition (full)', linewidth=1.0, zorder=3)
    ax.plot(rounds, basic_selective, '--', color=COLORS['fixed_eps'],
            label=f'Basic (selective, {mean_part*100:.0f}% part.)', linewidth=1.0, zorder=3)
    ax.plot(rounds, adv_cum, '-', color=COLORS['paid_fd'],
            label=r'Advanced (Thm 7, full)', linewidth=1.2, zorder=5)
    ax.plot(rounds, adv_selective, '--', color=COLORS['paid_fd'],
            label=f'Advanced (selective)', linewidth=1.2, zorder=5)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel(r'Cumulative $\varepsilon_{\mathrm{total}}$')
    ax.set_xlim(0, 100)
    ax.legend(loc='upper left', fontsize=6.5)

    fig.savefig(OUTDIR / 'tmc_fig10_privacy_composition.pdf')
    fig.savefig(OUTDIR / 'tmc_fig10_privacy_composition.png', dpi=300)
    plt.close(fig)
    print('[Fig 10] Privacy composition — done')


# ============================================================
# Bonus: Table data verification (print summary)
# ============================================================
def print_phase5_summary():
    """Print Phase 5 results for Table III completion."""
    print('\n' + '=' * 60)
    print('Phase 5 Pipeline Ablation (for Table III)')
    print('=' * 60)

    baseline_best = 61.4  # PAID-FD full
    for label in ['expI_noEMA_s42', 'expI_noMixedLoss_s42', 'expI_noPersistent_s42']:
        d = load_tmc(label)
        if d:
            s = d['summary']
            delta = s['best_acc'] * 100 - baseline_best
            print(f'  {label:30s}  best={s["best_acc"]*100:.2f}%  '
                  f'final={s["final_acc"]*100:.2f}%  Δ={delta:+.1f}pp')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('TMC Paper Figure Generator')
    print('=' * 60)

    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_fig8()
    plot_fig9()
    plot_fig10()

    print_phase5_summary()

    print('\n' + '=' * 60)
    print(f'All figures saved to: {OUTDIR}/')
    print('=' * 60)
