#!/usr/bin/env python3
"""
TMC Paper — Publication-Quality Figure Generator v2
====================================================
All data figures for the TMC paper (Fig 3–10 + new Fig 6a bar chart).
Fig 1–2 are conceptual diagrams (drawn manually).

Quality checklist per figure:
  ✓ Legend never overlaps data lines or bars
  ✓ Annotation arrows never cross data
  ✓ Font sizes ≥ 7pt everywhere
  ✓ Colorblind-friendly palette (blue/orange/green, no red-green)
  ✓ Single-column width 3.5in, tight bbox
  ✓ Vector PDF + 300dpi PNG
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

# ============================================================
# Global Style
# ============================================================
plt.rcParams.update({
    'figure.figsize': (3.5, 2.6),
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.85',
    'legend.handlelength': 1.5,
    'lines.linewidth': 1.3,
    'lines.markersize': 4.5,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'text.usetex': False,
})

# ============================================================
# Canonical method color map — USE THESE EVERYWHERE.
# Same method must always appear in the same color across figures.
# ============================================================
# --- Primary methods (with LDP) ---
MC_PAID_FD   = '#1565C0'   # PAID-FD (ours)            strong blue
MC_PAID_G3   = '#1565C0'   # PAID-FD γ=3               (same as main)
MC_PAID_G10  = '#5B9BD5'   # PAID-FD γ=10              lighter blue (still blue family)
MC_FAIR_E1   = '#00838F'   # Fair Fixed-ε=1             teal
MC_FAIR_E3   = '#26A69A'   # Fair Fixed-ε=3             lighter teal
MC_FAIR_E5   = '#80CBC4'   # Fair Fixed-ε=5             very light teal
MC_OLD_FIX   = '#BF360C'   # Old Fixed-ε=3              deep burnt orange
MC_CSRA      = '#E6AB00'   # CSRA                       amber/gold (distinct from orange)
# --- No-privacy baselines ---
MC_FEDGMKD   = '#6A1B9A'   # FedGMKD                   purple
MC_FEDAVG    = '#C62828'   # FedAvg                    dark red
MC_FEDMD     = '#4E342E'   # FedMD                     brown
# --- λ variants (all PAID-FD, shown in blue shades) ---
MC_LM05      = '#5B9BD5'   # λ_mult=0.5  light blue
MC_LM10      = '#1565C0'   # λ_mult=1.0  default blue
MC_LM20      = '#0D3F7A'   # λ_mult=2.0  dark blue
# --- Privacy composition (theory — warm=basic, cool=advanced) ---
MC_BASIC_F   = '#BF360C'   # Basic, full participation  burnt orange
MC_BASIC_S   = '#FF7043'   # Basic, selective           lighter orange
MC_ADV_F     = '#1565C0'   # Advanced, full             strong blue
MC_ADV_S     = '#5B9BD5'   # Advanced, selective        lighter blue
# --- Misc ---
C_RED    = '#CB181D'        # highlight star, breaking point
C_GRAY   = '#737373'
C_DGRAY  = '#525252'
# Legacy aliases kept so existing C_ references don't break
C_BLUE   = MC_PAID_FD
C_LBLUE  = MC_PAID_G10
C_DBLUE  = MC_LM20
C_ORANGE = MC_OLD_FIX
C_LORANGE= MC_CSRA
C_GREEN  = MC_FEDGMKD
C_LGREEN = MC_FEDAVG
C_DGREEN = MC_FEDMD

OUTDIR = Path('results/figures')
OUTDIR.mkdir(parents=True, exist_ok=True)
TMC = Path('results/experiments/tmc')
V101_3S = Path('results/experiments/v10_1_3seeds_20260409_0922.json')
V101_COMB = Path('results/experiments/v10_1_combined_20260409_2304.json')


def load_tmc(label):
    p = TMC / f'{label}.json'
    return json.load(open(p)) if p.exists() else None

def load_tmc_group(prefix):
    out = {}
    for f in sorted(TMC.glob(f'{prefix}*.json')):
        d = json.load(open(f))
        out[d['label']] = d
    return out

def load_v101_combined():
    return json.load(open(V101_COMB))

def load_v101_3seeds():
    return json.load(open(V101_3S))

def savefig(fig, name):
    fig.savefig(OUTDIR / f'{name}.pdf')
    fig.savefig(OUTDIR / f'{name}.png', dpi=300)
    plt.close(fig)


# ============================================================
# Fig 3: Proposition 2 Verification (ε* vs γ)
# Self-validation — no baselines needed
# ============================================================
def plot_fig3():
    comb = load_v101_combined()['summaries']
    gamma_eps = defaultdict(list)
    for k, v in comb.items():
        if k.startswith('g') and '_s' in k:
            g = int(k.split('_')[0][1:])
            gamma_eps[g].append(v.get('avg_eps', v.get('avg_eps_per_round', 0)))

    gammas = sorted(gamma_eps.keys())
    means = [np.mean(gamma_eps[g]) for g in gammas]
    stds = [np.std(gamma_eps[g]) for g in gammas]

    fig, ax = plt.subplots()
    ax.errorbar(gammas, means, yerr=stds, fmt='o-', color=C_BLUE,
                capsize=4, capthick=1, markeredgecolor='white',
                markeredgewidth=0.6, markersize=6, zorder=5)

    ax.set_xlabel(r'Server valuation $\gamma$')
    ax.set_ylabel(r'Equilibrium $\varepsilon^*$ (avg per round)')
    ax.set_xticks(gammas)
    ax.set_ylim(2.5, 3.4)

    # Annotation in empty upper-left (data is lower-right trending)
    ax.annotate('Proposition 2:\nmonotone increasing',
                xy=(5, means[1]), xytext=(3.3, 3.25),
                fontsize=6.5, fontstyle='italic', color=C_GRAY,
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=0.7))

    savefig(fig, 'tmc_fig3_prop2_monotonicity')
    print('[Fig 3] Prop 2 — done')


# ============================================================
# Fig 4: Efficiency Frontier (Payment vs Accuracy)
# Self-property — no baselines needed
# ============================================================
def plot_fig4():
    comb = load_v101_combined()['summaries']
    gd = defaultdict(lambda: {'acc': [], 'pay': [], 'part': []})
    for k, v in comb.items():
        if k.startswith('g') and '_s' in k:
            g = int(k.split('_')[0][1:])
            gd[g]['acc'].append(v['best_acc'] * 100)
            gd[g]['pay'].append(v.get('cumulative_payment', 0))
            gd[g]['part'].append(v.get('avg_participation', 0) * 100)

    gammas = sorted(gd.keys())
    ma = [np.mean(gd[g]['acc']) for g in gammas]
    sa = [np.std(gd[g]['acc']) for g in gammas]
    mp = [np.mean(gd[g]['pay']) / 1000 for g in gammas]

    fig, ax = plt.subplots()
    ax.errorbar(mp, ma, yerr=sa, fmt='s-', color=C_BLUE, capsize=4,
                capthick=1, markeredgecolor='white', markeredgewidth=0.6,
                markersize=6, zorder=5)

    # Labels: alternate above/below to avoid overlap
    offsets = [(10, 8), (10, -12), (10, 8), (10, -12)]
    for i, (g, pay, acc) in enumerate(zip(gammas, mp, ma)):
        part = np.mean(gd[g]['part'])
        ax.annotate(f'$\\gamma$={g} ({part:.0f}%)',
                    xy=(pay, acc), fontsize=6, color=C_DGRAY,
                    textcoords='offset points', xytext=offsets[i])

    ax.set_xlabel(r'Cumulative Payment ($\times 10^3$)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_ylim(60.2, 62.5)

    # Dashed plateau line
    ax.axhline(y=ma[-1], color=C_GRAY, ls=':', lw=0.6, zorder=1)

    savefig(fig, 'tmc_fig4_efficiency_frontier')
    print('[Fig 4] Efficiency frontier — done')


# ============================================================
# Fig 5: λ Sensitivity
# Self-property — no baselines needed
# ============================================================
def plot_fig5():
    comb = load_v101_combined()['summaries']
    lm_data = defaultdict(lambda: defaultdict(list))
    for k, v in comb.items():
        if k.startswith('lm'):
            parts = k.split('_')
            lm = float(parts[0][2:])
            g = int(parts[1][1:])
            lm_data[lm][g].append(v['best_acc'] * 100)

    # Fill missing default point(s) from baseline runs when lm=1.0 sweep is incomplete.
    # In current data, lm=1.0,gamma=3 is absent in lambda sweep, but exists in g3 seeds.
    for k, v in comb.items():
        if k.startswith('g') and '_s' in k:
            g = int(k.split('_')[0][1:])
            if g not in lm_data[1.0]:
                lm_data[1.0][g].append(v['best_acc'] * 100)

    fig, ax = plt.subplots()
    styles = {
        0.5: (MC_LM05, 'o', r'$\lambda_\mathrm{mult}$=0.5 (low priv.)'),
        1.0: (MC_LM10, 'D', r'$\lambda_\mathrm{mult}$=1.0 (default)'),
        2.0: (MC_LM20, '^', r'$\lambda_\mathrm{mult}$=2.0 (high priv.)'),
    }
    for lm in sorted(lm_data.keys()):
        c, mk, lab = styles.get(lm, (C_GRAY, 'x', f'λ×{lm}'))
        gs = sorted(lm_data[lm].keys())
        means = [np.mean(lm_data[lm][g]) for g in gs]
        stds = [np.std(lm_data[lm][g]) for g in gs]
        ax.plot(gs, means, marker=mk, color=c, label=lab,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)
        ax.errorbar(gs, means, yerr=stds, fmt='none', ecolor=c,
                    elinewidth=0.8, capsize=2, zorder=4)

        # Explicitly mark the default gamma=3 point so it's visually obvious.
        if lm == 1.0 and 3 in gs:
            i3 = gs.index(3)
            ax.scatter([3], [means[i3]], marker=mk, s=36, color=C_BLUE,
                       edgecolor='white', linewidth=0.6, zorder=7)

    ax.set_xlabel(r'Server valuation $\gamma$')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks([3, 5, 7, 10])
    ax.set_ylim(60.0, 62.1)
    # Legend in lower-right where data is sparse
    ax.legend(loc='lower right', borderpad=0.4)

    savefig(fig, 'tmc_fig5_lambda_sensitivity')
    print('[Fig 5] λ sensitivity — done')


# ============================================================
# Fig 6a (NEW): Method Comparison Bar Chart
# VI-E — THIS is the main cross-method figure
# ============================================================
def plot_fig6a():
    """Grouped bar chart: all methods side by side."""
    # Gather best/final acc for all methods
    methods = []

    # PAID-FD (Phase 1 baseline from v10.1, γ=5, 3 seeds)
    comb = load_v101_combined()['summaries']
    paid_accs = [v['best_acc'] * 100 for k, v in comb.items()
                 if k.startswith('g5_s')]
    methods.append(('PAID-FD\n(ours)', np.mean(paid_accs), np.std(paid_accs), MC_PAID_FD))

    # Fair Fixed-ε (Phase 4, single seed each)
    for eps_val, label, mc in [(1, 'Fair\nε=1', MC_FAIR_E1),
                                (3, 'Fair\nε=3', MC_FAIR_E3),
                                (5, 'Fair\nε=5', MC_FAIR_E5)]:
        d = load_tmc(f'expF_faireps{eps_val}_s42')
        if d:
            methods.append((label, d['summary']['best_acc'] * 100, 0, mc))

    # Old Fixed-eps (Phase 1, 3 seeds)
    old_accs = []
    for s in [42, 123, 456]:
        d = load_tmc(f'expA_fixedeps3_s{s}')
        if d: old_accs.append(d['summary']['best_acc'] * 100)
    if old_accs:
        methods.append(('Old\nFixed-ε=3', np.mean(old_accs), np.std(old_accs), MC_OLD_FIX))

    # CSRA
    csra_accs = []
    for s in [42, 123, 456]:
        d = load_tmc(f'expA_csra_s{s}')
        if d: csra_accs.append(d['summary']['best_acc'] * 100)
    if csra_accs:
        methods.append(('CSRA', np.mean(csra_accs), np.std(csra_accs), MC_CSRA))

    # No-privacy baselines
    for tag, color, name in [('fedgmkd', MC_FEDGMKD, 'FedGMKD'),
                              ('fedavg', MC_FEDAVG, 'FedAvg'),
                              ('fedmd', MC_FEDMD, 'FedMD')]:
        d = load_tmc(f'expAp_{tag}_s42')
        if d:
            methods.append((name, d['summary']['best_acc'] * 100, 0, color))

    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(methods))
    names = [m[0] for m in methods]
    vals = [m[1] for m in methods]
    errs = [m[2] for m in methods]
    colors = [m[3] for m in methods]

    bars = ax.bar(x, vals, 0.65, yerr=errs, color=colors,
                  edgecolor='white', linewidth=0.5,
                  capsize=2, error_kw={'linewidth': 0.7}, zorder=5)

    # Value labels on top
    for i, (v, e) in enumerate(zip(vals, errs)):
        y_top = v + e + 0.8
        if v < 10:
            ax.text(i, v + e + 0.5, f'{v:.1f}', ha='center', fontsize=5.5, color=C_DGRAY)
        else:
            ax.text(i, y_top, f'{v:.1f}', ha='center', fontsize=5.5, color=C_DGRAY)

    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=5.6, rotation=18, ha='right')
    ax.set_ylim(0, 72)

    # Divider line between LDP methods and no-privacy
    n_ldp = len([m for m in methods if m[0] not in ['FedGMKD', 'FedAvg', 'FedMD']])
    ax.axvline(x=n_ldp - 0.5, color=C_GRAY, ls='--', lw=0.6)
    ax.text((n_ldp - 1) / 2, 68.5, 'with LDP', fontsize=5.5, ha='center', color=C_GRAY)
    ax.text(n_ldp + (len(methods) - n_ldp - 1) / 2, 68.5, 'no privacy',
            fontsize=5.5, ha='center', color=C_GRAY)

    savefig(fig, 'tmc_fig6a_method_comparison')
    print('[Fig 6a] Method comparison bar — done')


# ============================================================
# Fig 6b: Convergence Curves (CIFAR-100)
# VI-E — show all methods including CSRA
# ============================================================
def plot_fig6b():
    v101 = load_v101_3seeds()
    paid_key = next(k for k in v101['runs'] if k.startswith('g3') or k.startswith('g5'))

    # Each curve: (label, color, linestyle, linewidth, data_dict, marker, markevery_offset)
    # Stagger markevery offset so markers don't stack at same x positions
    curve_specs = [
        ('PAID-FD',        MC_PAID_FD,  '-',  1.7, v101['runs'][paid_key],         'o', 0),
        ('Fair Fixed-ε=1', MC_FAIR_E1,  '-',  1.0, load_tmc('expF_faireps1_s42'),  's', 3),
        ('Old Fixed-ε=3',  MC_OLD_FIX,  '--', 1.0, load_tmc('expA_fixedeps3_s42'), '^', 6),
        ('CSRA',           MC_CSRA,     ':',  0.9, load_tmc('expA_csra_s42'),       'x', 1),
        ('FedGMKD',        MC_FEDGMKD,  '--', 0.9, load_tmc('expAp_fedgmkd_s42'),  'D', 4),
        ('FedAvg',         MC_FEDAVG,   '--', 0.9, load_tmc('expAp_fedavg_s42'),   'v', 7),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    rounds = np.arange(100)

    for name, color, ls, lw, data, mk, off in curve_specs:
        if data is None:
            continue
        accs = np.array(data['accuracies'][:100]) * 100
        zord = 6 if 'PAID' in name else 3
        ax.plot(rounds, accs, color=color, ls=ls, lw=lw, label=name,
                marker=mk, markevery=(off, 15), markersize=3.8,
                markeredgecolor='white', markeredgewidth=0.45, zorder=zord)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 68)

    # Legend goes above the plot to avoid ANY data overlap.
    # Extra top margin is provided via subplots_adjust.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=3, borderpad=0.3, labelspacing=0.2,
              handletextpad=0.4, fontsize=6.2, columnspacing=0.6)
    fig.subplots_adjust(top=0.83)

    # Annotate the Old Fixed-ε decline directly on line (end of curve)
    ax.annotate('degrades', xy=(90, 37.5), xytext=(70, 30),
                fontsize=5.5, color=MC_OLD_FIX,
                arrowprops=dict(arrowstyle='->', color=MC_OLD_FIX, lw=0.7))

    for ext in ['pdf', 'png']:
        fig.savefig(OUTDIR / ('tmc_fig6b_convergence_cifar100.' + ext),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
        fig.savefig(OUTDIR / ('tmc_fig6_convergence_cifar100.' + ext),
                    dpi=300 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print('[Fig 6b] Convergence CIFAR-100 — done')


# ============================================================
# Fig 7: Privacy-Utility Curve (武器圖)
# VI-F — self-property (ε sweep on identical pipeline)
# ============================================================
def plot_fig7():
    eps_data = []
    for f in sorted(TMC.glob('expG_*.json')):
        d = json.load(open(f))
        eps_str = d['label'].split('_')[1].replace('eps', '').replace('p', '.')
        eps = float(eps_str)
        eps_data.append((eps, d['summary']['best_acc'] * 100))
    eps_data.sort()

    epsilons = [e[0] for e in eps_data]
    bests = [e[1] for e in eps_data]

    fig, ax = plt.subplots()

    # Main curve
    ax.plot(epsilons, bests, 'o-', color=C_LBLUE, markeredgecolor='white',
            markeredgewidth=0.5, label='Fixed ε (same pipeline)', zorder=5)

    # PAID-FD star — prominent
    ax.plot(2.84, 61.4, '*', markersize=11, color=C_RED,
            markeredgecolor='white', markeredgewidth=0.4,
            label=r'PAID-FD ($\varepsilon^*\!\approx\!2.84$)', zorder=10)

    # Shaded plateau region
    ax.axhspan(60.5, 62.0, alpha=0.06, color=C_BLUE, zorder=1)

    # Breaking point annotation — arrow from right side
    ax.annotate('Breaking point',
                xy=(0.1, 56.2), xytext=(0.35, 54.0),
                fontsize=6.5, color=C_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.9))

    # Plateau label — top area, no arrow needed
    ax.text(3, 62.5, 'noise-resilient plateau', fontsize=6,
            color=C_GRAY, fontstyle='italic', ha='center')

    ax.set_xlabel(r'Privacy budget $\varepsilon$ per round')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xscale('log')
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(0.07, 15)
    ax.set_ylim(53, 64)

    # Legend in lower-right (data goes upper-right, so no overlap)
    ax.legend(loc='lower right', borderpad=0.4)

    savefig(fig, 'tmc_fig7_privacy_utility')
    print('[Fig 7] Privacy-utility — done')


# ============================================================
# Fig 8: N Scalability
# VI-H — add baseline reference lines
# ============================================================
def plot_fig8():
    expB = load_tmc_group('expB_')
    comb = load_v101_combined()['summaries']

    data = defaultdict(list)
    for label, d in expB.items():
        n = d['config']['n_devices']
        g = d['config']['gamma']
        data[(n, g)].append(d['summary']['best_acc'] * 100)
    for k, v in comb.items():
        if k.startswith('g') and '_s' in k:
            g = int(k.split('_')[0][1:])
            if g in [3, 10]:
                data[(50, g)].append(v['best_acc'] * 100)

    fig, ax = plt.subplots()
    ns = [20, 50, 80]
    x = np.arange(len(ns))
    width = 0.34

    # Use grouped bars instead of connecting lines (clearer for 3 discrete N settings)
    for gi, (gamma, color) in enumerate([(3, MC_PAID_G3), (10, MC_PAID_G10)]):
        means = [np.mean(data[(n, gamma)]) for n in ns]
        stds = [np.std(data[(n, gamma)]) for n in ns]
        off = (gi - 0.5) * width
        ax.bar(x + off, means, width * 0.9, yerr=stds, color=color,
               edgecolor='white', linewidth=0.5, capsize=2,
               error_kw={'linewidth': 0.7},
               label=f'PAID-FD $\\gamma$={gamma}', zorder=5)

    # Baselines shown as single N=50 reference points (no misleading horizontal lines)
    baseline_x = x[1] + 0.46
    baselines = [
        ('FedGMKD', 'expAp_fedgmkd_s42', MC_FEDGMKD, 'D'),
        ('FedAvg', 'expAp_fedavg_s42', MC_FEDAVG, 'P'),
        ('Old Fixed-ε=3', 'expA_fixedeps3_s42', MC_OLD_FIX, 'X'),
    ]
    for bname, btag, bcolor, bmark in baselines:
        d = load_tmc(btag)
        if d:
            val = d['summary']['best_acc'] * 100
            ax.scatter([baseline_x], [val], marker=bmark, s=34, color=bcolor,
                       edgecolor='white', linewidth=0.5,
                       label=f'{bname} (N=50)', zorder=7)

    ax.set_xlabel('Number of Devices ($N$)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in ns])
    ax.set_ylim(38, 65)

    # Legend above the bars so it never overlaps bars or reference points.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=3, borderpad=0.3, fontsize=6.0,
              labelspacing=0.2, columnspacing=0.6, handletextpad=0.4)
    fig.subplots_adjust(top=0.83)

    savefig(fig, 'tmc_fig8_scalability')
    print('[Fig 8] Scalability — done')


# ============================================================
# Fig 9: Non-IID Robustness (α sweep)
# VI-J — add Old Fixed-eps reference
# ============================================================
def plot_fig9():
    expE = load_tmc_group('expE_')
    comb = load_v101_combined()['summaries']

    data = defaultdict(list)
    for label, d in expE.items():
        alpha = d['config'].get('alpha', 0.5)
        gamma = d['config']['gamma']
        data[(alpha, gamma)].append(d['summary']['best_acc'] * 100)
    for k, v in comb.items():
        if k.startswith('g') and '_s' in k:
            g = int(k.split('_')[0][1:])
            if g in [3, 10]:
                data[(0.5, g)].append(v['best_acc'] * 100)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    alphas = [0.1, 0.5, 1.0]
    x = np.arange(len(alphas))
    width = 0.34

    # Redesign: cleaner grouped bars + baseline lines in legend only (no text overlay)
    for gi, (gamma, color) in enumerate([(3, MC_PAID_G3), (10, MC_PAID_G10)]):
        means = [np.mean(data[(a, gamma)]) for a in alphas]
        stds = [np.std(data[(a, gamma)]) for a in alphas]
        offset = (gi - 0.5) * width
        ax.bar(x + offset, means, width * 0.9, yerr=stds, color=color,
               edgecolor='white', linewidth=0.5, capsize=2,
               error_kw={'linewidth': 0.7},
               label=f'PAID-FD $\\gamma$={gamma}', zorder=5)

    # Reference lines drawn ABOVE bars (zorder=10) so they are never hidden
    old_d = load_tmc('expA_fixedeps3_s42')
    if old_d:
        val = old_d['summary']['best_acc'] * 100
        ax.axhline(y=val, color=MC_OLD_FIX, ls='--', lw=1.2, zorder=10)

    gm_d = load_tmc('expAp_fedgmkd_s42')
    if gm_d:
        val = gm_d['summary']['best_acc'] * 100
        ax.axhline(y=val, color=MC_FEDGMKD, ls='--', lw=1.2, zorder=10)

    ax.set_xlabel(r'Dirichlet $\alpha$ (non-IID severity)')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([r'$\alpha$=0.1'+'\n(extreme)',
                        r'$\alpha$=0.5'+'\n(moderate)',
                        r'$\alpha$=1.0'+'\n(mild)'])
    ax.set_ylim(38, 65)

    # Reference line value labels: place on the RIGHT edge to avoid bar overlap
    if old_d:
        old_val = old_d['summary']['best_acc'] * 100
        ax.text(2.55, old_val + 0.3, f'Old Fixed-ε=3 ({old_val:.1f}%)',
                fontsize=5.5, color=MC_OLD_FIX, ha='left', va='bottom')
    if gm_d:
        gm_val = gm_d['summary']['best_acc'] * 100
        # offset label downward if too close to old_d line
        label_y = gm_val - 1.2 if gm_d and old_d and abs(gm_val - old_d['summary']['best_acc']*100) < 2 else gm_val + 0.3
        ax.text(2.55, label_y, f'FedGMKD ({gm_val:.1f}%)',
                fontsize=5.5, color=MC_FEDGMKD, ha='left', va='bottom')

    # Legend above the chart (reference line entries are now in right-edge labels,
    # so only PAID-FD bars go in the legend)
    handles, labels = ax.get_legend_handles_labels()
    bar_h = [h for h, l in zip(handles, labels) if 'PAID' in l]
    bar_l = [l for l in labels if 'PAID' in l]
    ax.legend(bar_h, bar_l, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=2, borderpad=0.3, fontsize=6.2,
              labelspacing=0.2, columnspacing=0.6)
    fig.subplots_adjust(top=0.88, right=0.82)

    savefig(fig, 'tmc_fig9_noniid_alpha')
    print('[Fig 9] Non-IID robustness — done')


# ============================================================
# Fig 10: Privacy Composition Over Rounds
# VI-K — theory validation, no baselines needed
# ============================================================
def plot_fig10():
    v101 = load_v101_3seeds()
    key = 'g5_s42' if 'g5_s42' in v101['runs'] else list(v101['runs'].keys())[0]
    run = v101['runs'][key]
    avg_eps = np.array(run['avg_eps'][:100])
    part = np.array(run['participation_rates'][:100])
    rounds = np.arange(1, 101)

    eps_star = np.mean(avg_eps)
    delta = 1e-5
    mean_part = np.mean(part)

    # Basic composition
    basic_full = np.cumsum(avg_eps)
    basic_sel = basic_full * mean_part

    # Advanced composition
    R = rounds.astype(float)
    adv_full = eps_star * np.sqrt(2 * R * np.log(1 / delta)) + \
               R * eps_star * (np.exp(eps_star) - 1)
    R_sel = R * mean_part
    adv_sel = eps_star * np.sqrt(2 * R_sel * np.log(1 / delta)) + \
              R_sel * eps_star * (np.exp(eps_star) - 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(rounds, basic_full, '-', color=MC_BASIC_F, lw=1.0,
            marker='o', markevery=(0, 15), markersize=3.3,
            markeredgecolor='white', markeredgewidth=0.4,
            label='Basic (full part.)')
    ax.plot(rounds, basic_sel, '--', color=MC_BASIC_S, lw=1.0,
            marker='s', markevery=(4, 15), markersize=3.2,
            markeredgecolor='white', markeredgewidth=0.35,
            label='Basic (selective, %.0f%%)' % (mean_part * 100))
    ax.plot(rounds, adv_full, '-', color=MC_ADV_F, lw=1.5,
            marker='D', markevery=(8, 15), markersize=3.4,
            markeredgecolor='white', markeredgewidth=0.35,
            label='Advanced (Thm 7, full)')
    ax.plot(rounds, adv_sel, '--', color=MC_ADV_S, lw=1.5,
            marker='^', markevery=(12, 15), markersize=3.4,
            markeredgecolor='white', markeredgewidth=0.35,
            label='Advanced (selective)')

    ax.set_xlabel('Communication Round')
    ax.set_ylabel(r'Cumulative $\varepsilon_\mathrm{total}$')
    ax.set_xlim(1, 100)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    # On log scale the lines fan out from the lower-left corner, so
    # the lower-right corner is the clearest area for the legend.
    ax.legend(loc='lower right', borderpad=0.3, fontsize=6.5,
              labelspacing=0.3, handlelength=1.8)

    # Caption note below the x-axis explaining near-linear appearance
    fig.text(0.5, -0.04,
             r'Note: for $\varepsilon^*\!\approx\!2.84$, the term $R\varepsilon(e^\varepsilon-1)$ '
             r'dominates $\sqrt{R}$ in the advanced bound.',
             ha='center', fontsize=5.5, color=C_DGRAY, style='italic',
             wrap=True)

    savefig(fig, 'tmc_fig10_privacy_composition')
    print('[Fig 10] Privacy composition — done')


# ============================================================
# Summary Print
# ============================================================
def print_phase5():
    print('\n' + '=' * 50)
    print('Phase 5 Pipeline Ablation (Table III data)')
    print('=' * 50)
    base = 61.4
    for lab in ['expI_noEMA_s42', 'expI_noMixedLoss_s42', 'expI_noPersistent_s42']:
        d = load_tmc(lab)
        if d:
            s = d['summary']
            delta = s['best_acc'] * 100 - base
            print(f'  {lab:30s}  best={s["best_acc"]*100:.2f}%  Δ={delta:+.1f}pp')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('TMC Paper Figure Generator v2')
    print('=' * 50)

    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6a()   # NEW: method comparison bar chart
    plot_fig6b()   # convergence curves with all methods
    plot_fig7()
    plot_fig8()    # UPDATED: + baseline references
    plot_fig9()    # UPDATED: + baseline references
    plot_fig10()

    print_phase5()

    print('\n' + '=' * 50)
    print(f'All figures saved to: {OUTDIR}/')
    n_pdf = len(list(OUTDIR.glob('tmc_fig*.pdf')))
    print(f'Total: {n_pdf} PDFs')
    print('=' * 50)
