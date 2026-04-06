#!/usr/bin/env python3
"""
Plot v9.1 Results: 5 Figures for Paper Ablation Study
======================================================

These figures justify the FD pipeline hyperparameters (C, T, α)
and demonstrate γ differentiation. They form Section V-X "Parameter
Sensitivity Analysis" or "Ablation Study" in the paper.

Figure Layout (from v9.1's 8 configs):
─────────────────────────────────────────────────────────────────────

Fig A: "Effect of Temperature T" (isolate T effect at C=5)
  ├── D6  C=5, T=3  (flat teacher signal)
  ├── D3  C=5, T=2  (moderate)
  └── D2  C=5, T=1  (peaked teacher signal)
  → Convergence curves, 3 lines. Shows T=1 >> T=3.
  → Paper message: "Lower T preserves dark knowledge in 100-class setting"

Fig B: "Effect of Clip Bound C" (isolate C effect at T=1)
  ├── D5  C=2, T=1  (truncated logits)
  └── D2  C=5, T=1  (full logit magnitude)
  → 2 lines + noise analysis. Shows C=5 >> C=2.
  → Paper message: "Larger C preserves inter-class logit structure"

Fig C: "C × T Interaction — 2D Heatmap" (all 4 combos at γ=5)
  ├── D1  C=2, T=3  (worst)
  ├── D5  C=2, T=1
  ├── D6  C=5, T=3
  └── D2  C=5, T=1  (best)
  → 2×2 heatmap of final accuracy.
  → Paper message: "Both C and T matter; combined effect is multiplicative"

Fig D: "CE Anchor Effect" (at best C=5, T=1)
  ├── D2  CE=0    (pure KL)
  └── D4  CE=0.3  (KL + CE anchor)
  → 2 lines. Shows whether CE anchor helps or hurts.
  → Paper message: depends on results

Fig E: "γ Differentiation with Optimized Pipeline" (THE key figure)
  ├── D8  γ=3   (fewer participants, lower price)
  ├── D2  γ=5   (moderate)
  └── D7  γ=10  (all participants, highest price)
  → 3 convergence curves. THIS is the figure that validates the entire paper.
  → Paper message: "Higher γ → better accuracy through Stackelberg pricing"

─────────────────────────────────────────────────────────────────────

Usage:
  python scripts/plot_v9_1_figures.py                 # plot from saved results
  python scripts/plot_v9_1_figures.py --show           # display interactively
  python scripts/plot_v9_1_figures.py --format pdf     # save as PDF
"""

import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Matplotlib config for publication
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paper-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
    'lines.markersize': 4,
})

COLORS = {
    'blue': '#2196F3',
    'red': '#F44336',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'purple': '#9C27B0',
    'grey': '#9E9E9E',
}


def load_results(path="results/experiments/v9_1_distill_fix.json"):
    with open(path) as f:
        return json.load(f)


def smooth(values, window=5):
    """Simple moving average for smoother curves."""
    if len(values) <= window:
        return values
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(np.mean(values[lo:hi]))
    return result


def fig_a_temperature(data, out_dir, fmt='png'):
    """Fig A: Effect of Temperature T (C=5 fixed, γ=5)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    configs = [
        ("D2_C5_T1_g5", "T = 1", COLORS['blue'], '-'),
        ("D3_C5_T2_g5", "T = 2", COLORS['orange'], '--'),
        ("D6_C5_T3_g5", "T = 3", COLORS['red'], '-.'),
    ]

    for key, label, color, ls in configs:
        if key not in data['runs']:
            print("  [Fig A] Missing %s, skipping" % key)
            continue
        accs = [a * 100 for a in data['runs'][key]['accuracies']]
        rounds = list(range(len(accs)))
        accs_smooth = smooth(accs, 5)
        ax.plot(rounds, accs_smooth, label=label, color=color, linestyle=ls)
        ax.scatter(rounds[::10], [accs[i] for i in rounds[::10]], color=color, s=15, zorder=5)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('(a) Effect of Temperature $T$ (C=5, $\\gamma$=5)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    path = os.path.join(out_dir, 'fig_a_temperature.%s' % fmt)
    fig.savefig(path)
    plt.close(fig)
    print("  Saved: %s" % path)


def fig_b_clip_bound(data, out_dir, fmt='png'):
    """Fig B: Effect of Clip Bound C (T=1 fixed, γ=5)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    configs = [
        ("D2_C5_T1_g5", "C = 5", COLORS['blue'], '-'),
        ("D5_C2_T1_g5", "C = 2", COLORS['red'], '--'),
    ]

    for key, label, color, ls in configs:
        if key not in data['runs']:
            print("  [Fig B] Missing %s, skipping" % key)
            continue
        accs = [a * 100 for a in data['runs'][key]['accuracies']]
        rounds = list(range(len(accs)))
        accs_smooth = smooth(accs, 5)
        ax.plot(rounds, accs_smooth, label=label, color=color, linestyle=ls)
        ax.scatter(rounds[::10], [accs[i] for i in rounds[::10]], color=color, s=15, zorder=5)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('(b) Effect of Clip Bound $C$ (T=1, $\\gamma$=5)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    path = os.path.join(out_dir, 'fig_b_clip_bound.%s' % fmt)
    fig.savefig(path)
    plt.close(fig)
    print("  Saved: %s" % path)


def fig_c_heatmap(data, out_dir, fmt='png'):
    """Fig C: C × T Heatmap (γ=5, CE=0)."""
    # 2×2 grid: C={2,5} × T={1,3}
    grid = {
        (2, 3): "D1_C2_T3_g5",
        (2, 1): "D5_C2_T1_g5",
        (5, 3): "D6_C5_T3_g5",
        (5, 1): "D2_C5_T1_g5",
    }

    matrix = np.zeros((2, 2))  # rows=C, cols=T
    c_vals = [2, 5]
    t_vals = [1, 3]  # columns: T=1, T=3

    for i, c in enumerate(c_vals):
        for j, t in enumerate(t_vals):
            key = grid.get((c, t))
            if key and key in data['runs']:
                accs = data['runs'][key]['accuracies']
                matrix[i, j] = max(accs) * 100  # best accuracy
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=30, vmax=50)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['T = 1', 'T = 3'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['C = 2', 'C = 5'])
    ax.set_title('(c) Best Accuracy (%) — C × T Grid ($\\gamma$=5)')

    # Annotate cells
    for i in range(2):
        for j in range(2):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, '%.1f%%' % matrix[i, j],
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if matrix[i, j] < 38 else 'black')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Accuracy (%)')

    path = os.path.join(out_dir, 'fig_c_ct_heatmap.%s' % fmt)
    fig.savefig(path)
    plt.close(fig)
    print("  Saved: %s" % path)


def fig_d_ce_anchor(data, out_dir, fmt='png'):
    """Fig D: CE Anchor Effect (C=5, T=1, γ=5)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    configs = [
        ("D2_C5_T1_g5", "Pure KL ($\\alpha$=0)", COLORS['blue'], '-'),
        ("D4_C5_T1_CE_g5", "KL + CE ($\\alpha$=0.3)", COLORS['orange'], '--'),
    ]

    for key, label, color, ls in configs:
        if key not in data['runs']:
            print("  [Fig D] Missing %s, skipping" % key)
            continue
        accs = [a * 100 for a in data['runs'][key]['accuracies']]
        rounds = list(range(len(accs)))
        accs_smooth = smooth(accs, 5)
        ax.plot(rounds, accs_smooth, label=label, color=color, linestyle=ls)
        ax.scatter(rounds[::10], [accs[i] for i in rounds[::10]], color=color, s=15, zorder=5)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('(d) CE Anchor Effect (C=5, T=1, $\\gamma$=5)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    path = os.path.join(out_dir, 'fig_d_ce_anchor.%s' % fmt)
    fig.savefig(path)
    plt.close(fig)
    print("  Saved: %s" % path)


def fig_e_gamma_diff(data, out_dir, fmt='png'):
    """Fig E: γ Differentiation (C=5, T=1, CE=0) — THE KEY FIGURE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    configs = [
        ("D8_C5_T1_g3", "$\\gamma$ = 3", COLORS['red'], '--'),
        ("D2_C5_T1_g5", "$\\gamma$ = 5", COLORS['orange'], '-.'),
        ("D7_C5_T1_g10", "$\\gamma$ = 10", COLORS['blue'], '-'),
    ]

    # Left: Convergence curves
    for key, label, color, ls in configs:
        if key not in data['runs']:
            print("  [Fig E] Missing %s, skipping" % key)
            continue
        accs = [a * 100 for a in data['runs'][key]['accuracies']]
        rounds = list(range(len(accs)))
        accs_smooth = smooth(accs, 5)
        ax1.plot(rounds, accs_smooth, label=label, color=color, linestyle=ls)
        ax1.scatter(rounds[::10], [accs[i] for i in rounds[::10]], color=color, s=15, zorder=5)

    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(e1) Accuracy Convergence')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)

    # Right: Bar chart of key metrics
    bar_data = []
    for key, label, color, _ in configs:
        if key not in data['runs']:
            continue
        run = data['runs'][key]
        accs = run['accuracies']
        parts = run.get('participation_rates', [])
        eps_list = run.get('avg_eps', [])
        bar_data.append({
            'label': label.replace('$\\gamma$ = ', 'γ='),
            'color': color,
            'final': accs[-1] * 100,
            'best': max(accs) * 100,
            'part': np.mean(parts) * 100 if parts else 0,
            'eps': np.mean(eps_list) if eps_list else 0,
        })

    if bar_data:
        x = np.arange(len(bar_data))
        width = 0.35
        bars1 = ax2.bar(x - width/2, [d['best'] for d in bar_data], width,
                        label='Best Acc', color=[d['color'] for d in bar_data], alpha=0.7)
        bars2 = ax2.bar(x + width/2, [d['final'] for d in bar_data], width,
                        label='Final Acc', color=[d['color'] for d in bar_data], alpha=0.4,
                        hatch='//')

        ax2.set_xticks(x)
        ax2.set_xticklabels([d['label'] for d in bar_data])
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('(e2) Best vs Final Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Annotate participation rate
        for i, d in enumerate(bar_data):
            ax2.annotate('Part=%.0f%%\nε=%.2f' % (d['part'], d['eps']),
                         xy=(i, d['best'] + 0.5), ha='center', fontsize=8,
                         color=d['color'])

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_e_gamma_diff.%s' % fmt)
    fig.savefig(path)
    plt.close(fig)
    print("  Saved: %s" % path)


def fig_combined_summary(data, out_dir, fmt='png'):
    """Combined 2×2 figure for paper (A, B, D, E1)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Reuse individual plotting logic inline
    panels = [
        # (ax, configs, title)
        (axes[0, 0], [
            ("D2_C5_T1_g5", "T=1", COLORS['blue'], '-'),
            ("D3_C5_T2_g5", "T=2", COLORS['orange'], '--'),
            ("D6_C5_T3_g5", "T=3", COLORS['red'], '-.'),
        ], '(a) Temperature $T$ (C=5, $\\gamma$=5)'),

        (axes[0, 1], [
            ("D2_C5_T1_g5", "C=5", COLORS['blue'], '-'),
            ("D5_C2_T1_g5", "C=2", COLORS['red'], '--'),
        ], '(b) Clip Bound $C$ (T=1, $\\gamma$=5)'),

        (axes[1, 0], [
            ("D2_C5_T1_g5", "Pure KL", COLORS['blue'], '-'),
            ("D4_C5_T1_CE_g5", "KL+CE ($\\alpha$=0.3)", COLORS['orange'], '--'),
        ], '(c) CE Anchor (C=5, T=1, $\\gamma$=5)'),

        (axes[1, 1], [
            ("D8_C5_T1_g3", "$\\gamma$=3", COLORS['red'], '--'),
            ("D2_C5_T1_g5", "$\\gamma$=5", COLORS['orange'], '-.'),
            ("D7_C5_T1_g10", "$\\gamma$=10", COLORS['blue'], '-'),
        ], '(d) $\\gamma$ Differentiation (C=5, T=1)'),
    ]

    for ax, configs, title in panels:
        for key, label, color, ls in configs:
            if key not in data['runs']:
                continue
            accs = [a * 100 for a in data['runs'][key]['accuracies']]
            rounds = list(range(len(accs)))
            accs_smooth = smooth(accs, 5)
            ax.plot(rounds, accs_smooth, label=label, color=color, linestyle=ls)

        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_combined_ablation.%s' % fmt)
    fig.savefig(path)
    plt.close(fig)
    print("  Saved: %s" % path)


def fig_teacher_signal_theory(out_dir, fmt='png'):
    """Theoretical figure: Teacher signal strength vs (C, T).
    
    This is a PURE THEORY figure (no experiment results needed).
    Shows why C=5,T=1 is optimal for K=100 classes.
    """
    import torch
    import torch.nn.functional as F

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    K = 100
    n_devices = 50
    eps = 3.0
    unclipped_correct = 4.0
    n_mc = 3000

    # Left: correct class prob vs T for different C
    T_range = np.linspace(0.5, 5.0, 20)
    for C_val, color, ls in [(2.0, COLORS['red'], '--'), (5.0, COLORS['blue'], '-'), (8.0, COLORS['green'], ':')]:
        probs = []
        for T_val in T_range:
            clipped = min(unclipped_correct, C_val)
            logits = torch.zeros(K)
            logits[0] = clipped
            noise_scale = 2 * C_val / eps

            correct_p = []
            for _ in range(n_mc):
                noisy = []
                for d in range(n_devices):
                    noise = np.random.laplace(0, noise_scale, K).astype(np.float32)
                    noisy.append(logits + torch.from_numpy(noise))
                agg = torch.stack(noisy).mean(dim=0)
                teacher = F.softmax(agg / T_val, dim=0)
                correct_p.append(teacher[0].item())
            probs.append(np.mean(correct_p) * 100)

        ax1.plot(T_range, probs, label='C = %.0f' % C_val, color=color, linestyle=ls)

    ax1.axhline(y=1.0, color='grey', linestyle=':', alpha=0.5, label='Uniform (1%)')
    ax1.set_xlabel('Temperature $T$')
    ax1.set_ylabel('Correct Class Probability (%)')
    ax1.set_title('Teacher Signal Strength vs $T$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: correct class prob vs C for different T
    C_range = np.linspace(1.0, 10.0, 20)
    for T_val, color, ls in [(1.0, COLORS['blue'], '-'), (2.0, COLORS['orange'], '--'), (3.0, COLORS['red'], ':')]:
        probs = []
        for C_val in C_range:
            clipped = min(unclipped_correct, C_val)
            logits = torch.zeros(K)
            logits[0] = clipped
            noise_scale = 2 * C_val / eps

            correct_p = []
            for _ in range(n_mc):
                noisy = []
                for d in range(n_devices):
                    noise = np.random.laplace(0, noise_scale, K).astype(np.float32)
                    noisy.append(logits + torch.from_numpy(noise))
                agg = torch.stack(noisy).mean(dim=0)
                teacher = F.softmax(agg / T_val, dim=0)
                correct_p.append(teacher[0].item())
            probs.append(np.mean(correct_p) * 100)

        ax2.plot(C_range, probs, label='T = %.0f' % T_val, color=color, linestyle=ls)

    ax2.axhline(y=1.0, color='grey', linestyle=':', alpha=0.5, label='Uniform (1%)')
    ax2.set_xlabel('Clip Bound $C$')
    ax2.set_ylabel('Correct Class Probability (%)')
    ax2.set_title('Teacher Signal Strength vs $C$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig_teacher_signal_theory.%s' % fmt)
    fig.savefig(path)
    plt.close(fig)
    print("  Saved: %s" % path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot v9.1 ablation figures")
    parser.add_argument("--results", type=str,
                        default="results/experiments/v9_1_distill_fix.json")
    parser.add_argument("--outdir", type=str, default="results/figures/v9_1")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--theory-only", action="store_true",
                        help="Only plot theory figure (no experiment results needed)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.theory_only:
        print("Plotting theory-only figure...")
        fig_teacher_signal_theory(args.outdir, args.format)
        print("Done!")
        sys.exit(0)

    if not os.path.exists(args.results):
        print("Results file not found: %s" % args.results)
        print("Run v9.1 sweep first, or use --theory-only for theory figure.")
        sys.exit(1)

    data = load_results(args.results)
    print("Loaded: %s" % args.results)
    print("  Version: %s" % data.get('version'))
    print("  Configs: %s" % list(data.get('runs', {}).keys()))
    print()

    print("Generating figures...")
    fig_a_temperature(data, args.outdir, args.format)
    fig_b_clip_bound(data, args.outdir, args.format)
    fig_c_heatmap(data, args.outdir, args.format)
    fig_d_ce_anchor(data, args.outdir, args.format)
    fig_e_gamma_diff(data, args.outdir, args.format)
    fig_combined_summary(data, args.outdir, args.format)
    fig_teacher_signal_theory(args.outdir, args.format)

    print("\nAll figures saved to %s/" % args.outdir)

    if args.show:
        plt.show()
