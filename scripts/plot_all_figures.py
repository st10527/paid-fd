#!/usr/bin/env python3
"""
PAID-FD Plot Generator
======================

Generates all 11 figures + 1 table from experiment results.

Usage:
    # Generate all plots
    python scripts/plot_all_figures.py
    
    # Generate specific figure
    python scripts/plot_all_figures.py --fig 1
    python scripts/plot_all_figures.py --fig 3
    
    # Custom output directory
    python scripts/plot_all_figures.py --outdir results/figures/final
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ============================================================================
# Style Configuration (IEEE/TMC compatible)
# ============================================================================

def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


# Color palette (colorblind-friendly)
COLORS = {
    'PAID-FD': '#E63946',       # Red (our method - prominent)
    'FedAvg': '#457B9D',        # Steel blue
    'FedMD': '#2A9D8F',         # Teal
    'FedGMKD': '#E9C46A',       # Gold
    'CSRA': '#F4A261',          # Orange
    'Fixed-eps-1.0': '#A8DADC', # Light blue
    'Fixed-eps-5.0': '#264653', # Dark teal
    'Fixed-eps-10.0': '#6D6875',# Gray purple
    'Fixed-eps-0.5': '#B5838D', # Dusty rose
    'Fixed-eps-2.0': '#FFB4A2', # Light salmon
    'Fixed-eps-20.0': '#CDB4DB',# Light purple
}

MARKERS = {
    'PAID-FD': 'o',
    'FedAvg': 's',
    'FedMD': '^',
    'FedGMKD': 'D',
    'CSRA': 'v',
    'Fixed-eps-1.0': '<',
    'Fixed-eps-5.0': '>',
    'Fixed-eps-10.0': 'p',
}

METHOD_LABELS = {
    'PAID-FD': 'PAID-FD (Ours)',
    'FedAvg': 'FedAvg',
    'FedMD': 'FedMD',
    'FedGMKD': 'FedGMKD',
    'CSRA': 'CSRA',
    'Fixed-eps-1.0': r'Fixed-$\epsilon$=1.0',
    'Fixed-eps-5.0': r'Fixed-$\epsilon$=5.0',
    'Fixed-eps-10.0': r'Fixed-$\epsilon$=10.0',
}


# ============================================================================
# Data Loading Helpers
# ============================================================================

def load_result(phase_name: str) -> Optional[dict]:
    path = PROJECT_ROOT / "results" / "experiments" / f"{phase_name}.json"
    if not path.exists():
        print(f"  ⚠️  {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def get_mean_std(runs: list, key: str = 'final_accuracy'):
    """Get mean and std from list of run dicts."""
    values = [r[key] for r in runs]
    return np.mean(values), np.std(values)


def get_curves_mean_std(runs: list, key: str = 'accuracies'):
    """Get per-round mean and std from list of runs."""
    arrays = [np.array(r[key]) for r in runs]
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    stacked = np.stack(arrays)
    return stacked.mean(axis=0), stacked.std(axis=0)


def savefig(fig, name: str, outdir: str):
    """Save figure in both PDF and PNG."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)
    print(f"  📊 Saved: {name}.pdf / .png")


# ============================================================================
# Figure 1: Gamma Sensitivity (Phase 1.1)
# ============================================================================

def plot_fig1(outdir: str):
    """Fig 1: γ sensitivity - line with errorbar."""
    print("\n📈 Figure 1: Gamma Sensitivity")
    data = load_result('phase1_gamma')
    if not data:
        return
    
    gamma_values = sorted([float(g) for g in data['runs'].keys()])
    means = []
    stds = []
    
    for g in gamma_values:
        m, s = get_mean_std(data['runs'][str(int(g))])
        means.append(m * 100)
        stds.append(s * 100)
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(gamma_values, means, yerr=stds, 
                color=COLORS['PAID-FD'], marker='o', capsize=3,
                linewidth=2, markersize=6)
    
    # Mark the best point
    best_idx = np.argmax(means)
    ax.annotate(f'Best: γ={int(gamma_values[best_idx])}\n({means[best_idx]:.1f}%)',
                xy=(gamma_values[best_idx], means[best_idx]),
                xytext=(gamma_values[best_idx] + 100, means[best_idx] - 2),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='left')
    
    ax.set_xlabel(r'Server Valuation Coefficient $\gamma$')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(r'Sensitivity of $\gamma$')
    ax.grid(True, alpha=0.3)
    
    savefig(fig, 'fig1_gamma_sensitivity', outdir)


# ============================================================================
# Figure 2: Lambda Sensitivity (Phase 1.2)
# ============================================================================

def plot_fig2(outdir: str):
    """Fig 2: λ sensitivity - dual Y-axis (accuracy + ε*)."""
    print("\n📈 Figure 2: Lambda Sensitivity")
    data = load_result('phase1_lambda')
    if not data:
        return
    
    lambda_values = sorted([float(l) for l in data['runs'].keys()])
    acc_means, acc_stds = [], []
    eps_means, eps_stds = [], []
    
    for lam in lambda_values:
        m, s = get_mean_std(data['runs'][str(lam)])
        acc_means.append(m * 100)
        acc_stds.append(s * 100)
        
        # Get average ε from the runs
        eps_vals = [np.mean(r['avg_eps']) for r in data['runs'][str(lam)]]
        eps_means.append(np.mean(eps_vals))
        eps_stds.append(np.std(eps_vals))
    
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()
    
    line1 = ax1.errorbar(lambda_values, acc_means, yerr=acc_stds,
                         color=COLORS['PAID-FD'], marker='o', capsize=3,
                         linewidth=2, label='Accuracy')
    line2 = ax2.errorbar(lambda_values, eps_means, yerr=eps_stds,
                         color='#457B9D', marker='s', capsize=3,
                         linewidth=2, linestyle='--', label=r'$\epsilon^*$')
    
    ax1.set_xlabel(r'Privacy Sensitivity Multiplier $\lambda_{mult}$')
    ax1.set_ylabel('Test Accuracy (%)', color=COLORS['PAID-FD'])
    ax2.set_ylabel(r'Average $\epsilon^*$', color='#457B9D')
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor=COLORS['PAID-FD'])
    ax2.tick_params(axis='y', labelcolor='#457B9D')
    
    lines = [line1, line2]
    labels = ['Accuracy', r'$\epsilon^*$']
    ax1.legend(lines, labels, loc='center left')
    ax1.set_title(r'Sensitivity of $\lambda$')
    
    savefig(fig, 'fig2_lambda_sensitivity', outdir)


# ============================================================================
# Figure 3: Convergence Curves (Phase 2)
# ============================================================================

def plot_fig3(outdir: str):
    """Fig 3: Convergence curves - multi-line with shaded std."""
    print("\n📈 Figure 3: Convergence Curves")
    data = load_result('phase2_convergence')
    if not data:
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for method_name in data.get('methods', data['runs'].keys()):
        if method_name not in data['runs']:
            continue
        runs = data['runs'][method_name]
        mean, std = get_curves_mean_std(runs, 'accuracies')
        mean *= 100
        std *= 100
        rounds = np.arange(len(mean))
        
        color = COLORS.get(method_name, '#999999')
        marker = MARKERS.get(method_name, 'x')
        label = METHOD_LABELS.get(method_name, method_name)
        
        # Plot every 5 rounds for markers
        mark_every = max(1, len(rounds) // 10)
        ax.plot(rounds, mean, color=color, marker=marker, 
                markevery=mark_every, label=label, linewidth=1.5)
        ax.fill_between(rounds, mean - std, mean + std, 
                       color=color, alpha=0.15)
    
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Convergence Comparison')
    ax.legend(loc='lower right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    savefig(fig, 'fig3_convergence_curves', outdir)


# ============================================================================
# Figure 4: Final Accuracy Bar Chart (Phase 2)
# ============================================================================

def plot_fig4(outdir: str):
    """Fig 4: Final accuracy bar chart."""
    print("\n📈 Figure 4: Final Accuracy Bar Chart")
    data = load_result('phase2_convergence')
    if not data:
        return
    
    methods = []
    means = []
    stds = []
    colors = []
    
    # Sort by accuracy
    method_accs = {}
    for method_name, runs in data['runs'].items():
        m, s = get_mean_std(runs)
        method_accs[method_name] = (m * 100, s * 100)
    
    sorted_methods = sorted(method_accs.keys(), key=lambda x: method_accs[x][0], reverse=True)
    
    for method_name in sorted_methods:
        m, s = method_accs[method_name]
        methods.append(METHOD_LABELS.get(method_name, method_name))
        means.append(m)
        stds.append(s)
        colors.append(COLORS.get(method_name, '#999999'))
    
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(methods)), means, yerr=stds, 
                  color=colors, capsize=3, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                f'{m:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final Accuracy Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    savefig(fig, 'fig4_final_accuracy_bar', outdir)


# ============================================================================
# Figure 5: Privacy-Accuracy Tradeoff (Phase 3)
# ============================================================================

def plot_fig5(outdir: str):
    """Fig 5: Privacy-accuracy tradeoff - PAID-FD vs Fixed-ε."""
    print("\n📈 Figure 5: Privacy-Accuracy Tradeoff")
    data = load_result('phase3_privacy')
    if not data:
        return
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Fixed-ε curve
    fixed_eps = sorted([float(e) for e in data['fixed_eps_runs'].keys()])
    fixed_accs = []
    fixed_stds = []
    for eps in fixed_eps:
        m, s = get_mean_std(data['fixed_eps_runs'][str(eps)])
        fixed_accs.append(m * 100)
        fixed_stds.append(s * 100)
    
    ax.errorbar(fixed_eps, fixed_accs, yerr=fixed_stds,
                color='#264653', marker='s', capsize=3,
                linewidth=2, label=r'Fixed-$\epsilon$')
    
    # PAID-FD curve (plot ε* vs accuracy)
    paid_eps = []
    paid_accs = []
    paid_stds_acc = []
    
    for lam_str, runs in data['paid_fd_runs'].items():
        avg_eps = np.mean([np.mean(r['avg_eps']) for r in runs])
        m, s = get_mean_std(runs)
        paid_eps.append(avg_eps)
        paid_accs.append(m * 100)
        paid_stds_acc.append(s * 100)
    
    # Sort by ε
    sort_idx = np.argsort(paid_eps)
    paid_eps = [paid_eps[i] for i in sort_idx]
    paid_accs = [paid_accs[i] for i in sort_idx]
    paid_stds_acc = [paid_stds_acc[i] for i in sort_idx]
    
    ax.errorbar(paid_eps, paid_accs, yerr=paid_stds_acc,
                color=COLORS['PAID-FD'], marker='o', capsize=3,
                linewidth=2, label='PAID-FD (Ours)')
    
    ax.set_xlabel(r'Privacy Budget $\epsilon$')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xscale('log')
    ax.set_title('Privacy-Accuracy Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    savefig(fig, 'fig5_privacy_accuracy_tradeoff', outdir)


# ============================================================================
# Figure 6: Price Evolution (Phase 4)
# ============================================================================

def plot_fig6(outdir: str):
    """Fig 6: Optimal price p* evolution over rounds."""
    print("\n📈 Figure 6: Price Evolution")
    data = load_result('phase4_incentive')
    if not data:
        return
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    runs = data['runs']
    mean_prices, std_prices = get_curves_mean_std(runs, 'prices')
    rounds = np.arange(len(mean_prices))
    
    ax.plot(rounds, mean_prices, color=COLORS['PAID-FD'], linewidth=2)
    ax.fill_between(rounds, mean_prices - std_prices, mean_prices + std_prices,
                    color=COLORS['PAID-FD'], alpha=0.2)
    
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel(r'Optimal Price $p^*$')
    ax.set_title('Price Evolution')
    ax.grid(True, alpha=0.3)
    
    savefig(fig, 'fig6_price_evolution', outdir)


# ============================================================================
# Figure 7: Participation Rate (Phase 4)
# ============================================================================

def plot_fig7(outdir: str):
    """Fig 7: Participation rate over rounds."""
    print("\n📈 Figure 7: Participation Rate")
    data = load_result('phase4_incentive')
    if not data:
        return
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    runs = data['runs']
    mean_part, std_part = get_curves_mean_std(runs, 'participation_rates')
    mean_part *= 100
    std_part *= 100
    rounds = np.arange(len(mean_part))
    
    ax.plot(rounds, mean_part, color=COLORS['PAID-FD'], linewidth=2,
            label='PAID-FD')
    ax.fill_between(rounds, mean_part - std_part, mean_part + std_part,
                    color=COLORS['PAID-FD'], alpha=0.2)
    
    # Reference lines
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='All participate')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Participation Rate (%)')
    ax.set_title('Participation Dynamics')
    ax.legend(loc='center right')
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3)
    
    savefig(fig, 'fig7_participation_rate', outdir)


# ============================================================================
# Figure 8: Utility Distribution (Phase 4)
# ============================================================================

def plot_fig8(outdir: str):
    """Fig 8: Device utility distribution by device type (box/violin plot)."""
    print("\n📈 Figure 8: Utility Distribution")
    data = load_result('phase4_incentive')
    if not data:
        return
    
    # Extract per-device info from extras
    # We'll compute utility proxy from avg_s and avg_eps across runs
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    runs = data['runs']
    # Collect participation rates and avg_eps per run as a proxy for utility
    all_participation = [np.mean(r['participation_rates']) * 100 for r in runs]
    all_avg_eps = [np.mean(r['avg_eps']) for r in runs]
    all_avg_s = [np.mean(r['avg_s']) for r in runs]
    
    # Box plot of key metrics across seeds
    metrics = {
        'Participation\nRate (%)': all_participation,
        r'Avg $\epsilon^*$': all_avg_eps,
        r'Avg $s^*$': all_avg_s,
    }
    
    positions = range(len(metrics))
    for i, (label, values) in enumerate(metrics.items()):
        bp = ax.boxplot([values], positions=[i], widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=COLORS['PAID-FD'], alpha=0.5))
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics.keys())
    ax.set_title('PAID-FD Game Outcome Statistics')
    ax.grid(True, alpha=0.3, axis='y')
    
    savefig(fig, 'fig8_utility_distribution', outdir)


# ============================================================================
# Figure 9: Heterogeneity Impact (Phase 5)
# ============================================================================

def plot_fig9(outdir: str):
    """Fig 9: Heterogeneity impact - grouped bar chart."""
    print("\n📈 Figure 9: Heterogeneity Impact")
    data = load_result('phase5_heterogeneity')
    if not data:
        return
    
    het_levels = list(data['runs'].keys())
    methods = data['methods']
    
    x = np.arange(len(het_levels))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, method_name in enumerate(methods):
        means = []
        stds_list = []
        for level in het_levels:
            if method_name in data['runs'][level]:
                m, s = get_mean_std(data['runs'][level][method_name])
                means.append(m * 100)
                stds_list.append(s * 100)
            else:
                means.append(0)
                stds_list.append(0)
        
        color = COLORS.get(method_name, '#999999')
        label = METHOD_LABELS.get(method_name, method_name)
        offset = (i - len(methods)/2 + 0.5) * width
        
        bars = ax.bar(x + offset, means, width, yerr=stds_list,
                      color=color, label=label, capsize=2,
                      edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('Heterogeneity Level')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Impact of Device Heterogeneity')
    ax.set_xticks(x)
    ax.set_xticklabels(het_levels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    savefig(fig, 'fig9_heterogeneity_impact', outdir)


# ============================================================================
# Figure 10: Scalability (Phase 6)
# ============================================================================

def plot_fig10(outdir: str):
    """Fig 10: Scalability - accuracy and time vs n_devices."""
    print("\n📈 Figure 10: Scalability")
    data = load_result('phase6_scalability')
    if not data:
        return
    
    n_devices_list = sorted([int(n) for n in data['runs'].keys()])
    methods = data['methods']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # Panel A: Accuracy vs n_devices
    for method_name in methods:
        acc_means = []
        acc_stds = []
        for n_dev in n_devices_list:
            if method_name in data['runs'][str(n_dev)]:
                m, s = get_mean_std(data['runs'][str(n_dev)][method_name])
                acc_means.append(m * 100)
                acc_stds.append(s * 100)
            else:
                acc_means.append(0)
                acc_stds.append(0)
        
        color = COLORS.get(method_name, '#999999')
        marker = MARKERS.get(method_name, 'x')
        label = METHOD_LABELS.get(method_name, method_name)
        
        ax1.errorbar(n_devices_list, acc_means, yerr=acc_stds,
                     color=color, marker=marker, capsize=3,
                     linewidth=1.5, label=label)
    
    ax1.set_xlabel('Number of Devices')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Accuracy vs Scale')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Time per round vs n_devices
    for method_name in methods:
        time_means = []
        time_stds = []
        for n_dev in n_devices_list:
            if method_name in data['runs'][str(n_dev)]:
                runs = data['runs'][str(n_dev)][method_name]
                n_rounds = runs[0]['n_rounds']
                times = [r['elapsed_sec'] / n_rounds for r in runs]
                time_means.append(np.mean(times))
                time_stds.append(np.std(times))
            else:
                time_means.append(0)
                time_stds.append(0)
        
        color = COLORS.get(method_name, '#999999')
        marker = MARKERS.get(method_name, 'x')
        label = METHOD_LABELS.get(method_name, method_name)
        
        ax2.errorbar(n_devices_list, time_means, yerr=time_stds,
                     color=color, marker=marker, capsize=3,
                     linewidth=1.5, label=label)
    
    ax2.set_xlabel('Number of Devices')
    ax2.set_ylabel('Time per Round (sec)')
    ax2.set_title('(b) Computation Time vs Scale')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    savefig(fig, 'fig10_scalability', outdir)


# ============================================================================
# Figure 11: Ablation Study (Phase 7)
# ============================================================================

def plot_fig11(outdir: str):
    """Fig 11: Ablation study - bar chart."""
    print("\n📈 Figure 11: Ablation Study")
    data = load_result('phase7_ablation')
    if not data:
        return
    
    variants = data['variants']
    means = []
    stds = []
    colors_list = []
    
    ablation_colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']
    
    for i, variant in enumerate(variants):
        m, s = get_mean_std(data['runs'][variant])
        means.append(m * 100)
        stds.append(s * 100)
        colors_list.append(ablation_colors[i % len(ablation_colors)])
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(len(variants)), means, yerr=stds,
                  color=colors_list, capsize=3,
                  edgecolor='black', linewidth=0.5)
    
    # Value labels
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.3,
                f'{m:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Ablation Study')
    ax.grid(True, alpha=0.3, axis='y')
    
    savefig(fig, 'fig11_ablation', outdir)


# ============================================================================
# Table 1: Full Metrics Comparison (Phase 2)
# ============================================================================

def generate_table1(outdir: str):
    """Table 1: Complete metrics comparison."""
    print("\n📋 Table 1: Full Metrics")
    data = load_result('phase2_convergence')
    if not data:
        return
    
    rows = []
    for method_name, runs in data['runs'].items():
        acc_m, acc_s = get_mean_std(runs, 'final_accuracy')
        best_m, best_s = get_mean_std(runs, 'best_accuracy')
        
        # Get average ε (for privacy methods)
        eps_vals = [np.mean(r.get('avg_eps', [0])) for r in runs]
        avg_eps = np.mean(eps_vals)
        
        # Communication cost estimation
        if method_name in ['FedAvg', 'FedGMKD', 'CSRA']:
            comm_cost = "~44 MB"
        else:
            comm_cost = "~0.4 MB"
        
        # Privacy level
        if method_name == 'FedAvg':
            privacy = "∞ (none)"
        elif method_name == 'FedMD':
            privacy = "- (none)"
        elif method_name.startswith('Fixed-eps'):
            eps = method_name.split('-')[-1]
            privacy = f"ε={eps}"
        elif method_name == 'PAID-FD':
            privacy = f"ε*={avg_eps:.2f}"
        elif method_name == 'CSRA':
            privacy = f"ε≈{avg_eps:.1f}"
        else:
            privacy = "-"
        
        # Time per round
        n_rounds = runs[0]['n_rounds']
        times = [r['elapsed_sec'] / n_rounds for r in runs]
        time_per_round = np.mean(times)
        
        label = METHOD_LABELS.get(method_name, method_name)
        rows.append({
            'method': label,
            'accuracy': f"{acc_m*100:.2f}±{acc_s*100:.2f}",
            'best_acc': f"{best_m*100:.2f}±{best_s*100:.2f}",
            'privacy': privacy,
            'comm_cost': comm_cost,
            'time_per_round': f"{time_per_round:.1f}s",
        })
    
    # Sort by accuracy
    rows.sort(key=lambda x: float(x['accuracy'].split('±')[0]), reverse=True)
    
    # Write LaTeX table
    tex_path = os.path.join(outdir, 'table1_metrics.tex')
    os.makedirs(outdir, exist_ok=True)
    
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison on CIFAR-100}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Accuracy (\\%) & Privacy ($\\epsilon$) & Comm. & Time/Rnd \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(f"{row['method']} & {row['accuracy']} & {row['privacy']} "
                    f"& {row['comm_cost']} & {row['time_per_round']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"  📋 Saved: table1_metrics.tex")
    
    # Also print to console
    print("\n  " + "-" * 80)
    print(f"  {'Method':<22} {'Accuracy':<14} {'Privacy':<12} {'Comm':<10} {'Time/Rnd':<10}")
    print("  " + "-" * 80)
    for row in rows:
        print(f"  {row['method']:<22} {row['accuracy']:<14} {row['privacy']:<12} "
              f"{row['comm_cost']:<10} {row['time_per_round']:<10}")
    print("  " + "-" * 80)


# ============================================================================
# Plot Registry
# ============================================================================

PLOT_REGISTRY = {
    1: ('Fig 1: Gamma Sensitivity', plot_fig1),
    2: ('Fig 2: Lambda Sensitivity', plot_fig2),
    3: ('Fig 3: Convergence Curves', plot_fig3),
    4: ('Fig 4: Final Accuracy Bar', plot_fig4),
    5: ('Fig 5: Privacy-Accuracy Tradeoff', plot_fig5),
    6: ('Fig 6: Price Evolution', plot_fig6),
    7: ('Fig 7: Participation Rate', plot_fig7),
    8: ('Fig 8: Utility Distribution', plot_fig8),
    9: ('Fig 9: Heterogeneity Impact', plot_fig9),
    10: ('Fig 10: Scalability', plot_fig10),
    11: ('Fig 11: Ablation Study', plot_fig11),
    'T1': ('Table 1: Full Metrics', generate_table1),
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PAID-FD Plot Generator')
    parser.add_argument('--fig', type=str, default=None,
                       help='Specific figure to generate (1-11 or T1)')
    parser.add_argument('--all', action='store_true', default=True,
                       help='Generate all figures (default)')
    parser.add_argument('--outdir', type=str, default='results/figures',
                       help='Output directory')
    args = parser.parse_args()
    
    setup_style()
    outdir = str(PROJECT_ROOT / args.outdir)
    
    print("=" * 60)
    print("PAID-FD Figure Generator")
    print(f"Output: {outdir}")
    print("=" * 60)
    
    if args.fig:
        key = int(args.fig) if args.fig.isdigit() else args.fig
        if key in PLOT_REGISTRY:
            name, plotter = PLOT_REGISTRY[key]
            plotter(outdir)
        else:
            print(f"Unknown figure: {args.fig}")
            print(f"Available: {list(PLOT_REGISTRY.keys())}")
    else:
        for key in sorted(PLOT_REGISTRY.keys(), key=lambda x: str(x)):
            name, plotter = PLOT_REGISTRY[key]
            try:
                plotter(outdir)
            except Exception as e:
                print(f"  ⚠️  Failed: {name}: {e}")
    
    print(f"\n✅ Done! Figures saved to: {outdir}")


if __name__ == '__main__':
    main()
