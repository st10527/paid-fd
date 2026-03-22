#!/usr/bin/env python3
"""
Route B – Plot Exp 1: 6-Method Comparison
==========================================
Run after GPU experiments complete (routeB_exp1_comparison.json).

Generates:
  Fig 3a: Convergence curves (accuracy vs round, all methods)
  Fig 3b: Final accuracy bar chart with error bars
  Fig 3c: Communication cost comparison
  Fig 3d: Privacy budget comparison
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
DATA_PATH = "results/experiments/routeB_exp1_comparison.json"
os.makedirs(OUT_DIR, exist_ok=True)

# Color & style for each method
METHOD_STYLE = {
    "PAID-FD":        {"color": "#E64B35", "marker": "o", "ls": "-",  "lw": 2.5},
    "Fixed-eps-0.5":  {"color": "#4DBBD5", "marker": "s", "ls": "--", "lw": 2.0},
    "Fixed-eps-1.0":  {"color": "#00A087", "marker": "^", "ls": "--", "lw": 2.0},
    "FedAvg":         {"color": "#3C5488", "marker": "D", "ls": "-.", "lw": 2.0},
    "FedMD":          {"color": "#F39B7F", "marker": "v", "ls": ":",  "lw": 2.0},
    "FedGMKD":        {"color": "#8491B4", "marker": "P", "ls": "-.", "lw": 2.0},
    "CSRA":           {"color": "#91D1C2", "marker": "X", "ls": ":",  "lw": 2.0},
}

# Display names for paper
DISPLAY_NAMES = {
    "PAID-FD": "PAID-FD (ours)",
    "Fixed-eps-0.5": "Fixed-ε (ε=0.5)",
    "Fixed-eps-1.0": "Fixed-ε (ε=1.0)",
    "FedAvg": "FedAvg",
    "FedMD": "FedMD",
    "FedGMKD": "FedGMKD",
    "CSRA": "CSRA",
}


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("  Run `python scripts/run_routeB_gpu.py --exp 1` on GPU first.")
        sys.exit(1)
    with open(DATA_PATH) as f:
        return json.load(f)


def compute_stats(runs_list):
    """Compute mean ± std across seeds for a list of runs."""
    accs_all = [np.array(r["accuracies"]) for r in runs_list]
    n_rounds = min(len(a) for a in accs_all)
    accs_all = np.array([a[:n_rounds] for a in accs_all])
    
    return {
        "acc_mean": accs_all.mean(axis=0),
        "acc_std": accs_all.std(axis=0),
        "final_mean": np.mean([r["final_accuracy"] for r in runs_list]),
        "final_std": np.std([r["final_accuracy"] for r in runs_list]),
        "best_mean": np.mean([r["best_accuracy"] for r in runs_list]),
        "best_std": np.std([r["best_accuracy"] for r in runs_list]),
        "n_rounds": n_rounds,
        "avg_part": np.mean([np.mean(r.get("participation_rates", [0])) for r in runs_list]),
        "avg_eps": np.mean([np.mean(r.get("avg_eps", [0])) for r in runs_list]),
        "total_comm": np.mean([
            sum(e.get("communication", 0) for e in r.get("energy_history", []))
            for r in runs_list
        ]),
        "total_payment": np.mean([
            sum(e.get("price", 0) * e.get("total_quality", 0) 
                for e in r.get("extras", []))
            for r in runs_list
        ]),
    }


def main():
    data = load_data()
    runs = data["runs"]
    
    # Compute stats per method
    stats = {}
    for method_name in runs:
        stats[method_name] = compute_stats(runs[method_name])
    
    # ── Fig 3a: Convergence curves ──
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name in METHOD_STYLE:
        if method_name not in stats:
            continue
        s = stats[method_name]
        st = METHOD_STYLE[method_name]
        rounds = np.arange(s["n_rounds"])
        
        ax.plot(rounds, s["acc_mean"] * 100, color=st["color"], linestyle=st["ls"],
                linewidth=st["lw"], label=DISPLAY_NAMES.get(method_name, method_name))
        ax.fill_between(rounds, 
                        (s["acc_mean"] - s["acc_std"]) * 100,
                        (s["acc_mean"] + s["acc_std"]) * 100,
                        color=st["color"], alpha=0.15)
    
    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("(a) Convergence Comparison: 6 Methods on CIFAR-100", fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(s["n_rounds"] for s in stats.values()) - 1)
    
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3a_convergence_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3a_convergence_comparison.png", dpi=200, bbox_inches="tight")
    print("✓ Saved fig3a_convergence_comparison.pdf/png")
    plt.close(fig)
    
    # ── Fig 3b: Final accuracy bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods_ordered = [m for m in METHOD_STYLE if m in stats]
    x = np.arange(len(methods_ordered))
    means = [stats[m]["final_mean"] * 100 for m in methods_ordered]
    stds = [stats[m]["final_std"] * 100 for m in methods_ordered]
    colors = [METHOD_STYLE[m]["color"] for m in methods_ordered]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_ylabel("Final Accuracy (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods_ordered], 
                       rotation=15, ha="right", fontsize=10)
    ax.set_title("(b) Final Accuracy Comparison (mean ± std, 3 seeds)", fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3b_final_accuracy.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3b_final_accuracy.png", dpi=200, bbox_inches="tight")
    print("✓ Saved fig3b_final_accuracy.pdf/png")
    plt.close(fig)
    
    # ── Fig 3c: Multi-metric radar / bar comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 3c-1: Communication cost
    comm_vals = [stats[m]["total_comm"] / 1e6 for m in methods_ordered]
    bars = axes[0].barh(x, comm_vals, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels([DISPLAY_NAMES.get(m, m) for m in methods_ordered], fontsize=9)
    axes[0].set_xlabel("Total Comm. Cost (×10⁶ J)", fontsize=11)
    axes[0].set_title("Communication Cost", fontsize=12)
    axes[0].invert_yaxis()
    
    # 3c-2: Average ε
    eps_vals = [stats[m]["avg_eps"] for m in methods_ordered]
    bars = axes[1].barh(x, eps_vals, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([DISPLAY_NAMES.get(m, m) for m in methods_ordered], fontsize=9)
    axes[1].set_xlabel("Average Privacy Budget ε", fontsize=11)
    axes[1].set_title("Privacy Budget (lower = stronger)", fontsize=12)
    axes[1].invert_yaxis()
    
    # 3c-3: Participation rate
    part_vals = [stats[m]["avg_part"] * 100 for m in methods_ordered]
    bars = axes[2].barh(x, part_vals, color=colors, edgecolor="black", linewidth=0.8)
    axes[2].set_yticks(x)
    axes[2].set_yticklabels([DISPLAY_NAMES.get(m, m) for m in methods_ordered], fontsize=9)
    axes[2].set_xlabel("Avg. Participation Rate (%)", fontsize=11)
    axes[2].set_title("Participation Rate", fontsize=12)
    axes[2].invert_yaxis()
    
    fig.suptitle("(c) Multi-Metric Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3c_multi_metric.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3c_multi_metric.png", dpi=200, bbox_inches="tight")
    print("✓ Saved fig3c_multi_metric.pdf/png")
    plt.close(fig)
    
    # ── Print summary table ──
    print(f"\n{'='*90}")
    print("Table: 6-Method Comparison (CIFAR-100, 50 devices)")
    print(f"{'='*90}")
    print(f"{'Method':<20} {'Acc(%)':>8} {'±':>4} {'Part(%)':>8} {'Avg ε':>8} {'Comm(M)':>10}")
    print(f"{'-'*90}")
    for m in methods_ordered:
        s = stats[m]
        print(f"{DISPLAY_NAMES.get(m,m):<20} {s['final_mean']*100:8.1f} "
              f"{'±':>4}{s['final_std']*100:4.1f} {s['avg_part']*100:8.0f} "
              f"{s['avg_eps']:8.3f} {s['total_comm']/1e6:10.2f}")
    print(f"{'='*90}")
    
    print(f"\n✅ All Exp 1 plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
