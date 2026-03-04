#!/usr/bin/env python3
"""
Route B – Plot Exp 6: Ablation Study
======================================
Run after GPU experiments complete (routeB_exp6_ablation.json).

Generates:
  Fig 6a: Convergence curves for ablation variants
  Fig 6b: Final accuracy bar chart with component contribution
  Fig 6c: Component contribution breakdown (accuracy delta from Bare-FD)
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
DATA_PATH = "results/experiments/routeB_exp6_ablation.json"
os.makedirs(OUT_DIR, exist_ok=True)

# Ordered from full to bare
VARIANT_ORDER = [
    "Full (PAID-FD)",
    "No-CE (pure KL)",
    "No-BLUE",
    "No-EMA",
    "Bare-FD",
    "No-LDP (oracle)",
]

VARIANT_STYLE = {
    "Full (PAID-FD)":   {"color": "#E64B35", "ls": "-",  "lw": 2.5},
    "No-EMA":           {"color": "#4DBBD5", "ls": "--", "lw": 2.0},
    "No-BLUE":          {"color": "#00A087", "ls": "-.", "lw": 2.0},
    "No-CE (pure KL)":  {"color": "#3C5488", "ls": ":",  "lw": 2.0},
    "Bare-FD":          {"color": "#8491B4", "ls": "--", "lw": 2.0},
    "No-LDP (oracle)":  {"color": "#F39B7F", "ls": "-",  "lw": 2.0},
}

VARIANT_SHORT = {
    "Full (PAID-FD)":  "Full",
    "No-EMA":          "−EMA",
    "No-BLUE":         "−BLUE",
    "No-CE (pure KL)": "−CE",
    "Bare-FD":         "Bare",
    "No-LDP (oracle)": "No-LDP\n(oracle)",
}


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("  Run `python scripts/run_routeB_gpu.py --exp 6` on GPU first.")
        sys.exit(1)
    with open(DATA_PATH) as f:
        return json.load(f)


def compute_stats(runs_list):
    accs_all = [np.array(r["accuracies"]) for r in runs_list]
    n_rounds = min(len(a) for a in accs_all)
    accs_all = np.array([a[:n_rounds] for a in accs_all])
    
    return {
        "acc_mean": accs_all.mean(axis=0),
        "acc_std": accs_all.std(axis=0),
        "final_mean": np.mean([r["final_accuracy"] for r in runs_list]),
        "final_std": np.std([r["final_accuracy"] for r in runs_list]),
        "best_mean": np.mean([r["best_accuracy"] for r in runs_list]),
        "n_rounds": n_rounds,
    }


def main():
    data = load_data()
    runs = data["runs"]
    
    stats = {}
    for variant in VARIANT_ORDER:
        if variant in runs:
            stats[variant] = compute_stats(runs[variant])
    
    available = [v for v in VARIANT_ORDER if v in stats]
    
    # ── Fig 6a: Convergence curves ──
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for variant in available:
        s = stats[variant]
        st = VARIANT_STYLE[variant]
        rounds = np.arange(s["n_rounds"])
        
        ax.plot(rounds, s["acc_mean"] * 100, color=st["color"], linestyle=st["ls"],
                linewidth=st["lw"], label=variant)
        ax.fill_between(rounds,
                        (s["acc_mean"] - s["acc_std"]) * 100,
                        (s["acc_mean"] + s["acc_std"]) * 100,
                        color=st["color"], alpha=0.12)
    
    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("(a) Ablation Study: Component Contribution to PAID-FD", fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6a_ablation_convergence.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig6a_ablation_convergence.png", dpi=200, bbox_inches="tight")
    print("✓ Saved fig6a_ablation_convergence.pdf/png")
    plt.close(fig)
    
    # ── Fig 6b: Final accuracy bars ──
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x = np.arange(len(available))
    means = [stats[v]["final_mean"] * 100 for v in available]
    stds = [stats[v]["final_std"] * 100 for v in available]
    colors = [VARIANT_STYLE[v]["color"] for v in available]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_ylabel("Final Accuracy (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_SHORT.get(v, v) for v in available], fontsize=10)
    ax.set_title("(b) Final Accuracy by Ablation Variant", fontsize=14, pad=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6b_ablation_accuracy.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig6b_ablation_accuracy.png", dpi=200, bbox_inches="tight")
    print("✓ Saved fig6b_ablation_accuracy.pdf/png")
    plt.close(fig)
    
    # ── Fig 6c: Component contribution (delta from Bare-FD) ──
    if "Bare-FD" in stats:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bare_acc = stats["Bare-FD"]["final_mean"] * 100
        components = [v for v in available if v not in ["Bare-FD", "No-LDP (oracle)"]]
        
        deltas = [(bare_acc - stats[v]["final_mean"] * 100) * -1 for v in components]
        # Actually: contribution = full_acc - variant_acc (how much removing it hurts)
        if "Full (PAID-FD)" in stats:
            full_acc = stats["Full (PAID-FD)"]["final_mean"] * 100
            contrib = {
                "EMA Buffer": stats.get("No-EMA", {}).get("final_mean", 0) * 100,
                "BLUE Weights": stats.get("No-BLUE", {}).get("final_mean", 0) * 100,
                "CE Anchor": stats.get("No-CE (pure KL)", {}).get("final_mean", 0) * 100,
            }
            
            comp_names = list(contrib.keys())
            losses = [full_acc - v for v in contrib.values()]
            comp_colors = ["#4DBBD5", "#00A087", "#3C5488"]
            
            bars = ax.barh(range(len(comp_names)), losses, color=comp_colors,
                          edgecolor="black", linewidth=0.8)
            ax.set_yticks(range(len(comp_names)))
            ax.set_yticklabels(comp_names, fontsize=12)
            ax.set_xlabel("Accuracy Drop When Removed (%)", fontsize=13)
            ax.set_title("(c) Component Contribution (accuracy loss)", fontsize=14, pad=10)
            ax.grid(True, alpha=0.3, axis="x")
            ax.invert_yaxis()
            
            for bar, val in zip(bars, losses):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f"{val:+.1f}%", va="center", fontsize=11, fontweight="bold")
        
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/fig6c_component_contribution.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{OUT_DIR}/fig6c_component_contribution.png", dpi=200, bbox_inches="tight")
        print("✓ Saved fig6c_component_contribution.pdf/png")
        plt.close(fig)
    
    # ── Summary table ──
    print(f"\n{'='*70}")
    print("Table: Ablation Study (CIFAR-100, γ=5, 50 devices)")
    print(f"{'='*70}")
    print(f"{'Variant':<22} {'Final Acc(%)':>12} {'± std':>8} {'Best Acc(%)':>12}")
    print(f"{'-'*70}")
    for v in available:
        s = stats[v]
        print(f"{v:<22} {s['final_mean']*100:12.1f} {s['final_std']*100:8.1f} {s['best_mean']*100:12.1f}")
    print(f"{'='*70}")
    
    if "Full (PAID-FD)" in stats and "Bare-FD" in stats:
        full = stats["Full (PAID-FD)"]["final_mean"] * 100
        bare = stats["Bare-FD"]["final_mean"] * 100
        print(f"\n  Full pipeline improvement over Bare-FD: +{full - bare:.1f}%")
    
    if "No-LDP (oracle)" in stats and "Full (PAID-FD)" in stats:
        oracle = stats["No-LDP (oracle)"]["final_mean"] * 100
        full = stats["Full (PAID-FD)"]["final_mean"] * 100
        print(f"  Privacy cost (Full vs No-LDP oracle): {full - oracle:+.1f}%")
    
    print(f"\n✅ All Exp 6 plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
