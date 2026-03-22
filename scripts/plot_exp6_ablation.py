#!/usr/bin/env python3
"""
Route B - Plot Exp 6: Ablation Study (3 seeds merged)
=======================================================
Reads from routeB_exp6_merged_3seeds.json

Generates:
  fig6a: Convergence curves for all ablation variants
  fig6b: Final accuracy bar chart with delta annotations
  fig6c: Component contribution waterfall / breakdown
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
DATA_PATH = "results/experiments/routeB_exp6_merged_3seeds.json"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

VARIANT_ORDER = [
    "Full (PAID-FD)",
    "No-LDP (oracle)",
    "No-EMA",
    "No-BLUE",
    "No-CE (pure KL)",
    "Bare-FD",
]

VARIANT_STYLE = {
    "Full (PAID-FD)":   {"color": "#E64B35", "ls": "-",  "lw": 2.5, "marker": "o", "ms": 5},
    "No-LDP (oracle)":  {"color": "#F39B7F", "ls": "-",  "lw": 2.0, "marker": "d", "ms": 4},
    "No-EMA":           {"color": "#4DBBD5", "ls": "--", "lw": 2.0, "marker": "s", "ms": 4},
    "No-BLUE":          {"color": "#00A087", "ls": "-.", "lw": 2.0, "marker": "^", "ms": 4},
    "No-CE (pure KL)":  {"color": "#3C5488", "ls": ":",  "lw": 2.0, "marker": "v", "ms": 4},
    "Bare-FD":          {"color": "#8491B4", "ls": "--", "lw": 2.0, "marker": "X", "ms": 5},
}

VARIANT_SHORT = {
    "Full (PAID-FD)":  "Full\n(PAID-FD)",
    "No-LDP (oracle)": "No-LDP\n(Oracle)",
    "No-EMA":          "No-EMA",
    "No-BLUE":         "No-BLUE",
    "No-CE (pure KL)": "No-CE\n(pure KL)",
    "Bare-FD":         "Bare-FD",
}


def load_data():
    if not os.path.exists(DATA_PATH):
        print("Data not found: " + DATA_PATH)
        print("Run scripts/analyze_full_results.py first to merge seeds.")
        sys.exit(1)
    with open(DATA_PATH) as f:
        return json.load(f)


def compute_stats(runs_list):
    accs_all = [np.array(r["accuracies"]) for r in runs_list]
    n_rounds = min(len(a) for a in accs_all)
    accs_mat = np.array([a[:n_rounds] for a in accs_all])

    finals = [r["accuracies"][-1] for r in runs_list]
    bests  = [max(r["accuracies"]) for r in runs_list]

    return {
        "acc_mean":   accs_mat.mean(axis=0),
        "acc_std":    accs_mat.std(axis=0),
        "final_mean": float(np.mean(finals)),
        "final_std":  float(np.std(finals)),
        "best_mean":  float(np.mean(bests)),
        "best_std":   float(np.std(bests)),
        "n_rounds":   n_rounds,
        "n_seeds":    len(runs_list),
    }


# ────────────────────────────────────────────────────────────
# Fig 6a: Ablation Convergence Curves
# ────────────────────────────────────────────────────────────
def fig6a_convergence(stats):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for v in VARIANT_ORDER:
        if v not in stats:
            continue
        s  = stats[v]
        st = VARIANT_STYLE[v]
        rr = np.arange(s["n_rounds"])
        me = max(1, s["n_rounds"] // 10)

        ax.plot(rr, s["acc_mean"] * 100,
                color=st["color"], ls=st["ls"], lw=st["lw"],
                marker=st["marker"], ms=st["ms"], markevery=me,
                label=v)
        ax.fill_between(rr,
                        (s["acc_mean"] - s["acc_std"]) * 100,
                        (s["acc_mean"] + s["acc_std"]) * 100,
                        color=st["color"], alpha=0.12)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(loc="center right", framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 70)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6a_ablation_convergence.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig6a_ablation_convergence.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig6a_ablation_convergence")


# ────────────────────────────────────────────────────────────
# Fig 6b: Final Accuracy Bar Chart with Deltas
# ────────────────────────────────────────────────────────────
def fig6b_bar_chart(stats):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    available = [v for v in VARIANT_ORDER if v in stats]
    x      = np.arange(len(available))
    means  = [stats[v]["final_mean"] * 100 for v in available]
    stds   = [stats[v]["final_std"]  * 100 for v in available]
    colors = [VARIANT_STYLE[v]["color"]     for v in available]

    full_mean = stats["Full (PAID-FD)"]["final_mean"] * 100

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.88, width=0.6)

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        # Value label
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        # Delta label (skip Full itself)
        if available[i] != "Full (PAID-FD)":
            delta = mean - full_mean
            delta_color = "green" if delta >= 0 else "red"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"({delta:+.1f}%)", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_ylabel("Final Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_SHORT.get(v, v) for v in available], fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 72)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6b_ablation_accuracy.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig6b_ablation_accuracy.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig6b_ablation_accuracy")


# ────────────────────────────────────────────────────────────
# Fig 6c: Component Contribution (accuracy loss when removed)
# ────────────────────────────────────────────────────────────
def fig6c_contribution(stats):
    if "Full (PAID-FD)" not in stats:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    full_acc = stats["Full (PAID-FD)"]["final_mean"] * 100

    components = {
        "CE Anchor Loss":    stats.get("No-CE (pure KL)", {}).get("final_mean", 0) * 100,
        "BLUE + EMA + CE\n(Bare-FD removes all)": stats.get("Bare-FD", {}).get("final_mean", 0) * 100,
        "EMA Buffer":        stats.get("No-EMA", {}).get("final_mean", 0) * 100,
        "BLUE Aggregation":  stats.get("No-BLUE", {}).get("final_mean", 0) * 100,
        "LDP Noise\n(oracle upper bound)": stats.get("No-LDP (oracle)", {}).get("final_mean", 0) * 100,
    }

    comp_names = list(components.keys())
    losses = [full_acc - v for v in components.values()]
    comp_colors = ["#3C5488", "#8491B4", "#4DBBD5", "#00A087", "#F39B7F"]

    bars = ax.barh(range(len(comp_names)), losses,
                   color=comp_colors, edgecolor="black", linewidth=0.8, height=0.55)
    ax.set_yticks(range(len(comp_names)))
    ax.set_yticklabels(comp_names, fontsize=11)
    ax.set_xlabel("Accuracy Drop When Removed (%)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    for bar, val in zip(bars, losses):
        x_pos = max(bar.get_width(), 0) + 0.3
        sign = "+" if val < 0 else ""  # negative loss = oracle is better
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", fontsize=11, fontweight="bold")

    # Add reference line at 0
    ax.axvline(x=0, color="black", lw=0.8, ls="-")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6c_component_contribution.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig6c_component_contribution.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig6c_component_contribution")


# ────────────────────────────────────────────────────────────
def main():
    print("Loading merged 3-seed ablation data...")
    data = load_data()
    seeds = data.get("seeds", "?")
    print(f"  Seeds: {seeds}")

    stats = {}
    for v in VARIANT_ORDER:
        if v in data["runs"]:
            stats[v] = compute_stats(data["runs"][v])
    available = [v for v in VARIANT_ORDER if v in stats]
    print(f"  Variants: {available}")

    print(f"\nGenerating Exp 6 figures:")
    fig6a_convergence(stats)
    fig6b_bar_chart(stats)
    fig6c_contribution(stats)

    # Summary table
    full_acc = stats["Full (PAID-FD)"]["final_mean"] * 100
    print(f"\n{'='*70}")
    fmt = "  {:<22s} {:>12s} {:>10s}"
    print(fmt.format("Variant", "Acc (%)", "Delta (%)"))
    print(f"  {'-'*50}")
    for v in available:
        s = stats[v]
        acc = "{:.2f} +/- {:.2f}".format(s["final_mean"]*100, s["final_std"]*100)
        delta = s["final_mean"]*100 - full_acc
        print(fmt.format(v, acc, "{:+.2f}".format(delta)))
    print(f"  {'='*50}")

    print(f"\n  Full pipeline improvement over Bare-FD: "
          f"+{full_acc - stats['Bare-FD']['final_mean']*100:.1f}%")
    print(f"  Privacy cost (Full vs Oracle): "
          f"{stats['Full (PAID-FD)']['final_mean']*100 - stats['No-LDP (oracle)']['final_mean']*100:+.1f}%")
    print(f"\n  All Exp 6 figures -> {OUT_DIR}/fig6*.pdf")


if __name__ == "__main__":
    main()
