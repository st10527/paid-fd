#!/usr/bin/env python3
"""
Route B - Plot Exp 1: 7-Method Comparison (3 seeds merged)
============================================================
Reads from routeB_exp1_merged_3seeds.json

Generates:
  fig3a: Convergence curves (accuracy vs round, all methods)
  fig3b: Final accuracy bar chart with error bars
  fig3c: Multi-metric 3-panel (convergence speed, comm cost, privacy)
  fig3d: Accuracy-Privacy trade-off scatter
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
DATA_PATH = "results/experiments/routeB_exp1_merged_3seeds.json"
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

METHOD_STYLE = {
    "PAID-FD":        {"color": "#E64B35", "marker": "o", "ls": "-",  "lw": 2.5, "ms": 5},
    "Fixed-eps-0.5":  {"color": "#4DBBD5", "marker": "s", "ls": "--", "lw": 2.0, "ms": 4},
    "Fixed-eps-1.0":  {"color": "#00A087", "marker": "^", "ls": "--", "lw": 2.0, "ms": 4},
    "FedAvg":         {"color": "#3C5488", "marker": "D", "ls": "-.", "lw": 2.0, "ms": 4},
    "FedMD":          {"color": "#F39B7F", "marker": "v", "ls": ":",  "lw": 2.0, "ms": 4},
    "FedGMKD":        {"color": "#8491B4", "marker": "P", "ls": "-.", "lw": 2.0, "ms": 4},
    "CSRA":           {"color": "#91D1C2", "marker": "X", "ls": ":",  "lw": 2.0, "ms": 4},
}

DISPLAY_NAMES = {
    "PAID-FD":       "PAID-FD (Ours)",
    "Fixed-eps-0.5": r"Fixed-$\epsilon$ ($\epsilon$=0.5)",
    "Fixed-eps-1.0": r"Fixed-$\epsilon$ ($\epsilon$=1.0)",
    "FedAvg":        "FedAvg",
    "FedMD":         "FedMD",
    "FedGMKD":       "FedGMKD",
    "CSRA":          "CSRA",
}

PLOT_ORDER = ["PAID-FD", "Fixed-eps-0.5", "Fixed-eps-1.0",
              "FedMD", "FedAvg", "FedGMKD", "CSRA"]


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

    # convergence round: first round >= 90% of final
    conv_rounds = []
    for r in runs_list:
        accs = r["accuracies"]
        final = accs[-1]
        th = 0.9 * final if final > 0.02 else 0.02
        cr = next((i for i, a in enumerate(accs) if a >= th), len(accs))
        conv_rounds.append(cr)

    # avg epsilon from extras
    avg_eps_list = []
    total_comm_list = []
    for r in runs_list:
        extras = r.get("extras", [])
        if extras and isinstance(extras[0], dict):
            eps_vals = [e.get("avg_eps", e.get("fixed_epsilon", 0)) for e in extras]
            avg_eps_list.append(float(np.mean(eps_vals)) if eps_vals else 0.0)
        else:
            avg_eps_list.append(0.0)
        energy = r.get("energy_history", [])
        if energy:
            total_comm_list.append(sum(e.get("communication", 0) for e in energy))
        else:
            total_comm_list.append(0.0)

    return {
        "acc_mean":   accs_mat.mean(axis=0),
        "acc_std":    accs_mat.std(axis=0),
        "final_mean": float(np.mean(finals)),
        "final_std":  float(np.std(finals)),
        "best_mean":  float(np.mean(bests)),
        "best_std":   float(np.std(bests)),
        "conv_mean":  float(np.mean(conv_rounds)),
        "conv_std":   float(np.std(conv_rounds)),
        "avg_eps":    float(np.mean(avg_eps_list)),
        "total_comm": float(np.mean(total_comm_list)),
        "n_rounds":   n_rounds,
        "n_seeds":    len(runs_list),
    }


# ────────────────────────────────────────────────────────────
# Fig 3a: Convergence Curves
# ────────────────────────────────────────────────────────────
def fig3a_convergence(stats):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in PLOT_ORDER:
        if m not in stats:
            continue
        s  = stats[m]
        st = METHOD_STYLE[m]
        rr = np.arange(s["n_rounds"])
        me = max(1, s["n_rounds"] // 10)
        ax.plot(rr, s["acc_mean"] * 100,
                color=st["color"], ls=st["ls"], lw=st["lw"],
                marker=st["marker"], ms=st["ms"], markevery=me,
                label=DISPLAY_NAMES.get(m, m))
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
    fig.savefig(f"{OUT_DIR}/fig3a_convergence_comparison.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3a_convergence_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig3a_convergence_comparison")


# ────────────────────────────────────────────────────────────
# Fig 3b: Final Accuracy Bar Chart
# ────────────────────────────────────────────────────────────
def fig3b_bar_chart(stats):
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = [m for m in PLOT_ORDER if m in stats]
    x      = np.arange(len(methods))
    means  = [stats[m]["final_mean"] * 100 for m in methods]
    stds   = [stats[m]["final_std"]  * 100 for m in methods]
    colors = [METHOD_STYLE[m]["color"] for m in methods]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.88, width=0.65)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Final Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods],
                       rotation=20, ha="right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 72)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3b_final_accuracy.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3b_final_accuracy.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig3b_final_accuracy")


# ────────────────────────────────────────────────────────────
# Fig 3c: Multi-metric horizontal bars (3 panels)
# ────────────────────────────────────────────────────────────
def fig3c_multi_metric(stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    methods = [m for m in PLOT_ORDER if m in stats]
    colors  = [METHOD_STYLE[m]["color"]  for m in methods]
    labels  = [DISPLAY_NAMES.get(m, m)   for m in methods]
    y       = np.arange(len(methods))

    # (i) Convergence speed
    conv = [stats[m]["conv_mean"] for m in methods]
    axes[0].barh(y, conv, color=colors, edgecolor="black", lw=0.6, height=0.6)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("Rounds to 90% Final Acc")
    axes[0].set_title("(i) Convergence Speed")
    axes[0].invert_yaxis()
    for i, v in enumerate(conv):
        if v > 0:
            axes[0].text(v + 0.5, i, f"R{v:.0f}", va="center", fontsize=9)

    # (ii) Communication cost
    comm = [stats[m]["total_comm"] / 1e6 for m in methods]
    axes[1].barh(y, comm, color=colors, edgecolor="black", lw=0.6, height=0.6)
    axes[1].set_yticks(y); axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("Total Comm. Cost (MJ)")
    axes[1].set_title("(ii) Communication Cost")
    axes[1].invert_yaxis()

    # (iii) Privacy budget
    eps = [stats[m]["avg_eps"] for m in methods]
    axes[2].barh(y, eps, color=colors, edgecolor="black", lw=0.6, height=0.6)
    axes[2].set_yticks(y); axes[2].set_yticklabels(labels, fontsize=9)
    axes[2].set_xlabel(r"Average $\epsilon$")
    axes[2].set_title(r"(iii) Privacy Budget")
    axes[2].invert_yaxis()

    fig.tight_layout(w_pad=2.5)
    fig.savefig(f"{OUT_DIR}/fig3c_multi_metric.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3c_multi_metric.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig3c_multi_metric")


# ────────────────────────────────────────────────────────────
# Fig 3d: Accuracy vs Privacy Trade-off
# ────────────────────────────────────────────────────────────
def fig3d_tradeoff(stats):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    NO_DP_X = 5.5  # x-position for methods without DP

    for m in PLOT_ORDER:
        if m not in stats:
            continue
        s  = stats[m]
        st = METHOD_STYLE[m]
        acc = s["final_mean"] * 100

        if m in ("FedAvg", "FedGMKD", "FedMD"):
            eps_plot = NO_DP_X
        elif m == "CSRA":
            eps_plot = 1.0
        else:
            eps_plot = s["avg_eps"]

        ax.scatter(eps_plot, acc, color=st["color"], marker=st["marker"],
                   s=130, zorder=5, edgecolors="black", linewidth=0.8)
        offset = (8, 3) if eps_plot < 5 else (8, 0)
        ax.annotate(DISPLAY_NAMES.get(m, m), (eps_plot, acc),
                    textcoords="offset points", xytext=offset,
                    fontsize=9, color=st["color"])

    ax.axvspan(5, 6.2, alpha=0.07, color="gray")
    ax.text(NO_DP_X, 2, r"No DP ($\epsilon=\infty$)",
            ha="center", fontsize=9, color="gray")
    ax.set_xlabel(r"Privacy Budget $\epsilon$ (lower $\rightarrow$ stronger)")
    ax.set_ylabel("Final Accuracy (%)")
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3d_accuracy_privacy_tradeoff.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3d_accuracy_privacy_tradeoff.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  fig3d_accuracy_privacy_tradeoff")


# ────────────────────────────────────────────────────────────
def main():
    print("Loading merged 3-seed data...")
    data = load_data()
    seeds = data.get("seeds", "?")
    print(f"  Seeds: {seeds}")

    stats = {m: compute_stats(runs) for m, runs in data["runs"].items()}
    print(f"  Methods: {list(stats.keys())}")
    print(f"\nGenerating Exp 1 figures:")

    fig3a_convergence(stats)
    fig3b_bar_chart(stats)
    fig3c_multi_metric(stats)
    fig3d_tradeoff(stats)

    # Summary table
    print(f"\n{'='*80}")
    fmt = "  {:<22s} {:>12s} {:>6s} {:>8s} {:>10s}"
    print(fmt.format("Method", "Acc (%)", "Conv", "Avg eps", "Comm(MJ)"))
    print(f"  {'-'*68}")
    for m in PLOT_ORDER:
        if m not in stats:
            continue
        s = stats[m]
        acc = "{:.2f} +/- {:.2f}".format(s["final_mean"]*100, s["final_std"]*100)
        print(fmt.format(DISPLAY_NAMES.get(m,m), acc,
                         "R{:.0f}".format(s["conv_mean"]),
                         "{:.3f}".format(s["avg_eps"]),
                         "{:.2f}".format(s["total_comm"]/1e6)))
    print(f"  {'='*68}")
    print(f"\n  All Exp 1 figures -> {OUT_DIR}/fig3*.pdf")


if __name__ == "__main__":
    main()
