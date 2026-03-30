#!/usr/bin/env python3
"""
Diagnostic: Verify per-device eps_i and true BLUE variance at different gamma.

This script runs the Stackelberg game solver (no actual training) to print
individual device decisions at gamma={3,5,7,10}, comparing:
  1. True BLUE variance: Var = sigma^2 / (sum eps_i^2)
  2. Naive estimate: Var = sigma^2 / (n * avg_eps^2)

Also tests lambda_mult=0.1 to predict whether high-eps regime fixes things.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import ServerPricing

C = 2.0  # clip bound
Delta = 2 * C  # Laplace sensitivity for range [-C, C]

def analyze_game(gamma, devices, label=""):
    """Run game solver and analyze per-device eps distribution."""
    solver = ServerPricing(gamma=gamma, delta=0.01)
    price, decisions = solver.solve(devices)

    participating = [d for d in decisions if d.participates]
    n = len(participating)

    if n == 0:
        print("  %s gamma=%2d: 0 participants (price=%.3f)" % (label, gamma, price))
        return None

    eps_list = sorted([d.eps_star for d in participating], reverse=True)
    eps2_list = [e**2 for e in eps_list]
    sum_eps2 = sum(eps2_list)
    avg_eps = np.mean(eps_list)

    # BLUE weights: w_i = eps_i^2 / sum(eps_j^2)
    blue_weights = [e2 / sum_eps2 for e2 in eps2_list]

    # True BLUE variance per dimension:
    # aggregated = sum(w_i * z_i), Var(z_i) = 2*(Delta/eps_i)^2
    # Var(agg) = sum(w_i^2 * Var(z_i)) = sum((eps_i^2/S)^2 * 2*Delta^2/eps_i^2)
    #          = 2*Delta^2/S^2 * sum(eps_i^2)
    #          = 2*Delta^2 / S  where S = sum(eps_i^2)
    var_blue_true = 2 * Delta**2 / sum_eps2

    # Naive estimate: assume all eps equal to avg
    var_blue_naive = 2 * Delta**2 / (n * avg_eps**2)

    # SNR estimate (assume signal variance ~1 per dim for logits in [-2,2])
    signal_var = 0.5  # rough estimate for clipped logit signal
    snr_true = signal_var / var_blue_true

    print()
    print("  %s gamma=%2d: price=%.3f, n=%d/%d" % (label, gamma, price, n, len(devices)))
    print("    avg_eps=%.4f  sum(eps_i^2)=%.4f  n*avg_eps^2=%.4f" % (
        avg_eps, sum_eps2, n * avg_eps**2))
    print("    Var_BLUE(true) =%.4f   Var_BLUE(naive)=%.4f   ratio=%.2fx" % (
        var_blue_true, var_blue_naive, var_blue_naive / var_blue_true))
    print("    SNR_true=%.3f" % snr_true)

    # Top/bottom 5 devices
    print("    Top-5 eps:    %s" % ["%.4f" % e for e in eps_list[:5]])
    print("    Bottom-5 eps: %s" % ["%.4f" % e for e in eps_list[-5:]])
    print("    Top-5 BLUE w: %s" % ["%.4f" % w for w in blue_weights[:5]])
    print("    Bottom-5 w:   %s" % ["%.4f" % w for w in blue_weights[-5:]])

    # Effective number of devices (inverse Herfindahl)
    hhi = sum(w**2 for w in blue_weights)
    n_eff = 1.0 / hhi if hhi > 0 else 0

    print("    N_effective=%.1f (out of %d)" % (n_eff, n))

    return {
        'gamma': gamma, 'n': n, 'price': price,
        'avg_eps': avg_eps, 'sum_eps2': sum_eps2,
        'var_true': var_blue_true, 'var_naive': var_blue_naive,
        'snr': snr_true, 'n_eff': n_eff,
        'eps_list': eps_list,
    }


def run_analysis(lambda_mult, gammas=[3, 5, 7, 10]):
    label = "lam_mult=%.2f" % lambda_mult

    print("=" * 70)
    print("  BLUE VARIANCE ANALYSIS: lambda_mult = %.2f" % lambda_mult)
    print("=" * 70)

    override = {'privacy_sensitivity': {'lambda_mult': lambda_mult}} if lambda_mult != 1.0 else None
    gen = HeterogeneityGenerator(
        n_devices=50,
        config_path='config/devices/heterogeneity.yaml',
        config_override=override,
        seed=42,
    )
    devices = gen.generate()

    # Print lambda distribution
    lambdas = sorted([d.lambda_i for d in devices])
    print("  Lambda distribution (50 devices):")
    print("    min=%.4f  median=%.4f  max=%.4f  mean=%.4f" % (
        min(lambdas), np.median(lambdas), max(lambdas), np.mean(lambdas)))
    print("    Quartiles: [%.4f, %.4f, %.4f, %.4f, %.4f]" % (
        np.percentile(lambdas, 0), np.percentile(lambdas, 25),
        np.percentile(lambdas, 50), np.percentile(lambdas, 75),
        np.percentile(lambdas, 100)))

    results = []
    for g in gammas:
        r = analyze_game(g, devices, label)
        if r:
            results.append(r)

    # Summary comparison
    if len(results) >= 2:
        print()
        print("  --- COMPARISON TABLE ---")
        print("  %-8s %4s %7s %9s %9s %9s %6s %6s" % (
            "gamma", "n", "avg_eps", "sum_eps2", "Var_true", "Var_naive", "SNR", "N_eff"))
        print("  " + "-" * 62)
        for r in results:
            print("  %-8d %4d %7.4f %9.4f %9.4f %9.4f %6.3f %6.1f" % (
                r['gamma'], r['n'], r['avg_eps'], r['sum_eps2'],
                r['var_true'], r['var_naive'], r['snr'], r['n_eff']))

        # Monotonicity of true BLUE variance
        vars_true = [r['var_true'] for r in results]
        mono_dec = all(vars_true[i] >= vars_true[i+1] for i in range(len(vars_true)-1))
        print()
        print("  Var_BLUE monotonically decreasing with gamma: %s" % (
            "YES (BLUE helps!)" if mono_dec else "NO"))

        if len(results) >= 2:
            best_r = min(results, key=lambda r: r['var_true'])
            worst_r = max(results, key=lambda r: r['var_true'])
            ratio = worst_r['var_true'] / best_r['var_true']
            print("  Best: gamma=%d (Var=%.4f, SNR=%.3f)" % (
                best_r['gamma'], best_r['var_true'], best_r['snr']))
            print("  Worst: gamma=%d (Var=%.4f, SNR=%.3f)" % (
                worst_r['gamma'], worst_r['var_true'], worst_r['snr']))
            print("  Ratio: %.2fx" % ratio)

    return results


if __name__ == "__main__":
    print("PART 1: Current lambda (lambda_mult=1.0)")
    r1 = run_analysis(1.0)

    print("\n\n")
    print("PART 2: Reduced lambda (lambda_mult=0.1)")
    r2 = run_analysis(0.1)

    print("\n\n")
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if r1 and r2:
        print()
        print("Current lambda (mult=1.0):")
        for r in r1:
            print("  gamma=%d: n=%d, avg_eps=%.3f, SNR=%.3f" % (
                r['gamma'], r['n'], r['avg_eps'], r['snr']))

        print()
        print("Reduced lambda (mult=0.1):")
        for r in r2:
            print("  gamma=%d: n=%d, avg_eps=%.3f, SNR=%.3f" % (
                r['gamma'], r['n'], r['avg_eps'], r['snr']))

        # Check if lambda/10 puts us in workable SNR regime
        all_snr_ok = all(r['snr'] > 2.0 for r in r2)
        print()
        if all_snr_ok:
            print("  lambda_mult=0.1: ALL configs have SNR > 2 -> distillation should work!")
        else:
            low_snr = [r for r in r2 if r['snr'] <= 2.0]
            print("  lambda_mult=0.1: Some configs still have SNR <= 2:")
            for r in low_snr:
                print("    gamma=%d: SNR=%.3f" % (r['gamma'], r['snr']))

        # Check if gamma differentiation exists in BLUE variance
        vars_01 = [r['var_true'] for r in r2]
        if len(vars_01) >= 2:
            gap = max(vars_01) / min(vars_01)
            print("  Var_BLUE range at lambda/10: %.2fx" % gap)
