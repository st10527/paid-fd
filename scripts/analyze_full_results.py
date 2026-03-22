#!/usr/bin/env python3
"""
Comprehensive analysis of Route B full results (3 seeds × 100 rounds).
Merges seed 42 from git history with seeds 123, 456 from current files.
"""
import json
import subprocess
import sys
import numpy as np
from collections import defaultdict


def load_seed42_from_git(filename, commit="94ea7c1"):
    """Recover seed 42 data from git history."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{filename}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: Could not recover {filename} from git {commit}")
        print(result.stderr)
        return None
    return json.loads(result.stdout)


def merge_results(old_data, new_data):
    """Merge seed 42 (old) with seeds 123,456 (new)."""
    merged = {
        'seeds': [42] + new_data.get('seeds', []),
        'runs': {}
    }
    for name in new_data['runs']:
        old_runs = old_data['runs'].get(name, [])
        new_runs = new_data['runs'].get(name, [])
        merged['runs'][name] = old_runs + new_runs
    return merged


def analyze_exp1(data):
    """Analyze Exp 1: Method Comparison."""
    print("=" * 80)
    print("EXP 1: METHOD COMPARISON — 7 methods × 3 seeds × 100 rounds")
    print("=" * 80)

    results = {}
    for name, runs in data['runs'].items():
        finals = [r['accuracies'][-1] for r in runs]
        bests = [max(r['accuracies']) for r in runs]
        all_accs = [r['accuracies'] for r in runs]

        # Convergence: round where accuracy first exceeds 90% of final
        conv_rounds = []
        for r in runs:
            accs = r['accuracies']
            final = accs[-1]
            threshold = 0.9 * final
            conv_r = next((i for i, a in enumerate(accs) if a >= threshold), len(accs))
            conv_rounds.append(conv_r)

        results[name] = {
            'finals': finals,
            'mean_final': np.mean(finals),
            'std_final': np.std(finals),
            'mean_best': np.mean(bests),
            'std_best': np.std(bests),
            'conv_rounds': conv_rounds,
            'mean_conv': np.mean(conv_rounds),
            'seeds': [r.get('seed', '?') for r in runs],
            'all_accs': all_accs,
        }

    # Print per-seed details
    for name, r in results.items():
        print(f"\n--- {name} ---")
        for i, (seed, final) in enumerate(zip(r['seeds'], r['finals'])):
            accs = r['all_accs'][i]
            print(f"  seed={seed}: final={final:.4f}, best={max(accs):.4f}, "
                  f"R10={accs[9]:.4f}, R50={accs[49]:.4f}, conv@90%=R{r['conv_rounds'][i]}")

    # Summary table with mean ± std
    print("\n" + "=" * 80)
    print("EXP 1 SUMMARY (mean ± std across 3 seeds)")
    print("=" * 80)
    print(f"{'Method':<20} {'Final Acc':>15} {'Best Acc':>15} {'Conv Round':>12}")
    print("-" * 65)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_final']):
        final_str = f"{r['mean_final']:.4f} ± {r['std_final']:.4f}"
        best_str = f"{r['mean_best']:.4f} ± {r['std_best']:.4f}"
        conv_str = f"R{r['mean_conv']:.0f}"
        print(f"{name:<20} {final_str:>15} {best_str:>15} {conv_str:>12}")

    return results


def analyze_exp6(data):
    """Analyze Exp 6: Ablation Study."""
    print("\n" + "=" * 80)
    print("EXP 6: ABLATION STUDY — 6 variants × 3 seeds × 100 rounds")
    print("=" * 80)

    results = {}
    for name, runs in data['runs'].items():
        finals = [r['accuracies'][-1] for r in runs]
        bests = [max(r['accuracies']) for r in runs]
        all_accs = [r['accuracies'] for r in runs]

        # Trend analysis: is accuracy going up or down at the end?
        end_trends = []
        for r in runs:
            accs = r['accuracies']
            last20 = accs[-20:]
            first10 = np.mean(last20[:10])
            last10 = np.mean(last20[10:])
            end_trends.append(last10 - first10)

        results[name] = {
            'finals': finals,
            'mean_final': np.mean(finals),
            'std_final': np.std(finals),
            'mean_best': np.mean(bests),
            'std_best': np.std(bests),
            'end_trends': end_trends,
            'mean_trend': np.mean(end_trends),
            'seeds': [r.get('seed', '?') for r in runs],
            'all_accs': all_accs,
        }

    # Per-seed details
    for name, r in results.items():
        print(f"\n--- {name} ---")
        for i, (seed, final) in enumerate(zip(r['seeds'], r['finals'])):
            accs = r['all_accs'][i]
            trend = r['end_trends'][i]
            trend_sym = "↑" if trend > 0.002 else ("↓" if trend < -0.002 else "→")
            print(f"  seed={seed}: final={final:.4f}, best={max(accs):.4f}, "
                  f"R10={accs[9]:.4f}, R50={accs[49]:.4f}, end_trend={trend:+.4f}{trend_sym}")

    # Ablation summary table
    full_mean = results.get('Full (PAID-FD)', {}).get('mean_final', 0)
    print("\n" + "=" * 80)
    print("EXP 6 ABLATION SUMMARY (mean ± std across 3 seeds)")
    print("=" * 80)
    print(f"{'Variant':<25} {'Final Acc':>15} {'Δ vs Full':>10} {'End Trend':>10}")
    print("-" * 65)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean_final']):
        final_str = f"{r['mean_final']:.4f} ± {r['std_final']:.4f}"
        delta = r['mean_final'] - full_mean
        delta_str = f"{delta:+.4f}"
        trend_str = f"{r['mean_trend']:+.4f}"
        print(f"{name:<25} {final_str:>15} {delta_str:>10} {trend_str:>10}")

    return results


def cross_seed_consistency(exp1_results, exp6_results):
    """Check if results are consistent across seeds."""
    print("\n" + "=" * 80)
    print("CROSS-SEED CONSISTENCY CHECK")
    print("=" * 80)

    issues = []

    # Exp 1
    print("\n--- Exp 1: Method Comparison ---")
    for name, r in exp1_results.items():
        cv = r['std_final'] / r['mean_final'] if r['mean_final'] > 0.01 else float('inf')
        status = "✅" if cv < 0.05 else ("⚠️" if cv < 0.10 else "❌")
        print(f"  {status} {name:<20} CV={cv:.4f}  "
              f"finals={[f'{f:.4f}' for f in r['finals']]}")
        if cv >= 0.10:
            issues.append(f"Exp1 {name}: high CV={cv:.4f}")

    # Exp 6
    print("\n--- Exp 6: Ablation ---")
    for name, r in exp6_results.items():
        cv = r['std_final'] / r['mean_final'] if r['mean_final'] > 0.01 else float('inf')
        status = "✅" if cv < 0.05 else ("⚠️" if cv < 0.10 else "❌")
        print(f"  {status} {name:<25} CV={cv:.4f}  "
              f"finals={[f'{f:.4f}' for f in r['finals']]}")
        if cv >= 0.10:
            issues.append(f"Exp6 {name}: high CV={cv:.4f}")

    return issues


def route_b_goal_evaluation(exp1_results, exp6_results):
    """Evaluate results against Route B paper goals."""
    print("\n" + "=" * 80)
    print("ROUTE B GOAL EVALUATION")
    print("=" * 80)

    goals = []
    paid = exp1_results['PAID-FD']
    
    # Goal 1: PAID-FD achieves competitive accuracy (~60%)
    g1 = paid['mean_final'] >= 0.55
    goals.append(('G1: PAID-FD accuracy ≥ 55%', g1,
                   f"{paid['mean_final']:.4f} ± {paid['std_final']:.4f}"))

    # Goal 2: PAID-FD beats FedAvg
    fedavg = exp1_results['FedAvg']
    gap_fedavg = paid['mean_final'] - fedavg['mean_final']
    g2 = gap_fedavg > 0.05
    goals.append(('G2: PAID-FD >> FedAvg', g2,
                   f"gap = {gap_fedavg:+.4f}"))

    # Goal 3: PAID-FD beats FedGMKD
    fedgmkd = exp1_results['FedGMKD']
    gap_gmkd = paid['mean_final'] - fedgmkd['mean_final']
    g3 = gap_gmkd > 0.05
    goals.append(('G3: PAID-FD >> FedGMKD', g3,
                   f"gap = {gap_gmkd:+.4f}"))

    # Goal 4: PAID-FD beats FedMD (validates BLUE+EMA)
    fedmd = exp1_results['FedMD']
    gap_md = paid['mean_final'] - fedmd['mean_final']
    g4 = gap_md > 0.05
    goals.append(('G4: PAID-FD >> FedMD (BLUE+EMA value)', g4,
                   f"gap = {gap_md:+.4f}"))

    # Goal 5: CSRA shows parameter-DP failure
    csra = exp1_results['CSRA']
    g5 = csra['mean_final'] < 0.05
    goals.append(('G5: CSRA param-DP failure (< 5%)', g5,
                   f"{csra['mean_final']:.4f}"))

    # Goal 6: Fixed-eps close to PAID-FD (same accuracy, different mechanism)
    for fe_name in ['Fixed-eps-0.5', 'Fixed-eps-1.0']:
        fe = exp1_results[fe_name]
        gap = abs(paid['mean_final'] - fe['mean_final'])
        g = gap < 0.05
        goals.append((f'G6: {fe_name} ≈ PAID-FD (< 5% gap)', g,
                       f"gap = {gap:.4f}"))

    # Goal 7: Ablation - CE is critical
    full_abl = exp6_results['Full (PAID-FD)']
    noce = exp6_results['No-CE (pure KL)']
    ce_gap = full_abl['mean_final'] - noce['mean_final']
    g7 = ce_gap > 0.20
    goals.append(('G7: CE anchor critical (Δ > 20%)', g7,
                   f"Δ = {ce_gap:.4f}"))

    # Goal 8: Ablation - Bare-FD is much worse
    bare = exp6_results['Bare-FD']
    bare_gap = full_abl['mean_final'] - bare['mean_final']
    g8 = bare_gap > 0.20
    goals.append(('G8: Bare-FD much worse (Δ > 20%)', g8,
                   f"Δ = {bare_gap:.4f}"))

    # Goal 9: No-LDP oracle ≈ Full (noise doesn't hurt much)
    noldp = exp6_results['No-LDP (oracle)']
    ldp_gap = noldp['mean_final'] - full_abl['mean_final']
    g9 = abs(ldp_gap) < 0.03
    goals.append(('G9: No-LDP ≈ Full (noise tolerance)', g9,
                   f"Δ = {ldp_gap:+.4f}"))

    # Goal 10: Convergence speed - PAID-FD converges faster than FedAvg/FedGMKD
    conv_paid = paid['mean_conv']
    conv_fedavg = fedavg['mean_conv']
    conv_gmkd = fedgmkd['mean_conv']
    g10 = conv_paid < min(conv_fedavg, conv_gmkd)
    goals.append(('G10: PAID-FD converges faster than FedAvg/FedGMKD', g10,
                   f"PAID-FD R{conv_paid:.0f} vs FedAvg R{conv_fedavg:.0f}, FedGMKD R{conv_gmkd:.0f}"))

    print()
    n_pass = 0
    for name, ok, detail in goals:
        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            n_pass += 1
        print(f"  {status}: {name}")
        print(f"          {detail}")

    print(f"\n  Score: {n_pass}/{len(goals)} goals met")
    return goals


def paper_table_preview(exp1_results, exp6_results):
    """Generate paper-ready table previews."""
    print("\n" + "=" * 80)
    print("PAPER TABLE PREVIEW: Table 1 — Method Comparison")
    print("=" * 80)
    
    # Order for paper
    order = ['PAID-FD', 'Fixed-eps-0.5', 'Fixed-eps-1.0', 'FedMD', 'FedAvg', 'FedGMKD', 'CSRA']
    print(f"{'Method':<18} {'Final Acc (%)':>15} {'Best Acc (%)':>15} {'Privacy':>12} {'Conv':>6}")
    print("-" * 70)
    for name in order:
        r = exp1_results[name]
        final_str = f"{r['mean_final']*100:.2f} ± {r['std_final']*100:.2f}"
        best_str = f"{r['mean_best']*100:.2f} ± {r['std_best']*100:.2f}"
        
        # Privacy annotation
        if name == 'PAID-FD':
            priv = "LDP (game)"
        elif 'Fixed-eps' in name:
            priv = f"LDP (ε={name.split('-')[-1]})"
        elif name == 'FedMD':
            priv = "None"
        elif name == 'FedAvg':
            priv = "None (params)"
        elif name == 'FedGMKD':
            priv = "None (proto)"
        elif name == 'CSRA':
            priv = "Param-DP"
        else:
            priv = "?"
        
        print(f"{name:<18} {final_str:>15} {best_str:>15} {priv:>12} R{r['mean_conv']:.0f}")

    print("\n" + "=" * 80)
    print("PAPER TABLE PREVIEW: Table 2 — Ablation Study")
    print("=" * 80)
    full_mean = exp6_results['Full (PAID-FD)']['mean_final']
    abl_order = ['Full (PAID-FD)', 'No-LDP (oracle)', 'No-EMA', 'No-BLUE', 'No-CE (pure KL)', 'Bare-FD']
    print(f"{'Variant':<25} {'Final Acc (%)':>15} {'Δ (%)':>10} {'Components Removed':>25}")
    print("-" * 80)
    comp_map = {
        'Full (PAID-FD)': 'None (full system)',
        'No-LDP (oracle)': 'LDP noise',
        'No-EMA': 'EMA buffer',
        'No-BLUE': 'BLUE aggregation',
        'No-CE (pure KL)': 'CE anchor loss',
        'Bare-FD': 'EMA + BLUE + CE',
    }
    for name in abl_order:
        r = exp6_results[name]
        final_str = f"{r['mean_final']*100:.2f} ± {r['std_final']*100:.2f}"
        delta = (r['mean_final'] - full_mean) * 100
        delta_str = f"{delta:+.2f}"
        comp = comp_map.get(name, '?')
        print(f"{name:<25} {final_str:>15} {delta_str:>10} {comp:>25}")


def convergence_analysis(exp1_data, exp6_data):
    """Detailed convergence curve analysis."""
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Exp 1: accuracy at key milestones
    milestones = [10, 25, 50, 75, 100]
    print(f"\n{'Method':<18}", end="")
    for m in milestones:
        print(f" {'R'+str(m):>8}", end="")
    print()
    print("-" * 60)
    
    order = ['PAID-FD', 'Fixed-eps-0.5', 'Fixed-eps-1.0', 'FedMD', 'FedAvg', 'FedGMKD', 'CSRA']
    for name in order:
        runs = exp1_data['runs'][name]
        print(f"{name:<18}", end="")
        for m in milestones:
            vals = [r['accuracies'][min(m-1, len(r['accuracies'])-1)] for r in runs]
            mean_v = np.mean(vals)
            print(f" {mean_v*100:>7.2f}%", end="")
        print()

    # Exp 6: accuracy at milestones
    print(f"\n{'Ablation':<25}", end="")
    for m in milestones:
        print(f" {'R'+str(m):>8}", end="")
    print()
    print("-" * 70)

    abl_order = ['Full (PAID-FD)', 'No-LDP (oracle)', 'No-EMA', 'No-BLUE', 'No-CE (pure KL)', 'Bare-FD']
    for name in abl_order:
        runs = exp6_data['runs'][name]
        print(f"{name:<25}", end="")
        for m in milestones:
            vals = [r['accuracies'][min(m-1, len(r['accuracies'])-1)] for r in runs]
            mean_v = np.mean(vals)
            print(f" {mean_v*100:>7.2f}%", end="")
        print()


def game_analysis(exp1_data):
    """Analyze PAID-FD game parameters across seeds."""
    print("\n" + "=" * 80)
    print("GAME MECHANISM ANALYSIS (PAID-FD)")
    print("=" * 80)
    
    for run in exp1_data['runs']['PAID-FD']:
        seed = run.get('seed', '?')
        extras = run.get('extras', [])
        if not extras:
            continue
        e0 = extras[0]
        e_last = extras[-1]
        print(f"\n  seed={seed}:")
        print(f"    Price (γ=5): {e0.get('price', '?'):.4f}")
        print(f"    Avg ε:       {e0.get('avg_eps', '?'):.4f}")
        print(f"    Avg s:       {e0.get('avg_s', '?'):.4f}")
        print(f"    Server U:    {e0.get('server_utility', '?'):.2f}")
        print(f"    Total Q:     {e0.get('total_quality', '?'):.2f}")
        print(f"    Cumulative privacy (R100): max={e_last.get('max_privacy_spent', '?'):.2f}, "
              f"avg={e_last.get('avg_privacy_spent', '?'):.2f}")


def main():
    print("Loading and merging results from 3 seeds...\n")

    # Load seed 42 from git history
    exp1_s42 = load_seed42_from_git("results/experiments/routeB_exp1_comparison.json", "94ea7c1")
    exp6_s42 = load_seed42_from_git("results/experiments/routeB_exp6_ablation.json", "94ea7c1")

    # Also check acdb2c7 for exp6 seed42
    if exp6_s42 is None:
        exp6_s42 = load_seed42_from_git("results/experiments/routeB_exp6_ablation.json", "acdb2c7")

    # Load current files (seeds 123, 456)
    with open('results/experiments/routeB_exp1_comparison.json') as f:
        exp1_new = json.load(f)
    with open('results/experiments/routeB_exp6_ablation.json') as f:
        exp6_new = json.load(f)

    # Merge
    if exp1_s42:
        exp1_merged = merge_results(exp1_s42, exp1_new)
        print(f"Exp1: Merged seeds {exp1_merged['seeds']}")
    else:
        exp1_merged = exp1_new
        print("WARNING: Could not recover Exp1 seed 42, using 2 seeds only")

    if exp6_s42:
        exp6_merged = merge_results(exp6_s42, exp6_new)
        print(f"Exp6: Merged seeds {exp6_merged['seeds']}")
    else:
        exp6_merged = exp6_new
        print("WARNING: Could not recover Exp6 seed 42, using 2 seeds only")

    # Verify merge
    for name, runs in exp1_merged['runs'].items():
        n = len(runs)
        if n != 3:
            print(f"  ⚠️  {name}: expected 3 runs, got {n}")
        break

    # Run all analyses
    exp1_results = analyze_exp1(exp1_merged)
    exp6_results = analyze_exp6(exp6_merged)
    issues = cross_seed_consistency(exp1_results, exp6_results)
    goals = route_b_goal_evaluation(exp1_results, exp6_results)
    paper_table_preview(exp1_results, exp6_results)
    convergence_analysis(exp1_merged, exp6_merged)
    game_analysis(exp1_merged)

    # Save merged data for future plotting
    with open('results/experiments/routeB_exp1_merged_3seeds.json', 'w') as f:
        json.dump(exp1_merged, f, indent=2)
    with open('results/experiments/routeB_exp6_merged_3seeds.json', 'w') as f:
        json.dump(exp6_merged, f, indent=2)
    print("\n✅ Merged 3-seed data saved to:")
    print("   results/experiments/routeB_exp1_merged_3seeds.json")
    print("   results/experiments/routeB_exp6_merged_3seeds.json")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    n_goals_pass = sum(1 for _, ok, _ in goals if ok)
    n_goals = len(goals)

    if issues:
        print(f"\n⚠️  Consistency issues: {issues}")
    else:
        print("\n✅ Cross-seed consistency: ALL GOOD")

    print(f"✅ Goals met: {n_goals_pass}/{n_goals}")

    if n_goals_pass >= 8:
        print("\n🎉 Results are STRONG — ready for paper writing & final plots!")
    elif n_goals_pass >= 6:
        print("\n⚠️  Results are DECENT — some goals need investigation")
    else:
        print("\n❌ Results need significant attention")


if __name__ == '__main__':
    main()
