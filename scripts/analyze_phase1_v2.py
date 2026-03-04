#!/usr/bin/env python3
"""
Comprehensive Phase 1 Result Analysis
======================================
"""
import json, os, glob
import numpy as np

RESULTS_DIR = "results/experiments"

def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)

def analyze_run(run):
    acc = run.get('accuracies', [])
    losses = run.get('losses', [])
    parts = run.get('participation_rates', [])
    prices = run.get('prices', [])
    avg_eps = run.get('avg_eps', [])
    energy = run.get('energy_history', [])
    if not acc:
        return None
    best_acc = max(acc)
    best_round = acc.index(best_acc)
    final_acc = acc[-1]
    n_rounds = len(acc)
    thr98 = best_acc * 0.98
    first_98 = next((i for i, v in enumerate(acc) if v >= thr98), n_rounds)
    checkpoints = {}
    for r in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]:
        if r < len(acc):
            checkpoints[r] = acc[r]
    part_avg = np.mean(parts) if parts else 0
    part_first = parts[0] if parts else 0
    eps_first = avg_eps[0] if avg_eps else 0
    price_first = prices[0] if prices else 0
    if len(acc) >= 40:
        last20 = np.mean(acc[-20:])
        prev20 = np.mean(acc[-40:-20])
        delta_last = last20 - prev20
    else:
        last20 = prev20 = delta_last = 0
    return {
        'n_rounds': n_rounds, 'best_acc': best_acc, 'best_round': best_round,
        'final_acc': final_acc, 'first_98pct': first_98, 'checkpoints': checkpoints,
        'part_rate': part_first, 'part_avg': part_avg,
        'eps_avg_first': eps_first, 'price_first': price_first,
        'last20_avg': last20, 'prev20_avg': prev20, 'delta_last20': delta_last,
        'acc_history': acc, 'loss_history': losses,
        'part_history': parts, 'price_history': prices, 'eps_history': avg_eps,
    }

SEP = '=' * 80

# ============================================================
# Phase 1.1: Gamma
# ============================================================
print(SEP)
print("PHASE 1.1: GAMMA SENSITIVITY")
print(SEP)

data_gamma = load("phase1_gamma_seed42.json")
print(f"Gamma values: {data_gamma.get('gamma_values', 'N/A')}")
print()

gamma_results = {}
for g_str, runs in data_gamma.get('runs', {}).items():
    r = analyze_run(runs[0])
    if r:
        gamma_results[g_str] = r

print(f"{'g':>4} | {'Best%':>7} | {'Final%':>7} | {'BestR':>5} | {'98%@':>5} | {'Part%':>6} | {'avg_e':>7} | {'Price':>6} | {'dLast20':>8}")
print("-" * 80)
for g in sorted(gamma_results, key=lambda x: float(x)):
    r = gamma_results[g]
    print(f"{g:>4} | {r['best_acc']*100:>6.2f}% | {r['final_acc']*100:>6.2f}% | "
          f"R{r['best_round']:>3} | R{r['first_98pct']:>3} | "
          f"{r['part_rate']*100:>5.1f}% | {r['eps_avg_first']:>7.3f} | "
          f"{r['price_first']:>6.3f} | {r['delta_last20']:>+7.4f}")

accs_b = [gamma_results[g]['best_acc'] for g in gamma_results]
accs_f = [gamma_results[g]['final_acc'] for g in gamma_results]
print()
print(f"Best spread:  {(max(accs_b)-min(accs_b))*100:.2f}%  (max={max(accs_b)*100:.2f}%, min={min(accs_b)*100:.2f}%)")
print(f"Final spread: {(max(accs_f)-min(accs_f))*100:.2f}%  (max={max(accs_f)*100:.2f}%, min={min(accs_f)*100:.2f}%)")

print()
print("Trajectory:")
header = f"{'R':>5}"
for g in sorted(gamma_results, key=lambda x: float(x)):
    header += f" | {'g='+g:>8}"
print(header)
print("-" * len(header))
for r in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]:
    line = f"R{r:>4}"
    for g in sorted(gamma_results, key=lambda x: float(x)):
        cp = gamma_results[g]['checkpoints']
        if r in cp:
            line += f" | {cp[r]*100:>7.2f}%"
        else:
            line += f" |     N/A "
    print(line)

print()
print("Participation & Epsilon across rounds:")
for g in sorted(gamma_results, key=lambda x: float(x)):
    r = gamma_results[g]
    ph = r['part_history']
    eh = r['eps_history']
    if ph:
        unique_parts = len(set([round(p, 4) for p in ph]))
        parts_str = f"R0={ph[0]*100:.1f}%"
        if len(ph) > 50:
            parts_str += f", R50={ph[50]*100:.1f}%"
        parts_str += f", R99={ph[-1]*100:.1f}%"
        vary = "CONSTANT" if unique_parts <= 2 else f"VARIES({unique_parts} unique)"
        print(f"  g={g}: parts=[{parts_str}] {vary}")
    if eh:
        print(f"         eps: R0={eh[0]:.4f}, R99={eh[-1]:.4f}")

# ============================================================
# Phase 1.2: Lambda
# ============================================================
print()
print(SEP)
print("PHASE 1.2: LAMBDA SENSITIVITY")
print(SEP)

data_lambda = load("phase1_lambda_seed42.json")
print(f"Lambda values: {data_lambda.get('lambda_values', 'N/A')}")
print(f"Best gamma used: {data_lambda.get('best_gamma', 'N/A')}")
print()

lambda_results = {}
for l_str, runs in data_lambda.get('runs', {}).items():
    r = analyze_run(runs[0])
    if r:
        lambda_results[l_str] = r

print(f"{'lam':>5} | {'Best%':>7} | {'Final%':>7} | {'BestR':>5} | {'98%@':>5} | {'Part%':>6} | {'avg_e':>7} | {'Price':>6} | {'dLast20':>8}")
print("-" * 82)
for l in sorted(lambda_results, key=lambda x: float(x)):
    r = lambda_results[l]
    print(f"{l:>5} | {r['best_acc']*100:>6.2f}% | {r['final_acc']*100:>6.2f}% | "
          f"R{r['best_round']:>3} | R{r['first_98pct']:>3} | "
          f"{r['part_rate']*100:>5.1f}% | {r['eps_avg_first']:>7.3f} | "
          f"{r['price_first']:>6.3f} | {r['delta_last20']:>+7.4f}")

accs_bl = [lambda_results[l]['best_acc'] for l in lambda_results]
accs_fl = [lambda_results[l]['final_acc'] for l in lambda_results]
print()
print(f"Best spread:  {(max(accs_bl)-min(accs_bl))*100:.2f}%  (max={max(accs_bl)*100:.2f}%, min={min(accs_bl)*100:.2f}%)")
print(f"Final spread: {(max(accs_fl)-min(accs_fl))*100:.2f}%  (max={max(accs_fl)*100:.2f}%, min={min(accs_fl)*100:.2f}%)")

print()
print("Trajectory:")
header = f"{'R':>5}"
for l in sorted(lambda_results, key=lambda x: float(x)):
    header += f" | {'l='+l:>8}"
print(header)
print("-" * len(header))
for r in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]:
    line = f"R{r:>4}"
    for l in sorted(lambda_results, key=lambda x: float(x)):
        cp = lambda_results[l]['checkpoints']
        if r in cp:
            line += f" | {cp[r]*100:>7.2f}%"
        else:
            line += f" |     N/A "
    print(line)

print()
print("Participation & Epsilon across rounds:")
for l in sorted(lambda_results, key=lambda x: float(x)):
    r = lambda_results[l]
    ph = r['part_history']
    eh = r['eps_history']
    if ph:
        unique_parts = len(set([round(p, 4) for p in ph]))
        parts_str = f"R0={ph[0]*100:.1f}%"
        if len(ph) > 50:
            parts_str += f", R50={ph[50]*100:.1f}%"
        parts_str += f", R99={ph[-1]*100:.1f}%"
        vary = "CONSTANT" if unique_parts <= 2 else f"VARIES({unique_parts} unique)"
        print(f"  l={l}: parts=[{parts_str}] {vary}")
    if eh:
        print(f"        eps: R0={eh[0]:.4f}, R99={eh[-1]:.4f}")

# ============================================================
# NOISE ANALYSIS
# ============================================================
print()
print(SEP)
print("NOISE & SIGNAL ANALYSIS")
print(SEP)

print()
print("Per-gamma noise breakdown:")
print(f"{'g':>4} | {'eps':>6} | {'C':>3} | {'noise_sc':>9} | {'N_part':>6} | {'eff_noise':>10} | {'SNR':>6}")
print("-" * 65)
for g in sorted(gamma_results, key=lambda x: float(x)):
    r = gamma_results[g]
    run = data_gamma['runs'][g][0]
    config = run.get('config', {})
    mc = config.get('method_config', {})
    eps = r['eps_avg_first']
    C = mc.get('clip_bound', 2.0)
    n_devices = config.get('n_devices', 50)
    n_part = max(int(r['part_rate'] * n_devices), 1)
    noise_sc = 2 * C / eps if eps > 0 else float('inf')
    eff = noise_sc / np.sqrt(n_part)
    snr = C / eff if eff > 0 else float('inf')
    print(f"{g:>4} | {eps:>6.3f} | {C:>3.0f} | {noise_sc:>9.3f} | {n_part:>6} | {eff:>10.4f} | {snr:>6.2f}")

print()
print("Per-lambda noise breakdown:")
print(f"{'lam':>5} | {'eps':>6} | {'C':>3} | {'noise_sc':>9} | {'N_part':>6} | {'eff_noise':>10} | {'SNR':>6}")
print("-" * 65)
for l in sorted(lambda_results, key=lambda x: float(x)):
    r = lambda_results[l]
    run = data_lambda['runs'][l][0]
    config = run.get('config', {})
    mc = config.get('method_config', {})
    eps = r['eps_avg_first']
    C = mc.get('clip_bound', 2.0)
    n_devices = config.get('n_devices', 50)
    n_part = max(int(r['part_rate'] * n_devices), 1)
    noise_sc = 2 * C / eps if eps > 0 else float('inf')
    eff = noise_sc / np.sqrt(n_part)
    snr = C / eff if eff > 0 else float('inf')
    print(f"{l:>5} | {eps:>6.3f} | {C:>3.0f} | {noise_sc:>9.3f} | {n_part:>6} | {eff:>10.4f} | {snr:>6.2f}")

# ============================================================
# LOSS ANALYSIS
# ============================================================
print()
print(SEP)
print("LOSS TRAJECTORY ANALYSIS")
print(SEP)
print()
print("Gamma - Loss at key rounds:")
for g in sorted(gamma_results, key=lambda x: float(x)):
    r = gamma_results[g]
    lh = r['loss_history']
    if lh:
        parts = []
        for rr in [0, 20, 50, 80, 99]:
            if rr < len(lh):
                parts.append(f"R{rr}={lh[rr]:.4f}")
        print(f"  g={g}: {', '.join(parts)}")

print()
print("Lambda - Loss at key rounds:")
for l in sorted(lambda_results, key=lambda x: float(x)):
    r = lambda_results[l]
    lh = r['loss_history']
    if lh:
        parts = []
        for rr in [0, 20, 50, 80, 99]:
            if rr < len(lh):
                parts.append(f"R{rr}={lh[rr]:.4f}")
        print(f"  l={l}: {', '.join(parts)}")

# ============================================================
# Check extras / config deep dive
# ============================================================
print()
print(SEP)
print("CONFIG DEEP DIVE (from first gamma run)")
print(SEP)
first_g = sorted(gamma_results, key=lambda x: float(x))[0]
run0 = data_gamma['runs'][first_g][0]
config0 = run0.get('config', {})
extras0 = run0.get('extras', {})
print(f"  Config keys: {sorted(config0.keys())}")
print(f"  method_config: {config0.get('method_config', {})}")
print(f"  heterogeneity: {config0.get('heterogeneity', {})}")
print(f"  Extras keys: {sorted(extras0.keys()) if extras0 else 'NONE'}")
if extras0:
    for k, v in sorted(extras0.items()):
        if isinstance(v, (int, float, str, bool)):
            print(f"    {k}: {v}")
        elif isinstance(v, list) and len(v) <= 5:
            print(f"    {k}: {v}")
        else:
            print(f"    {k}: type={type(v).__name__}, len={len(v) if hasattr(v,'__len__') else 'N/A'}")

# ============================================================
# NOHUP check
# ============================================================
print()
print(SEP)
print("NOHUP.OUT CHECK")
print(SEP)
for nf in ['nohup.out', 'scripts/nohup.out']:
    if os.path.exists(nf):
        with open(nf) as f:
            lines = f.readlines()
        errors = [l.strip() for l in lines if any(k in l.lower() for k in ['error', 'fail', 'exception', 'traceback'])]
        print(f"\n{nf}: {len(lines)} lines, {len(errors)} error lines")
        if errors:
            for e in errors[:10]:
                print(f"  ! {e}")
        print(f"  Last 5 lines:")
        for l in lines[-5:]:
            print(f"    {l.rstrip()}")
