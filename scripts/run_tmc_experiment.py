#!/usr/bin/env python3
"""
TMC Paper — Master Experiment Runner
======================================
All 54 experiments for the TMC paper, organized by phase.

Phase 1 (Week 1): A + A' + B + C = 33 runs  — CIFAR-100, core story
Phase 2 (Week 2): D = 9 runs                 — CIFAR-10 validation
Phase 3 (Week 3): E = 12 runs                — Non-IID α sweep

Usage:
  # List all configs in a phase
  python scripts/run_tmc_experiment.py --phase 1 --list

  # Run a specific task (for SLURM array jobs)
  python scripts/run_tmc_experiment.py --phase 1 --task-id 5

  # Dry-run (show config without executing)
  python scripts/run_tmc_experiment.py --phase 1 --task-id 5 --dry-run

  # Run on specific device
  python scripts/run_tmc_experiment.py --phase 1 --task-id 0 --device cuda:0
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import argparse
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from scripts.run_all_experiments import run_single_experiment, save_json


# ============================================================
# Output directory
# ============================================================
OUTDIR = os.environ.get("TMC_V2_OUTDIR", "results/experiments/tmc")


# ============================================================
# v10.1 PAID-FD base config (proven: 5/5 efficiency, 61% acc)
# ============================================================
PAID_FD_METHOD = {
    'delta': 0.01,
    'clip_bound': 2.0,
    'ema_momentum': 0.9,
    'distill_alpha': 0.7,
    'temperature': 3.0,
    'pretrain_epochs': 10,
    'pretrain_lr': 0.1,
    'use_blue': True,
    'use_ema': True,
    'use_mixed_loss': True,
    'use_ldp': True,
    'use_denoising': False,
}

# ============================================================
# Training config template (shared across all methods)
# ============================================================
def make_training_config(n_devices=50, gamma=5.0, alpha=0.5,
                         lambda_mult=1.0, method_config=None):
    """Build the training config dict."""
    return {
        'n_devices': n_devices,
        'gamma': gamma,
        'alpha': alpha,
        'local_epochs': 5,
        'local_lr': 0.01,
        'local_momentum': 0.9,
        'distill_epochs': 1,
        'distill_lr': 0.001,
        'temperature': 3.0,
        'public_samples': 20000,
        'synthetic': False,
        'heterogeneity': {
            'config_file': 'config/devices/heterogeneity.yaml',
            'overrides': {
                'privacy_sensitivity': {
                    'lambda_mult': lambda_mult
                }
            }
        },
        'method_config': method_config or {},
    }


# ============================================================
# PHASE 1: Core experiments (33 runs)
# A: Privacy-preserving comparison (9)
# A': No-privacy reference (3)
# B: N sweep / scalability (12)
# C: Ablation study (9)
# ============================================================

def build_phase1():
    """33 runs for Week 1."""
    configs = []

    # ---- Exp A: Privacy-preserving comparison ----
    # These methods ALL do LDP — fair accuracy comparison
    # PAID-FD data already exists from v10.1 sweep (γ={3,5,7,10} × 3 seeds)

    # Fixed-ε-1: strong privacy (ε=1), heavy noise
    for seed in [42, 123, 456]:
        mc = {'pretrain_epochs': 10, 'pretrain_lr': 0.1,
              'clip_bound': 2.0, 'temperature': 3.0,
              'distill_epochs': 1, 'distill_lr': 0.001,
              'participation_rate': 1.0, 'use_denoising': False}
        configs.append({
            'label': 'expA_fixedeps1_s%d' % seed,
            'exp': 'A', 'method': 'Fixed-eps-1',
            'seed': seed,
            'config': make_training_config(method_config=mc),
            'desc': 'Fixed ε=1 (strong privacy)',
        })

    # Fixed-ε-3: matches PAID-FD's avg ε≈2.8
    for seed in [42, 123, 456]:
        mc = {'pretrain_epochs': 10, 'pretrain_lr': 0.1,
              'clip_bound': 2.0, 'temperature': 3.0,
              'distill_epochs': 1, 'distill_lr': 0.001,
              'participation_rate': 1.0, 'use_denoising': False}
        configs.append({
            'label': 'expA_fixedeps3_s%d' % seed,
            'exp': 'A', 'method': 'Fixed-eps-3',
            'seed': seed,
            'config': make_training_config(method_config=mc),
            'desc': 'Fixed ε=3 (≈PAID-FD avg ε)',
        })

    # CSRA: reverse auction + DP on parameters
    for seed in [42, 123, 456]:
        mc = {'budget_per_round': 50.0,
              'local_epochs': 5, 'local_lr': 0.01}
        configs.append({
            'label': 'expA_csra_s%d' % seed,
            'exp': 'A', 'method': 'CSRA',
            'seed': seed,
            'config': make_training_config(method_config=mc),
            'desc': 'CSRA (reverse auction + param DP)',
        })

    # ---- Exp A': No-privacy reference (upper bound) ----
    # These methods have NO LDP — reference only, NOT competitors

    # FedAvg: classic parameter aggregation
    mc = {'local_epochs': 5, 'local_lr': 0.01, 'participation_rate': 0.5}
    configs.append({
        'label': 'expAp_fedavg_s42',
        'exp': "A'", 'method': 'FedAvg',
        'seed': 42,
        'config': make_training_config(method_config=mc),
        'desc': 'FedAvg (no privacy, param agg)',
    })

    # FedMD: logit distillation, no incentive, no privacy
    mc = {'pretrain_epochs': 10, 'pretrain_lr': 0.1,
          'distill_epochs': 1, 'distill_lr': 0.001,
          'temperature': 3.0, 'clip_bound': 5.0, 'use_denoising': False}
    configs.append({
        'label': 'expAp_fedmd_s42',
        'exp': "A'", 'method': 'FedMD',
        'seed': 42,
        'config': make_training_config(method_config=mc),
        'desc': 'FedMD (no privacy, logit distill)',
    })

    # FedGMKD: prototype-based, no privacy
    mc = {'alpha': 0.5, 'beta': 0.5, 'dat_temperature': 1.0,
          'local_epochs': 5, 'local_lr': 0.01}
    configs.append({
        'label': 'expAp_fedgmkd_s42',
        'exp': "A'", 'method': 'FedGMKD',
        'seed': 42,
        'config': make_training_config(method_config=mc),
        'desc': 'FedGMKD (no privacy, prototype)',
    })

    # ---- Exp B: N sweep (scalability) ----
    # N={20,80} × γ={3,10} × 3 seeds (N=50 already done in v10.1)

    for N in [20, 80]:
        for gamma in [3, 10]:
            for seed in [42, 123, 456]:
                mc = {**PAID_FD_METHOD, 'gamma': gamma}
                configs.append({
                    'label': 'expB_n%d_g%d_s%d' % (N, gamma, seed),
                    'exp': 'B', 'method': 'PAID-FD',
                    'seed': seed,
                    'config': make_training_config(
                        n_devices=N, gamma=gamma, method_config=mc),
                    'desc': 'PAID-FD N=%d γ=%d' % (N, gamma),
                })

    # ---- Exp C: Ablation study ----
    # γ=5 baseline (already done in v10.1 3-seed), remove one component at a time

    # C1: BLUE off (uniform aggregation instead of BLUE weighting)
    for seed in [42, 123, 456]:
        mc = {**PAID_FD_METHOD, 'gamma': 5, 'use_blue': False}
        configs.append({
            'label': 'expC_noblue_s%d' % seed,
            'exp': 'C', 'method': 'PAID-FD',
            'seed': seed,
            'config': make_training_config(gamma=5, method_config=mc),
            'desc': 'Ablation: BLUE off (uniform agg)',
        })

    # C2: Full participation (γ=100 forces 100% participation)
    # Shows game selection saves cost without hurting accuracy
    for seed in [42, 123, 456]:
        mc = {**PAID_FD_METHOD, 'gamma': 100}
        configs.append({
            'label': 'expC_fullpart_s%d' % seed,
            'exp': 'C', 'method': 'PAID-FD',
            'seed': seed,
            'config': make_training_config(gamma=100, method_config=mc),
            'desc': 'Ablation: forced full participation (γ=100)',
        })

    # C3: Oracle — no LDP noise (accuracy ceiling)
    for seed in [42, 123, 456]:
        mc = {**PAID_FD_METHOD, 'gamma': 5, 'use_ldp': False}
        configs.append({
            'label': 'expC_noldp_s%d' % seed,
            'exp': 'C', 'method': 'PAID-FD',
            'seed': seed,
            'config': make_training_config(gamma=5, method_config=mc),
            'desc': 'Ablation: no LDP (oracle upper bound)',
        })

    assert len(configs) == 33, "Phase 1 should have 33 configs, got %d" % len(configs)
    return configs


# ============================================================
# PHASE 2: CIFAR-10 cross-dataset validation (9 runs)
# Privacy-preserving methods only
# ============================================================

def build_phase2():
    """9 runs for Week 2 — CIFAR-10."""
    configs = []

    # PAID-FD on CIFAR-10 (γ=5, representative middle point)
    for seed in [42, 123, 456]:
        mc = {**PAID_FD_METHOD, 'gamma': 5}
        configs.append({
            'label': 'expD_paidfd_cifar10_s%d' % seed,
            'exp': 'D', 'method': 'PAID-FD',
            'seed': seed,
            'dataset': 'cifar10',
            'config': make_training_config(gamma=5, method_config=mc),
            'desc': 'PAID-FD on CIFAR-10',
        })

    # Fixed-ε-3 on CIFAR-10
    for seed in [42, 123, 456]:
        mc = {'pretrain_epochs': 10, 'pretrain_lr': 0.1,
              'clip_bound': 2.0, 'temperature': 3.0,
              'distill_epochs': 1, 'distill_lr': 0.001,
              'participation_rate': 1.0, 'use_denoising': False}
        configs.append({
            'label': 'expD_fixedeps3_cifar10_s%d' % seed,
            'exp': 'D', 'method': 'Fixed-eps-3',
            'seed': seed,
            'dataset': 'cifar10',
            'config': make_training_config(method_config=mc),
            'desc': 'Fixed ε=3 on CIFAR-10',
        })

    # CSRA on CIFAR-10
    for seed in [42, 123, 456]:
        mc = {'budget_per_round': 50.0,
              'local_epochs': 5, 'local_lr': 0.01}
        configs.append({
            'label': 'expD_csra_cifar10_s%d' % seed,
            'exp': 'D', 'method': 'CSRA',
            'seed': seed,
            'dataset': 'cifar10',
            'config': make_training_config(method_config=mc),
            'desc': 'CSRA on CIFAR-10',
        })

    assert len(configs) == 9, "Phase 2 should have 9 configs, got %d" % len(configs)
    return configs


# ============================================================
# PHASE 3: Non-IID α sweep (12 runs)
# α={0.1, 1.0} × γ={3,10} × 3 seeds (α=0.5 already done)
# ============================================================

def build_phase3():
    """12 runs for Week 3 — Dirichlet α sweep."""
    configs = []

    for alpha in [0.1, 1.0]:
        for gamma in [3, 10]:
            for seed in [42, 123, 456]:
                mc = {**PAID_FD_METHOD, 'gamma': gamma}
                configs.append({
                    'label': 'expE_a%s_g%d_s%d' % (
                        str(alpha).replace('.', ''), gamma, seed),
                    'exp': 'E', 'method': 'PAID-FD',
                    'seed': seed,
                    'config': make_training_config(
                        gamma=gamma, alpha=alpha, method_config=mc),
                    'desc': 'PAID-FD α=%.1f γ=%d' % (alpha, gamma),
                })

    assert len(configs) == 12, "Phase 3 should have 12 configs, got %d" % len(configs)
    return configs


# ============================================================
# PHASE 4: Reviewer-defense experiments (12 runs)
# F: Fair Fixed-ε (same PAID-FD pipeline, fixed epsilon) (3 runs)
# G: ε sweep for privacy-utility curve (6 runs)
# H: Heterogeneous-λ BLUE validation (3 runs)
# ============================================================

def build_phase4():
    """12 runs — critical experiments to counter reviewer attacks."""
    configs = []

    # ---- Exp F: Fair Fixed-ε ----
    # Same pipeline as PAID-FD (persistent models, EMA, mixed loss, persistent Adam)
    # but with fixed epsilon instead of Stackelberg game.
    # This isolates the game's contribution from the pipeline's contribution.
    # Reviewer attack: "Gains come from pipeline, not game."
    # Defense: If Fair-Fixed-ε < PAID-FD, game has genuine contribution.
    for eps in [1, 3, 5]:
        mc = {**PAID_FD_METHOD, 'gamma': 5,
              'fixed_epsilon': float(eps),
              'use_blue': False}  # equal weights since all eps are same
        configs.append({
            'label': 'expF_faireps%d_s42' % eps,
            'exp': 'F', 'method': 'PAID-FD',
            'seed': 42,
            'config': make_training_config(gamma=5, method_config=mc),
            'desc': 'Fair Fixed-ε=%d (PAID-FD pipeline)' % eps,
        })

    # ---- Exp G: Privacy-utility curve ----
    # ε ∈ {0.1, 0.5, 1, 2, 5, 10} × seed=42
    # Plots the standard privacy-utility tradeoff curve (reviewer must-have).
    # Uses PAID-FD pipeline with fixed epsilon to show noise impact.
    for eps_val in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        eps_label = str(eps_val).replace('.', 'p')
        mc = {**PAID_FD_METHOD, 'gamma': 5,
              'fixed_epsilon': eps_val,
              'use_blue': False}
        configs.append({
            'label': 'expG_eps%s_s42' % eps_label,
            'exp': 'G', 'method': 'PAID-FD',
            'seed': 42,
            'config': make_training_config(gamma=5, method_config=mc),
            'desc': 'Privacy-utility: ε=%.1f' % eps_val,
        })

    # ---- Exp H: Heterogeneous-λ BLUE validation ----
    # High λ diversity (lambda_mult=5.0) → wider eps_star range → BLUE matters.
    # Compare BLUE-on vs BLUE-off under heterogeneous privacy regime.
    # Defense: "BLUE degenerates to uniform in homogeneous regime (Prop 6)
    #           but provides significant gains under heterogeneity."

    # H1: PAID-FD with BLUE under heterogeneous λ
    mc_blue = {**PAID_FD_METHOD, 'gamma': 5, 'use_blue': True}
    configs.append({
        'label': 'expH_hetlam_blue_s42',
        'exp': 'H', 'method': 'PAID-FD',
        'seed': 42,
        'config': make_training_config(gamma=5, lambda_mult=5.0,
                                       method_config=mc_blue),
        'desc': 'Hetero-λ (×5) with BLUE',
    })

    # H2: PAID-FD without BLUE under heterogeneous λ
    mc_noblue = {**PAID_FD_METHOD, 'gamma': 5, 'use_blue': False}
    configs.append({
        'label': 'expH_hetlam_noblue_s42',
        'exp': 'H', 'method': 'PAID-FD',
        'seed': 42,
        'config': make_training_config(gamma=5, lambda_mult=5.0,
                                       method_config=mc_noblue),
        'desc': 'Hetero-λ (×5) without BLUE',
    })

    # H3: Baseline (original homogeneous λ with BLUE) for reference
    mc_base = {**PAID_FD_METHOD, 'gamma': 5, 'use_blue': True}
    configs.append({
        'label': 'expH_homlam_blue_s42',
        'exp': 'H', 'method': 'PAID-FD',
        'seed': 42,
        'config': make_training_config(gamma=5, lambda_mult=1.0,
                                       method_config=mc_base),
        'desc': 'Homo-λ (baseline) with BLUE',
    })

    assert len(configs) == 12, "Phase 4 should have 12 configs, got %d" % len(configs)
    return configs


# ============================================================
# PHASE 5: Pipeline Internal Ablation (3 runs)
# I: No EMA / No Mixed Loss / No Persistent Models
# ============================================================

def build_phase5(seed: int = 42):
    """Pipeline component ablation: each run disables ONE component.

    Args:
        seed: Random seed for reproducibility (default 42; use 123/456 for
              multi-seed validation).  The output label includes ``_s{seed}``.
    """
    configs = []
    s = seed

    # Ablation 1: No EMA (single-round logits, no cross-round averaging)
    mc = dict(PAID_FD_METHOD)
    mc['use_ema'] = False
    configs.append({
        'label': 'expI_noEMA_s%d' % s,
        'exp': 'I',
        'method': 'PAID-FD',
        'seed': s,
        'desc': 'No EMA (single-round logits)',
        'config': make_training_config(gamma=5.0, method_config=mc),
    })

    # Ablation 2: No Mixed Loss (pure KL, no CE anchor)
    mc2 = dict(PAID_FD_METHOD)
    mc2['use_mixed_loss'] = False
    configs.append({
        'label': 'expI_noMixedLoss_s%d' % s,
        'exp': 'I',
        'method': 'PAID-FD',
        'seed': s,
        'desc': 'No mixed loss (pure KL distillation)',
        'config': make_training_config(gamma=5.0, method_config=mc2),
    })

    # Ablation 3: No Persistent Models (fresh copy from server each round)
    mc3 = dict(PAID_FD_METHOD)
    mc3['persistent_local_models'] = False
    configs.append({
        'label': 'expI_noPersistent_s%d' % s,
        'exp': 'I',
        'method': 'PAID-FD',
        'seed': s,
        'desc': 'No persistent local models (fresh each round)',
        'config': make_training_config(gamma=5.0, method_config=mc3),
    })

    return configs


# ============================================================
# All phases
# ============================================================
ALL_PHASES = {
    1: build_phase1,
    2: build_phase2,
    3: build_phase3,
    4: build_phase4,
    5: build_phase5,   # accepts optional seed= kwarg
}


# ============================================================
# Extract summary from run result
# ============================================================
def extract_summary(run, spec):
    """Extract key metrics from run result."""
    accs = run.get("accuracies", [])
    extras = run.get("extras", [])

    final_acc = accs[-1] if accs else 0
    best_acc = max(accs) if accs else 0
    cum_payment = extras[-1].get("cumulative_payment", 0) if extras else 0
    avg_part = np.mean(run.get("participation_rates", [])) or 0
    avg_priv = extras[-1].get("avg_privacy_spent", 0) if extras else 0
    max_priv = extras[-1].get("max_privacy_spent", 0) if extras else 0
    avg_price = np.mean(run.get("prices", [])) if run.get("prices") else 0
    avg_eps = np.mean(run.get("avg_eps", [])) if run.get("avg_eps") else 0

    # Time-to-target
    targets = [0.45, 0.50, 0.55, 0.58, 0.60]
    ttt = {}
    for t in targets:
        reached = [i for i, a in enumerate(accs) if a >= t]
        ttt[str(t)] = reached[0] if reached else None

    return {
        "label": spec["label"],
        "exp": spec["exp"],
        "method": spec["method"],
        "seed": spec["seed"],
        "desc": spec.get("desc", ""),
        "final_acc": final_acc,
        "best_acc": best_acc,
        "avg_participation": avg_part,
        "avg_price": avg_price,
        "avg_eps_per_round": avg_eps,
        "cumulative_payment": cum_payment,
        "avg_privacy_spent": avg_priv,
        "max_privacy_spent": max_priv,
        "time_to_targets": ttt,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="TMC Paper — Master Experiment Runner")
    parser.add_argument("--phase", type=int, required=True,
                        choices=[1, 2, 3, 4, 5], help="Phase number")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Task ID (0-indexed) for SLURM array")
    parser.add_argument("--list", action="store_true",
                        help="List all configs in the phase")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config without executing")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed for phase 5 (default: use hard-coded 42)")
    args = parser.parse_args()

    # Phase 5 supports explicit seed override
    if args.phase == 5 and args.seed is not None:
        configs = build_phase5(seed=args.seed)
    else:
        configs = ALL_PHASES[args.phase]()

    # ---- List mode ----
    if args.list:
        print("Phase %d: %d experiments" % (args.phase, len(configs)))
        print("-" * 80)
        for i, c in enumerate(configs):
            print("  [%2d] %-30s  %s  seed=%d  %s" % (
                i, c['label'], c['method'], c['seed'], c['desc']))
        print("-" * 80)

        # Group summary
        exps = {}
        for c in configs:
            e = c['exp']
            exps[e] = exps.get(e, 0) + 1
        for e, n in sorted(exps.items()):
            print("  Exp %s: %d runs" % (e, n))
        print("  Total: %d runs" % len(configs))
        return

    # ---- Validate task-id ----
    if args.task_id is None:
        print("ERROR: --task-id required (0 to %d)" % (len(configs) - 1))
        sys.exit(1)

    if args.task_id < 0 or args.task_id >= len(configs):
        print("ERROR: task-id %d out of range [0, %d)" % (
            args.task_id, len(configs)))
        sys.exit(1)

    spec = configs[args.task_id]
    label = spec['label']
    outfile = Path(OUTDIR) / ("%s.json" % label)

    # ---- Skip if already done ----
    if outfile.exists():
        print("[SKIP] %s already exists: %s" % (label, outfile))
        return

    # ---- Dry-run mode ----
    if args.dry_run:
        print("DRY-RUN: Phase %d, Task %d" % (args.phase, args.task_id))
        print("  Label:  %s" % label)
        print("  Method: %s" % spec['method'])
        print("  Seed:   %d" % spec['seed'])
        print("  Desc:   %s" % spec['desc'])
        print("  Config: %s" % json.dumps(spec['config'], indent=2, default=str))
        print("  Output: %s" % outfile)
        return

    # ---- Execute ----
    print("=" * 70)
    print("  TMC Experiment — Phase %d, Task %d/%d" % (
        args.phase, args.task_id + 1, len(configs)))
    print("  Label:  %s" % label)
    print("  Method: %s" % spec['method'])
    print("  Seed:   %d" % spec['seed'])
    print("  Desc:   %s" % spec['desc'])
    print("  Device: %s" % args.device)
    print("=" * 70)

    # Handle CIFAR-10 (Phase 2)
    dataset = spec.get('dataset', 'cifar100')
    if dataset == 'cifar10':
        # Override num_classes and model for CIFAR-10
        spec['config']['num_classes'] = 10
        spec['config']['model'] = 'resnet18'
        spec['config']['dataset'] = 'cifar10'

    t0 = time.time()
    run = run_single_experiment(
        method_name=spec['method'],
        config=spec['config'],
        seed=spec['seed'],
        device=args.device,
        n_rounds=args.rounds,
        save_decisions=False,
        verbose=True,
    )
    elapsed = time.time() - t0

    summary = extract_summary(run, spec)
    summary["elapsed_sec"] = elapsed

    # Save result
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    result = {
        "label": label,
        "phase": args.phase,
        "exp": spec['exp'],
        "method": spec['method'],
        "seed": spec['seed'],
        "desc": spec['desc'],
        "dataset": dataset,
        "summary": summary,
        "accuracies": run["accuracies"],
        "participation_rates": run["participation_rates"],
        "prices": run["prices"],
        "avg_eps": run["avg_eps"],
        "config": spec['config'],
        "completed_at": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
    }
    save_json(result, str(outfile))

    print("\n" + "=" * 70)
    print("  DONE: %s" % label)
    print("  best=%.2f%%  final=%.2f%%  part=%.0f%%  cost=%.0f" % (
        summary["best_acc"] * 100, summary["final_acc"] * 100,
        summary["avg_participation"] * 100, summary["cumulative_payment"]))
    print("  time=%.0fs (%.1f min)" % (elapsed, elapsed / 60))
    print("  saved: %s" % outfile)
    print("=" * 70)


if __name__ == "__main__":
    main()
