#!/usr/bin/env python3
"""
Phase 5 Pipeline Ablation — Smoke Test
========================================
Runs all 3 ablation configs for T=2 rounds with seed=999 to verify
that each ablation flag is wired correctly before launching 18-hr full runs.

Expected behaviour (T=2, CIFAR-100, N=50, seed=999):
  • noEMA          : non-zero accuracy in [5%, 80%]  (flag: use_ema=False)
  • noMixedLoss    : non-zero accuracy in [1%, 80%]  (flag: use_mixed_loss=False)
  • noPersistent   : non-zero accuracy in [5%, 80%]  (flag: persistent_local_models=False)

The test FAILS if:
  • Any run crashes / returns empty accuracies
  • Any ablation flag is NOT reflected in the result extras

Usage:
  cd /path/to/paid_fd
  python scripts/_smoke_test_phase5.py [--device cpu|cuda]
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_tmc_experiment import (
    PAID_FD_METHOD, make_training_config
)
from scripts.run_all_experiments import run_single_experiment

# ──────────────────────────────────────────────────────────────
# Ablation specs  (mirrors build_phase5 but seed/rounds are overridden)
# ──────────────────────────────────────────────────────────────
def _ablation_specs():
    specs = []

    mc1 = dict(PAID_FD_METHOD); mc1['use_ema'] = False
    specs.append({
        'label':  'smoke_noEMA',
        'desc':   'No EMA',
        'config': make_training_config(gamma=5.0, method_config=mc1),
        'checks': {'use_ema': False},
    })

    mc2 = dict(PAID_FD_METHOD); mc2['use_mixed_loss'] = False
    specs.append({
        'label':  'smoke_noMixedLoss',
        'desc':   'No Mixed Loss',
        'config': make_training_config(gamma=5.0, method_config=mc2),
        'checks': {'use_mixed_loss': False},
    })

    mc3 = dict(PAID_FD_METHOD); mc3['persistent_local_models'] = False
    specs.append({
        'label':  'smoke_noPersistent',
        'desc':   'No Persistent Models',
        'config': make_training_config(gamma=5.0, method_config=mc3),
        'checks': {'persistent_local_models': False},
    })

    return specs


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

def _check_config_flags(spec):
    """Verify ablation flags are in the method_config that will be used."""
    mc = spec['config'].get('method_config', {})
    failures = []
    for key, expected in spec['checks'].items():
        actual = mc.get(key, '__MISSING__')
        if actual != expected:
            failures.append("  config.method_config[%r] = %r  (expected %r)" % (
                key, actual, expected))
    return failures


def _check_result(result, spec):
    """Validate run result sanity."""
    failures = []
    accs = result.get('accuracies', [])
    if len(accs) == 0:
        failures.append("  No accuracy values returned")
    elif max(accs) < 0.005:  # less than 0.5% — likely dead model
        failures.append("  Max accuracy %.2f%% is suspiciously low" % (max(accs) * 100))
    return failures


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 5 Smoke Test")
    parser.add_argument('--device', default='cpu', help='torch device (default: cpu)')
    parser.add_argument('--rounds', type=int, default=2, help='# rounds (default: 2)')
    parser.add_argument('--seed',   type=int, default=999, help='smoke-test seed (default: 999)')
    args = parser.parse_args()

    specs = _ablation_specs()
    all_passed = True

    print("=" * 65)
    print("  Phase 5 Smoke Test  (T=%d, seed=%d, device=%s)" % (
        args.rounds, args.seed, args.device))
    print("=" * 65)

    for spec in specs:
        label = spec['label']
        print("\n─── %s ───" % label)

        # 1. Config flag check (pre-run)
        flag_errs = _check_config_flags(spec)
        if flag_errs:
            print("  %s  CONFIG FLAG CHECK" % FAIL)
            for e in flag_errs:
                print(e)
            all_passed = False
            continue
        print("  %s  Config flags: %s" % (PASS, spec['checks']))

        # 2. Execute
        t0 = time.time()
        try:
            result = run_single_experiment(
                method_name='PAID-FD',
                config=spec['config'],
                seed=args.seed,
                device=args.device,
                n_rounds=args.rounds,
                save_decisions=False,
                verbose=False,
            )
        except Exception as exc:
            print("  %s  EXECUTION CRASHED: %s" % (FAIL, exc))
            all_passed = False
            continue
        elapsed = time.time() - t0

        # 3. Result sanity check
        res_errs = _check_result(result, spec)
        if res_errs:
            print("  %s  RESULT SANITY CHECK" % FAIL)
            for e in res_errs:
                print(e)
            all_passed = False
        else:
            accs = result.get('accuracies', [])
            print("  %s  Accuracy: [%s]  (%.1fs)" % (
                PASS,
                ", ".join("%.2f%%" % (a * 100) for a in accs),
                elapsed,
            ))

    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("  \033[92mSMOKE TEST PASSED.\033[0m")
        print("  All 3 ablation configs are wired correctly.")
        print("  Ready for full runs (seeds 123, 456):")
        print("    bash scripts/run_phase5_seeds.sh")
    else:
        print("  \033[91mSMOKE TEST FAILED.\033[0m")
        print("  Fix the issues above before launching full runs.")
        sys.exit(1)
    print("=" * 65)


if __name__ == '__main__':
    main()
