#!/usr/bin/env python3
"""Check convergence from Fix 17 GPU results to determine optimal round count."""
import json
import numpy as np

with open('results/experiments/phase1_gamma_seed42.json') as f:
    data = json.load(f)

for gamma_str, runs in data.get('runs', {}).items():
    for run in runs:
        hist = run.get('accuracies', run.get('accuracy_history', []))
        if not hist:
            continue
        
        best = max(hist)
        best_idx = hist.index(best)
        threshold_98 = best * 0.98
        threshold_995 = best * 0.995
        first_98 = next((i for i, v in enumerate(hist) if v >= threshold_98), len(hist))
        first_995 = next((i for i, v in enumerate(hist) if v >= threshold_995), len(hist))
        
        print(f"gamma={gamma_str}: {len(hist)} rounds")
        print(f"  best={best:.4f} at R{best_idx}, final={hist[-1]:.4f}")
        print(f"  98%  of best (>{threshold_98:.4f}) first at R{first_98}")
        print(f"  99.5% of best (>{threshold_995:.4f}) first at R{first_995}")
        
        # Trajectory at key checkpoints
        checkpoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        parts = []
        for r in checkpoints:
            if r < len(hist):
                parts.append(f"R{r}={hist[r]:.2f}")
        print(f"  trajectory: {', '.join(parts)}")
        
        # Improvement in last 20 vs prev 20
        if len(hist) >= 40:
            last20 = np.mean(hist[-20:])
            prev20 = np.mean(hist[-40:-20])
            print(f"  R60-79 avg={prev20:.4f}, R80-99 avg={last20:.4f}, delta={last20-prev20:+.4f}")
        
        # Is it still climbing at R100?
        if len(hist) >= 10:
            last10 = np.mean(hist[-10:])
            prev10 = np.mean(hist[-20:-10]) if len(hist) >= 20 else np.mean(hist[:10])
            still_climbing = last10 > prev10 + 0.001
            print(f"  Still climbing? {'YES' if still_climbing else 'NO'} (last10={last10:.4f} vs prev10={prev10:.4f})")
        print()
