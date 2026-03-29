#!/usr/bin/env python3
"""Parse Phase 0.1 Diagnostic Results - detailed analysis"""
import json

def main():
    data = json.load(open("results/experiments/v8_phase0_diagnostic.json"))
    runs = data["runs"]
    
    print("=" * 75)
    print("V8 PHASE 0.1 DIAGNOSTIC — FULL ANALYSIS")
    print("Seed: %d   Rounds: %d" % (data["seed"], data["n_rounds"]))
    print("=" * 75)
    
    summaries = []
    
    for name in sorted(runs.keys()):
        r = runs[name]
        accs = r.get("accuracies", [])
        losses = r.get("losses", [])
        parts = r.get("participation_rates", [])
        prices = r.get("prices", [])
        avg_eps_list = r.get("avg_eps", [])
        avg_s_list = r.get("avg_s", [])
        cfg = r.get("config", {})
        elapsed = r.get("elapsed_sec", 0)
        
        print()
        print("-" * 65)
        print("  %s" % name)
        print("-" * 65)
        
        # Config
        print("  method=%s  gamma=%s" % (cfg.get("method","?"), cfg.get("gamma","?")))
        print("  clip_bound=%s  distill_lr=%s  distill_epochs=%s" % (
            cfg.get("clip_bound","?"), cfg.get("distill_lr","?"), cfg.get("distill_epochs","?")))
        print("  temperature=%s  pretrain_epochs=%s  pretrain_lr=%s" % (
            cfg.get("temperature","?"), cfg.get("pretrain_epochs","?"), cfg.get("pretrain_lr","?")))
        print("  elapsed: %.0fs (%.1fmin)" % (elapsed, elapsed/60))
        
        # Accuracy trajectory
        first = accs[0] if accs else 0
        last = accs[-1] if accs else 0
        best = max(accs) if accs else 0
        worst = min(accs) if accs else 0
        change = last - first
        
        direction = "UP" if change > 0 else "DOWN"
        print("\n  Accuracy: R1=%.4f -> R%d=%.4f  (change=%+.4f, %s)" % (
            first, len(accs), last, change, direction))
        print("  Best=%.4f  Worst=%.4f" % (best, worst))
        print("  Trajectory: %s" % [round(a,4) for a in accs])
        
        # Loss trajectory
        if losses:
            print("  Loss: R1=%.4f -> R%d=%.4f" % (losses[0], len(losses), losses[-1]))
        
        # Participation
        if parts:
            avg_part = sum(parts) / len(parts)
            print("  Avg participation: %.1f%%  (constant=%s)" % (avg_part*100, len(set(parts))==1))
        else:
            avg_part = 0
        
        # Epsilon
        if avg_eps_list:
            avg_eps = sum(avg_eps_list) / len(avg_eps_list)
            print("  Avg epsilon: %.4f" % avg_eps)
        else:
            avg_eps = 0
        
        # Prices
        if prices:
            print("  Avg price: %.4f" % (sum(prices)/len(prices)))
        
        # Upload quality s
        if avg_s_list:
            print("  Avg s* (upload): %.4f" % (sum(avg_s_list)/len(avg_s_list)))
        
        # SNR estimate for noisy configs
        if avg_eps > 0 and cfg.get("method") != "FedMD":
            C = float(cfg.get("clip_bound", 2.0))
            noise_scale = 2 * C / avg_eps
            noise_var = 2 * noise_scale ** 2
            n_part = max(1, int(avg_part * 50))
            agg_var = noise_var / (n_part ** 2)
            snr = C / max(agg_var ** 0.5, 1e-10)
            print("  [SNR] C=%.1f, noise_scale=%.2f, noise_var=%.1f" % (C, noise_scale, noise_var))
            print("         n_part=%d, agg_var=%.4f, SNR~%.2f" % (n_part, agg_var, snr))
        
        summaries.append({
            "name": name,
            "first": first, "last": last, "change": change,
            "best": best, "avg_part": avg_part, "avg_eps": avg_eps,
            "elapsed": elapsed, "accs": accs
        })
    
    # ================================================================
    print("\n" + "=" * 75)
    print("COMPARISON TABLE")
    print("=" * 75)
    print("%-40s %6s %6s %7s %6s %5s %6s" % ("Config", "R1", "R20", "Chg", "Best", "Part", "Eps"))
    print("-" * 76)
    for s in summaries:
        short = s["name"][:38]
        print("%-40s %6.4f %6.4f %+7.4f %6.4f %4.0f%% %6.3f" % (
            short, s["first"], s["last"], s["change"], s["best"],
            s["avg_part"]*100, s["avg_eps"]))
    
    # ================================================================
    print("\n" + "=" * 75)
    print("DIAGNOSIS")
    print("=" * 75)
    
    oracle = None
    for s in summaries:
        if "C6" in s["name"]:
            oracle = s
    
    print("\n[1] ORACLE CHECK (C6: FedMD, no noise):")
    if oracle:
        print("    R1=%.4f -> R20=%.4f (%+.4f)" % (oracle["first"], oracle["last"], oracle["change"]))
        if oracle["change"] < 0:
            print("    *** ORACLE ALSO DEGRADES! ***")
            print("    => Problem is NOT just noise/SNR")
            print("    => v8 FD flow (fresh copy + pure KL) is FUNDAMENTALLY BROKEN")
    
    print("\n[2] ALL CONFIGS DEGRADE:")
    all_degrade = all(s["change"] < 0 for s in summaries)
    print("    All degrade: %s" % all_degrade)
    best_s = max(summaries, key=lambda s: s["last"])
    worst_s = min(summaries, key=lambda s: s["last"])
    print("    Best final:  %s -> %.4f" % (best_s["name"][:30], best_s["last"]))
    print("    Worst final: %s -> %.4f" % (worst_s["name"][:30], worst_s["last"]))
    
    print("\n[3] ROOT CAUSE:")
    print("    v8 design: each round starts from FRESH pretrained copy")
    print("    KL distillation alone destroys pretrained features")
    print("    Even noise-free oracle cannot maintain pretrain accuracy")
    print("    => The 'fresh copy + pure KL' design is the fundamental flaw")

if __name__ == "__main__":
    main()
