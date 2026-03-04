"""Quick check of Phase 1 data structure."""
import json

for fname, label in [
    ("results/experiments/phase1_gamma_seed42.json", "GAMMA"),
    ("results/experiments/phase1_lambda_seed42.json", "LAMBDA"),
]:
    print(f"=== {label} ===")
    with open(fname) as f:
        data = json.load(f)
    
    print(f"Top keys: {list(data.keys())}")
    runs = data["runs"]
    print(f"Runs type: {type(runs).__name__}, keys/len: {list(runs.keys()) if isinstance(runs, dict) else len(runs)}")
    
    for k in list(runs.keys())[:2]:
        seed_runs = runs[k]
        print(f"\n  Key={k}: {len(seed_runs)} runs, type={type(seed_runs[0]).__name__}")
        r0 = seed_runs[0]
        print(f"  Sub-keys: {list(r0.keys())}")
        print(f"  final_acc={r0.get('final_accuracy')}, best_acc={r0.get('best_accuracy')}")
        print(f"  n_accs={len(r0.get('accuracies', []))}")
        extras = r0.get("extras", [])
        if extras:
            print(f"  extras[0]: {extras[0]}")
        eh = r0.get("energy_history", [])
        if eh:
            print(f"  energy[0]: {eh[0]}")
        print(f"  part_rates[:3]={r0.get('participation_rates', [])[:3]}")
    print()
