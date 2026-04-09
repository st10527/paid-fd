#!/usr/bin/env python3
"""Verify dedup and count runs needed."""
SEED_RUNS = []
for g in [3, 5, 7, 10]:
    for s in [42, 123, 456]:
        SEED_RUNS.append({"gamma": g, "seed": s, "lambda_mult": 1.0,
                          "label": "g%d_s%d" % (g, s)})
LAMBDA_RUNS = []
for lm in [0.5, 1.0, 2.0]:
    for g in [3, 5, 7, 10]:
        LAMBDA_RUNS.append({"gamma": g, "seed": 42, "lambda_mult": lm,
                            "label": "lm%s_g%d" % (lm, g)})

ALL = SEED_RUNS + LAMBDA_RUNS
print("Total before dedup: %d" % len(ALL))

seen = {}
deduped = []
for run in ALL:
    key = (run["gamma"], run["seed"], run["lambda_mult"])
    if key in seen:
        seen[key]["aliases"].append(run["label"])
    else:
        run["aliases"] = [run["label"]]
        seen[key] = run
        deduped.append(run)

print("After dedup: %d" % len(deduped))
for run in deduped:
    if len(run["aliases"]) > 1:
        print("  OVERLAP: %s" % str(run["aliases"]))

existing = {"g3_s42", "g3_s123", "lm0.5_g3"}
need = [r for r in deduped if not any(l in existing for l in r["aliases"])]
print("\nAlready done: %d" % (len(deduped) - len(need)))
print("Need to run: %d" % len(need))
print("Est time: ~%.1f hours" % (len(need) * 20 / 60))
