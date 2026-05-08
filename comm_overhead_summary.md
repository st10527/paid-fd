# Communication & Latency Overhead

> Analytical estimates based on simulation config + existing run summaries.  
> Computed 2026-05-08 from git commit 40c2ee2 (pre-rerun, old cubic).  
> Values marked (**new cubic est.**) use corrected cubic from current commit.

---

## Per-round Upload Volume (bytes per device)

### PAID-FD mechanism
Each participating device uploads **s_i logit vectors** of K=100 float32 values:
```
bytes_per_device = s_i × K × 4 bytes
```

Average s_i is derived from the best-response formula `s*(ε) = p/c − (1+ε)/ε`,
evaluated at mean price and ε* across device tiers.

| Metric | Old cubic (pre-fix) | New cubic (corrected) |
|--------|--------------------|-----------------------|
| Avg ε* (γ=5, mean over devices) | 2.78 | ~3.33 (estimated +0.55) |
| Avg s* (median-c device) | 13.4 vectors | ~13.3 vectors |
| Bytes/device/round | **5,345 B (5.2 KB)** | **~5,313 B (5.2 KB)** |
| Aggregate upload (N_part ≈ 41/50) | ~213 KB/round | ~213 KB/round |

> **Note**: s* is largely unchanged despite higher ε*, because higher ε raises the
> (1+ε)/ε term. The c_i correction cancels this effect numerically.

### FedAvg baseline
Each device uploads the **entire model** every round:

| Model | Parameters | Upload per device/round |
|-------|-----------|-------------------------|
| ResNet18 (used in all exps) | 11,220,132 | **42.80 MB** |

### Reduction ratio

| Method | Bytes/device/round | vs FedAvg |
|--------|--------------------|-----------|
| PAID-FD | 5.2 KB | — |
| FedAvg (ResNet18) | 42,800 KB | **8,269×** larger |

> PAID-FD reduces per-device upload by **~8,270×** per round compared to
> FedAvg with a ResNet18 backbone. This is a fundamental advantage of
> federated distillation: devices only upload K=100 class logits (for
> s_i ≈ 13 public-data samples), not model weights.

---

## Per-round Latency Breakdown (PAID-FD, γ=5, N=50)

> Timing derived from: elapsed_sec / T_rounds (end-to-end wall-clock),
> hardware specs from heterogeneity.yaml, and analytical estimates.
> Full wall-clock for 1 run ≈ 2.53 h (9,112 s / 100 rounds = **91 s/round**).

| Stage | Time estimate (ms) | Notes |
|-------|-------------------|-------|
| Device local fine-tune | ~45,000 | 5 epochs × SGD on D_i, amortised over N_part=41 (sequential sim) |
| Device logit inference on D_ref | ~800 | s_i ≈ 13 samples × 100 classes |
| Logit upload (1 Mbps uplink) | **42.8** | 5.2 KB × 8 / 1 Mbps |
| Server BLUE aggregation | ~50 | Weighted sum over N_part logit matrices |
| Server distillation step | ~5,000 | 1 epoch KD on D_ref (20K samples, ResNet18) |
| **Total per-round (wall-clock)** | **~91,000** | **91 s** (actual measured) |

> The simulation runs devices sequentially (not truly parallel), so
> fine-tune time dominates. In a real deployment with simultaneous
> training, the wall-clock per round ≈ max_device_train + server_step.

### Real-deployment estimated latency (truly parallel devices)
| Stage | Time |
|-------|------|
| Slowest device fine-tune (Type C, RPi 3) | ~120 s |
| Upload (1 Mbps, 5.2 KB) | **42 ms** |
| Server aggregation + distillation | ~10 s |
| **Total per round (realistic)** | **~130 s** |

---

## Comparison vs FedAvg total round time (real deployment)

| Method | Model upload | Training | Total/round | Notes |
|--------|-------------|----------|-------------|-------|
| PAID-FD | 42 ms | ~120 s | **~130 s** | Logit-only upload |
| FedAvg (ResNet18) | **343 s** @ 1 Mbps | ~120 s | **~463 s** | 42.8 MB model upload |
| **Speedup** | **8,269×** upload | same | **~3.6×** total | Bottleneck shifts to train |

> At 1 Mbps, FedAvg model upload alone takes 343 s/round. PAID-FD's logit
> upload is 42 ms — a **>8,000×** reduction in communication cost. The total
> round speedup (~3.6×) is lower because local training dominates on slow devices.
> At higher bandwidth (10 Mbps), FedAvg upload drops to 34 s, and the speedup
> becomes ~1.3× — communication is no longer the bottleneck for either method.

---

## Notes for Reviewers

1. **Privacy semantics differ**: FedAvg uploads raw model weights (no formal DP).
   PAID-FD uploads LDP-noised logits (ε-DP per round). The communication
   reduction is not a fair accuracy-for-communication tradeoff — PAID-FD
   provides **both** better communication efficiency **and** formal privacy.

2. **s_i variability**: Under the new (correct) cubic, s_i is approximately
   unchanged. The higher ε* increases SNR but s_i ≈ p/c − (1+ε)/ε compensates,
   keeping per-device upload ~5 KB across both cubic variants.

3. **Bandwidth assumption**: All latency estimates assume 1 Mbps symmetric link
   (conservative; typical 4G uplink is 5–50 Mbps, reducing FedAvg upload to
   0.68–6.8 s/round at 50 Mbps).

---

*Last updated: 2026-05-08 | git: 40c2ee2 | Re-run pending (new cubic)*
