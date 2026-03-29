# Phase 0.1 Diagnostic Report — v8 Distillation Failure Analysis

**Date**: 2026-03-29  
**Experiment**: `v8_phase0_diagnostic.json` (6 configs × 20 rounds, seed=42)

---

## 1. Summary Table

| Config | Description | R1 acc | R20 acc | Δ from pretrain | R1→R20 |
|--------|-------------|--------|---------|-----------------|--------|
| Pretrain | — | 44.4% | — | baseline | — |
| C1 | C=1.0, lr=0.001 | 25.0% | 7.5% | -36.9pp | -17.5pp |
| C2 | C=2.0, lr=0.0001 | 40.2% | 16.4% | -28.0pp | -23.8pp |
| C3 | C=1.0, lr=0.0001 | 37.0% | 14.7% | -29.7pp | -22.3pp |
| C4 | λ×0.1 (ε≈0.78) | 31.9% | 9.2% | -35.2pp | -22.7pp |
| C5 | λ×0.1+C=1.0+lr=1e-4 | 37.1% | 15.4% | -29.0pp | -21.7pp |
| **C6** | **FedMD oracle (no noise)** | **36.0%** | **16.0%** | **-28.4pp** | **-20.0pp** |

**Key fact**: ALL 6 configs degrade **monotonically** over 20 rounds. Even the
noise-free oracle (C6) drops from 36% → 16%.

---

## 2. Critical Finding: Problem is NOT Noise

C6 (FedMD, no noise, no game) also degrades catastrophically:
```
C6 trajectory: [36.0, 36.3, 34.3, 32.0, 32.3, 29.8, 27.8, 27.3, 
                24.5, 25.1, 22.3, 21.6, 20.3, 18.4, 17.2, 19.3, 
                17.9, 17.5, 16.8, 16.0]
```

This proves the **v8 FD flow itself is fundamentally broken**, regardless of
noise. The SNR analysis from Phase 0 was a red herring — while noise makes
things worse, the core problem is in the distillation mechanism.

---

## 3. Root Cause Analysis: TWO BUGS

### Bug 1: Adam Optimizer State Leak (CRITICAL)

In both `paid_fd.py` and `fedmd.py`:
```python
# __init__() — created ONCE
self.distill_optimizer = torch.optim.Adam(
    self.server_model.parameters(), lr=self.config.distill_lr
)

# _distill_to_server() — reused EVERY round
optimizer = self.distill_optimizer  # ← same Adam with accumulated state!
```

Adam maintains per-parameter running estimates:
- `m_t` = first moment (mean of gradients)
- `v_t` = second moment (mean of squared gradients)

These **accumulate across rounds**, causing:
- Round 1: Fresh Adam, cautious steps → moderate damage
- Round 10: Accumulated momentum, larger effective lr → more damage
- Round 20: Adam "locked in" to a destructive direction

This explains the **monotonic degradation** — even with identical inputs each
round, the optimizer state makes each subsequent round more destructive.

### Bug 2: Cumulative Server Model Drift

The server model is **never reset**:
```
Round 1: pretrained_model → distill → model₁ (slightly degraded)
Round 2: model₁ → devices copy this → local train → logits → distill → model₂ (more degraded)
Round 3: model₂ → ... → model₃ (even more degraded)
```

Each round, devices start from an **increasingly degraded** server model.
Their local training can't compensate, so the aggregated teacher logits
come from worse starting points. This creates a death spiral.

### Bug 3 (Design): No Anchor Loss

Pure KL divergence loss:
```python
loss = KL(student || teacher) * T²
```

With T=3: loss is multiplied by **9×**. There is **no ground-truth anchor** —
nothing prevents the model from drifting arbitrarily far from the pretrained
representation. In standard knowledge distillation, the student also has a CE
loss on true labels as anchor. We have none.

---

## 4. Why Lower lr Helps R1 but Doesn't Fix the Trend

C2 (lr=0.0001) preserves 40.2% after R1 vs C1's 25.0%:
- 10× smaller lr → 10× smaller parameter updates per step
- Round 1 distillation barely changes the model → preserves pretrain

But over 20 rounds, C2 **still** drops to 16.4% because:
- Adam accumulates momentum regardless of lr
- Each round adds ~1pp of cumulative drift
- The lr only controls the speed, not the destination

---

## 5. Proposed Fix: v8.1

Three changes, each addressing one root cause:

### Fix 1: Reset Optimizer Each Round
```python
# In _distill_to_server(), create FRESH optimizer each time
def _distill_to_server(self, teacher_probs, public_images):
    optimizer = torch.optim.SGD(              # SGD, not Adam
        self.server_model.parameters(),
        lr=self.config.distill_lr,
        momentum=0.9, weight_decay=5e-4
    )
```
- SGD has no persistent state issues
- Or: create fresh Adam each round (but SGD is simpler)

### Fix 2: Add CE Anchor Loss
```python
# Blended loss: anchor to public data truth + learn from teacher
loss = alpha * CE(model(x), true_labels) + (1-alpha) * KL(model, teacher) * T²
```
- `alpha ∈ [0, 1]` controls anchor strength
- `alpha = 0.5` → equal weight to ground truth and teacher
- Prevents drift: model always stays close to pretrained performance on public data
- The teacher logits ADD knowledge from private data ON TOP of this anchor

### Fix 3: Moderate the T² Scaling
- With T=3, KL loss is multiplied by 9
- Consider T=2 (4× scaling) or even T=1 (no scaling)
- Or: adjust alpha to compensate

### Expected Outcome
- CE anchor ensures model never drops below pretrain level on public data
- Teacher logits provide ADDITIONAL knowledge from private data
- Higher γ → more/better participants → better teacher → more improvement above pretrain
- This is exactly the γ-accuracy gap we need for the paper

---

## 6. Verification Plan

Create `run_phase0_v8_1.py` with configs:

| Config | Description | 
|--------|-------------|
| V1 | v8.1: SGD + CE anchor (α=0.5) |
| V2 | v8.1: SGD + CE anchor (α=0.3) |  
| V3 | v8.1: fresh Adam + CE anchor (α=0.5) |
| V4 | v8.1: SGD only (no CE anchor, α=0) |
| V5 | v8.1: CE anchor only (old Adam, α=0.5) |
| V6 | FedMD oracle with same v8.1 fixes |

If V1/V2 show **improvement** over rounds (or at least stability), the fix works.
If V6 also improves, the core FD mechanism is restored.

---

## 7. Impact on Paper

This fix is **theory-compatible**:
- The Stackelberg game is unchanged
- Device decisions (ε*, s*) are unchanged
- The only change is how the server uses aggregated logits
- CE anchor is standard practice in knowledge distillation (Hinton 2015)
- We can present this as "anchored federated distillation"

The γ-accuracy mechanism:
```
γ ↑ → price p* ↑ → more devices participate (participation_rate ↑)
  → aggregated teacher logits have more signal, less noise variance
  → KL component provides better guidance
  → (1-α) * KL adds more improvement above CE anchor
  → final accuracy ↑
```
