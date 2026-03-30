# v8.2 Phase 0.2 Results Analysis
## Class-Conditional Denoising Verification

**Date**: 2025-02-10
**Version**: v8.2
**Seed**: 42, 20 rounds, CIFAR-100, 50 devices, Dirichlet α=0.5

---

## 1. Summary Table

| Config | Description | R1 | Final | Best | Delta | Part | Verdict |
|--------|-------------|-----|-------|------|-------|------|---------|
| D1 | denoise + pure KL (γ=5) | 34.4% | 25.3% | 35.1% | **-9.1%** | 70% | ❌ |
| D2 | denoise + pure KL (γ=3) | 34.0% | 24.1% | 34.9% | **-9.9%** | 38% | ❌ |
| D3 | denoise + pure KL (γ=10) | 34.0% | 24.5% | 34.7% | **-9.5%** | 100% | ❌ |
| D4 | v8.0 no denoise (γ=5) | 39.3% | 38.7% | 40.5% | **-0.6%** | 70% | ✅ Stable |
| D5 | CE anchor α=0.5 (γ=5) | 43.4% | 45.0% | 45.1% | **+1.6%** | 70% | ✅ Best |
| D6 | denoise + CE α=0.3 (γ=5) | 40.2% | 43.4% | 43.4% | **+3.2%** | 70% | ✅ OK |
| D7 | FedMD oracle (no noise) | 43.5% | 41.2% | 43.5% | **-2.3%** | 100% | ⚠️ |
| D8 | FedMD oracle + denoise | 40.2% | 28.8% | 40.2% | **-11.4%** | 100% | 💀 |

---

## 2. Smoking Gun: D7 vs D8

**D8 is the critical finding.**

- D7 (FedMD, clean logits, no denoise): 43.5% → 41.2% (mild -2.3%)
- D8 (FedMD, clean logits, WITH denoise): 40.2% → 28.8% (**catastrophic -11.4%**)

**Class-conditional denoising destroys information even on perfectly clean logits.**

This proves the problem is NOT "denoising isn't strong enough." The denoising operation itself is fundamentally harmful.

---

## 3. Root Cause: Why Denoising Destroys Information

### The flawed theoretical assumption:

> "Each sample's logit = class signal + noise → class-conditional mean averages out noise."

### Why this is wrong:

In federated distillation, the "signal" is NOT just the class identity. It's the **per-sample soft probability distribution** — the "dark knowledge" (Hinton 2015).

Example: Consider two "dog" images:
1. A dog that looks like a cat → high P(cat), moderate P(dog)
2. A clearly recognizable dog → high P(dog), low everything else

These two samples have **legitimately different** logit vectors. Both are "signal," not noise.

Class-conditional mean replaces both with the average dog logit, destroying:
- Inter-sample variation within the same class
- Confusion patterns (which classes look alike)
- The entire point of soft labels vs hard labels

### Formal statement:

Let $z_i$ = true logit for sample $i$ of class $c$. The class-conditional mean is:

$$\bar{z}_c = \frac{1}{n_c} \sum_{i \in \text{class } c} z_i$$

This operation has information loss:

$$\text{Lost} = \sum_{i \in \text{class } c} \|z_i - \bar{z}_c\|^2$$

This "lost" information IS the dark knowledge. With CIFAR-100 having 200 samples/class, we collapse 200 distinct soft distributions into 1 prototype per class. The student essentially learns from 100 prototypes instead of 20,000 samples.

---

## 4. Good News: Fresh SGD Already Fixed Catastrophic Degradation

### Phase 0.1 (v8.0 with Adam, from previous experiment):
- C1 (γ=5): 25.0% → 7.5% over 20 rounds (catastrophic)
- C6 (FedMD oracle): 36.0% → 16.0% (even oracle collapsed)

### Phase 0.2 (v8.2 with fresh SGD, this experiment):
- D4 (γ=5, no denoise): 39.3% → 38.7% over 20 rounds (**stable!**)
- D7 (FedMD oracle): 43.5% → 41.2% (**mild drift only**)

**Conclusion: The Adam state leak was the primary cause of Phase 0.1's catastrophic failure. Fresh SGD per round resolves it.**

---

## 5. Analysis of Each Configuration

### D1-D3: Denoising + Pure KL (the v8.2 hypothesis)
All three degrade ~9-10% over 20 rounds. Denoising destroys dark knowledge, pure KL then trains on degraded signal. **The v8.2 hypothesis is falsified.**

γ differentiation is weak and inverted for D3:
- D2 (γ=3): 24.1% (worst, as expected — fewer participants)
- D1 (γ=5): 25.3%
- D3 (γ=10): 24.5% (should be best, but isn't)

### D4: v8.0 baseline (no denoise, no anchor)
Stable at 38.7%. Fresh SGD prevents collapse. But no improvement either — distillation with noisy logits adds slight noise each round.

### D5: CE anchor α=0.5 (no denoise)
Best result: 45.0%. Loss = 0.5 * CE(labels) + 0.5 * KL(noisy_teacher) * T².
The CE half trains correctly on public labels. The KL half contributes noisy knowledge transfer. Net effect is positive because CE dominates.

**BUT** — as noted in our Claude Web discussion: if CE dominates, γ is irrelevant. All γ values would give ~45% because CE doesn't depend on γ. This is the same trap as v7.

### D6: Denoise + CE α=0.3
43.4%. CE compensates for denoising damage. Lower α (0.3) means less CE anchor but enough to prevent collapse. The denoising still hurts (starts at 40.2% vs D5's 43.4%).

### D7/D8: Oracle sanity check
D7 (clean, no denoise): Mild drift (-2.3%). Expected — pure KL on clean logits without CE should slowly improve, but fresh model copy each round limits accumulation.

D8 (clean, WITH denoise): Catastrophic -11.4%. **Proves denoising is intrinsically harmful.**

---

## 6. Revised Understanding

### What works:
1. **Fresh SGD per round** → prevents catastrophic collapse ✅
2. **CE anchor** → provides stability and improvement via public labels ✅
3. **BLUE aggregation** → already handles noise optimally across devices ✅

### What doesn't work:
1. **Class-conditional denoising** → destroys dark knowledge ❌
2. **Pure KL on noisy logits** → mild degradation (not catastrophic with fresh SGD, but not helpful) ⚠️

### The fundamental tension:
- FD's value comes from **per-sample** soft distributions (dark knowledge)
- LDP noise corrupts these per-sample distributions
- Class-conditional mean removes noise BUT ALSO removes dark knowledge
- No free lunch: can't remove noise without losing signal at per-sample granularity

---

## 7. What Now? Options for Claude Web Discussion

### Option A: Accept CE Anchor as Primary Mechanism
- Use α * CE + (1-α) * KL with α ≈ 0.5
- Accept that most learning comes from CE on public labels
- γ differentiation through KL half (noisy but non-zero contribution)
- Risk: γ gap may be tiny (same as v7 concern)

### Option B: Per-Sample Shrinkage Estimator (instead of full mean replacement)
Instead of replacing with class mean, BLEND per-sample logit with class mean:

$$\hat{z}_i = (1-\lambda) \cdot z_i + \lambda \cdot \bar{z}_c$$

where $\lambda \in [0,1]$ controls the shrinkage strength. When noise is high, $\lambda → 1$ (more denoising). When noise is low, $\lambda → 0$ (preserve per-sample info).

Optimal James-Stein-style $\lambda$ depends on estimated SNR.

**γ mechanism**: Higher γ → more participants → lower post-BLUE noise → lower optimal $\lambda$ → more dark knowledge preserved → better accuracy.

### Option C: Selective Denoising (only denoise noisy classes)
Some classes have few private samples → high noise. Others have many → low noise. Only denoise the high-noise classes, preserving dark knowledge for the rest.

### Option D: BLUE Is Already Enough
BLUE already optimally combines noisy estimates. With 35 participants (γ=5), BLUE variance is:

$$\text{Var}_{\text{BLUE}} = \left(\sum_i w_i^2 \sigma_i^2\right) \approx \frac{\sigma^2}{35}$$

This is already a √35 ≈ 6× noise reduction. Combined with fresh SGD (no accumulation), the remaining noise may be tolerable. The question is: can we see γ differentiation through BLUE quality alone?

Test: Run D4 at γ=3/5/10 to see if BLUE-only shows γ effect.

### Option E: Noise-Aware Temperature Scaling
Instead of denoising the logits, adjust the temperature used in softmax based on estimated SNR:

$$T_{\text{effective}} = T \cdot \sqrt{1 + \text{noise\_var} / \text{signal\_var}}$$

Higher noise → higher temperature → softer distribution → noise has less effect.

---

## 8. Recommended Next Step

**Immediate test (Option D verification):**

Run v8.0-style (fresh SGD, no denoise, no CE anchor, pure KL) at γ = {3, 5, 10} for 30 rounds. This tests whether BLUE aggregation quality alone creates γ differentiation.

If yes → BLUE is the mechanism, no denoising needed.
If no → Need Option B (shrinkage estimator) to get γ differentiation.

**Remove denoising from codebase or make it opt-in with default OFF.**

---

## 9. Key Takeaway for Paper

The class-conditional denoising approach was theoretically motivated but empirically disproven. The fundamental insight is:

> **In FD, the signal IS the per-sample distribution, not the per-class identity. Any operation that collapses per-sample variation destroys the value proposition of knowledge distillation over simple label sharing.**

This is actually an important negative result that strengthens the paper — it explains WHY naive denoising fails and motivates more sophisticated approaches.
