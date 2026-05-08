# Channel Model Configuration (from actual simulation)

> Extracted from `config/devices/heterogeneity.yaml` — these are the exact values
> used in all PAID-FD simulation runs.

---

## Device Types (Hardware Tiers)

| Tier | Device Name | CPU Freq | Memory | Compute Cap | Power (idle/peak) | Ratio |
|------|-------------|----------|--------|-------------|-------------------|-------|
| Type A (high-end) | Jetson Nano | 1.5 GHz | 4096 MB | 1.0× | 5.0 / 10.0 W | 30% |
| Type B (mid-range) | Raspberry Pi 4 | 1.2 GHz | 4096 MB | 0.8× | 3.0 / 7.5 W | 40% |
| Type C (low-end / straggler) | Raspberry Pi 3 | 0.8 GHz | 1024 MB | 0.5× | 1.5 / 5.0 W | 30% |

---

## Energy Model

### Formula
```
E_train = κ_cpu × f_i² × C_train × |D_i| × epochs
E_inf   = κ_cpu × f_i² × C_inf   × s_i
E_comm  = P_tx  × (data_bits / rate)
```

### Parameters
| Parameter | Value | Units |
|-----------|-------|-------|
| κ_cpu (CPU energy coefficient) | 1.0 × 10⁻²⁸ | J/cycle³ |
| Cycles per training sample (C_train) | 1.0 × 10⁶ | cycles |
| Cycles per inference op (C_inf) | 1.0 × 10⁴ | cycles |
| Transmission power P_tx | 0.1 | W |
| Bandwidth B | 1.0 × 10⁶ | Hz (1 MHz) |

---

## Wireless Channel Model

### Formula
```
Rate = B × log2(1 + SNR × channel_gain)
channel_gain = (d / d_0)^(−η)   [path loss model]
```

### Parameters
| Parameter | Value |
|-----------|-------|
| Reference distance d₀ | 1.0 m |
| Path loss exponent η | 3.0 (typical urban: 2.7–3.5) |
| Device distance range | [10, 100] m from server (uniform) |
| Noise power | 1.0 × 10⁻¹⁰ W |
| Reference SNR at d₀ | 30 dB |

### Implied SNR range across devices
- At 10 m: SNR ≈ 30 − 10×3.0×log10(10/1.0) = 30 − 30 = **0 dB** (conservative lower bound)
- At 100 m: SNR ≈ 30 − 10×3.0×log10(100) = 30 − 60 = **−30 dB** (outer edge)
- In practice simulation clips to positive SNR → effective range **≈ [0, 30] dB**

---

## Cost Function Parameters

> Marginal cost c_i = c_inf_i + c_comm_i, normalized for game dynamics.

### Inference cost per device tier
| Tier | c_inf multiplier | c_inf (base = 0.1) |
|------|------------------|--------------------|
| Type A (Jetson Nano) | 0.5 | 0.05 |
| Type B (RPi 4) | 1.0 | 0.10 |
| Type C (RPi 3) | 2.0 | 0.20 |

### Communication cost
| Parameter | Value |
|-----------|-------|
| c_comm range | [0.05, 0.20] (uniform) |
| c_comm mid-point | 0.125 |

### Total marginal cost c_i range
| Tier | c_inf | c_comm (range) | Total c_i range |
|------|-------|----------------|-----------------|
| Type A | 0.05 | [0.05, 0.20] | **[0.10, 0.25]** |
| Type B | 0.10 | [0.05, 0.20] | **[0.15, 0.30]** |
| Type C | 0.20 | [0.05, 0.20] | **[0.25, 0.40]** |

**Overall c_i range across all devices: [0.10, 0.40]** (consistent with cubic fix test range)

---

## Privacy Sensitivity Distribution

> λ_i represents device i's cost per unit of ε. Higher λ → more noise preferred.

| Level | Base λ | Ratio | Description |
|-------|--------|-------|-------------|
| Very low | 0.05 | 15% | Public data devices (weather stations, etc.) |
| Low | 0.15 | 25% | Non-sensitive (environmental sensors, etc.) |
| Medium | 0.40 | 25% | Moderate sensitivity (activity recognition) |
| High | 0.80 | 20% | Sensitive data (health monitoring) |
| Very high | 1.50 | 15% | Highly sensitive (medical devices) |

- Per-device jitter: λ_i = λ_base × Uniform(0.7, 1.3) (±30%)
- `lambda_mult` experiment parameter scales all λ_i uniformly (used in Fig 5)

---

## Heterogeneity Summary for Paper

| Parameter | Value | Source |
|-----------|-------|--------|
| N devices | 50 (default) | Config |
| Device tiers | 3 (30/40/30%) | Config |
| CPU freq range | [0.8, 1.5] GHz | Config |
| Cost c_i range | [0.10, 0.40] | Derived |
| Privacy λ_i range | [0.035, 1.95] (with ±30% jitter) | Config |
| Path loss exponent | 3.0 | Config |
| Distance range | [10, 100] m | Config |
| Bandwidth | 1 MHz | Config |

---

*Last updated: 2026-05-08 (cubic solver bug fix, git: 40c2ee2)*
