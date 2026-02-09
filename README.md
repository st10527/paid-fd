# PAID-FD: Privacy-Aware Incentive-Driven Federated Distillation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simulation framework for **Privacy-Aware Incentive-Driven Federated Distillation** with Stackelberg game-based incentive mechanism.

> 📝 **Paper**: Submitted to IEEE Transactions on Mobile Computing (TMC)

## 🌟 Highlights

- **Stackelberg Game Mechanism**: One-shot broadcast pricing with optimal device response
- **Three-Dimensional Heterogeneity**: Communication, privacy sensitivity, and computation costs
- **Adaptive Privacy**: Device-specific ε allocation via game equilibrium  
- **Energy Efficient**: ~99% energy savings compared to traditional FL
- **Cross-Domain Distillation**: CIFAR-100 (private) + STL-10 (public)

## 📁 Project Structure

```
paid_fd/
├── config/                     # YAML configurations
│   ├── default.yaml            # Default settings
│   ├── experiments/            # Experiment configs
│   └── methods/                # Method configs
├── src/
│   ├── data/                   # Data loading & partitioning
│   │   ├── datasets.py         # CIFAR-100, STL-10, Synthetic
│   │   └── partition.py        # Dirichlet Non-IID
│   ├── devices/                # Device simulation
│   │   ├── heterogeneity.py    # 3D heterogeneity model
│   │   └── energy.py           # Energy consumption
│   ├── game/                   # Game theory
│   │   ├── stackelberg.py      # Algorithms 1 & 2
│   │   └── utility.py          # Quality functions
│   ├── privacy/                # Privacy mechanisms
│   │   └── ldp.py              # Laplace/Gaussian DP
│   ├── models/                 # Neural networks
│   │   ├── resnet.py           # ResNet-18/34
│   │   └── cnn.py              # Lightweight CNNs
│   ├── methods/                # FL methods
│   │   ├── paid_fd.py          # Our method
│   │   └── fixed_eps.py        # Ablation baseline
│   └── utils/                  # Utilities
├── experiments/
│   └── run_experiment.py       # Unified runner
├── results/                    # Output directory
├── scripts/                    # Helper scripts
└── tests/                      # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/paid-fd.git
cd paid-fd

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Quick test with synthetic data (no download, CPU friendly)
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method PAID-FD \
    --synthetic \
    --rounds 10

# Full experiment with real data (requires GPU)
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method PAID-FD

# Run multiple methods
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method all

# Specify device
python experiments/run_experiment.py \
    --config exp2_convergence \
    --method PAID-FD \
    --device cuda:0
```

### Test Core Components (No PyTorch Required)

```python
from src.devices.heterogeneity import HeterogeneityGenerator
from src.game.stackelberg import StackelbergSolver

# Generate 50 heterogeneous devices
gen = HeterogeneityGenerator(n_devices=50, seed=42)
devices = gen.generate()

# Solve Stackelberg game
solver = StackelbergSolver(gamma=10.0)
result = solver.solve(devices)

print(f"Optimal price: {result['price']:.4f}")
print(f"Participation: {result['participation_rate']:.0%}")
print(f"Avg ε*: {result['avg_eps']:.3f}")
```

## ⚙️ Configuration

Edit `config/default.yaml`:

```yaml
system:
  n_devices: 50
  seed: 42

data:
  partition:
    method: dirichlet
    alpha: 0.5  # Non-IID level

paid_fd:
  gamma: 10.0      # Server valuation
  clip_bound: 5.0  # LDP clipping

training:
  n_rounds: 200
  local_epochs: 1
  distill_epochs: 5
```

## 📊 Experiments

| Exp | Description | Config |
|-----|-------------|--------|
| 1 | Algorithm Efficiency | `exp1_efficiency.yaml` |
| 2 | Convergence & Accuracy | `exp2_convergence.yaml` |
| 3 | Privacy-Accuracy Tradeoff | `exp3_privacy.yaml` |
| 4 | Energy Analysis | `exp4_energy.yaml` |
| 5 | Heterogeneity Impact | `exp5_heterogeneity.yaml` |
| 6 | Incentive Analysis | `exp6_incentive.yaml` |
| 7 | Scalability | `exp7_scalability.yaml` |

## 📈 Results

Results are saved to `results/experiments/{exp_name}/`:

```python
from src.utils.results import ResultManager

manager = ResultManager()

# List all results
files = manager.list_results("exp2_convergence")

# Compare methods
comparison = manager.compare_results("exp2_convergence", metric="final_accuracy")
print(comparison)
```

## 🔧 Methods

| Method | Description | Type | Status |
|--------|-------------|------|--------|
| **PAID-FD** | Stackelberg game + adaptive ε (ours) | FD + LDP | ✅ |
| **Fixed-ε** | Fixed privacy budget ablation | FD + LDP | ✅ |
| **FedMD** | FD baseline, no privacy (Li & Wang, NeurIPS 2019) | FD | ✅ |
| **FedAvg** | Parameter averaging (McMahan et al., 2017) | Param-Avg | ✅ |
| **CSRA** | Reverse auction DPFL (Yang et al., TIFS 2024) | Param-Avg + DP | ✅ |
| **FedGMKD** | GMM prototype KD + DAT (Zhang et al., 2024) | Prototype | ✅ |

## 💻 Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Development | CPU | GPU (any) |
| Single Experiment | GTX 1080 | RTX 3090 |
| Full Experiments | RTX 3090 | A100 / Multi-GPU |

**Memory**: ~3GB VRAM for ResNet-18 + CIFAR-100

## 📄 Citation

```bibtex
@article{paid_fd_2026,
  title={Privacy-Aware Incentive-Driven Federated Distillation 
         for Heterogeneous Edge Networks},
  author={...},
  journal={IEEE Transactions on Mobile Computing},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CIFAR-100 and STL-10 datasets
- PyTorch team
- Federated learning community
