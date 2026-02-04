#!/usr/bin/env python3
"""
Quick test script to verify installation.

Usage:
    python scripts/quick_test.py
    
    # Or from anywhere:
    python /path/to/paid-fd/scripts/quick_test.py
"""

import sys
import os
from pathlib import Path

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change to project root
os.chdir(PROJECT_ROOT)


def test_core_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from src.devices.heterogeneity import HeterogeneityGenerator
        from src.devices.energy import EnergyCalculator
        from src.game.stackelberg import StackelbergSolver
        from src.privacy.ldp import LaplaceDP, add_noise_to_logits
        from src.utils.results import ResultManager, ExperimentTracker
        from src.utils.seed import set_seed
        print("  ✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_torch_imports():
    """Test PyTorch-dependent imports."""
    print("Testing PyTorch imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
        
        from src.models import get_model, count_parameters
        from src.methods import PAIDFD, FixedEpsilon
        print("  ✓ Model and method imports successful")
        return True
    except ImportError as e:
        print(f"  ⚠ PyTorch not available: {e}")
        return False


def test_heterogeneity():
    """Test device heterogeneity generation."""
    print("Testing heterogeneity generator...")
    
    from src.devices.heterogeneity import HeterogeneityGenerator
    
    gen = HeterogeneityGenerator(n_devices=20, seed=42)
    devices = gen.generate()
    stats = gen.get_statistics(devices)
    
    print(f"  ✓ Generated {len(devices)} devices")
    print(f"  ✓ Types: {stats['type_distribution']}")
    return True


def test_stackelberg():
    """Test Stackelberg game solver."""
    print("Testing Stackelberg solver...")
    
    from src.devices.heterogeneity import HeterogeneityGenerator
    from src.game.stackelberg import StackelbergSolver
    import numpy as np
    
    gen = HeterogeneityGenerator(n_devices=20, seed=42)
    devices = gen.generate()
    
    # Assign data sizes
    for dev in devices:
        dev.data_size = np.random.randint(50, 200)
    
    solver = StackelbergSolver(gamma=10.0)
    result = solver.solve(devices)
    
    print(f"  ✓ Optimal price: {result['price']:.4f}")
    print(f"  ✓ Participation: {result['participation_rate']:.0%}")
    print(f"  ✓ Avg ε*: {result['avg_eps']:.3f}")
    return True


def test_ldp():
    """Test LDP mechanism."""
    print("Testing LDP privacy...")
    
    import numpy as np
    from src.privacy.ldp import add_noise_to_logits
    
    logits = np.random.randn(50, 100).astype(np.float32)
    noisy = add_noise_to_logits(logits, epsilon=1.0, clip_bound=5.0)
    
    noise_mag = np.abs(noisy - np.clip(logits, -5, 5)).mean()
    print(f"  ✓ Noise magnitude (ε=1.0): {noise_mag:.4f}")
    return True


def test_energy():
    """Test energy calculator."""
    print("Testing energy model...")
    
    from src.devices.energy import EnergyCalculator
    
    calc = EnergyCalculator()
    comparison = calc.compare_fd_vs_fl(cpu_freq=1.2, data_size=100, s_i=50)
    
    print(f"  ✓ FD savings: {comparison['fd_savings_percent']:.1f}%")
    return True


def main():
    print("=" * 60)
    print("PAID-FD Quick Test")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Core tests (no PyTorch required)
    all_passed &= test_core_imports()
    all_passed &= test_heterogeneity()
    all_passed &= test_stackelberg()
    all_passed &= test_ldp()
    all_passed &= test_energy()
    
    # PyTorch tests
    torch_ok = test_torch_imports()
    
    print()
    print("=" * 60)
    if all_passed:
        print("✓ All core tests passed!")
        if torch_ok:
            print("✓ PyTorch ready - can run full experiments")
        else:
            print("⚠ PyTorch not available - install for full functionality")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
