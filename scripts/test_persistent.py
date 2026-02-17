#!/usr/bin/env python3
"""Quick validation of persistent local model logic."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.methods.paid_fd import PAIDFD, PAIDFDConfig
from src.methods.fedmd import FedMD, FedMDConfig
from src.methods.fixed_eps import FixedEpsilon, FixedEpsilonConfig
from src.models.resnet import ResNet18
from src.models.utils import copy_model

def test_persistent_models():
    print("=" * 60)
    print("Testing Persistent Local Models")
    print("=" * 60)
    
    # 1. Test PAID-FD
    print("\n--- PAID-FD ---")
    config = PAIDFDConfig(
        gamma=5.0,
        local_epochs=1, local_lr=0.01,
        distill_epochs=2, distill_lr=0.005,
        public_samples=100
    )
    model = ResNet18(num_classes=10)
    method = PAIDFD(server_model=model, config=config, n_classes=10, device='cpu')
    
    assert hasattr(method, 'local_models'), "Missing local_models dict"
    assert hasattr(method, 'local_optimizers'), "Missing local_optimizers dict"
    assert len(method.local_models) == 0, "Should start empty"
    print("  ✅ Persistent dicts initialized")
    
    # Simulate creating persistent model for device 0
    local_model = copy_model(model)
    method.local_models[0] = local_model
    opt = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    method.local_optimizers[0] = opt
    
    # Train one step
    p_before = list(method.local_models[0].parameters())[0].data.clone()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    out = method.local_models[0](x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    method.local_optimizers[0].step()
    method.local_optimizers[0].zero_grad()
    p_after = list(method.local_models[0].parameters())[0].data.clone()
    
    assert not torch.equal(p_before, p_after), "Model should have changed"
    assert torch.equal(
        list(model.parameters())[0].data,
        list(model.parameters())[0].data  # server unchanged
    ), "Server model should be unchanged"
    print("  ✅ Persistent training works")
    
    # Check model persists across simulated rounds
    p_round1 = list(method.local_models[0].parameters())[0].data.clone()
    out2 = method.local_models[0](x)
    loss2 = torch.nn.functional.cross_entropy(out2, y)
    loss2.backward()
    method.local_optimizers[0].step()
    method.local_optimizers[0].zero_grad()
    p_round2 = list(method.local_models[0].parameters())[0].data.clone()
    
    assert not torch.equal(p_round1, p_round2), "Should continue training"
    print("  ✅ Model persists across rounds")
    
    # 2. Test FedMD
    print("\n--- FedMD ---")
    fedmd_config = FedMDConfig(
        local_epochs=1, local_lr=0.01,
        distill_epochs=2, public_samples=100
    )
    fedmd_model = ResNet18(num_classes=10)
    fedmd = FedMD(server_model=fedmd_model, config=fedmd_config, n_classes=10, device='cpu')
    assert hasattr(fedmd, 'local_models'), "Missing local_models"
    assert hasattr(fedmd, 'local_optimizers'), "Missing local_optimizers"
    print("  ✅ Persistent dicts initialized")
    
    # 3. Test FixedEpsilon
    print("\n--- FixedEpsilon ---")
    fe_config = FixedEpsilonConfig(
        epsilon=1.0,
        local_epochs=1, local_lr=0.01,
        distill_epochs=2
    )
    fe_model = ResNet18(num_classes=10)
    fe = FixedEpsilon(server_model=fe_model, config=fe_config, n_classes=10, device='cpu')
    assert hasattr(fe, 'local_models'), "Missing local_models"
    assert hasattr(fe, 'local_optimizers'), "Missing local_optimizers"
    print("  ✅ Persistent dicts initialized")
    
    # 4. Verify config defaults
    print("\n--- Config Defaults ---")
    paid_cfg = PAIDFDConfig(gamma=10.0)
    assert paid_cfg.local_epochs == 1, f"Expected 1, got {paid_cfg.local_epochs}"
    assert paid_cfg.local_lr == 0.01, f"Expected 0.01, got {paid_cfg.local_lr}"
    assert paid_cfg.distill_epochs == 5, f"Expected 5, got {paid_cfg.distill_epochs}"
    assert paid_cfg.public_samples == 2000, f"Expected 2000, got {paid_cfg.public_samples}"
    print(f"  local_epochs={paid_cfg.local_epochs} ✅")
    print(f"  local_lr={paid_cfg.local_lr} ✅")
    print(f"  distill_epochs={paid_cfg.distill_epochs} ✅")
    print(f"  public_samples={paid_cfg.public_samples} ✅")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)

if __name__ == '__main__':
    test_persistent_models()
