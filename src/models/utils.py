"""
Model utilities for PAID-FD.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


def get_model(
    name: str,
    num_classes: int = 100,
    **kwargs
) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        name: Model name ('resnet18', 'resnet34', 'cnn', 'simple_cnn')
        num_classes: Number of output classes
        **kwargs: Additional arguments for the model
        
    Returns:
        PyTorch model
    """
    from .cnn import SimpleCNN, CNN_CIFAR
    from .resnet import ResNet18, ResNet34
    
    models = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'cnn': CNN_CIFAR,
        'cnn_cifar': CNN_CIFAR,
        'simple_cnn': SimpleCNN,
        'simplecnn': SimpleCNN,
    }
    
    name_lower = name.lower()
    if name_lower not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name_lower](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def initialize_weights(model: nn.Module, method: str = 'kaiming'):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        method: Initialization method ('kaiming', 'xavier', 'normal')
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if method == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif method == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif method == 'normal':
                nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            if method == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif method == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif method == 'normal':
                nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in megabytes
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def copy_model(model: nn.Module, device: str = None) -> nn.Module:
    """
    Create a deep copy of a model.
    
    Args:
        model: PyTorch model
        device: Target device (None = same as original)
        
    Returns:
        Copy of the model on specified device
    """
    import copy
    model_copy = copy.deepcopy(model)
    
    if device is not None:
        model_copy = model_copy.to(device)
    
    return model_copy


def average_models(models: list, weights: Optional[list] = None) -> Dict[str, torch.Tensor]:
    """
    Average model parameters (for FedAvg-style aggregation).
    
    Args:
        models: List of models to average
        weights: Optional weights for weighted average
        
    Returns:
        Dictionary of averaged state dict
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    
    # Get state dicts
    state_dicts = [m.state_dict() for m in models]
    
    # Average
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )
    
    return avg_state


def model_difference_norm(model1: nn.Module, model2: nn.Module) -> float:
    """
    Compute L2 norm of the difference between two models.
    
    Useful for measuring model drift.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        L2 norm of parameter difference
    """
    diff_norm = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        diff_norm += torch.norm(p1 - p2).item() ** 2
    return diff_norm ** 0.5
