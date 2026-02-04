"""
Simple CNN models for CIFAR-100.

These are lightweight models suitable for edge devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN for CIFAR-100.
    
    Architecture:
    - 2 Conv blocks (Conv + BN + ReLU + MaxPool)
    - 2 FC layers
    
    ~200K parameters - suitable for resource-constrained devices.
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1: 3 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Conv block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Conv block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward - returns logits."""
        return self.forward(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature representation before classifier."""
        x = self.features(x)
        return x.view(x.size(0), -1)


class CNN_CIFAR(nn.Module):
    """
    Larger CNN for CIFAR-100 with better capacity.
    
    Architecture:
    - 4 Conv blocks with increasing channels
    - Global average pooling
    - 1 FC layer
    
    ~1.2M parameters - good balance of accuracy and efficiency.
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.layer1 = self._make_layer(in_channels, 64, dropout)
        self.layer2 = self._make_layer(64, 128, dropout)
        self.layer3 = self._make_layer(128, 256, dropout)
        self.layer4 = self._make_layer(256, 512, dropout)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self.num_classes = num_classes
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)  # 32 -> 16
        x = self.layer2(x)  # 16 -> 8
        x = self.layer3(x)  # 8 -> 4
        x = self.layer4(x)  # 4 -> 2
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class LeNet5(nn.Module):
    """
    LeNet-5 style network for CIFAR.
    
    Very lightweight (~60K params) - for extremely constrained devices.
    """
    
    def __init__(self, num_classes: int = 100):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
