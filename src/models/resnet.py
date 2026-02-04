"""
ResNet models for CIFAR-100.

Modified for 32x32 images (smaller initial conv, no initial pooling).

Reference:
- Deep Residual Learning for Image Recognition (He et al., 2016)
"""

import torch
import torch.nn as nn
from typing import Type, List, Optional


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet for CIFAR (32x32 images).
    
    Modified from standard ResNet:
    - Initial conv: 3x3 with stride 1 (instead of 7x7 stride 2)
    - No initial max pooling
    """
    
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 100,
        in_channels: int = 3
    ):
        super().__init__()
        
        self.in_planes = 64
        self.num_classes = num_classes
        
        # Initial conv layer (modified for CIFAR)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature representation before FC layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class ResNet18(ResNet):
    """ResNet-18 for CIFAR-100 (~11M parameters)."""
    
    def __init__(self, num_classes: int = 100, in_channels: int = 3):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


class ResNet34(ResNet):
    """ResNet-34 for CIFAR-100 (~21M parameters)."""
    
    def __init__(self, num_classes: int = 100, in_channels: int = 3):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


# Convenience functions
def resnet18(num_classes: int = 100, **kwargs) -> ResNet18:
    """Create ResNet-18 model."""
    return ResNet18(num_classes=num_classes, **kwargs)


def resnet34(num_classes: int = 100, **kwargs) -> ResNet34:
    """Create ResNet-34 model."""
    return ResNet34(num_classes=num_classes, **kwargs)
