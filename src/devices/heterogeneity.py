"""
Three-Dimensional Device Heterogeneity Generator

Models heterogeneity across:
1. Communication cost (c_i^comm): channel-dependent, continuous
2. Privacy sensitivity (λ_i): user-dependent, discrete levels  
3. Computation cost (c_i^inf): hardware-dependent, device types

Based on real edge device specifications:
- Type A: NVIDIA Jetson Nano (high-end)
- Type B: Raspberry Pi 4 4GB (mid-range)
- Type C: Raspberry Pi 3 (low-end, potential straggler)
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from enum import Enum


class DeviceType(Enum):
    """Edge device types based on real hardware."""
    TYPE_A = "jetson_nano"    # High-end: fast inference
    TYPE_B = "rpi4_4gb"       # Mid-range: baseline
    TYPE_C = "rpi3"           # Low-end: potential straggler


class PrivacyLevel(Enum):
    """Privacy sensitivity levels."""
    LOW = 0.1      # Not privacy-sensitive
    MEDIUM = 0.5   # Moderately sensitive
    HIGH = 1.0     # Highly sensitive (e.g., medical data)


@dataclass
class DeviceProfile:
    """
    Complete profile for a simulated edge device.
    
    Attributes:
        device_id: Unique identifier
        device_type: Hardware type (A/B/C)
        c_comm: Communication cost coefficient
        c_inf: Inference cost coefficient  
        lambda_i: Privacy sensitivity coefficient
        cpu_freq: CPU frequency in GHz
        memory: Memory in MB
        data_size: Number of local training samples
        distance: Distance to edge server (meters)
        channel_gain: Current channel gain (updated per round)
    """
    device_id: int
    device_type: DeviceType
    
    # Three-dimensional heterogeneity
    c_comm: float           # Communication cost
    c_inf: float            # Inference cost
    lambda_i: float         # Privacy sensitivity
    
    # Hardware specs
    cpu_freq: float = 1.0   # GHz
    memory: int = 2048      # MB
    
    # Data characteristics
    data_size: int = 0      # Set by data partitioner
    data_classes: List[int] = field(default_factory=list)  # Classes present in local data
    
    # Wireless parameters
    distance: float = 50.0  # meters
    channel_gain: float = 1.0  # Updated per round
    
    @property
    def c_total(self) -> float:
        """Aggregate marginal cost (for Stackelberg game)."""
        return self.c_comm + self.c_inf
    
    @property
    def is_straggler(self) -> bool:
        """Check if device is a potential straggler."""
        return self.device_type == DeviceType.TYPE_C
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['device_type'] = self.device_type.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DeviceProfile':
        """Create from dictionary."""
        d = d.copy()
        d['device_type'] = DeviceType(d['device_type'])
        return cls(**d)


class HeterogeneityGenerator:
    """
    Generates heterogeneous device profiles for simulation.
    
    Configuration follows the experimental plan:
    - Device types: 30% Type-A, 40% Type-B, 30% Type-C
    - Privacy levels: 40% low, 40% medium, 20% high
    - Communication costs: Uniform in [c_min, c_max]
    
    Usage:
        generator = HeterogeneityGenerator(n_devices=50, seed=42)
        devices = generator.generate()
    """
    
    # Device type configurations based on real hardware
    DEVICE_CONFIGS = {
        DeviceType.TYPE_A: {
            "name": "Jetson Nano",
            "cpu_freq": 1.5,      # GHz (quad-core ARM Cortex-A57)
            "memory": 4096,       # MB
            "c_inf_mult": 0.5,    # Relative to baseline
            "ratio": 0.30,        # 30% of devices
        },
        DeviceType.TYPE_B: {
            "name": "RPi 4 (4GB)",
            "cpu_freq": 1.2,      # GHz (quad-core Cortex-A72)
            "memory": 4096,       # MB
            "c_inf_mult": 1.0,    # Baseline
            "ratio": 0.40,        # 40% of devices
        },
        DeviceType.TYPE_C: {
            "name": "RPi 3",
            "cpu_freq": 0.8,      # GHz (quad-core Cortex-A53)
            "memory": 1024,       # MB
            "c_inf_mult": 2.0,    # 2x slower
            "ratio": 0.30,        # 30% of devices
        }
    }
    
    # Privacy sensitivity distribution
    # Note: Lower lambda = less privacy-sensitive = willing to provide higher ε
    PRIVACY_CONFIG = {
        PrivacyLevel.LOW: {"value": 0.01, "ratio": 0.40},     # Very low sensitivity
        PrivacyLevel.MEDIUM: {"value": 0.05, "ratio": 0.40},  # Low sensitivity  
        PrivacyLevel.HIGH: {"value": 0.1, "ratio": 0.20},     # Moderate sensitivity
    }
    
    def __init__(
        self,
        n_devices: int,
        c_inf_base: float = 0.01,      # Lowered for more samples
        c_comm_range: Tuple[float, float] = (0.005, 0.02),  # Lowered
        distance_range: Tuple[float, float] = (10.0, 100.0),
        seed: int = 42
    ):
        """
        Initialize the generator.
        
        Args:
            n_devices: Number of devices to generate
            c_inf_base: Base inference cost coefficient (normalized, ~0.1)
            c_comm_range: Range for communication cost (normalized)
            distance_range: Range for device distances (meters)
            seed: Random seed for reproducibility
        """
        self.n_devices = n_devices
        self.c_inf_base = c_inf_base
        self.c_comm_range = c_comm_range
        self.distance_range = distance_range
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> List[DeviceProfile]:
        """
        Generate all device profiles.
        
        Returns:
            List of DeviceProfile instances
        """
        devices = []
        
        # Assign device types according to ratios
        device_types = self._assign_device_types()
        
        # Assign privacy sensitivities
        lambdas = self._assign_privacy_levels()
        
        for i in range(self.n_devices):
            dtype = device_types[i]
            config = self.DEVICE_CONFIGS[dtype]
            
            # Communication cost (channel-dependent, continuous)
            c_comm = self.rng.uniform(*self.c_comm_range)
            
            # Inference cost (hardware-dependent, from config)
            c_inf = self.c_inf_base * config["c_inf_mult"]
            
            # Distance (affects channel gain)
            distance = self.rng.uniform(*self.distance_range)
            
            device = DeviceProfile(
                device_id=i,
                device_type=dtype,
                c_comm=c_comm,
                c_inf=c_inf,
                lambda_i=lambdas[i],
                cpu_freq=config["cpu_freq"],
                memory=config["memory"],
                distance=distance,
                channel_gain=self._compute_channel_gain(distance)
            )
            devices.append(device)
        
        return devices
    
    def _assign_device_types(self) -> List[DeviceType]:
        """Assign device types according to specified ratios."""
        types = []
        
        for dtype, config in self.DEVICE_CONFIGS.items():
            count = int(self.n_devices * config["ratio"])
            types.extend([dtype] * count)
        
        # Handle rounding - fill remaining with Type B (most common)
        while len(types) < self.n_devices:
            types.append(DeviceType.TYPE_B)
        
        # Shuffle to randomize positions
        self.rng.shuffle(types)
        return types[:self.n_devices]
    
    def _assign_privacy_levels(self) -> List[float]:
        """Assign privacy sensitivities according to specified ratios."""
        lambdas = []
        
        for level, config in self.PRIVACY_CONFIG.items():
            count = int(self.n_devices * config["ratio"])
            lambdas.extend([config["value"]] * count)
        
        # Handle rounding
        while len(lambdas) < self.n_devices:
            lambdas.append(PrivacyLevel.MEDIUM.value)
        
        self.rng.shuffle(lambdas)
        return lambdas[:self.n_devices]
    
    def _compute_channel_gain(self, distance: float, alpha: float = 3.0) -> float:
        """
        Compute path loss based channel gain.
        
        Args:
            distance: Distance in meters
            alpha: Path loss exponent (typically 2-4)
            
        Returns:
            Channel gain (normalized)
        """
        # Simple path loss model: g = (d0/d)^alpha
        d0 = 1.0  # Reference distance
        return (d0 / max(distance, d0)) ** alpha
    
    def update_channel_gains(
        self, 
        devices: List[DeviceProfile],
        fading: bool = True
    ) -> List[DeviceProfile]:
        """
        Update channel gains for all devices (call each round).
        
        Args:
            devices: List of device profiles
            fading: Whether to include Rayleigh fading
            
        Returns:
            Updated device profiles
        """
        for device in devices:
            base_gain = self._compute_channel_gain(device.distance)
            
            if fading:
                # Rayleigh fading: |h|^2 is exponentially distributed
                fading_gain = self.rng.exponential(1.0)
                device.channel_gain = base_gain * fading_gain
            else:
                device.channel_gain = base_gain
            
            # Update communication cost based on channel gain
            # Lower channel gain -> higher communication cost
            device.c_comm = self.c_comm_range[0] + (
                self.c_comm_range[1] - self.c_comm_range[0]
            ) * (1.0 - min(device.channel_gain, 1.0))
        
        return devices
    
    def get_statistics(self, devices: List[DeviceProfile]) -> Dict:
        """
        Compute statistics of the device population.
        
        Returns:
            Dictionary with various statistics
        """
        stats = {
            "n_devices": len(devices),
            "type_distribution": {},
            "lambda_stats": {},
            "c_comm_stats": {},
            "c_inf_stats": {},
            "c_total_stats": {},
            "straggler_ratio": 0.0
        }
        
        # Type distribution
        for dtype in DeviceType:
            count = sum(1 for d in devices if d.device_type == dtype)
            stats["type_distribution"][dtype.value] = count
        
        # Lambda statistics
        lambdas = [d.lambda_i for d in devices]
        stats["lambda_stats"] = {
            "mean": float(np.mean(lambdas)),
            "std": float(np.std(lambdas)),
            "min": float(np.min(lambdas)),
            "max": float(np.max(lambdas)),
            "distribution": {
                "low": sum(1 for l in lambdas if l <= 0.2),
                "medium": sum(1 for l in lambdas if 0.2 < l <= 0.7),
                "high": sum(1 for l in lambdas if l > 0.7)
            }
        }
        
        # Cost statistics
        c_comms = [d.c_comm for d in devices]
        c_infs = [d.c_inf for d in devices]
        c_totals = [d.c_total for d in devices]
        
        for name, values in [("c_comm", c_comms), ("c_inf", c_infs), ("c_total", c_totals)]:
            stats[f"{name}_stats"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
        
        # Straggler ratio
        stats["straggler_ratio"] = sum(1 for d in devices if d.is_straggler) / len(devices)
        
        return stats
    
    def create_custom_distribution(
        self,
        type_ratios: Dict[str, float] = None,
        lambda_ratios: Dict[str, float] = None
    ) -> List[DeviceProfile]:
        """
        Create devices with custom type and privacy distributions.
        
        Useful for Exp 5: Heterogeneity Analysis.
        
        Args:
            type_ratios: {"type_a": 0.2, "type_b": 0.5, "type_c": 0.3}
            lambda_ratios: {"low": 0.3, "medium": 0.4, "high": 0.3}
        """
        # Temporarily update configs
        if type_ratios:
            self.DEVICE_CONFIGS[DeviceType.TYPE_A]["ratio"] = type_ratios.get("type_a", 0.3)
            self.DEVICE_CONFIGS[DeviceType.TYPE_B]["ratio"] = type_ratios.get("type_b", 0.4)
            self.DEVICE_CONFIGS[DeviceType.TYPE_C]["ratio"] = type_ratios.get("type_c", 0.3)
        
        if lambda_ratios:
            self.PRIVACY_CONFIG[PrivacyLevel.LOW]["ratio"] = lambda_ratios.get("low", 0.4)
            self.PRIVACY_CONFIG[PrivacyLevel.MEDIUM]["ratio"] = lambda_ratios.get("medium", 0.4)
            self.PRIVACY_CONFIG[PrivacyLevel.HIGH]["ratio"] = lambda_ratios.get("high", 0.2)
        
        devices = self.generate()
        
        # Reset to defaults
        self.DEVICE_CONFIGS[DeviceType.TYPE_A]["ratio"] = 0.30
        self.DEVICE_CONFIGS[DeviceType.TYPE_B]["ratio"] = 0.40
        self.DEVICE_CONFIGS[DeviceType.TYPE_C]["ratio"] = 0.30
        self.PRIVACY_CONFIG[PrivacyLevel.LOW]["ratio"] = 0.40
        self.PRIVACY_CONFIG[PrivacyLevel.MEDIUM]["ratio"] = 0.40
        self.PRIVACY_CONFIG[PrivacyLevel.HIGH]["ratio"] = 0.20
        
        return devices


def create_devices(
    n_devices: int,
    seed: int = 42,
    **kwargs
) -> List[DeviceProfile]:
    """
    Convenience function to create device profiles.
    
    Args:
        n_devices: Number of devices
        seed: Random seed
        **kwargs: Additional arguments for HeterogeneityGenerator
        
    Returns:
        List of DeviceProfile instances
    """
    generator = HeterogeneityGenerator(n_devices=n_devices, seed=seed, **kwargs)
    return generator.generate()
