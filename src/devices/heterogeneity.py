"""
Three-Dimensional Device Heterogeneity Generator

Models heterogeneity across:
1. Communication cost (c_i^comm): channel-dependent, continuous
2. Privacy sensitivity (λ_i): user-dependent, discrete levels  
3. Computation cost (c_i^inf): hardware-dependent, device types

Configuration is loaded from config/devices/heterogeneity.yaml
All parameters should have literature support or experimental justification.
"""

import numpy as np
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
from pathlib import Path


class DeviceType(Enum):
    """Edge device types based on real hardware."""
    TYPE_A = "jetson_nano"    # High-end: fast inference
    TYPE_B = "rpi4_4gb"       # Mid-range: baseline
    TYPE_C = "rpi3"           # Low-end: potential straggler


class PrivacyLevel(Enum):
    """Privacy sensitivity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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
    memory: int = 4096      # MB
    
    # Data assignment (set after initialization)
    data_size: int = 0
    data_classes: List[int] = field(default_factory=list)
    
    # Channel state (can be updated per round)
    distance: float = 50.0  # meters
    channel_gain: float = 1.0
    
    @property
    def c_total(self) -> float:
        """Total marginal cost c_i = c_inf + c_comm."""
        return self.c_inf + self.c_comm
    
    @property
    def is_straggler(self) -> bool:
        """Check if device is a potential straggler (Type C)."""
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


def load_heterogeneity_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load heterogeneity configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Find project root and load default config
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        config_path = project_root / "config" / "devices" / "heterogeneity.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict[str, Any]:
    """Return default configuration if file not found."""
    return {
        "device_types": {
            "type_a": {"cpu_freq_ghz": 1.5, "memory_mb": 4096, "compute_capability": 1.0, "ratio": 0.30},
            "type_b": {"cpu_freq_ghz": 1.2, "memory_mb": 4096, "compute_capability": 0.8, "ratio": 0.40},
            "type_c": {"cpu_freq_ghz": 0.8, "memory_mb": 1024, "compute_capability": 0.5, "ratio": 0.30},
        },
        "cost_parameters": {
            "c_inf_base": 0.1,
            "c_inf_multipliers": {"type_a": 0.5, "type_b": 1.0, "type_c": 2.0},
            "c_comm_range": [0.05, 0.2],
        },
        "privacy_sensitivity": {
            "lambda_jitter": 0.3,
            "levels": {
                "very_low": {"value": 0.05, "ratio": 0.15},
                "low": {"value": 0.15, "ratio": 0.25},
                "medium": {"value": 0.4, "ratio": 0.25},
                "high": {"value": 0.8, "ratio": 0.20},
                "very_high": {"value": 1.5, "ratio": 0.15},
            }
        },
        "communication_model": {
            "distance_range_m": [10.0, 100.0],
            "path_loss_exponent": 3.0,
        }
    }


class HeterogeneityGenerator:
    """
    Generates heterogeneous device profiles for simulation.
    
    All parameters are loaded from configuration file for:
    1. Reproducibility
    2. Easy parameter sweeps for experiments
    3. Clear documentation of parameter choices
    
    Usage:
        # Default config
        generator = HeterogeneityGenerator(n_devices=50, seed=42)
        
        # Custom config
        generator = HeterogeneityGenerator(
            n_devices=50, 
            config_path="config/devices/custom.yaml",
            seed=42
        )
        
        devices = generator.generate()
    """
    
    # Mapping from config keys to DeviceType enum
    TYPE_MAPPING = {
        "type_a": DeviceType.TYPE_A,
        "type_b": DeviceType.TYPE_B,
        "type_c": DeviceType.TYPE_C,
    }
    
    def __init__(
        self,
        n_devices: int,
        config_path: Optional[str] = None,
        config_override: Optional[Dict] = None,
        seed: int = 42
    ):
        """
        Initialize the generator.
        
        Args:
            n_devices: Number of devices to generate
            config_path: Path to heterogeneity config YAML
            config_override: Dict to override specific config values
            seed: Random seed for reproducibility
        """
        self.n_devices = n_devices
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Load configuration
        self.config = load_heterogeneity_config(config_path)
        
        # Apply overrides if provided
        if config_override:
            self._apply_overrides(config_override)
        
        # Extract commonly used parameters
        self._parse_config()
    
    def _apply_overrides(self, overrides: Dict):
        """Apply configuration overrides."""
        def deep_update(base: Dict, updates: Dict):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        deep_update(self.config, overrides)
    
    def _parse_config(self):
        """Parse configuration into usable format."""
        # Device types
        self.device_configs = {}
        for type_key, type_config in self.config["device_types"].items():
            device_type = self.TYPE_MAPPING[type_key]
            self.device_configs[device_type] = {
                "cpu_freq": type_config.get("cpu_freq_ghz", 1.0),
                "memory": type_config.get("memory_mb", 4096),
                "compute_capability": type_config.get("compute_capability", 1.0),
                "ratio": type_config.get("ratio", 0.33),
            }
        
        # Cost parameters
        cost_config = self.config["cost_parameters"]
        self.c_inf_base = cost_config["c_inf_base"]
        self.c_inf_multipliers = {
            self.TYPE_MAPPING[k]: v 
            for k, v in cost_config["c_inf_multipliers"].items()
        }
        self.c_comm_range = tuple(cost_config["c_comm_range"])
        
        # Privacy sensitivity
        self.privacy_levels = {}
        for level_key, level_config in self.config["privacy_sensitivity"]["levels"].items():
            self.privacy_levels[level_key] = {
                "value": level_config["value"],
                "ratio": level_config["ratio"],
            }
        
        # Communication model
        comm_config = self.config.get("communication_model", {})
        self.distance_range = tuple(comm_config.get("distance_range_m", [10.0, 100.0]))
    
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
            device_type = device_types[i]
            config = self.device_configs[device_type]
            
            # Compute costs
            c_inf = self.c_inf_base * self.c_inf_multipliers[device_type]
            c_comm = self.rng.uniform(*self.c_comm_range)
            
            # Distance and channel
            distance = self.rng.uniform(*self.distance_range)
            channel_gain = self._compute_channel_gain(distance)
            
            device = DeviceProfile(
                device_id=i,
                device_type=device_type,
                c_comm=c_comm,
                c_inf=c_inf,
                lambda_i=lambdas[i],
                cpu_freq=config["cpu_freq"],
                memory=config["memory"],
                distance=distance,
                channel_gain=channel_gain
            )
            devices.append(device)
        
        return devices
    
    def _assign_device_types(self) -> List[DeviceType]:
        """Assign device types according to configured ratios."""
        types = []
        ratios = [(dt, cfg["ratio"]) for dt, cfg in self.device_configs.items()]
        
        for device_type, ratio in ratios:
            count = int(self.n_devices * ratio)
            types.extend([device_type] * count)
        
        # Fill remaining slots with most common type (Type B)
        while len(types) < self.n_devices:
            types.append(DeviceType.TYPE_B)
        
        self.rng.shuffle(types)
        return types[:self.n_devices]
    
    def _assign_privacy_levels(self) -> List[float]:
        """Assign privacy sensitivity values according to configured ratios.
        
        Supports lambda_mult override: if config['privacy_sensitivity']['lambda_mult']
        is set, ALL lambda values are multiplied by this factor.
        """
        # Read optional global multiplier and jitter
        ps_config = self.config.get("privacy_sensitivity", {})
        lambda_mult = ps_config.get("lambda_mult", 1.0)
        lambda_jitter = ps_config.get("lambda_jitter", 0.3)  # ±30% default
        
        lambdas = []
        for level_key, level_config in self.privacy_levels.items():
            count = int(self.n_devices * level_config["ratio"])
            base_val = level_config["value"] * lambda_mult
            for _ in range(count):
                # Add uniform jitter so each device has a unique λ
                jitter = 1.0 + self.rng.uniform(-lambda_jitter, lambda_jitter)
                lambdas.append(base_val * jitter)
        
        # Fill remaining with medium level
        medium_base = self.privacy_levels.get("medium", {"value": 0.05})["value"] * lambda_mult
        while len(lambdas) < self.n_devices:
            jitter = 1.0 + self.rng.uniform(-lambda_jitter, lambda_jitter)
            lambdas.append(medium_base * jitter)
        
        self.rng.shuffle(lambdas)
        return lambdas[:self.n_devices]
    
    def _compute_channel_gain(self, distance: float) -> float:
        """
        Compute channel gain based on distance.
        
        Uses simplified path loss model: g = (d0/d)^α
        where d0 is reference distance and α is path loss exponent.
        """
        d0 = 1.0  # Reference distance
        alpha = self.config.get("communication_model", {}).get("path_loss_exponent", 3.0)
        return (d0 / distance) ** alpha
    
    def update_channel_gains(self, devices: List[DeviceProfile]) -> None:
        """
        Update channel gains for all devices (simulating channel variation).
        
        Args:
            devices: List of devices to update
        """
        for device in devices:
            # Add small random variation (shadowing)
            base_gain = self._compute_channel_gain(device.distance)
            shadowing = self.rng.lognormal(0, 0.5)  # Log-normal shadowing
            device.channel_gain = base_gain * shadowing
    
    def get_statistics(self, devices: List[DeviceProfile]) -> Dict:
        """
        Compute statistics about the generated devices.
        
        Args:
            devices: List of DeviceProfile
            
        Returns:
            Dictionary with statistics
        """
        type_counts = {}
        for dt in DeviceType:
            type_counts[dt.value] = sum(1 for d in devices if d.device_type == dt)
        
        lambdas = [d.lambda_i for d in devices]
        c_totals = [d.c_total for d in devices]
        
        return {
            "n_devices": len(devices),
            "type_distribution": type_counts,
            "straggler_ratio": type_counts.get("rpi3", 0) / len(devices),
            "lambda_stats": {
                "min": min(lambdas),
                "max": max(lambdas),
                "mean": np.mean(lambdas),
                "std": np.std(lambdas),
            },
            "cost_stats": {
                "min": min(c_totals),
                "max": max(c_totals),
                "mean": np.mean(c_totals),
            },
            "config_source": "heterogeneity.yaml"
        }
    
    def get_config_summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "Heterogeneity Configuration:",
            f"  Devices: {self.n_devices}",
            f"  Seed: {self.seed}",
            "",
            "  Device Types:",
        ]
        for dt, cfg in self.device_configs.items():
            lines.append(f"    {dt.value}: {cfg['ratio']*100:.0f}%, c_inf_mult={self.c_inf_multipliers[dt]}")
        
        lines.append("")
        lines.append("  Privacy Levels:")
        for level, cfg in self.privacy_levels.items():
            lines.append(f"    {level}: λ={cfg['value']}, {cfg['ratio']*100:.0f}%")
        
        lines.append("")
        lines.append(f"  Cost: c_inf_base={self.c_inf_base}, c_comm_range={self.c_comm_range}")
        
        return "\n".join(lines)


# Convenience function for backward compatibility
def create_devices(
    n_devices: int,
    config_path: Optional[str] = None,
    seed: int = 42
) -> List[DeviceProfile]:
    """
    Convenience function to create devices.
    
    Args:
        n_devices: Number of devices
        config_path: Optional path to config file
        seed: Random seed
        
    Returns:
        List of DeviceProfile
    """
    generator = HeterogeneityGenerator(
        n_devices=n_devices,
        config_path=config_path,
        seed=seed
    )
    return generator.generate()
