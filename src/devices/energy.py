"""
Energy Consumption Model for PAID-FD

Computes energy breakdown:
- E_train: Local model training energy
- E_inf: Inference energy (computing logits on public data)
- E_comm: Communication energy (uploading logits)
- E_opt: Optimization energy (bisection, negligible)

Based on DVFS (Dynamic Voltage and Frequency Scaling) model.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EnergyParams:
    """
    Energy model parameters based on literature values.
    
    References:
    - DVFS model: κ_cpu from mobile computing literature
    - Communication: Based on LTE/WiFi power models
    """
    # DVFS parameters
    kappa_cpu: float = 1e-28        # Effective switched capacitance (J/cycle/Hz²)
    
    # Computation parameters (CPU cycles per operation)
    C_train: float = 1e6            # Cycles per training sample (CNN forward+backward)
    C_inf: float = 1e4              # Cycles per inference sample (forward only)
    
    # Communication parameters
    P_tx: float = 0.1               # Transmit power (Watts)
    bandwidth: float = 1e6          # Bandwidth (Hz) - 1 MHz baseline
    noise_power: float = 1e-10      # Noise power (Watts)
    
    # Payload sizes
    logit_size_bytes: int = 400     # Size per logit vector (100 classes × 4 bytes)
    gradient_size_bytes: int = 44_000_000  # ResNet-18 gradients (~11M params × 4 bytes)
    
    # Protocol overhead
    protocol_overhead_ms: float = 50.0  # Per-message overhead (handshake, ACK)


class EnergyCalculator:
    """
    Calculate energy consumption for FD operations.
    
    Usage:
        calc = EnergyCalculator()
        energy = calc.compute_total_energy(
            cpu_freq=1.2,      # GHz
            data_size=500,     # training samples
            s_i=1000,          # logits to upload
            snr_db=10.0        # channel quality
        )
    """
    
    def __init__(self, params: EnergyParams = None):
        self.params = params or EnergyParams()
    
    def compute_training_energy(
        self,
        cpu_freq: float,          # GHz
        data_size: int,           # Number of training samples
        epochs: int = 1,
        batch_size: int = 32
    ) -> float:
        """
        Compute local training energy.
        
        E_train = κ_cpu × f² × C_train × |D| × epochs
        
        Args:
            cpu_freq: CPU frequency in GHz
            data_size: Number of local training samples
            epochs: Number of local epochs
            batch_size: Batch size (affects only iteration count, not energy here)
            
        Returns:
            Training energy in Joules
        """
        f = cpu_freq * 1e9  # Convert to Hz
        
        # Total cycles = cycles_per_sample × samples × epochs
        total_cycles = self.params.C_train * data_size * epochs
        
        # Energy = κ × f² × cycles
        energy = self.params.kappa_cpu * (f ** 2) * total_cycles
        
        return energy
    
    def compute_inference_energy(
        self,
        cpu_freq: float,          # GHz  
        s_i: int                  # Number of logits to compute
    ) -> float:
        """
        Compute inference energy for generating logits.
        
        E_inf = κ_cpu × f² × C_inf × s_i
        
        This is the energy for computing logits on s_i public data samples.
        
        Args:
            cpu_freq: CPU frequency in GHz
            s_i: Number of logits to compute (upload volume)
            
        Returns:
            Inference energy in Joules
        """
        f = cpu_freq * 1e9
        
        total_cycles = self.params.C_inf * s_i
        energy = self.params.kappa_cpu * (f ** 2) * total_cycles
        
        return energy
    
    def compute_communication_energy(
        self,
        payload_bytes: int,
        snr_db: float = 10.0,
        channel_gain: float = 1.0
    ) -> float:
        """
        Compute communication energy.
        
        E_comm = P_tx × transmission_time
        transmission_time = payload_bits / rate
        rate = B × log2(1 + SNR × channel_gain)
        
        Args:
            payload_bytes: Data to transmit in bytes
            snr_db: Signal-to-noise ratio in dB
            channel_gain: Channel gain factor (0-1, affected by fading)
            
        Returns:
            Communication energy in Joules
        """
        # Convert SNR to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Effective SNR with channel gain
        effective_snr = snr_linear * channel_gain
        
        # Shannon capacity (bits/second)
        rate = self.params.bandwidth * np.log2(1 + effective_snr)
        
        # Transmission time
        payload_bits = payload_bytes * 8
        tx_time = payload_bits / rate
        
        # Energy
        energy = self.params.P_tx * tx_time
        
        return energy
    
    def compute_fd_communication_energy(
        self,
        s_i: int,
        snr_db: float = 10.0,
        channel_gain: float = 1.0
    ) -> float:
        """
        Compute communication energy for FD (uploading logits).
        
        Args:
            s_i: Number of logit vectors to upload
            snr_db: SNR in dB
            channel_gain: Channel gain
            
        Returns:
            Communication energy in Joules
        """
        payload = s_i * self.params.logit_size_bytes
        return self.compute_communication_energy(payload, snr_db, channel_gain)
    
    def compute_fl_communication_energy(
        self,
        snr_db: float = 10.0,
        channel_gain: float = 1.0
    ) -> float:
        """
        Compute communication energy for FL (uploading gradients).
        
        Args:
            snr_db: SNR in dB
            channel_gain: Channel gain
            
        Returns:
            Communication energy in Joules
        """
        return self.compute_communication_energy(
            self.params.gradient_size_bytes, snr_db, channel_gain
        )
    
    def compute_optimization_energy(self) -> float:
        """
        Compute energy for running the bisection algorithm.
        
        This is negligible compared to training/inference/communication.
        We include it for completeness.
        
        Returns:
            Optimization energy in Joules (very small)
        """
        # Bisection requires ~20 iterations, each with ~10 FLOPs
        # At 1 GHz, this is ~200 ns, energy ~1e-9 J
        return 1e-9
    
    def compute_total_energy(
        self,
        cpu_freq: float,
        data_size: int,
        s_i: int,
        snr_db: float = 10.0,
        channel_gain: float = 1.0,
        epochs: int = 1,
        participates: bool = True
    ) -> Dict[str, float]:
        """
        Compute total energy consumption with breakdown.
        
        Args:
            cpu_freq: CPU frequency in GHz
            data_size: Number of local training samples
            s_i: Number of logits to upload
            snr_db: Channel SNR in dB
            channel_gain: Channel gain factor
            epochs: Number of local training epochs
            participates: Whether device participates this round
            
        Returns:
            Dictionary with energy breakdown:
            {
                "training": E_train,
                "inference": E_inf,
                "communication": E_comm,
                "optimization": E_opt,
                "total": E_total
            }
        """
        if not participates or s_i <= 0:
            return {
                "training": 0.0,
                "inference": 0.0,
                "communication": 0.0,
                "optimization": 0.0,
                "total": 0.0
            }
        
        E_train = self.compute_training_energy(cpu_freq, data_size, epochs)
        E_inf = self.compute_inference_energy(cpu_freq, s_i)
        E_comm = self.compute_fd_communication_energy(s_i, snr_db, channel_gain)
        E_opt = self.compute_optimization_energy()
        
        return {
            "training": E_train,
            "inference": E_inf,
            "communication": E_comm,
            "optimization": E_opt,
            "total": E_train + E_inf + E_comm + E_opt
        }
    
    def compute_marginal_cost(
        self,
        cpu_freq: float,
        snr_db: float = 10.0,
        channel_gain: float = 1.0
    ) -> Tuple[float, float]:
        """
        Compute marginal costs c_inf and c_comm for the Stackelberg game.
        
        These are the per-logit costs used in the utility function.
        
        Args:
            cpu_freq: CPU frequency in GHz
            snr_db: Channel SNR in dB
            channel_gain: Channel gain
            
        Returns:
            (c_inf, c_comm) tuple
        """
        # c_inf: energy per inference sample
        c_inf = self.compute_inference_energy(cpu_freq, s_i=1)
        
        # c_comm: energy per logit transmission
        c_comm = self.compute_fd_communication_energy(s_i=1, snr_db=snr_db, channel_gain=channel_gain)
        
        return c_inf, c_comm
    
    def compare_fd_vs_fl(
        self,
        cpu_freq: float,
        data_size: int,
        s_i: int,
        snr_db: float = 10.0,
        channel_gain: float = 1.0
    ) -> Dict[str, float]:
        """
        Compare energy consumption between FD and FL.
        
        Args:
            cpu_freq: CPU frequency in GHz
            data_size: Local data size
            s_i: Number of logits for FD
            snr_db: Channel SNR
            channel_gain: Channel gain
            
        Returns:
            Comparison dictionary with ratios
        """
        # FD energy
        fd_energy = self.compute_total_energy(
            cpu_freq, data_size, s_i, snr_db, channel_gain
        )
        
        # FL energy (same training, but upload gradients instead)
        E_train = fd_energy["training"]
        E_comm_fl = self.compute_fl_communication_energy(snr_db, channel_gain)
        fl_total = E_train + E_comm_fl
        
        # Ratios
        fd_total = fd_energy["total"]
        
        return {
            "fd": fd_energy,
            "fl": {
                "training": E_train,
                "inference": 0.0,  # FL doesn't need inference on public data
                "communication": E_comm_fl,
                "optimization": 0.0,
                "total": fl_total
            },
            "total_ratio": fd_total / fl_total if fl_total > 0 else 0,
            "comm_ratio": fd_energy["communication"] / E_comm_fl if E_comm_fl > 0 else 0,
            "fd_savings_percent": (1 - fd_total / fl_total) * 100 if fl_total > 0 else 0
        }
    
    def estimate_protocol_overhead(
        self,
        n_communications: int
    ) -> float:
        """
        Estimate protocol overhead time.
        
        Args:
            n_communications: Number of communication rounds
            
        Returns:
            Overhead time in milliseconds
        """
        return n_communications * self.params.protocol_overhead_ms
    
    def compare_protocol_overhead(
        self,
        n_devices: int
    ) -> Dict[str, float]:
        """
        Compare protocol overhead between PAID-FD and auction-based methods.
        
        PAID-FD (Stackelberg, one-shot):
        - 1 downlink (price broadcast)
        - 1 uplink per participant (logits)
        
        CSRA (Reverse Auction):
        - 1 downlink (budget broadcast)
        - 1 uplink (bids)
        - 1 downlink (winner announcement)
        - 1 uplink (gradients)
        
        Returns:
            Overhead comparison in milliseconds
        """
        # PAID-FD: 2 communications per round
        paid_fd_comms = 2
        paid_fd_overhead = self.estimate_protocol_overhead(paid_fd_comms)
        
        # CSRA: 4 communications per round
        csra_comms = 4
        csra_overhead = self.estimate_protocol_overhead(csra_comms)
        
        return {
            "paid_fd": {
                "n_communications": paid_fd_comms,
                "overhead_ms": paid_fd_overhead
            },
            "csra": {
                "n_communications": csra_comms,
                "overhead_ms": csra_overhead
            },
            "reduction_ms": csra_overhead - paid_fd_overhead,
            "reduction_percent": (1 - paid_fd_overhead / csra_overhead) * 100
        }


def print_energy_comparison(
    calc: EnergyCalculator,
    cpu_freq: float = 1.2,
    data_size: int = 500,
    s_i: int = 1000
):
    """
    Print a formatted comparison of FD vs FL energy.
    
    Useful for verification and debugging.
    """
    comparison = calc.compare_fd_vs_fl(cpu_freq, data_size, s_i)
    
    print("\n" + "="*60)
    print("Energy Comparison: FD vs FL")
    print("="*60)
    print(f"Configuration:")
    print(f"  CPU Frequency: {cpu_freq} GHz")
    print(f"  Local Data Size: {data_size} samples")
    print(f"  Logits to Upload (FD): {s_i}")
    print(f"  Gradient Size (FL): {calc.params.gradient_size_bytes / 1e6:.1f} MB")
    print(f"  Logit Total Size (FD): {s_i * calc.params.logit_size_bytes / 1e3:.1f} KB")
    print()
    
    print("FD Energy Breakdown:")
    for k, v in comparison["fd"].items():
        print(f"  {k:15s}: {v:.6f} J")
    
    print("\nFL Energy Breakdown:")
    for k, v in comparison["fl"].items():
        print(f"  {k:15s}: {v:.6f} J")
    
    print(f"\nComparison:")
    print(f"  Total Energy Ratio (FD/FL): {comparison['total_ratio']:.4f}")
    print(f"  Comm Energy Ratio (FD/FL): {comparison['comm_ratio']:.6f}")
    print(f"  FD Savings: {comparison['fd_savings_percent']:.1f}%")
    print("="*60 + "\n")
