"""
Data Partitioning for Federated Learning

Implements:
- Dirichlet distribution for Non-IID partitioning
- IID partitioning (baseline)
- Configurable heterogeneity levels

Reference:
- Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import Dataset, Subset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    DataLoader = None


@dataclass
class PartitionInfo:
    """Information about a data partition."""
    client_id: int
    n_samples: int
    class_distribution: Dict[int, int]  # {class_id: count}
    
    @property
    def n_classes(self) -> int:
        return len([c for c, count in self.class_distribution.items() if count > 0])


class DirichletPartitioner:
    """
    Partition data using Dirichlet distribution for Non-IID splits.
    
    The Dirichlet parameter α controls heterogeneity:
    - α → 0: Each client gets data from only one class (extreme Non-IID)
    - α → ∞: Each client gets uniform distribution (IID)
    - α = 0.5: Moderate heterogeneity (commonly used)
    - α = 1.0: Mild heterogeneity
    
    Usage:
        partitioner = DirichletPartitioner(alpha=0.5, n_clients=50)
        client_indices = partitioner.partition(dataset)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        n_clients: int = 50,
        min_samples_per_client: int = 10,
        seed: int = 42
    ):
        """
        Args:
            alpha: Dirichlet concentration parameter (lower = more heterogeneous)
            n_clients: Number of clients to partition data among
            min_samples_per_client: Minimum samples each client should receive
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.n_clients = n_clients
        self.min_samples_per_client = min_samples_per_client
        self.rng = np.random.RandomState(seed)
    
    def partition(
        self,
        dataset: Any,
        targets: Optional[np.ndarray] = None
    ) -> Dict[int, List[int]]:
        """
        Partition dataset indices among clients using Dirichlet distribution.
        
        Args:
            dataset: Dataset with .targets attribute or provide targets separately
            targets: Optional array of class labels
            
        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        # Get targets
        if targets is None:
            if hasattr(dataset, 'targets'):
                targets = np.array(dataset.targets)
            elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
                targets = np.array(dataset.dataset.targets)
            else:
                raise ValueError("Could not find targets. Please provide them explicitly.")
        
        targets = np.asarray(targets)
        n_samples = len(targets)
        n_classes = len(np.unique(targets))
        
        # Get indices for each class
        class_indices = {c: np.where(targets == c)[0] for c in range(n_classes)}
        
        # Initialize client data
        client_indices = {i: [] for i in range(self.n_clients)}
        
        # For each class, distribute samples according to Dirichlet
        for c in range(n_classes):
            indices = class_indices[c].copy()
            self.rng.shuffle(indices)
            
            # Sample from Dirichlet to get proportions for each client
            proportions = self.rng.dirichlet([self.alpha] * self.n_clients)
            
            # Convert proportions to counts
            counts = (proportions * len(indices)).astype(int)
            
            # Adjust to ensure all samples are assigned
            diff = len(indices) - counts.sum()
            for i in range(abs(diff)):
                idx = i % self.n_clients
                counts[idx] += 1 if diff > 0 else -1
            counts = np.maximum(counts, 0)
            
            # Assign indices to clients
            start = 0
            for client_id in range(self.n_clients):
                end = start + counts[client_id]
                client_indices[client_id].extend(indices[start:end].tolist())
                start = end
        
        # Ensure minimum samples per client
        self._ensure_minimum_samples(client_indices, n_samples)
        
        return client_indices
    
    def _ensure_minimum_samples(
        self,
        client_indices: Dict[int, List[int]],
        n_samples: int
    ):
        """Redistribute samples if any client has too few."""
        # Find clients with too few samples
        poor_clients = [
            c for c, idx in client_indices.items() 
            if len(idx) < self.min_samples_per_client
        ]
        
        if not poor_clients:
            return
        
        # Find clients with extra samples
        rich_clients = [
            c for c, idx in client_indices.items()
            if len(idx) > self.min_samples_per_client * 2
        ]
        
        for poor in poor_clients:
            needed = self.min_samples_per_client - len(client_indices[poor])
            for rich in rich_clients:
                if needed <= 0:
                    break
                available = len(client_indices[rich]) - self.min_samples_per_client
                transfer = min(needed, available // 2)
                if transfer > 0:
                    # Transfer samples
                    transferred = client_indices[rich][-transfer:]
                    client_indices[rich] = client_indices[rich][:-transfer]
                    client_indices[poor].extend(transferred)
                    needed -= transfer
    
    def get_partition_info(
        self,
        client_indices: Dict[int, List[int]],
        targets: np.ndarray
    ) -> List[PartitionInfo]:
        """Get detailed information about each client's partition."""
        info_list = []
        n_classes = len(np.unique(targets))
        
        for client_id, indices in client_indices.items():
            client_targets = targets[indices]
            class_dist = {}
            for c in range(n_classes):
                class_dist[c] = int(np.sum(client_targets == c))
            
            info = PartitionInfo(
                client_id=client_id,
                n_samples=len(indices),
                class_distribution=class_dist
            )
            info_list.append(info)
        
        return info_list
    
    def compute_heterogeneity_metrics(
        self,
        client_indices: Dict[int, List[int]],
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics quantifying the heterogeneity of the partition.
        
        Returns:
            Dictionary with:
            - avg_classes_per_client: Average number of classes per client
            - label_distribution_skew: KL divergence from uniform
            - sample_imbalance: Coefficient of variation of sample counts
        """
        targets = np.asarray(targets)
        n_classes = len(np.unique(targets))
        
        # Classes per client
        classes_per_client = []
        for indices in client_indices.values():
            if len(indices) > 0:
                client_targets = targets[indices]
                n_client_classes = len(np.unique(client_targets))
                classes_per_client.append(n_client_classes)
            else:
                classes_per_client.append(0)
        
        avg_classes = np.mean(classes_per_client)
        
        # Sample count statistics
        sample_counts = [len(idx) for idx in client_indices.values()]
        sample_cv = np.std(sample_counts) / np.mean(sample_counts) if np.mean(sample_counts) > 0 else 0
        
        # Label distribution skew (average KL divergence from uniform)
        uniform = np.ones(n_classes) / n_classes
        kl_divs = []
        
        for indices in client_indices.values():
            if len(indices) == 0:
                continue
            client_targets = targets[indices]
            client_dist = np.zeros(n_classes)
            for c in range(n_classes):
                client_dist[c] = np.sum(client_targets == c)
            if client_dist.sum() > 0:
                client_dist = client_dist / client_dist.sum()
                # KL divergence (with smoothing to avoid log(0))
                client_dist = np.clip(client_dist, 1e-10, 1)
                kl = np.sum(client_dist * np.log(client_dist / uniform))
                kl_divs.append(kl)
        
        avg_kl = np.mean(kl_divs) if kl_divs else 0
        
        return {
            "avg_classes_per_client": float(avg_classes),
            "max_classes": n_classes,
            "label_distribution_skew": float(avg_kl),
            "sample_imbalance_cv": float(sample_cv),
            "min_samples": int(min(sample_counts)),
            "max_samples": int(max(sample_counts)),
            "total_samples": int(sum(sample_counts))
        }


class IIDPartitioner:
    """
    Partition data uniformly (IID) among clients.
    
    Each client receives approximately the same number of samples
    with similar class distributions.
    """
    
    def __init__(
        self,
        n_clients: int = 50,
        seed: int = 42
    ):
        self.n_clients = n_clients
        self.rng = np.random.RandomState(seed)
    
    def partition(
        self,
        dataset: Any,
        targets: Optional[np.ndarray] = None
    ) -> Dict[int, List[int]]:
        """
        Partition dataset indices uniformly among clients.
        """
        if targets is not None:
            n_samples = len(targets)
        elif hasattr(dataset, 'targets'):
            n_samples = len(dataset.targets)
        elif hasattr(dataset, '__len__'):
            n_samples = len(dataset)
        else:
            raise ValueError("Cannot determine dataset size")
        
        # Shuffle all indices
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        # Split into n_clients parts
        splits = np.array_split(indices, self.n_clients)
        
        client_indices = {
            i: splits[i].tolist() for i in range(self.n_clients)
        }
        
        return client_indices


def create_client_loaders(
    dataset: Any,
    client_indices: Dict[int, List[int]],
    batch_size: int = 32
) -> Dict[int, Any]:
    """
    Create DataLoaders for each client.
    
    Args:
        dataset: Full dataset (can be a Subset or wrapped dataset)
        client_indices: Partition from partitioner (indices into dataset)
        batch_size: Batch size for DataLoaders
        
    Returns:
        Dictionary mapping client_id -> DataLoader
    """
    if not TORCH_AVAILABLE:
        # Return raw data indices if torch not available
        return client_indices
    
    # =======================================================
    # [TMC Fix] Handle nested Subset correctly
    # =======================================================
    # If dataset is a Subset, we need to map client_indices through the Subset's indices
    # client_indices are positions within 'dataset', not the underlying base dataset
    
    if isinstance(dataset, Subset):
        # dataset is Subset(base, subset_indices)
        # client_indices are positions 0..len(dataset)-1
        # We need to map them to the actual base dataset indices
        base_dataset = dataset.dataset
        subset_indices = np.array(dataset.indices)
        
        client_loaders = {}
        for client_id, indices in client_indices.items():
            if len(indices) == 0:
                continue
            # Map indices through the subset: actual_idx = subset_indices[idx]
            actual_indices = subset_indices[indices].tolist()
            client_subset = Subset(base_dataset, actual_indices)
            client_loaders[client_id] = DataLoader(
                client_subset,
                batch_size=min(batch_size, len(actual_indices)),
                shuffle=True,
                drop_last=False,
                pin_memory=True
            )
        return client_loaders
    
    # Handle other wrapped datasets (e.g., CIFAR10Wrapper)
    if hasattr(dataset, 'dataset') and not isinstance(dataset, Subset):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset
    
    client_loaders = {}
    
    for client_id, indices in client_indices.items():
        if len(indices) == 0:
            continue
        
        client_subset = Subset(base_dataset, indices)
        client_loaders[client_id] = DataLoader(
            client_subset,
            batch_size=min(batch_size, len(indices)),
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )
    
    return client_loaders


def print_partition_summary(
    client_indices: Dict[int, List[int]],
    targets: np.ndarray,
    partitioner_name: str = "Unknown"
):
    """Print a summary of the partition."""
    targets = np.asarray(targets)
    n_clients = len(client_indices)
    n_classes = len(np.unique(targets))
    
    sample_counts = [len(idx) for idx in client_indices.values()]
    
    # Classes per client
    classes_per_client = []
    for indices in client_indices.values():
        if len(indices) > 0:
            client_targets = targets[indices]
            classes_per_client.append(len(np.unique(client_targets)))
        else:
            classes_per_client.append(0)
    
    print(f"\n{'='*50}")
    print(f"Partition Summary: {partitioner_name}")
    print(f"{'='*50}")
    print(f"Total samples: {sum(sample_counts)}")
    print(f"Number of clients: {n_clients}")
    print(f"Number of classes: {n_classes}")
    print(f"\nSamples per client:")
    print(f"  Min: {min(sample_counts)}")
    print(f"  Max: {max(sample_counts)}")
    print(f"  Mean: {np.mean(sample_counts):.1f}")
    print(f"  Std: {np.std(sample_counts):.1f}")
    print(f"\nClasses per client:")
    print(f"  Min: {min(classes_per_client)}")
    print(f"  Max: {max(classes_per_client)}")
    print(f"  Mean: {np.mean(classes_per_client):.1f}")
    print(f"{'='*50}\n")
