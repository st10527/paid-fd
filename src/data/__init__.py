"""
Data loading and partitioning modules for PAID-FD.
"""

from .datasets import (
    load_cifar100,
    load_stl10,
    get_data_loaders,
    DatasetInfo,
    SyntheticDataset,
    create_synthetic_datasets
)
from .partition import (
    DirichletPartitioner,
    IIDPartitioner,
    create_client_loaders,
    print_partition_summary
)

__all__ = [
    "load_cifar100",
    "load_stl10",
    "get_data_loaders",
    "DatasetInfo",
    "SyntheticDataset",
    "create_synthetic_datasets",
    "DirichletPartitioner",
    "IIDPartitioner",
    "create_client_loaders",
    "print_partition_summary"
]
