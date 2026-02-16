"""
Base class for federated learning methods.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class RoundResult:
    """Result of a single training round."""
    round_idx: int
    accuracy: float
    loss: float
    participation_rate: float
    n_participants: int
    energy: Dict[str, float]
    extra: Dict[str, Any]


class FederatedMethod(ABC):
    """
    Abstract base class for federated learning methods.
    
    Subclasses must implement:
    - run_round(): Execute one round of training
    - aggregate(): Aggregate updates from clients
    """
    
    def __init__(
        self,
        server_model: nn.Module,
        n_classes: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            server_model: Global model on server
            n_classes: Number of output classes
            device: Device to run on
        """
        self.server_model = server_model.to(device)
        self.n_classes = n_classes
        self.device = device
        self.round_history: List[RoundResult] = []
        self.current_round = 0
    
    @abstractmethod
    def run_round(
        self,
        round_idx: int,
        devices: List[Any],
        client_loaders: Dict[int, Any],
        public_loader: Any
    ) -> RoundResult:
        """
        Execute one round of federated training.
        
        Args:
            round_idx: Current round index
            devices: List of DeviceProfile
            client_loaders: Dict mapping client_id -> DataLoader
            public_loader: DataLoader for public dataset
            
        Returns:
            RoundResult with metrics
        """
        pass
    
    @abstractmethod
    def aggregate(self, updates: List[Dict], weights: List[float]) -> None:
        """
        Aggregate updates into the server model.
        
        Args:
            updates: List of update dictionaries (logits or gradients)
            weights: Aggregation weights
        """
        pass
    
    def evaluate(
        self,
        test_loader: Any,
        model: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test DataLoader
            model: Model to evaluate (default: server_model)
            
        Returns:
            Dictionary with accuracy, loss, etc.
        """
        if model is None:
            model = self.server_model
        
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * len(data)
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(data)
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "loss": total_loss / total if total > 0 else 0,
            "n_samples": total
        }
    
    def train_local(
        self,
        model: nn.Module,
        train_loader: Any,
        epochs: int = 1,
        lr: float = 0.01
    ) -> float:
        """
        Train model locally on client data.
        
        Args:
            model: Local model
            train_loader: Local training DataLoader
            epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            Average training loss
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        n_batches = 0
        
        for _ in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def compute_logits(
        self,
        model: nn.Module,
        data_loader: Any,
        n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute logits on a dataset.
        
        Args:
            model: Model to use
            data_loader: DataLoader
            n_samples: Max samples to use (None = all)
            
        Returns:
            Tensor of logits (N, n_classes)
        """
        model.eval()
        all_logits = []
        count = 0
        
        with torch.no_grad():
            for data, _ in data_loader:
                if n_samples is not None and count >= n_samples:
                    break
                
                data = data.to(self.device)
                logits = model(data)
                all_logits.append(logits.cpu())
                count += len(data)
        
        result = torch.cat(all_logits, dim=0)
        if n_samples is not None:
            result = result[:n_samples]
        
        return result
    
    def get_round_history(self) -> List[Dict]:
        """Get history of all rounds."""
        return [
            {
                "round": r.round_idx,
                "accuracy": r.accuracy,
                "loss": r.loss,
                "participation_rate": r.participation_rate,
                "energy": r.energy
            }
            for r in self.round_history
        ]
    
    def get_best_accuracy(self) -> float:
        """Get best accuracy achieved."""
        if not self.round_history:
            return 0.0
        return max(r.accuracy for r in self.round_history)
    
    def get_final_accuracy(self) -> float:
        """Get accuracy from last round."""
        if not self.round_history:
            return 0.0
        return self.round_history[-1].accuracy
