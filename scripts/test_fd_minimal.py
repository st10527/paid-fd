#!/usr/bin/env python3
"""
Minimal Federated Distillation Test

This script tests the core FD logic with CIFAR-10 (10 classes)
to verify the approach works before scaling to CIFAR-100.

No game theory, no LDP noise - just pure FD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ==================== Data ====================
    print("\n1. Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Split into "private" (for local training) and "public" (for distillation)
    n_train = len(train_data)
    indices = np.random.permutation(n_train)
    private_indices = indices[:40000]  # 40k for local training
    public_indices = indices[40000:]   # 10k as public data
    
    private_data = Subset(train_data, private_indices)
    public_data = Subset(train_data, public_indices)
    
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    public_loader = DataLoader(public_data, batch_size=64, shuffle=False)
    
    print(f"   Private: {len(private_data)}, Public: {len(public_data)}, Test: {len(test_data)}")
    
    # ==================== Partition to Clients ====================
    print("\n2. Partitioning to 5 clients (Non-IID)...")
    n_clients = 5
    samples_per_client = len(private_data) // n_clients
    
    client_loaders = []
    for i in range(n_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        client_indices = private_indices[start:end]
        client_subset = Subset(train_data, client_indices)
        client_loaders.append(DataLoader(client_subset, batch_size=32, shuffle=True))
        print(f"   Client {i}: {len(client_subset)} samples")
    
    # ==================== Model ====================
    print("\n3. Creating models...")
    
    # Simple CNN for speed
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, num_classes)
            self.dropout = nn.Dropout(0.25)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    server_model = SimpleCNN(num_classes=10).to(device)
    
    def evaluate(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return correct / total * 100
    
    init_acc = evaluate(server_model)
    print(f"   Initial server accuracy: {init_acc:.2f}%")
    
    # ==================== Federated Distillation ====================
    print("\n4. Running Federated Distillation...")
    
    T = 3.0  # Temperature
    n_rounds = 10
    
    for round_idx in range(n_rounds):
        print(f"\n   Round {round_idx + 1}/{n_rounds}")
        
        # Step A: Each client trains locally and computes logits on public data
        all_probs = []
        
        for client_id, client_loader in enumerate(client_loaders):
            # Create local model (copy of server)
            local_model = SimpleCNN(num_classes=10).to(device)
            local_model.load_state_dict(server_model.state_dict())
            
            # Local training
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            local_model.train()
            for epoch in range(3):  # 3 local epochs
                for data, target in client_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Compute soft labels on public data
            local_model.eval()
            client_probs = []
            with torch.no_grad():
                for data, _ in public_loader:
                    data = data.to(device)
                    logits = local_model(data)
                    probs = F.softmax(logits / T, dim=1)
                    client_probs.append(probs.cpu())
            
            client_probs = torch.cat(client_probs, dim=0)
            all_probs.append(client_probs)
            
            del local_model
        
        # Step B: Aggregate (average probabilities)
        ensemble_probs = torch.stack(all_probs).mean(dim=0)  # [n_public, n_classes]
        print(f"      Ensemble probs shape: {ensemble_probs.shape}")
        print(f"      Ensemble entropy: {-(ensemble_probs * torch.log(ensemble_probs + 1e-10)).sum(dim=1).mean():.3f}")
        
        # Step C: Distill to server
        server_model.train()
        optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        
        ensemble_probs = ensemble_probs.to(device)
        
        for epoch in range(5):  # 5 distillation epochs
            idx = 0
            total_loss = 0
            n_batches = 0
            for data, _ in public_loader:
                batch_size = data.size(0)
                data = data.to(device)
                target_probs = ensemble_probs[idx:idx + batch_size]
                
                student_logits = server_model(data)
                
                # KL divergence
                loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    target_probs,
                    reduction='batchmean'
                ) * (T * T)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                idx += batch_size
        
        # Evaluate
        acc = evaluate(server_model)
        print(f"      Server accuracy: {acc:.2f}%")
    
    print("\n" + "="*50)
    print(f"Final accuracy: {acc:.2f}%")
    print("="*50)
    
    if acc > 30:
        print("✓ SUCCESS: FD is working!")
    elif acc > 15:
        print("⚠ PARTIAL: Some learning, but not great")
    else:
        print("✗ FAIL: FD not working")


if __name__ == '__main__':
    main()
