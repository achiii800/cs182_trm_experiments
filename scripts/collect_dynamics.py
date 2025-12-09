#!/usr/bin/env python3
"""
Collect Training Dynamics Data for TRM Router

Run training with a fixed inner solver while logging dynamics at each step.
This data is used to train the TRM router offline.

Usage:
    python scripts/collect_dynamics.py --solver dual_ascent --epochs 30 --output results/dynamics_da.pkl
    python scripts/collect_dynamics.py --solver admm --epochs 30 --output results/dynamics_admm.pkl
    python scripts/collect_dynamics.py --solver frank_wolfe --epochs 30 --output results/dynamics_fw.pkl
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import get_model
from muon import MuonSGD, get_inner_solver
from trm import TRMDataCollector


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
    """Get CIFAR-10 train/test loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def train_with_dynamics_collection(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    collector: TRMDataCollector,
    epochs: int,
    device: torch.device,
):
    """Train model and collect dynamics data."""
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute gradient norm
            loss.backward()
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Record pre-step state
            collector.pre_step(
                loss=loss.item(),
                grad_norm=grad_norm,
                epoch=epoch,
                step=step,
            )
            
            # Optimizer step
            optimizer.step()
            
            # Compute next loss for labeling
            with torch.no_grad():
                outputs_next = model(inputs)
                next_loss = criterion(outputs_next, targets).item()
            
            # Record post-step outcome
            collector.post_step(next_loss=next_loss)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Collect dynamics for TRM training")
    
    parser.add_argument("--solver", type=str, required=True,
                       choices=["spectral_clip", "dual_ascent", "quasi_newton", 
                               "frank_wolfe", "admm"],
                       help="Inner solver to use")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path for dynamics data")
    parser.add_argument("--model", type=str, default="small_cnn",
                       choices=["small_cnn", "resnet18", "tiny_vit", "mlp_mixer", "mlp"],
                       help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1,
                       help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="Momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                       help="Weight decay")
    parser.add_argument("--spectral-budget", type=float, default=0.1,
                       help="Spectral norm budget")
    parser.add_argument("--width-mult", type=float, default=1.0,
                       help="Model width multiplier")
    parser.add_argument("--record-every", type=int, default=1,
                       help="Record dynamics every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    steps_per_epoch = len(train_loader)
    
    # Model
    model = get_model(args.model, num_classes=10, width_mult=args.width_mult)
    model = model.to(device)
    
    # Find first Linear layer for tracking
    tracked_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tracked_layer = name
            break
    
    print(f"Model: {args.model} (width_mult={args.width_mult})")
    print(f"Tracking layer: {tracked_layer}")
    print(f"Solver: {args.solver}")
    
    # Inner solver
    inner_solver = get_inner_solver(args.solver)
    
    # Optimizer
    optimizer = MuonSGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        spectral_budget=args.spectral_budget,
        inner_solver=inner_solver,
    )
    
    # Data collector
    collector = TRMDataCollector(
        model=model,
        solver_name=args.solver,
        tracked_layer_name=tracked_layer,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        width_mult=args.width_mult,
        record_every=args.record_every,
    )
    
    # Train
    train_with_dynamics_collection(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        collector=collector,
        epochs=args.epochs,
        device=device,
    )
    
    # Save
    collector.save(args.output)
    print(f"\nSaved {len(collector.records)} dynamics records to {args.output}")


if __name__ == "__main__":
    main()
