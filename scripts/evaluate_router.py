#!/usr/bin/env python3
"""
Evaluate TRM Router on held-out data.

Usage:
    python scripts/evaluate_router.py --model results/trm_router.pt --data results/merged_dynamics.pkl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trm import TRMRouter, create_trm_router, DynamicsDataset


def evaluate_router(
    model_path: str,
    data_path: str,
    batch_size: int = 64,
    use_ema: bool = True,
):
    """Evaluate trained router."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    model = TRMRouter(
        input_dim=config['input_dim'],
        num_solvers=config['num_solvers'],
        action_dim=config['action_dim'],
        latent_dim=config['latent_dim'],
        n_inner_steps=config['n_inner_steps'],
        n_cycles=config['n_cycles'],
    )
    
    if use_ema and 'ema_state' in checkpoint:
        model.load_state_dict(checkpoint['ema_state'])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint['model_state'])
        print("Loaded model weights")
    
    model.to(device)
    model.eval()
    
    # Load data
    dataset = DynamicsDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Evaluate
    total_correct = 0
    total_samples = 0
    
    solver_correct = [0] * 5
    solver_total = [0] * 5
    
    solver_names = ["spectral_clip", "dual_ascent", "quasi_newton", "frank_wolfe", "admm"]
    confusion_matrix = torch.zeros(5, 5, dtype=torch.long)
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            labels = batch['solver_label'].to(device)
            
            outputs = model(features)
            preds = outputs['solver_logits'].argmax(dim=-1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += features.size(0)
            
            for i in range(5):
                mask = labels == i
                if mask.any():
                    solver_total[i] += mask.sum().item()
                    solver_correct[i] += ((preds == labels) & mask).sum().item()
            
            for t, p in zip(labels, preds):
                confusion_matrix[t, p] += 1
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {100*total_correct/total_samples:.2f}%")
    print(f"\nPer-solver Accuracy:")
    
    for i, name in enumerate(solver_names):
        if solver_total[i] > 0:
            acc = 100 * solver_correct[i] / solver_total[i]
            print(f"  {name:15s}: {acc:5.2f}% ({solver_correct[i]}/{solver_total[i]})")
    
    print(f"\nConfusion Matrix:")
    print(f"{'':15s}", end="")
    for name in solver_names:
        print(f"{name[:8]:>10s}", end="")
    print()
    
    for i, true_name in enumerate(solver_names):
        print(f"{true_name:15s}", end="")
        for j in range(5):
            print(f"{confusion_matrix[i, j].item():>10d}", end="")
        print()
    
    # Training history
    if 'history' in checkpoint:
        history = checkpoint['history']
        print(f"\nTraining History:")
        print(f"  Final train acc: {history[-1].get('train_acc', 0):.4f}")
        print(f"  Final val acc: {history[-1].get('val_acc', 0):.4f}")
        print(f"  Best val acc: {max(h.get('val_acc', 0) for h in history):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRM Router")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to dynamics data")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--no-ema", action="store_true",
                       help="Don't use EMA weights")
    
    args = parser.parse_args()
    
    evaluate_router(
        model_path=args.model,
        data_path=args.data,
        batch_size=args.batch_size,
        use_ema=not args.no_ema,
    )


if __name__ == "__main__":
    main()
