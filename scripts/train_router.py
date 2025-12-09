#!/usr/bin/env python3
"""
Train TRM Router from Collected Dynamics Data

This script trains the TRM router using collected dynamics data.

Usage:
    python scripts/train_router.py --data results/merged_dynamics.pkl --output results/trm_router.pt
    python scripts/train_router.py --data results/merged_dynamics.pkl --deep-supervision --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from trm import TRMRouter, create_trm_router, DynamicsDataset, create_dataloaders


class TRMRouterTrainer:
    """Trainer for TRM router."""
    
    def __init__(
        self,
        model: TRMRouter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        epochs: int = 100,
        use_deep_supervision: bool = True,
        regression_weight: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.use_deep_supervision = use_deep_supervision
        self.regression_weight = regression_weight
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=lr / 100)
        
        if use_ema:
            self.ema_model = self._create_ema_model()
        
        self.history = []
        
    def _create_ema_model(self) -> TRMRouter:
        """Create EMA copy of model."""
        ema_model = create_trm_router(size='small', input_dim=self.model.input_dim)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        return ema_model
    
    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model weights."""
        if not self.use_ema:
            return
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss."""
        solver_labels = batch['solver_label'].to(self.device)
        loss_deltas = batch['loss_deltas'].to(self.device)
        
        losses = {}
        
        # Classification loss
        if self.use_deep_supervision and 'all_solver_logits' in outputs:
            ce_loss = sum(
                F.cross_entropy(logits, solver_labels)
                for logits in outputs['all_solver_logits']
            ) / len(outputs['all_solver_logits'])
        else:
            ce_loss = F.cross_entropy(outputs['solver_logits'], solver_labels)
        
        losses['ce_loss'] = ce_loss.item()
        total_loss = ce_loss
        
        # Regression loss
        if self.regression_weight > 0:
            pred_probs = F.softmax(outputs['solver_logits'], dim=-1)
            expected_delta = (pred_probs * loss_deltas).sum(dim=-1).mean()
            reg_loss = self.regression_weight * expected_delta
            losses['reg_loss'] = expected_delta.item()
            total_loss = total_loss + reg_loss
        
        return total_loss, losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            labels = batch['solver_label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features, return_all_cycles=self.use_deep_supervision)
            loss, _ = self._compute_loss(outputs, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self._update_ema()
            
            total_loss += loss.item() * features.size(0)
            preds = outputs['solver_logits'].argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += features.size(0)
        
        return {
            'train_loss': total_loss / total_samples,
            'train_acc': total_correct / total_samples,
        }
    
    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> Dict[str, float]:
        """Evaluate on validation set."""
        model = self.ema_model if (use_ema and self.use_ema) else self.model
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        solver_correct = [0] * 5
        solver_total = [0] * 5
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            labels = batch['solver_label'].to(self.device)
            
            outputs = model(features)
            loss, _ = self._compute_loss(outputs, batch)
            
            total_loss += loss.item() * features.size(0)
            preds = outputs['solver_logits'].argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += features.size(0)
            
            for i in range(5):
                mask = labels == i
                if mask.any():
                    solver_total[i] += mask.sum().item()
                    solver_correct[i] += ((preds == labels) & mask).sum().item()
        
        per_solver_acc = {}
        solver_names = ["spectral_clip", "dual_ascent", "quasi_newton", "frank_wolfe", "admm"]
        for i, name in enumerate(solver_names):
            if solver_total[i] > 0:
                per_solver_acc[name] = solver_correct[i] / solver_total[i]
        
        return {
            'val_loss': total_loss / total_samples,
            'val_acc': total_correct / total_samples,
            'per_solver_acc': per_solver_acc,
        }
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            self.scheduler.step()
            
            metrics = {
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                **train_metrics,
                **val_metrics,
            }
            self.history.append(metrics)
            
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                best_state = self.model.state_dict().copy()
            
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Acc: {train_metrics['train_acc']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_acc']:.4f}"
                )
        
        return {
            'best_val_acc': best_val_acc,
            'best_state': best_state,
            'history': self.history,
        }
    
    def save(self, path: str, save_ema: bool = True):
        """Save trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state': self.model.state_dict(),
            'config': {
                'input_dim': self.model.input_dim,
                'num_solvers': self.model.num_solvers,
                'action_dim': self.model.action_dim,
                'latent_dim': self.model.latent_dim,
                'n_inner_steps': self.model.n_inner_steps,
                'n_cycles': self.model.n_cycles,
            },
            'history': self.history,
        }
        
        if save_ema and self.use_ema:
            state['ema_state'] = self.ema_model.state_dict()
        
        torch.save(state, path)
        print(f"Saved model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train TRM Router")
    
    parser.add_argument("--data", type=str, required=True,
                       help="Path to dynamics data (merged)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save trained model")
    parser.add_argument("--size", type=str, default="small",
                       choices=["tiny", "small", "base"],
                       help="TRM size variant")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--deep-supervision", action="store_true",
                       help="Use deep supervision")
    parser.add_argument("--regression-weight", type=float, default=0.0,
                       help="Weight for regression loss")
    parser.add_argument("--no-ema", action="store_true",
                       help="Disable EMA")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader = create_dataloaders(
        data_path=args.data,
        batch_size=args.batch_size,
        train_split=0.8,
    )
    
    print(f"Loaded data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # Create model
    model = create_trm_router(size=args.size)
    print(f"TRM Router: {model.num_parameters:,} parameters")
    
    # Train
    trainer = TRMRouterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        use_deep_supervision=args.deep_supervision,
        regression_weight=args.regression_weight,
        use_ema=not args.no_ema,
    )
    
    results = trainer.train()
    trainer.save(args.output)
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    print(f"\nFinal Val Accuracy: {final_metrics['val_acc']:.4f}")
    print("Per-solver accuracy:")
    for name, acc in final_metrics.get('per_solver_acc', {}).items():
        print(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
