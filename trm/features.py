"""
Training Dynamics Feature Extraction for TRM Router

This module extracts informative features from the training state that the TRM
router uses to make solver selection decisions. Features are designed to capture:

1. Loss landscape characteristics (current loss, loss trajectory)
2. Gradient statistics (norm, noise scale)
3. Spectral properties (top singular values, effective rank)
4. Manifold constraint satisfaction (orthogonality error)
5. Temporal context (epoch/step progress)

Features are normalized and packed into a fixed-size vector for the TRM input.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math

import torch
import torch.nn as nn
import numpy as np


@dataclass
class TrainingState:
    """
    Container for training state at a given step.
    
    Captures all relevant statistics for the TRM router decision.
    """
    
    # Loss information
    current_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    # Gradient statistics
    grad_norm: float = 0.0
    grad_norm_history: List[float] = field(default_factory=list)
    
    # Spectral properties (for selected layer)
    top_singular_value: float = 1.0
    effective_rank: float = 1.0
    condition_number: float = 1.0
    
    # Manifold constraint metrics
    orthogonality_error: float = 0.0
    
    # Update properties
    update_spectral_norm: float = 0.0
    update_effective_rank: float = 1.0
    
    # Progress indicators
    epoch: int = 0
    step: int = 0
    total_epochs: int = 100
    steps_per_epoch: int = 390
    
    # Model width (for muP context)
    width_mult: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'current_loss': self.current_loss,
            'grad_norm': self.grad_norm,
            'top_singular_value': self.top_singular_value,
            'effective_rank': self.effective_rank,
            'orthogonality_error': self.orthogonality_error,
            'update_spectral_norm': self.update_spectral_norm,
            'update_effective_rank': self.update_effective_rank,
            'epoch': self.epoch,
            'step': self.step,
        }


class DynamicsFeatureExtractor:
    """
    Extracts and normalizes training dynamics features for TRM input.
    
    Features (12 total):
        1. Normalized current loss (log-scale, clipped)
        2. Short-term loss delta (5-step moving average trend)
        3. Normalized gradient norm (log-scale)
        4. Gradient noise scale proxy
        5. Top singular value of selected weight matrix
        6. Effective rank of last update
        7. Condition number estimate (log-scale)
        8. Orthogonality error (log-scale)
        9. Epoch progress (0 to 1)
        10. Step-within-epoch progress (0 to 1)
        11. Width multiplier (log-scale, for muP context)
        12. Loss stability indicator (variance over recent history)
    
    All features are normalized to approximately [-1, 1] range.
    """
    
    FEATURE_DIM = 12
    FEATURE_NAMES = [
        'log_loss',
        'loss_trend',
        'log_grad_norm',
        'gns_proxy',
        'top_sv',
        'eff_rank',
        'log_cond',
        'log_ortho_err',
        'epoch_progress',
        'step_progress',
        'log_width',
        'loss_stability',
    ]
    
    def __init__(
        self,
        history_length: int = 50,
        loss_scale: float = 2.5,   # Expected max log-loss
        grad_scale: float = 3.0,   # Expected max log-grad-norm
    ):
        """
        Args:
            history_length: Length of history buffers
            loss_scale: Normalization scale for log-loss
            grad_scale: Normalization scale for log-grad-norm
        """
        self.history_length = history_length
        self.loss_scale = loss_scale
        self.grad_scale = grad_scale
        
        # Running statistics for normalization
        self._loss_history: List[float] = []
        self._grad_history: List[float] = []
        
    def reset(self):
        """Reset history buffers."""
        self._loss_history.clear()
        self._grad_history.clear()
        
    def update_history(self, state: TrainingState):
        """Update internal history with new state."""
        self._loss_history.append(state.current_loss)
        self._grad_history.append(state.grad_norm)
        
        # Trim to max length
        if len(self._loss_history) > self.history_length:
            self._loss_history = self._loss_history[-self.history_length:]
        if len(self._grad_history) > self.history_length:
            self._grad_history = self._grad_history[-self.history_length:]
    
    def extract_features(
        self,
        state: TrainingState,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Extract normalized feature vector from training state.
        
        Args:
            state: Current training state
            device: Target device for output tensor
            
        Returns:
            Feature tensor of shape (FEATURE_DIM,)
        """
        if device is None:
            device = torch.device('cpu')
        
        features = []
        
        # 1. Normalized log-loss
        log_loss = math.log(max(state.current_loss, 1e-8))
        norm_log_loss = log_loss / self.loss_scale
        norm_log_loss = max(-1.0, min(1.0, norm_log_loss))
        features.append(norm_log_loss)
        
        # 2. Short-term loss trend
        if len(self._loss_history) >= 5:
            recent = self._loss_history[-5:]
            trend = (recent[-1] - recent[0]) / (max(recent) - min(recent) + 1e-8)
            trend = max(-1.0, min(1.0, trend))
        else:
            trend = 0.0
        features.append(trend)
        
        # 3. Normalized log gradient norm
        log_grad = math.log(max(state.grad_norm, 1e-12))
        norm_log_grad = log_grad / self.grad_scale
        norm_log_grad = max(-1.0, min(1.0, norm_log_grad))
        features.append(norm_log_grad)
        
        # 4. Gradient noise scale proxy (variance / mean^2 of recent grads)
        if len(self._grad_history) >= 3:
            grads = np.array(self._grad_history[-10:])
            mean_g = np.mean(grads)
            var_g = np.var(grads)
            gns = var_g / (mean_g ** 2 + 1e-8)
            gns = math.log(max(gns, 1e-8)) / 5.0  # Normalize
            gns = max(-1.0, min(1.0, gns))
        else:
            gns = 0.0
        features.append(gns)
        
        # 5. Top singular value (normalized, assume ~O(1) range)
        top_sv = state.top_singular_value
        norm_sv = (top_sv - 1.0) / 2.0  # Center around 1, scale by 2
        norm_sv = max(-1.0, min(1.0, norm_sv))
        features.append(norm_sv)
        
        # 6. Effective rank (log-normalized)
        eff_rank = max(state.effective_rank, 1.0)
        log_rank = math.log(eff_rank) / 5.0  # Normalize
        log_rank = max(-1.0, min(1.0, log_rank))
        features.append(log_rank)
        
        # 7. Condition number (log-scale)
        cond = max(state.condition_number, 1.0)
        log_cond = math.log(cond) / 10.0  # Normalize
        log_cond = max(-1.0, min(1.0, log_cond))
        features.append(log_cond)
        
        # 8. Orthogonality error (log-scale)
        ortho_err = max(state.orthogonality_error, 1e-12)
        log_ortho = math.log(ortho_err) / 15.0 + 1.0  # Shift and normalize
        log_ortho = max(-1.0, min(1.0, log_ortho))
        features.append(log_ortho)
        
        # 9. Epoch progress (0 to 1, shifted to -1 to 1)
        epoch_prog = state.epoch / max(state.total_epochs, 1)
        epoch_prog = 2.0 * epoch_prog - 1.0
        features.append(epoch_prog)
        
        # 10. Step-within-epoch progress
        step_prog = (state.step % state.steps_per_epoch) / max(state.steps_per_epoch, 1)
        step_prog = 2.0 * step_prog - 1.0
        features.append(step_prog)
        
        # 11. Width multiplier (log-scale, for muP context)
        log_width = math.log(state.width_mult) / 2.0
        log_width = max(-1.0, min(1.0, log_width))
        features.append(log_width)
        
        # 12. Loss stability (coefficient of variation over recent history)
        if len(self._loss_history) >= 5:
            recent_loss = np.array(self._loss_history[-20:])
            cv = np.std(recent_loss) / (np.mean(recent_loss) + 1e-8)
            stability = -math.log(max(cv, 1e-8)) / 5.0  # High stability = positive
            stability = max(-1.0, min(1.0, stability))
        else:
            stability = 0.0
        features.append(stability)
        
        return torch.tensor(features, dtype=torch.float32, device=device)
    
    def extract_batch(
        self,
        states: List[TrainingState],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Extract features for a batch of states.
        
        Args:
            states: List of training states
            device: Target device
            
        Returns:
            Feature tensor of shape (B, FEATURE_DIM)
        """
        features = [self.extract_features(s, device) for s in states]
        return torch.stack(features, dim=0)


def compute_weight_spectral_stats(
    weight: torch.Tensor,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute spectral statistics of a weight matrix.
    
    Args:
        weight: Weight tensor (2D or will be reshaped)
        top_k: Number of singular values for effective rank computation
        
    Returns:
        Dict with 'top_sv', 'effective_rank', 'condition_number'
    """
    if weight.ndim != 2:
        weight = weight.view(weight.size(0), -1)
    
    with torch.no_grad():
        try:
            svs = torch.linalg.svdvals(weight)
            
            top_sv = svs[0].item()
            
            # Effective rank via spectral entropy
            svs_normalized = svs / (svs.sum() + 1e-8)
            entropy = -(svs_normalized * torch.log(svs_normalized + 1e-12)).sum()
            effective_rank = torch.exp(entropy).item()
            
            # Condition number
            min_sv = svs[-1].item()
            condition_number = top_sv / (min_sv + 1e-8)
            
        except RuntimeError:
            # Fallback for numerical issues
            top_sv = 1.0
            effective_rank = 1.0
            condition_number = 1.0
    
    return {
        'top_sv': top_sv,
        'effective_rank': effective_rank,
        'condition_number': condition_number,
    }


def compute_orthogonality_error(weight: torch.Tensor) -> float:
    """
    Compute deviation from orthogonality: ||W^T W - I||_F.
    
    For non-square matrices, computes ||W^T W - I||_F where I is appropriately sized.
    """
    if weight.ndim != 2:
        weight = weight.view(weight.size(0), -1)
    
    with torch.no_grad():
        WtW = weight.T @ weight
        n = min(WtW.size(0), WtW.size(1))
        I = torch.eye(n, device=weight.device, dtype=weight.dtype)
        error = torch.norm(WtW[:n, :n] - I, p='fro').item()
    
    return error


def compute_update_stats(
    update: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute statistics of a weight update (delta W).
    
    Args:
        update: Update tensor (should be 2D)
        
    Returns:
        Dict with 'spectral_norm', 'effective_rank', 'frobenius_norm'
    """
    if update.ndim != 2:
        update = update.view(update.size(0), -1)
    
    with torch.no_grad():
        try:
            svs = torch.linalg.svdvals(update)
            spectral_norm = svs[0].item()
            
            # Effective rank
            svs_normalized = svs / (svs.sum() + 1e-8)
            entropy = -(svs_normalized * torch.log(svs_normalized + 1e-12)).sum()
            effective_rank = torch.exp(entropy).item()
            
        except RuntimeError:
            spectral_norm = torch.norm(update, p='fro').item()
            effective_rank = 1.0
        
        frobenius_norm = torch.norm(update, p='fro').item()
    
    return {
        'spectral_norm': spectral_norm,
        'effective_rank': effective_rank,
        'frobenius_norm': frobenius_norm,
    }


class LiveDynamicsTracker:
    """
    Real-time tracker for training dynamics during optimization.
    
    Integrates with the training loop to maintain running statistics
    and provide TrainingState objects for the TRM router.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tracked_layer_name: str = None,
        history_length: int = 50,
        total_epochs: int = 100,
        steps_per_epoch: int = 390,
        width_mult: float = 1.0,
    ):
        """
        Args:
            model: The model being trained
            tracked_layer_name: Name of layer to track for spectral stats
            history_length: Length of history buffers
            total_epochs: Total epochs for progress calculation
            steps_per_epoch: Steps per epoch
            width_mult: Model width multiplier
        """
        self.model = model
        self.tracked_layer_name = tracked_layer_name
        self.history_length = history_length
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.width_mult = width_mult
        
        # Find tracked layer
        self._tracked_weight = None
        if tracked_layer_name:
            for name, module in model.named_modules():
                if name == tracked_layer_name and hasattr(module, 'weight'):
                    self._tracked_weight = module.weight
                    break
        
        # If not found, use first Linear layer
        if self._tracked_weight is None:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    self._tracked_weight = module.weight
                    break
        
        # State tracking
        self._loss_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._epoch = 0
        self._step = 0
        self._last_update: Optional[torch.Tensor] = None
        
    def update(
        self,
        loss: float,
        grad_norm: Optional[float] = None,
        epoch: int = None,
        step: int = None,
        update: Optional[torch.Tensor] = None,
    ):
        """Update tracker with current training state."""
        self._loss_history.append(loss)
        if len(self._loss_history) > self.history_length:
            self._loss_history = self._loss_history[-self.history_length:]
        
        if grad_norm is not None:
            self._grad_norm_history.append(grad_norm)
            if len(self._grad_norm_history) > self.history_length:
                self._grad_norm_history = self._grad_norm_history[-self.history_length:]
        
        if epoch is not None:
            self._epoch = epoch
        if step is not None:
            self._step = step
        if update is not None:
            self._last_update = update.clone().detach()
    
    def get_state(self) -> TrainingState:
        """Get current training state for TRM router."""
        state = TrainingState(
            current_loss=self._loss_history[-1] if self._loss_history else 0.0,
            loss_history=self._loss_history.copy(),
            grad_norm=self._grad_norm_history[-1] if self._grad_norm_history else 0.0,
            grad_norm_history=self._grad_norm_history.copy(),
            epoch=self._epoch,
            step=self._step,
            total_epochs=self.total_epochs,
            steps_per_epoch=self.steps_per_epoch,
            width_mult=self.width_mult,
        )
        
        # Compute spectral stats for tracked weight
        if self._tracked_weight is not None:
            spec_stats = compute_weight_spectral_stats(self._tracked_weight)
            state.top_singular_value = spec_stats['top_sv']
            state.effective_rank = spec_stats['effective_rank']
            state.condition_number = spec_stats['condition_number']
            state.orthogonality_error = compute_orthogonality_error(self._tracked_weight)
        
        # Compute update stats
        if self._last_update is not None:
            update_stats = compute_update_stats(self._last_update)
            state.update_spectral_norm = update_stats['spectral_norm']
            state.update_effective_rank = update_stats['effective_rank']
        
        return state
    
    def reset(self):
        """Reset all tracked state."""
        self._loss_history.clear()
        self._grad_norm_history.clear()
        self._epoch = 0
        self._step = 0
        self._last_update = None
