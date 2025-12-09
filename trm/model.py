"""
Tiny Recursive Model (TRM) for Meta-Optimization

Implements the core TRM architecture adapted for optimizer routing.
The model recursively refines a latent reasoning state z and action embedding y
to select the optimal inner solver configuration.

Architecture Overview:
    Input: x = embedded training dynamics features
    States: y = action embedding (solver choice), z = latent reasoning state
    
    For T recursive steps:
        z = TRMBlock(x, y, z)  # Update latent reasoning
    y = AnswerHead(y, z)       # Refine action embedding
    
    Output: logits over discrete action space (solver choice + hyperparameter bucket)

Reference: Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning
           with Tiny Networks." arXiv:2510.04871
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class TRMBlock(nn.Module):
    """
    Core TRM block: updates latent reasoning state z given (x, y, z).
    
    This is a 2-layer MLP with residual connection, following the TRM paper's
    finding that shallow networks with recursive application outperform
    deeper networks with single forward passes.
    
    Architecture:
        z' = z + MLP([x; y; z])
        
    where MLP is a 2-layer network with SwiGLU activation.
    """
    
    def __init__(
        self,
        input_dim: int,   # x dimension
        action_dim: int,  # y dimension
        latent_dim: int,  # z dimension
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        total_in = input_dim + action_dim + latent_dim
        
        self.norm = RMSNorm(total_in)
        
        # SwiGLU-style: gate * linear
        self.w_gate = nn.Linear(total_in, hidden_dim, bias=False)
        self.w_up = nn.Linear(total_in, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, latent_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,  # (B, input_dim) input features
        y: torch.Tensor,  # (B, action_dim) action embedding
        z: torch.Tensor,  # (B, latent_dim) latent reasoning
    ) -> torch.Tensor:
        """
        Update latent state z given (x, y, z).
        
        Returns:
            Updated z with shape (B, latent_dim)
        """
        # Concatenate inputs
        h = torch.cat([x, y, z], dim=-1)
        h = self.norm(h)
        
        # SwiGLU: gate * linear
        gate = F.silu(self.w_gate(h))
        hidden = self.w_up(h)
        h = gate * hidden
        
        # Project back to latent dim with residual
        delta_z = self.dropout(self.w_down(h))
        
        return z + delta_z


class AnswerHead(nn.Module):
    """
    Updates action embedding y given (y, z).
    
    This refines the action selection based on accumulated reasoning.
    """
    
    def __init__(
        self,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.norm = RMSNorm(action_dim + latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim + latent_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim, bias=False),
        )
        
    def forward(
        self,
        y: torch.Tensor,  # (B, action_dim)
        z: torch.Tensor,  # (B, latent_dim)
    ) -> torch.Tensor:
        h = torch.cat([y, z], dim=-1)
        h = self.norm(h)
        return y + self.mlp(h)


class HaltingModule(nn.Module):
    """
    Adaptive halting mechanism for TRM.
    
    Learns when to stop recursive reasoning based on current (y, z) state.
    Outputs a probability of halting; training uses expected computation.
    
    During inference, can use a threshold or expected depth.
    """
    
    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(action_dim + latent_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = torch.cat([y, z], dim=-1)
        return torch.sigmoid(self.linear(h)).squeeze(-1)


class TRMRouter(nn.Module):
    """
    Tiny Recursive Model for optimizer routing.
    
    Given training dynamics features, outputs a distribution over:
        - Inner solver choice: {DualAscent, ADMM, FrankWolfe, QuasiNewton, SpectralClip}
        - LR multiplier bucket: {0.5x, 1.0x, 2.0x}
    
    Architecture:
        1. Embed input features x
        2. Initialize y (action embedding) and z (latent reasoning)
        3. For T reasoning cycles:
           a. For n inner steps: z = TRMBlock(x, y, z)
           b. y = AnswerHead(y, z)
        4. Project y to logits over action space
    
    Args:
        input_dim: Dimension of input features (training dynamics)
        num_solvers: Number of inner solver choices
        num_lr_buckets: Number of LR multiplier buckets
        action_dim: Dimension of action embedding
        latent_dim: Dimension of latent reasoning state
        hidden_dim: Hidden dimension in MLP layers
        n_inner_steps: Number of inner reasoning steps per cycle
        n_cycles: Number of supervision/refinement cycles
        use_halting: If True, use adaptive halting
        dropout: Dropout rate
    """
    
    # Solver names for reference
    SOLVER_NAMES = ["spectral_clip", "dual_ascent", "quasi_newton", "frank_wolfe", "admm"]
    LR_MULTIPLIERS = [0.5, 1.0, 2.0]
    
    def __init__(
        self,
        input_dim: int = 12,
        num_solvers: int = 5,
        num_lr_buckets: int = 3,
        action_dim: int = 64,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_inner_steps: int = 4,
        n_cycles: int = 3,
        use_halting: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_solvers = num_solvers
        self.num_lr_buckets = num_lr_buckets
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.n_inner_steps = n_inner_steps
        self.n_cycles = n_cycles
        self.use_halting = use_halting
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Learnable initial states
        self.init_y = nn.Parameter(torch.zeros(action_dim))
        self.init_z = nn.Parameter(torch.zeros(latent_dim))
        
        # TRM core block (single block applied recursively)
        self.trm_block = TRMBlock(
            input_dim=action_dim,  # x is embedded
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # Answer refinement head
        self.answer_head = AnswerHead(action_dim, latent_dim, hidden_dim // 2)
        
        # Adaptive halting (optional)
        if use_halting:
            self.halting = HaltingModule(action_dim, latent_dim)
        
        # Output heads
        self.solver_head = nn.Linear(action_dim, num_solvers)
        self.lr_head = nn.Linear(action_dim, num_lr_buckets)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using truncated normal."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize learnable states
        nn.init.trunc_normal_(self.init_y, std=0.02)
        nn.init.trunc_normal_(self.init_z, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_cycles: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with recursive reasoning.
        
        Args:
            x: Input features (B, input_dim)
            return_all_cycles: If True, return outputs from all cycles (for deep supervision)
            
        Returns:
            Dict with keys:
                - 'solver_logits': (B, num_solvers) logits for solver choice
                - 'lr_logits': (B, num_lr_buckets) logits for LR multiplier
                - 'halt_probs': (B, n_cycles) halting probabilities if use_halting
                - 'all_solver_logits': List of (B, num_solvers) if return_all_cycles
        """
        B = x.size(0)
        device = x.device
        
        # Embed input
        x_emb = self.input_embed(x)  # (B, action_dim)
        
        # Initialize states
        y = self.init_y.unsqueeze(0).expand(B, -1).clone()  # (B, action_dim)
        z = self.init_z.unsqueeze(0).expand(B, -1).clone()  # (B, latent_dim)
        
        all_solver_logits = []
        all_lr_logits = []
        halt_probs = []
        
        for cycle in range(self.n_cycles):
            # Inner reasoning steps (update z)
            for _ in range(self.n_inner_steps):
                z = self.trm_block(x_emb, y, z)
            
            # Refine answer embedding
            y = self.answer_head(y, z)
            
            # Compute outputs for this cycle
            if return_all_cycles or cycle == self.n_cycles - 1:
                solver_logits = self.solver_head(y)
                lr_logits = self.lr_head(y)
                all_solver_logits.append(solver_logits)
                all_lr_logits.append(lr_logits)
            
            # Halting probability
            if self.use_halting:
                halt_probs.append(self.halting(y, z))
            
            # Detach for next cycle (deep supervision without full BPTT)
            if cycle < self.n_cycles - 1:
                y = y.detach()
                z = z.detach()
        
        result = {
            'solver_logits': all_solver_logits[-1],
            'lr_logits': all_lr_logits[-1],
        }
        
        if return_all_cycles:
            result['all_solver_logits'] = all_solver_logits
            result['all_lr_logits'] = all_lr_logits
        
        if self.use_halting:
            result['halt_probs'] = torch.stack(halt_probs, dim=1)
        
        return result
    
    def predict(self, x: torch.Tensor) -> Tuple[str, float]:
        """
        Inference mode: return discrete action.
        
        Args:
            x: Input features (1, input_dim) or (input_dim,)
            
        Returns:
            Tuple of (solver_name, lr_multiplier)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            solver_idx = outputs['solver_logits'].argmax(dim=-1).item()
            lr_idx = outputs['lr_logits'].argmax(dim=-1).item()
        
        solver_name = self.SOLVER_NAMES[solver_idx]
        lr_mult = self.LR_MULTIPLIERS[lr_idx]
        
        return solver_name, lr_mult
    
    def get_action_distribution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get probability distributions over actions.
        
        Args:
            x: Input features (B, input_dim)
            
        Returns:
            Dict with 'solver_probs' and 'lr_probs' as probability distributions
        """
        outputs = self.forward(x)
        return {
            'solver_probs': F.softmax(outputs['solver_logits'], dim=-1),
            'lr_probs': F.softmax(outputs['lr_logits'], dim=-1),
        }
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TRMRouterSmall(TRMRouter):
    """
    Compact TRM router with ~100K parameters.
    
    Designed for minimal overhead during training.
    """
    
    def __init__(self, input_dim: int = 12, **kwargs):
        super().__init__(
            input_dim=input_dim,
            action_dim=32,
            latent_dim=64,
            hidden_dim=128,
            n_inner_steps=3,
            n_cycles=2,
            **kwargs
        )


class TRMRouterTiny(TRMRouter):
    """
    Ultra-compact TRM router with ~20K parameters.
    
    Absolute minimal version for proof-of-concept experiments.
    """
    
    def __init__(self, input_dim: int = 12, **kwargs):
        super().__init__(
            input_dim=input_dim,
            action_dim=16,
            latent_dim=32,
            hidden_dim=64,
            n_inner_steps=2,
            n_cycles=2,
            **kwargs
        )


def create_trm_router(
    size: str = "small",
    input_dim: int = 12,
    **kwargs
) -> TRMRouter:
    """
    Factory function for TRM router variants.
    
    Args:
        size: One of 'tiny', 'small', 'base'
        input_dim: Dimension of input features
        **kwargs: Additional arguments passed to constructor
        
    Returns:
        TRMRouter instance
    """
    if size == "tiny":
        return TRMRouterTiny(input_dim=input_dim, **kwargs)
    elif size == "small":
        return TRMRouterSmall(input_dim=input_dim, **kwargs)
    elif size == "base":
        return TRMRouter(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown size: {size}. Choose from 'tiny', 'small', 'base'")
