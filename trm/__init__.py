"""
TRM Router: Tiny Recursive Model for Meta-Optimization of Inner Solvers

This module implements a TRM-based controller that observes training dynamics
and routes between inner solvers (Dual Ascent, ADMM, Frank-Wolfe, etc.) based
on learned heuristics.

Key Components:
    - TRMRouter: Core TRM architecture for solver selection
    - DynamicsFeatureExtractor: Extracts training state features
    - TRMDataCollector: Collects data for offline training

Reference:
    - Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning
      with Tiny Networks." arXiv:2510.04871
"""

from .model import (
    TRMRouter,
    TRMRouterSmall,
    TRMRouterTiny,
    TRMBlock,
    AnswerHead,
    HaltingModule,
    RMSNorm,
    create_trm_router,
)

from .features import (
    DynamicsFeatureExtractor,
    TrainingState,
    LiveDynamicsTracker,
    compute_weight_spectral_stats,
    compute_orthogonality_error,
    compute_update_stats,
)

from .dataset import (
    TRMDataCollector,
    DynamicsDataset,
    StepRecord,
    merge_solver_runs,
    create_dataloaders,
)

__all__ = [
    # Model
    "TRMRouter",
    "TRMRouterSmall",
    "TRMRouterTiny",
    "TRMBlock",
    "AnswerHead",
    "HaltingModule",
    "RMSNorm",
    "create_trm_router",
    # Features
    "DynamicsFeatureExtractor",
    "TrainingState",
    "LiveDynamicsTracker",
    "compute_weight_spectral_stats",
    "compute_orthogonality_error",
    "compute_update_stats",
    # Dataset
    "TRMDataCollector",
    "DynamicsDataset",
    "StepRecord",
    "merge_solver_runs",
    "create_dataloaders",
]
