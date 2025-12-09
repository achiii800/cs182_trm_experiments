"""
Muon optimizer module for TRM router experiments.

This is a streamlined version focused on the inner solvers being evaluated
by the TRM router.
"""

from .inner_solvers import (
    BaseInnerSolver,
    SpectralClipSolver,
    DualAscentSolver,
    QuasiNewtonDualSolver,
    FrankWolfeSolver,
    ADMMSolver,
    TangentSpaceProjector,
    get_inner_solver,
    SOLVER_REGISTRY,
)

from .muon_sgd import (
    MuonSGD,
    MuonAdamW,
    create_optimizer,
)

from .metrics import (
    compute_spectral_norms,
    estimate_sharpness,
    estimate_gradient_noise_scale,
)

__all__ = [
    # Inner solvers
    'BaseInnerSolver',
    'SpectralClipSolver',
    'DualAscentSolver',
    'QuasiNewtonDualSolver',
    'FrankWolfeSolver',
    'ADMMSolver',
    'TangentSpaceProjector',
    'get_inner_solver',
    'SOLVER_REGISTRY',
    # Optimizers
    'MuonSGD',
    'MuonAdamW',
    'create_optimizer',
    # Metrics
    'compute_spectral_norms',
    'estimate_sharpness',
    'estimate_gradient_noise_scale',
]
