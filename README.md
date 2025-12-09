# TRM Router Experiments: Meta-Optimization for Manifold Muon

**EECS 182 Final Project Extension - Tiny Reasoning Model for Optimizer Routing**

This module implements a Tiny Reasoning Model (TRM) router that learns to select the optimal inner solver for Manifold Muon optimization based on observed training dynamics.

## Overview

### Motivation

Our main project compares several inner solvers (Dual Ascent, ADMM, Frank-Wolfe, Quasi-Newton) for enforcing spectral-norm constraints in Manifold Muon. We observe that different solvers excel in different regimes—ADMM converges faster but requires more computation per step, while Frank-Wolfe produces low-rank updates that may generalize better in certain phases.

This raises a natural question: **Can we learn to route between solvers adaptively based on observed training dynamics?**

### TRM Architecture

Following Jolicoeur-Martineau (2025), we employ a small recursive network that:
1. Takes training dynamics features (loss, gradients, spectral stats) as input
2. Maintains a latent reasoning state z that evolves over recursive steps
3. Outputs a distribution over solver choices

The key insight is that recursive computation with a tiny model can outperform deeper single-pass networks while maintaining minimal overhead.

## Directory Structure

```
trm_router_experiments/
├── README.md
├── requirements.txt
├── muon/                          # Core optimizer (copied from main project)
│   ├── __init__.py
│   ├── inner_solvers.py
│   └── muon_sgd.py
├── trm/                           # TRM implementation
│   ├── __init__.py
│   ├── model.py                   # TRM architecture
│   ├── features.py                # Dynamics feature extraction
│   └── dataset.py                 # Data handling
├── scripts/
│   ├── collect_dynamics.py        # Collect training dynamics
│   ├── train_router.py            # Train TRM offline
│   ├── evaluate_router.py         # Evaluate on held-out data
│   └── online_routing.py          # Online TRM routing (experimental)
├── notebooks/
│   └── TRM_Experiments.ipynb      # Interactive experiments
├── configs/
│   └── default.yaml               # Default hyperparameters
└── results/                       # Output directory
```

## Quick Start

### 1. Collect Training Dynamics

Run fixed-solver training while logging dynamics features:

```bash
# Collect dynamics for each solver
python scripts/collect_dynamics.py --solver dual_ascent --epochs 30 --output results/dynamics_da.pkl
python scripts/collect_dynamics.py --solver admm --epochs 30 --output results/dynamics_admm.pkl
python scripts/collect_dynamics.py --solver frank_wolfe --epochs 30 --output results/dynamics_fw.pkl
```

### 2. Merge and Prepare Dataset

```bash
python scripts/prepare_dataset.py \
    --inputs results/dynamics_da.pkl results/dynamics_admm.pkl results/dynamics_fw.pkl \
    --output results/merged_dynamics.pkl
```

### 3. Train TRM Router (Offline)

```bash
python scripts/train_router.py \
    --data results/merged_dynamics.pkl \
    --output results/trm_router.pt \
    --epochs 100 \
    --deep-supervision
```

### 4. Evaluate

```bash
# Evaluate on held-out steps
python scripts/evaluate_router.py \
    --model results/trm_router.pt \
    --data results/merged_dynamics.pkl \
    --test-split 0.2
```

### 5. Online Routing (Experimental)

```bash
python scripts/online_routing.py \
    --router results/trm_router.pt \
    --epochs 30
```

## TRM Features (Input Vector)

The router receives a 12-dimensional feature vector at each step:

| Feature | Description |
|---------|-------------|
| `log_loss` | Normalized current training loss (log-scale) |
| `loss_trend` | 5-step moving average trend |
| `log_grad_norm` | Normalized gradient norm |
| `gns_proxy` | Gradient noise scale estimate |
| `top_sv` | Top singular value of tracked weight |
| `eff_rank` | Effective rank of last update |
| `log_cond` | Condition number estimate |
| `log_ortho_err` | Orthogonality violation (log-scale) |
| `epoch_progress` | Epoch progress (0→1) |
| `step_progress` | Within-epoch progress |
| `log_width` | Model width multiplier (for muP) |
| `loss_stability` | Recent loss variance |

## Results

### Offline Routing Accuracy

Training the TRM on 10K steps from 3-solver runs:

| Metric | Value |
|--------|-------|
| Oracle Agreement | --- |
| Expected Loss Δ vs Random | ---% |
| Expected Loss Δ vs Always-ADMM | ---% |

### Online Routing (Preliminary)

TRM-routed training shows:
- Comparable final accuracy to best fixed solver
- Adaptive solver selection based on training phase
- Early: prefers high-rank updates (ADMM/DA)
- Late: shifts toward low-rank (FW) for regularization

## Connection to Main Project

This extension demonstrates that optimizer dynamics contain learnable structure that small models can exploit. Key connections:

1. **Spectral Constraints as Trust Regions**: The TRM learns when tighter spectral control helps
2. **Update Rank as Regularization**: Routes to FW when effective rank should decrease
3. **Condition Monitoring**: Avoids quasi-Newton when condition number is extreme

## References

1. Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning with Tiny Networks."
2. Bernstein, J. (2025). "Manifold Muon: Geometry-Aware Optimization."
3. Yang, G. & Hu, E. (2021). "Tensor Programs V: µP."

## License

GPL-3.0 License
