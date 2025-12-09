"""
Data Collection and Dataset for TRM Router Training

This module provides infrastructure for:
    1. Collecting training dynamics during fixed-solver runs
    2. Building supervised datasets with oracle labels
    3. Creating train/val dataloaders

Workflow:
    1. Run training with each fixed solver, logging dynamics
    2. Merge runs and compute oracle labels (which solver was best at each step)
    3. Train TRM to predict optimal solver given features
"""

import json
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .features import TrainingState, DynamicsFeatureExtractor, LiveDynamicsTracker


@dataclass
class StepRecord:
    """
    Record of training dynamics at a single step.
    
    Captures features before the step and the outcome after.
    """
    
    # Step identification
    epoch: int
    step: int
    solver_name: str
    
    # Features (raw, before normalization)
    current_loss: float
    loss_5step_avg: float
    grad_norm: float
    top_singular_value: float
    effective_rank: float
    condition_number: float
    orthogonality_error: float
    update_spectral_norm: float
    update_effective_rank: float
    width_mult: float
    
    # Outcome (for labeling)
    next_loss: float
    loss_delta: float  # next_loss - current_loss (negative is good)
    
    # Optional: comparison across solvers (filled in post-hoc)
    oracle_solver: Optional[str] = None
    solver_loss_deltas: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TRMDataCollector:
    """
    Collects training dynamics data for offline TRM training.
    
    Usage:
        collector = TRMDataCollector(model, solver_name="admm")
        
        for epoch in range(epochs):
            for batch in loader:
                # Before step
                collector.pre_step(loss, grad_norm, epoch, step)
                
                optimizer.step()
                
                # After step (with next batch)
                collector.post_step(next_loss)
        
        collector.save("dynamics_admm.pkl")
    """
    
    def __init__(
        self,
        model: nn.Module,
        solver_name: str,
        tracked_layer_name: str = None,
        total_epochs: int = 50,
        steps_per_epoch: int = 390,
        width_mult: float = 1.0,
        record_every: int = 1,
    ):
        """
        Args:
            model: Model being trained
            solver_name: Name of inner solver being used
            tracked_layer_name: Layer to track for spectral stats
            total_epochs: Total epochs (for progress features)
            steps_per_epoch: Steps per epoch
            width_mult: Model width multiplier
            record_every: Record every N steps (for reducing data volume)
        """
        self.solver_name = solver_name
        self.record_every = record_every
        
        self.tracker = LiveDynamicsTracker(
            model=model,
            tracked_layer_name=tracked_layer_name,
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            width_mult=width_mult,
        )
        
        self.records: List[StepRecord] = []
        self._pending_state: Optional[TrainingState] = None
        self._step_counter = 0
        
    def pre_step(
        self,
        loss: float,
        grad_norm: float,
        epoch: int,
        step: int,
    ):
        """
        Record state before optimizer step.
        
        Call this after computing loss/grad but before optimizer.step().
        """
        self._step_counter += 1
        
        if self._step_counter % self.record_every != 0:
            return
        
        self.tracker.update(
            loss=loss,
            grad_norm=grad_norm,
            epoch=epoch,
            step=step,
        )
        self._pending_state = self.tracker.get_state()
    
    def post_step(
        self,
        next_loss: float,
        update: Optional[torch.Tensor] = None,
    ):
        """
        Record outcome after optimizer step.
        
        Call this after optimizer.step() with the loss on the same or next batch.
        """
        if self._pending_state is None:
            return
        
        if update is not None:
            self.tracker.update(update=update)
        
        state = self._pending_state
        
        # Compute loss-related history features
        loss_history = state.loss_history
        loss_5step_avg = np.mean(loss_history[-5:]) if len(loss_history) >= 5 else state.current_loss
        
        record = StepRecord(
            epoch=state.epoch,
            step=state.step,
            solver_name=self.solver_name,
            current_loss=state.current_loss,
            loss_5step_avg=loss_5step_avg,
            grad_norm=state.grad_norm,
            top_singular_value=state.top_singular_value,
            effective_rank=state.effective_rank,
            condition_number=state.condition_number,
            orthogonality_error=state.orthogonality_error,
            update_spectral_norm=state.update_spectral_norm,
            update_effective_rank=state.update_effective_rank,
            width_mult=state.width_mult,
            next_loss=next_loss,
            loss_delta=next_loss - state.current_loss,
        )
        
        self.records.append(record)
        self._pending_state = None
    
    def save(self, path: Union[str, Path]):
        """Save collected records to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'solver_name': self.solver_name,
            'num_records': len(self.records),
            'records': [r.to_dict() for r in self.records],
        }
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
        print(f"Saved {len(self.records)} records to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Load collected records from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        # Reconstruct StepRecord objects
        data['records'] = [StepRecord(**r) for r in data['records']]
        return data
    
    def reset(self):
        """Clear collected data."""
        self.records.clear()
        self.tracker.reset()
        self._pending_state = None
        self._step_counter = 0


def merge_solver_runs(
    run_paths: Dict[str, Union[str, Path]],
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Merge data from runs with different solvers and compute oracle labels.
    
    Args:
        run_paths: Dict mapping solver_name -> path to saved data
        output_path: Path to save merged dataset
        
    Returns:
        Merged data dictionary
    """
    # Load all runs
    runs = {}
    for solver_name, path in run_paths.items():
        runs[solver_name] = TRMDataCollector.load(path)
    
    # Check alignment
    num_records = None
    for solver_name, data in runs.items():
        if num_records is None:
            num_records = data['num_records']
        elif data['num_records'] != num_records:
            raise ValueError(
                f"Mismatched record counts: {solver_name} has {data['num_records']}, "
                f"expected {num_records}"
            )
    
    # Compute oracle labels for each step
    merged_records = []
    solver_names = list(runs.keys())
    
    for i in range(num_records):
        # Get loss deltas from each solver
        loss_deltas = {}
        for solver_name, data in runs.items():
            record = data['records'][i]
            loss_deltas[solver_name] = record.loss_delta
        
        # Find oracle (best solver for this step)
        oracle_solver = min(loss_deltas.keys(), key=lambda k: loss_deltas[k])
        
        # Use one solver's record as base
        base_record = runs[solver_names[0]]['records'][i]
        
        # Update with oracle info
        base_record.oracle_solver = oracle_solver
        base_record.solver_loss_deltas = loss_deltas
        
        merged_records.append(base_record)
    
    # Save merged data
    merged_data = {
        'solver_names': solver_names,
        'num_records': len(merged_records),
        'records': [r.to_dict() for r in merged_records],
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(merged_data, f)
    
    # Print summary
    oracle_counts = {}
    for r in merged_records:
        oracle_counts[r.oracle_solver] = oracle_counts.get(r.oracle_solver, 0) + 1
    
    print(f"Merged {len(merged_records)} records from {len(solver_names)} solvers")
    print("Oracle distribution:")
    for solver, count in sorted(oracle_counts.items(), key=lambda x: -x[1]):
        print(f"  {solver}: {count} ({100*count/len(merged_records):.1f}%)")
    
    return merged_data


class DynamicsDataset(Dataset):
    """
    PyTorch Dataset for TRM router training.
    
    Loads pre-collected dynamics data and provides (features, labels) pairs.
    """
    
    SOLVER_NAMES = ["spectral_clip", "dual_ascent", "quasi_newton", "frank_wolfe", "admm"]
    
    def __init__(
        self,
        data_path: Union[str, Path],
        label_mode: str = 'classification',
        include_lr_labels: bool = False,
    ):
        """
        Args:
            data_path: Path to merged dynamics data
            label_mode: 'classification' or 'regression'
            include_lr_labels: If True, also return LR bucket labels
        """
        self.label_mode = label_mode
        self.include_lr_labels = include_lr_labels
        
        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.solver_names = data.get('solver_names', self.SOLVER_NAMES)
        self.records = [StepRecord(**r) if isinstance(r, dict) else r 
                       for r in data['records']]
        
        # Build solver name to index mapping
        self.solver_to_idx = {name: i for i, name in enumerate(self.SOLVER_NAMES)}
        
        # Initialize feature extractor
        self.feature_extractor = DynamicsFeatureExtractor()
        
        # Pre-compute features
        self._precompute_features()
    
    def _precompute_features(self):
        """Pre-compute normalized features for all records."""
        self.features = []
        self.solver_labels = []
        self.loss_deltas = []
        
        for record in self.records:
            # Build TrainingState from record
            state = TrainingState(
                current_loss=record.current_loss,
                loss_history=[record.loss_5step_avg] * 5,
                grad_norm=record.grad_norm,
                grad_norm_history=[record.grad_norm],
                top_singular_value=record.top_singular_value,
                effective_rank=record.effective_rank,
                condition_number=record.condition_number,
                orthogonality_error=record.orthogonality_error,
                update_spectral_norm=record.update_spectral_norm,
                update_effective_rank=record.update_effective_rank,
                epoch=record.epoch,
                step=record.step,
                width_mult=record.width_mult,
            )
            
            # Extract features
            features = self.feature_extractor.extract_features(state)
            self.features.append(features)
            
            # Labels
            if record.oracle_solver:
                self.solver_labels.append(self.solver_to_idx.get(record.oracle_solver, 0))
            else:
                self.solver_labels.append(self.solver_to_idx.get(record.solver_name, 0))
            
            if record.solver_loss_deltas:
                deltas = [record.solver_loss_deltas.get(s, 0.0) for s in self.SOLVER_NAMES]
                self.loss_deltas.append(deltas)
            else:
                self.loss_deltas.append([record.loss_delta] * len(self.SOLVER_NAMES))
        
        self.features = torch.stack(self.features)
        self.solver_labels = torch.tensor(self.solver_labels, dtype=torch.long)
        self.loss_deltas = torch.tensor(self.loss_deltas, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        item = {
            'features': self.features[idx],
            'solver_label': self.solver_labels[idx],
            'loss_deltas': self.loss_deltas[idx],
        }
        
        if self.include_lr_labels:
            item['lr_label'] = torch.tensor(1, dtype=torch.long)  # Default: 1.0x
        
        return item
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        solver_counts = {}
        for label in self.solver_labels.tolist():
            name = self.SOLVER_NAMES[label]
            solver_counts[name] = solver_counts.get(name, 0) + 1
        
        return {
            'num_samples': len(self),
            'feature_dim': self.features.shape[1],
            'num_solvers': len(self.SOLVER_NAMES),
            'solver_distribution': solver_counts,
            'mean_features': self.features.mean(dim=0).tolist(),
            'std_features': self.features.std(dim=0).tolist(),
        }


def create_dataloaders(
    data_path: Union[str, Path],
    batch_size: int = 64,
    train_split: float = 0.8,
    num_workers: int = 0,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders from collected data.
    
    Args:
        data_path: Path to dynamics data
        batch_size: Batch size
        train_split: Fraction for training
        num_workers: DataLoader workers
        **dataset_kwargs: Additional args for DynamicsDataset
        
    Returns:
        (train_loader, val_loader)
    """
    dataset = DynamicsDataset(data_path, **dataset_kwargs)
    
    # Split indices
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader
