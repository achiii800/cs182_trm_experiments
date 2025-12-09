#!/usr/bin/env python3
"""
Merge dynamics data from multiple solver runs.

Usage:
    python scripts/prepare_dataset.py \
        --inputs results/dynamics_da.pkl results/dynamics_admm.pkl results/dynamics_fw.pkl \
        --output results/merged_dynamics.pkl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trm import merge_solver_runs


def main():
    parser = argparse.ArgumentParser(description="Merge solver dynamics data")
    
    parser.add_argument("--inputs", type=str, nargs="+", required=True,
                       help="Input dynamics files (format: path or solver_name:path)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output merged file path")
    
    args = parser.parse_args()
    
    # Parse inputs
    run_paths = {}
    for inp in args.inputs:
        if ":" in inp:
            name, path = inp.split(":", 1)
        else:
            # Infer solver name from filename
            path = inp
            stem = Path(path).stem
            # Try to extract solver name
            for solver in ["dual_ascent", "admm", "frank_wolfe", "quasi_newton", "spectral_clip"]:
                if solver in stem or solver.replace("_", "") in stem:
                    name = solver
                    break
            else:
                name = stem
        
        run_paths[name] = path
        print(f"  {name}: {path}")
    
    print(f"\nMerging {len(run_paths)} solver runs...")
    
    merged = merge_solver_runs(run_paths, args.output)
    
    print(f"\nSaved merged dataset to {args.output}")


if __name__ == "__main__":
    main()
