#!/usr/bin/env python3
"""Benchmark script for DA vs GP performance comparison."""

import argparse
import csv
import sys
import time
import subprocess
from pathlib import Path

DEFAULT_BACKENDS = ["sklearn", "dapper"]


def run_once(backend: str, n_obs: int, grid_size: int = None) -> float:
    """Run single benchmark and return elapsed time."""
    cmd = [
        sys.executable, "-m", "da_gp.src.cli", 
        "--backend", backend, 
        "--n_obs", str(n_obs)
    ]
    
    if grid_size is not None:
        cmd.extend(["--grid_size", str(grid_size)])
    
    print(f"Running {backend} with {n_obs} observations, grid_size={grid_size or 'default'}...")
    t0 = time.perf_counter()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - t0
        
        # Extract CSV line from output if available (format: backend,n_obs,grid_size,time_s,rmse)
        for line in result.stdout.split('\n'):
            if line.startswith('CSV:'):
                parts = line.split(',')
                if len(parts) >= 4:
                    return float(parts[3])  # Use reported time (now at index 3)
        
        return elapsed
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {backend}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return float('inf')  # Mark as failed


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark DA vs GP performance")
    
    # Observation sweep parameters
    parser.add_argument(
        "--n_obs_grid", 
        type=int, 
        nargs="+",
        help="Grid of observation counts to test"
    )
    parser.add_argument(
        "--n_obs_fixed",
        type=int,
        default=2000,
        help="Fixed observation count when sweeping dimensions (default: 2000)"
    )
    
    # Dimension sweep parameters  
    parser.add_argument(
        "--dim_grid",
        type=int,
        nargs="+", 
        help="Grid of state dimensions (grid sizes) to test"
    )
    parser.add_argument(
        "--grid_size_fixed",
        type=int,
        default=2000,
        help="Fixed grid size when sweeping observations (default: 2000)"
    )
    
    parser.add_argument(
        "--csv", 
        default="data/bench.csv",
        help="Output CSV file (default: data/bench.csv)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["sklearn", "dapper", "pdaf"],
        help="Backends to test (default: sklearn dapper)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments - need either obs_grid or dim_grid
    if not args.n_obs_grid and not args.dim_grid:
        parser.error("Must specify either --n_obs_grid or --dim_grid (or both)")
    
    # Ensure output directory exists
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Benchmarking backends: {args.backends}")
    if args.n_obs_grid:
        print(f"Observation counts: {args.n_obs_grid} (grid_size={args.grid_size_fixed})")
    if args.dim_grid:
        print(f"Grid sizes: {args.dim_grid} (n_obs={args.n_obs_fixed})")
    print(f"Output: {args.csv}")
    
    with open(args.csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["backend", "n_obs", "grid_size", "time_s"])
        
        # Independent sweeps as specified in plan
        for d in (args.dim_grid or [args.grid_size_fixed]):
            for m in (args.n_obs_grid or [args.n_obs_fixed]):
                for backend in args.backends:
                    elapsed = run_once(backend, m, d)
                    writer.writerow([backend, m, d, f"{elapsed:.3f}"])
                    print(f"  {backend} (n_obs={m}, grid_size={d}): {elapsed:.3f}s")
    
    print(f"Benchmark complete. Results saved to {args.csv}")


if __name__ == "__main__":
    main()