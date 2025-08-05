#!/usr/bin/env python3
"""Benchmark script for DA vs GP performance comparison."""

import argparse
import csv
import sys
import time
import subprocess
from pathlib import Path

DEFAULT_BACKENDS = ["sklearn", "dapper"]


def run_once(backend: str, n_obs: int) -> float:
    """Run single benchmark and return elapsed time."""
    cmd = [
        sys.executable, "-m", "src.cli", 
        "--backend", backend, 
        "--n_obs", str(n_obs)
    ]
    
    print(f"Running {backend} with {n_obs} observations...")
    t0 = time.perf_counter()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - t0
        
        # Extract CSV line from output if available
        for line in result.stdout.split('\n'):
            if line.startswith('CSV:'):
                parts = line.split(',')
                if len(parts) >= 3:
                    return float(parts[2])  # Use reported time
        
        return elapsed
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {backend}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return float('inf')  # Mark as failed


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark DA vs GP performance")
    parser.add_argument(
        "--obs_grid", 
        type=int, 
        nargs="+", 
        required=True,
        help="Grid of observation counts to test"
    )
    parser.add_argument(
        "--csv", 
        default="bench.csv",
        help="Output CSV file (default: bench.csv)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["sklearn", "dapper", "pdaf"],
        help="Backends to test (default: sklearn dapper)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Benchmarking backends: {args.backends}")
    print(f"Observation counts: {args.obs_grid}")
    print(f"Output: {args.csv}")
    
    with open(args.csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["backend", "n_obs", "time_s"])
        
        for n_obs in args.obs_grid:
            for backend in args.backends:
                elapsed = run_once(backend, n_obs)
                writer.writerow([backend, n_obs, f"{elapsed:.3f}"])
                print(f"  {backend}: {elapsed:.3f}s")
    
    print(f"Benchmark complete. Results saved to {args.csv}")


if __name__ == "__main__":
    main()