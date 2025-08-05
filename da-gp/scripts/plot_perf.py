#!/usr/bin/env python3
"""Generate clean performance scaling plots from benchmark data."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    """Generate performance scaling plot."""
    parser = argparse.ArgumentParser(description="Generate performance plots")
    parser.add_argument(
        "csv_file", 
        help="CSV file from bench.py (e.g., bench.csv)"
    )
    parser.add_argument(
        "--out", 
        default="figures/perf_scaling.pdf",
        help="Output figure path"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sklearn", "dapper"],
        help="Backends to plot (default: sklearn dapper)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.csv_file).exists():
        print(f"Error: {args.csv_file} not found")
        return
    
    # Ensure output directory exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    # Read benchmark data
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading {args.csv_file}: {e}")
        return
    
    # Validate required columns
    required_cols = ["backend", "n_obs", "time_s"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols} in {args.csv_file}")
        return
    
    # Filter to requested backends and remove failed runs
    df = df[df['backend'].isin(args.backends)]
    df = df[df['time_s'] != float('inf')]
    df = df[df['time_s'] > 0]
    
    if df.empty:
        print("Error: No valid benchmark data found for specified backends")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot styling
    colors = {'sklearn': 'blue', 'dapper': 'red', 'pdaf': 'green'}
    markers = {'sklearn': 'o', 'dapper': 's', 'pdaf': '^'}
    
    # Plot each backend
    for backend in args.backends:
        backend_data = df[df['backend'] == backend]
        if backend_data.empty:
            continue
            
        # Sort by n_obs for clean lines
        backend_data = backend_data.sort_values("n_obs")
        
        color = colors.get(backend, 'black')
        marker = markers.get(backend, 'v')
        
        plt.loglog(
            backend_data.n_obs, backend_data.time_s, 
            marker=marker, 
            label=backend.upper(),
            color=color,
            linewidth=2,
            markersize=8
        )
    
    plt.xlabel("Number of observations")
    plt.ylabel("Wall-clock time (s)")
    plt.title("Performance Scaling: DA vs GP")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved performance plot to {args.out}")
    
    # Print summary statistics
    print("\nBenchmark Summary:")
    for backend in args.backends:
        backend_data = df[df['backend'] == backend]
        if not backend_data.empty:
            max_obs = backend_data.n_obs.max()
            max_time = backend_data.time_s.max()
            min_time = backend_data.time_s.min()
            print(f"  {backend.upper()}: {min_time:.3f}s - {max_time:.3f}s (max {max_obs:,} obs)")


if __name__ == "__main__":
    main()