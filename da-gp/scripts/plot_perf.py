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
        "--x", 
        choices=["obs", "dim"],
        required=True,
        help="X-axis variable: 'obs' (observations) or 'dim' (dimensions/grid_size)"
    )
    parser.add_argument(
        "--out", 
        default=None,
        help="Output figure path (default: auto-generated based on --x)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sklearn", "dapper"],
        help="Backends to plot (default: sklearn dapper)"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output filename based on x-axis
    if args.out is None:
        if args.x == "obs":
            args.out = "figures/perf_scaling_obs.pdf"
        else:  # dim
            args.out = "figures/perf_scaling_dim.pdf"
    
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
    required_cols = ["backend", "n_obs", "grid_size", "time_s"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols} in {args.csv_file}")
        print(f"Expected columns: {required_cols}")
        print(f"Found columns: {list(df.columns)}")
        return
    
    # Filter to requested backends and remove failed runs
    df = df[df['backend'].isin(args.backends)]
    df = df[df['time_s'] != float('inf')]
    df = df[df['time_s'] > 0]
    
    if df.empty:
        print("Error: No valid benchmark data found for specified backends")
        return
    
    # For each plot type, we need to filter on the complementary fixed value
    # Find the most common fixed value to filter by
    if args.x == "obs":
        # X-axis is observations, so filter by fixed grid_size
        fixed_col = "grid_size"
        vary_col = "n_obs"
        xlabel = "Number of observations"
        fixed_val = df[fixed_col].mode().iloc[0] if not df[fixed_col].empty else df[fixed_col].iloc[0]
    else:  # dim
        # X-axis is dimensions, so filter by fixed n_obs
        fixed_col = "n_obs" 
        vary_col = "grid_size"
        xlabel = "State dimension (grid size)"
        fixed_val = df[fixed_col].mode().iloc[0] if not df[fixed_col].empty else df[fixed_col].iloc[0]
    
    # Filter by fixed value
    df_filtered = df[df[fixed_col] == fixed_val]
    
    if df_filtered.empty:
        print(f"Error: No data found with {fixed_col}={fixed_val}")
        print(f"Available {fixed_col} values: {sorted(df[fixed_col].unique())}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot styling
    colors = {'sklearn': 'blue', 'dapper': 'red', 'pdaf': 'green'}
    markers = {'sklearn': 'o', 'dapper': 's', 'pdaf': '^'}
    
    # Plot each backend
    for backend in args.backends:
        backend_data = df_filtered[df_filtered['backend'] == backend]
        if backend_data.empty:
            continue
            
        # Sort by varying column for clean lines
        backend_data = backend_data.sort_values(vary_col)
        
        color = colors.get(backend, 'black')
        marker = markers.get(backend, 'v')
        
        plt.loglog(
            backend_data[vary_col], backend_data['time_s'], 
            marker=marker, 
            label=backend.upper(),
            color=color,
            linewidth=2,
            markersize=8
        )
    
    plt.xlabel(xlabel)
    plt.ylabel("Wall-clock time (s)")
    
    # Update title based on plot type
    if args.x == "obs":
        plt.title(f"Performance vs Observations (d={fixed_val})")
    else:
        plt.title(f"Performance vs State Dimension (m={fixed_val})")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved performance plot to {args.out}")
    
    # Print summary statistics
    print(f"\nBenchmark Summary (filtered by {fixed_col}={fixed_val}):")
    for backend in args.backends:
        backend_data = df_filtered[df_filtered['backend'] == backend]
        if not backend_data.empty:
            max_vary = backend_data[vary_col].max()
            max_time = backend_data['time_s'].max()
            min_time = backend_data['time_s'].min()
            if args.x == "obs":
                print(f"  {backend.upper()}: {min_time:.3f}s - {max_time:.3f}s (max {max_vary:,} obs)")
            else:
                print(f"  {backend.upper()}: {min_time:.3f}s - {max_time:.3f}s (max {max_vary:,} dim)")


if __name__ == "__main__":
    main()