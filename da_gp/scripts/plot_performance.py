#!/usr/bin/env python3
"""Generate a combined performance scaling plot for the paper."""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# tueplots is the key for publication-quality figures
from tueplots import bundles, figsizes


def plot_ax(ax, df, x_col, fixed_col, backends, colors): # Now accepts colors dict
    """Helper function to plot data on a given axes object."""
    # Determine fixed value from the data (most common value)
    if not df[fixed_col].empty:
        fixed_val = df[fixed_col].mode().iloc[0]
    else:
        print(f"Warning: No data to determine fixed value for {fixed_col}")
        return

    # Filter data for the plot
    df_filtered = df[df[fixed_col] == fixed_val]
    if df_filtered.empty:
        print(f"Warning: No data found for {fixed_col}={fixed_val}")
        return

    # --- FIX 1: Use the consistent color map passed from main ---
    markers = {'sklearn': 'o', 'dapper_enkf': 's', 'dapper_letkf': '^'}

    for backend in backends:
        backend_data = df_filtered[df_filtered['backend'] == backend].sort_values(x_col)
        if backend_data.empty:
            continue

        ax.loglog(
            backend_data[x_col], backend_data['time_s'],
            marker=markers.get(backend, 'v'),
            label=backend.upper(),
            color=colors.get(backend, 'black'), # Use the consistent color
            linewidth=2,
            markersize=6
        )

    # Set labels and title
    if x_col == "n_obs":
        ax.set_xlabel("Number of observations ($m$)")
        ax.set_title(f"Scaling vs. Observations ($d={fixed_val:,}$)")
    else:
        ax.set_xlabel("State dimension ($d$)")
        ax.set_title(f"Scaling vs. Dimension ($m={fixed_val:,}$)")

    ax.set_ylabel("Wall-clock time (s)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Generate combined performance plots.")
    parser.add_argument("obs_csv", help="CSV for observation scaling (e.g., data/bench_obs.csv)")
    parser.add_argument("dim_csv", help="CSV for dimension scaling (e.g., data/bench_dim.csv)")
    parser.add_argument("--out", default="figures/perf_scaling.pdf", help="Output figure path")
    parser.add_argument("--backends", nargs="+", default=["sklearn", "dapper_enkf", "dapper_letkf"], help="Backends to plot")

    args = parser.parse_args()

    # --- FIX 1: Centralize the color scheme generation ---
    # This ensures both plotting scripts use the same colors for the same backends.
    color_map = plt.colormaps['viridis']
    colors = {
        backend: color_map(i / len(args.backends))
        for i, backend in enumerate(args.backends)
    }
    # Check if input files exist
    if not Path(args.obs_csv).exists():
        print(f"Error: {args.obs_csv} not found")
        return 1

    if not Path(args.dim_csv).exists():
        print(f"Error: {args.dim_csv} not found")
        return 1

    # Use tueplots to set figure size and font style for JMLR
    # This specifies a figure with 1 row and 2 columns of subplots.
    plt.rcParams.update(bundles.jmlr2001(nrows=1, ncols=2))

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2)

    try:
        df_obs = pd.read_csv(args.obs_csv)
        df_dim = pd.read_csv(args.dim_csv)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return 1

    # Validate CSV structure
    required_cols = ['backend', 'n_obs', 'grid_size', 'time_s']
    for df, name in [(df_obs, args.obs_csv), (df_dim, args.dim_csv)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Error: {name} missing columns: {missing}")
            return 1

    # Plot observation scaling on the left, passing the colors dictionary
    plot_ax(axs[0], df_obs, x_col="n_obs", fixed_col="grid_size", backends=args.backends, colors=colors)

    # Plot dimension scaling on the right, passing the same colors dictionary
    plot_ax(axs[1], df_dim, x_col="grid_size", fixed_col="n_obs", backends=args.backends, colors=colors)

    # Adjust layout and save
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved combined performance plot to {args.out}")

    return 0


if __name__ == "__main__":
    exit(main())