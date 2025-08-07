#!/usr/bin/env python3
# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate dual-curve log-log timing plots from benchmark data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Import unified figure styling
from da_gp.figstyle import setup_figure_style


def plot_timing_curves(
    df: pd.DataFrame,
    output_path: str = "figures/timing_comparison.pdf",
    colorblind_friendly: bool = False,
    show_legend: bool = True,
):
    """Create dual-curve log-log plots showing fit and predict times."""

    # Setup unified styling
    backend_styles = setup_figure_style(colorblind_friendly=colorblind_friendly)

    # Auto-detect sweep type based on data variation
    n_obs_unique = df["n_obs"].nunique()
    grid_size_unique = df["grid_size"].nunique()

    if n_obs_unique > 1 and grid_size_unique == 1:
        # Observation sweep
        x_col = "n_obs"
        x_label = "observations (m)"
        sort_col = "n_obs"
    elif grid_size_unique > 1 and n_obs_unique == 1:
        # Dimension sweep
        x_col = "grid_size"
        x_label = "state dimension (d)"
        sort_col = "grid_size"
    else:
        # Fallback - use n_obs if both vary or neither vary
        x_col = "n_obs"
        x_label = "observations (m)"
        sort_col = "n_obs"
        print(
            f"Warning: Ambiguous sweep type (n_obs unique: {n_obs_unique}, grid_size unique: {grid_size_unique}), defaulting to n_obs"
        )

    # Create figure with two subplots side by side
    fig, (ax_fit, ax_pred) = plt.subplots(1, 2)

    # Plot fit times (left subplot)
    ax_fit.set_title("Training Time")
    ax_fit.set_xlabel(x_label)
    ax_fit.set_ylabel("wall-clock time [s]")

    for backend, style in backend_styles.items():
        backend_data = df[df["backend"] == backend]
        if len(backend_data) > 0:
            # Hardening: Skip backends with insufficient data points
            if backend_data.shape[0] < 2:
                print(
                    f"Warning: Skipping {backend} - insufficient data points ({backend_data.shape[0]} < 2)"
                )
                continue

            # Sort by detected sweep variable for clean line plots
            backend_data = backend_data.sort_values(sort_col)

            ax_fit.loglog(
                backend_data[x_col],
                backend_data["fit_time"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linestyle=style["linestyle"],
                alpha=0.8,
                markersize=6,
            )

    ax_fit.grid(True, alpha=0.3)
    if show_legend:
        ax_fit.legend()

    # Explicit log scale setting for clarity (although loglog already forces it)
    ax_fit.set_xscale("log")
    ax_fit.set_yscale("log")

    # Plot predict times (right subplot)
    ax_pred.set_title("Prediction Time")
    ax_pred.set_xlabel(x_label)
    ax_pred.set_ylabel("wall-clock time [s]")

    for backend, style in backend_styles.items():
        backend_data = df[df["backend"] == backend]
        if len(backend_data) > 0:
            # Hardening: Skip backends with insufficient data points
            if backend_data.shape[0] < 2:
                print(
                    f"Warning: Skipping {backend} - insufficient data points ({backend_data.shape[0]} < 2)"
                )
                continue

            backend_data = backend_data.sort_values(sort_col)

            ax_pred.loglog(
                backend_data[x_col],
                backend_data["predict_time"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linestyle=style["linestyle"],
                alpha=0.8,
                markersize=6,
            )

    ax_pred.grid(True, alpha=0.3)
    if show_legend:
        ax_pred.legend()

    # Explicit log scale setting for clarity
    ax_pred.set_xscale("log")
    ax_pred.set_yscale("log")

    # Adjust layout and save
    plt.tight_layout()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Timing comparison plot saved to: {output_path}")

    return fig


def plot_dimension_scaling(
    df: pd.DataFrame,
    output_path: str = "figures/dimension_scaling.pdf",
    fixed_n_obs: int = None,
    colorblind_friendly: bool = False,
    show_legend: bool = True,
):
    """Create plots showing scaling with state dimension."""

    # Setup unified styling
    backend_styles = setup_figure_style(colorblind_friendly=colorblind_friendly)

    # Filter for dimension scaling (fixed n_obs, varying grid_size)
    if fixed_n_obs is None:
        # Find the most common n_obs value - this should be our "fixed" value
        n_obs_counts = df["n_obs"].value_counts()
        if len(n_obs_counts) == 0:
            print("No data available for dimension scaling plot")
            return None
        fixed_n_obs = n_obs_counts.index[0]

    dim_data = df[df["n_obs"] == fixed_n_obs]

    if len(dim_data) == 0:
        print(f"No data found for fixed n_obs={fixed_n_obs}")
        return None

    # Create figure with two subplots
    fig, (ax_fit, ax_pred) = plt.subplots(1, 2)

    # Plot fit times vs grid size
    ax_fit.set_title(f"Training Time (n_obs={fixed_n_obs})")
    ax_fit.set_xlabel("state dimension (d)")
    ax_fit.set_ylabel("wall-clock time [s]")

    for backend, style in backend_styles.items():
        backend_data = dim_data[dim_data["backend"] == backend]
        if len(backend_data) > 0:
            # Hardening: Skip backends with insufficient data points
            if backend_data.shape[0] < 2:
                print(
                    f"Warning: Skipping {backend} - insufficient data points ({backend_data.shape[0]} < 2)"
                )
                continue

            backend_data = backend_data.sort_values("grid_size")

            ax_fit.loglog(
                backend_data["grid_size"],
                backend_data["fit_time"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linestyle=style["linestyle"],
                alpha=0.8,
                markersize=6,
            )

    ax_fit.grid(True, alpha=0.3)
    if show_legend:
        ax_fit.legend()
    ax_fit.set_xscale("log")
    ax_fit.set_yscale("log")

    # Plot predict times vs grid size
    ax_pred.set_title(f"Prediction Time (n_obs={fixed_n_obs})")
    ax_pred.set_xlabel("state dimension (d)")
    ax_pred.set_ylabel("wall-clock time [s]")

    for backend, style in backend_styles.items():
        backend_data = dim_data[dim_data["backend"] == backend]
        if len(backend_data) > 0:
            # Hardening: Skip backends with insufficient data points
            if backend_data.shape[0] < 2:
                print(
                    f"Warning: Skipping {backend} - insufficient data points ({backend_data.shape[0]} < 2)"
                )
                continue

            backend_data = backend_data.sort_values("grid_size")

            ax_pred.loglog(
                backend_data["grid_size"],
                backend_data["predict_time"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linestyle=style["linestyle"],
                alpha=0.8,
                markersize=6,
            )

    ax_pred.grid(True, alpha=0.3)
    if show_legend:
        ax_pred.legend()
    ax_pred.set_xscale("log")
    ax_pred.set_yscale("log")

    plt.tight_layout()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Dimension scaling plot saved to: {output_path}")

    return fig


def main():
    """Main plotting script."""
    parser = argparse.ArgumentParser(description="Generate timing comparison plots")
    parser.add_argument("csv_file", help="CSV file containing benchmark timing data")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Output directory for plots (default: figures)",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--fixed-n-obs",
        type=int,
        help="Fixed n_obs value for dimension scaling plot (default: auto-detect)",
    )
    parser.add_argument(
        "--fixed-grid",
        type=int,
        help="Fixed grid_size value for observation scaling plot (default: auto-detect)",
    )
    parser.add_argument(
        "--colorblind-friendly",
        action="store_true",
        help="Use color-blind friendly palette",
    )
    parser.add_argument(
        "--no-legend", action="store_true", help="Hide legends on plots"
    )

    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading data from {args.csv_file}: {e}")
        return 1

    # Sanity guard: Fail fast if wrong CSV is used
    n_obs_unique = df["n_obs"].nunique()
    grid_size_unique = df["grid_size"].nunique()

    if n_obs_unique <= 1 and grid_size_unique <= 1:
        raise ValueError(
            f"Dataset has insufficient variation for timing plots: "
            f"n_obs has {n_obs_unique} unique values, "
            f"grid_size has {grid_size_unique} unique values. "
            f"At least one dimension must have multiple values for meaningful timing analysis."
        )

    print(f"Loaded {len(df)} timing records from {args.csv_file}")
    print(f"Backends: {sorted(df['backend'].unique())}")
    print(f"N_obs range: {df['n_obs'].min()}-{df['n_obs'].max()}")
    print(f"Grid size range: {df['grid_size'].min()}-{df['grid_size'].max()}")
    print()

    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    show_legend = not args.no_legend

    # Only generate plots that make sense for the data
    if n_obs_unique > 1:
        # Observation scaling plot (fit + predict vs n_obs)
        obs_plot_path = output_dir / "timing_vs_observations.pdf"
        plot_timing_curves(
            df,
            str(obs_plot_path),
            colorblind_friendly=args.colorblind_friendly,
            show_legend=show_legend,
        )
        print(f"Generated observation scaling plot: {obs_plot_path}")
    else:
        print(
            f"Skipping observation scaling plot - only {n_obs_unique} unique n_obs value(s)"
        )

    if grid_size_unique > 1:
        # Dimension scaling plot (fit + predict vs grid_size)
        dim_plot_path = output_dir / "timing_vs_dimensions.pdf"
        plot_dimension_scaling(
            df,
            str(dim_plot_path),
            fixed_n_obs=args.fixed_n_obs,
            colorblind_friendly=args.colorblind_friendly,
            show_legend=show_legend,
        )
        print(f"Generated dimension scaling plot: {dim_plot_path}")
    else:
        print(
            f"Skipping dimension scaling plot - only {grid_size_unique} unique grid_size value(s)"
        )

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("Plotting complete!")
    return 0


if __name__ == "__main__":
    exit(main())
