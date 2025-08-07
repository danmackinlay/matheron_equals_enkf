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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication-ready plot settings
try:
    from tueplots import bundles
    plt.rcParams.update(bundles.jmlr2001(nrows=1, ncols=2))
except ImportError:
    print("Warning: tueplots not available, using default matplotlib settings")
    plt.rcParams.update({
        'figure.figsize': (12, 5),
        'font.size': 10,
        'axes.labelsize': 11,
        'legend.fontsize': 9
    })

# Backend styling
BACKEND_STYLES = {
    'sklearn': {'marker': 'o', 'color': 'C0', 'label': 'Sklearn GP'},
    'dapper_enkf': {'marker': 's', 'color': 'C1', 'label': 'EnKF'},
    'dapper_letkf': {'marker': '^', 'color': 'C2', 'label': 'LETKF'},
}


def plot_timing_curves(df: pd.DataFrame, output_path: str = "figures/timing_comparison.pdf"):
    """Create dual-curve log-log plots showing fit and predict times."""
    
    # Create figure with two subplots side by side
    fig, (ax_fit, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot fit times (left subplot)
    ax_fit.set_title("Training Time")
    ax_fit.set_xlabel("observations (m)")
    ax_fit.set_ylabel("wall-clock time [s]")
    
    for backend, style in BACKEND_STYLES.items():
        backend_data = df[df['backend'] == backend]
        if len(backend_data) > 0:
            # Sort by n_obs for clean line plots
            backend_data = backend_data.sort_values('n_obs')
            
            ax_fit.loglog(
                backend_data['n_obs'], 
                backend_data['fit_time'],
                marker=style['marker'],
                color=style['color'],
                label=style['label'],
                linestyle='-',
                alpha=0.8,
                markersize=6
            )
    
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend()
    
    # Plot predict times (right subplot)
    ax_pred.set_title("Prediction Time")
    ax_pred.set_xlabel("observations (m)")
    ax_pred.set_ylabel("wall-clock time [s]")
    
    for backend, style in BACKEND_STYLES.items():
        backend_data = df[df['backend'] == backend]
        if len(backend_data) > 0:
            backend_data = backend_data.sort_values('n_obs')
            
            ax_pred.loglog(
                backend_data['n_obs'], 
                backend_data['predict_time'],
                marker=style['marker'],
                color=style['color'], 
                label=style['label'],
                linestyle='-',
                alpha=0.8,
                markersize=6
            )
    
    ax_pred.grid(True, alpha=0.3)
    ax_pred.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Timing comparison plot saved to: {output_path}")
    
    return fig


def plot_dimension_scaling(df: pd.DataFrame, output_path: str = "figures/dimension_scaling.pdf"):
    """Create plots showing scaling with state dimension."""
    
    # Filter for dimension scaling (fixed n_obs, varying grid_size)
    # Find the most common n_obs value - this should be our "fixed" value
    n_obs_counts = df['n_obs'].value_counts()
    if len(n_obs_counts) == 0:
        print("No data available for dimension scaling plot")
        return None
        
    fixed_n_obs = n_obs_counts.index[0]
    dim_data = df[df['n_obs'] == fixed_n_obs]
    
    if len(dim_data) == 0:
        print(f"No data found for fixed n_obs={fixed_n_obs}")
        return None
    
    # Create figure with two subplots
    fig, (ax_fit, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot fit times vs grid size
    ax_fit.set_title(f"Training Time (n_obs={fixed_n_obs})")
    ax_fit.set_xlabel("state dimension (d)")
    ax_fit.set_ylabel("wall-clock time [s]")
    
    for backend, style in BACKEND_STYLES.items():
        backend_data = dim_data[dim_data['backend'] == backend]
        if len(backend_data) > 0:
            backend_data = backend_data.sort_values('grid_size')
            
            ax_fit.loglog(
                backend_data['grid_size'], 
                backend_data['fit_time'],
                marker=style['marker'],
                color=style['color'],
                label=style['label'],
                linestyle='-',
                alpha=0.8,
                markersize=6
            )
    
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend()
    
    # Plot predict times vs grid size
    ax_pred.set_title(f"Prediction Time (n_obs={fixed_n_obs})")
    ax_pred.set_xlabel("state dimension (d)")
    ax_pred.set_ylabel("wall-clock time [s]")
    
    for backend, style in BACKEND_STYLES.items():
        backend_data = dim_data[dim_data['backend'] == backend]
        if len(backend_data) > 0:
            backend_data = backend_data.sort_values('grid_size')
            
            ax_pred.loglog(
                backend_data['grid_size'], 
                backend_data['predict_time'],
                marker=style['marker'],
                color=style['color'],
                label=style['label'],
                linestyle='-',
                alpha=0.8,
                markersize=6
            )
    
    ax_pred.grid(True, alpha=0.3)
    ax_pred.legend()
    
    plt.tight_layout()
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Dimension scaling plot saved to: {output_path}")
    
    return fig


def main():
    """Main plotting script."""
    parser = argparse.ArgumentParser(description="Generate timing comparison plots")
    parser.add_argument(
        "csv_file",
        help="CSV file containing benchmark timing data"
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Output directory for plots (default: figures)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively"
    )
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading data from {args.csv_file}: {e}")
        return 1
        
    print(f"Loaded {len(df)} timing records from {args.csv_file}")
    print(f"Backends: {sorted(df['backend'].unique())}")
    print(f"N_obs range: {df['n_obs'].min()}-{df['n_obs'].max()}")
    print(f"Grid size range: {df['grid_size'].min()}-{df['grid_size'].max()}")
    print()
    
    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Observation scaling plot (fit + predict vs n_obs)
    obs_plot_path = output_dir / "timing_vs_observations.pdf"
    plot_timing_curves(df, str(obs_plot_path))
    
    # Dimension scaling plot (fit + predict vs grid_size)
    dim_plot_path = output_dir / "timing_vs_dimensions.pdf"
    plot_dimension_scaling(df, str(dim_plot_path))
    
    if args.show:
        plt.show()
    else:
        plt.close('all')
    
    print("Plotting complete!")
    return 0


if __name__ == "__main__":
    exit(main())