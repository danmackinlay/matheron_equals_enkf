#!/usr/bin/env python3
"""Generate posterior comparison plots with truth, observations, and samples."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gp_common import X_grid, get_truth_and_mask, get_observations


def run_backend(backend: str, truth: np.ndarray, mask: np.ndarray, obs: np.ndarray, n_obs: int, n_draws: int = 200):
    """Run a specific backend with given data."""
    if backend == "sklearn":
        from src.gp_sklearn import run
        return run(n_obs=n_obs, truth=truth, mask=mask, obs=obs, n_ens=n_draws)
    elif backend == "dapper":
        from src.gp_dapper import run
        return run(n_ens=n_draws, n_obs=n_obs, truth=truth, mask=mask, obs=obs)
    elif backend == "pdaf":
        from src.gp_pdaf import run
        return run(n_ens=n_draws, n_obs=n_obs, truth_in=truth, mask_in=mask, obs_in=obs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main():
    """Generate posterior comparison plot."""
    parser = argparse.ArgumentParser(description="Generate posterior plots with dual-backend comparison")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sklearn", "dapper"],
        choices=["sklearn", "dapper", "pdaf"],
        help="Backends to compare"
    )
    parser.add_argument(
        "--n_obs",
        type=int,
        default=2000,
        help="Number of observations"
    )
    parser.add_argument(
        "--n_draws",
        type=int,
        default=30,
        help="Number of posterior draws per backend"
    )
    parser.add_argument(
        "--out",
        default="figures/posterior_samples.pdf",
        help="Output figure path"
    )
    parser.add_argument(
        "--show_truth",
        action="store_true",
        help="Show truth as thin black line"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Generate consistent data for all backends
    print(f"Generating data with {args.n_obs} observations...")
    truth, mask = get_truth_and_mask(args.n_obs)
    obs = get_observations(truth, mask, noise_std=0.1)

    # Set up plotting
    plt.figure(figsize=(12, 6))
    x = X_grid.flatten()

    # Color scheme
    colors = {
        'sklearn': 'blue',
        'dapper': 'red',
        'pdaf': 'green'
    }

    # Plot for each backend
    for i, backend in enumerate(args.backends):
        try:
            print(f"Running {backend} backend...")
            result = run_backend(backend, truth, mask, obs, args.n_obs, args.n_draws)

            samples = result["posterior_samples"]
            mean = result["posterior_mean"]
            color = colors.get(backend, f'C{i}')

            # Plot 10 random posterior samples
            n_plot_samples = min(10, len(samples))
            sample_indices = np.random.choice(len(samples), n_plot_samples, replace=False)

            for j, idx in enumerate(sample_indices):
                label = f"{backend.upper()}-samples" if j == 0 else None
                plt.plot(x, samples[idx],
                        lw=0.5, alpha=0.2, color=color, label=label)

            # Plot posterior mean
            plt.plot(x, mean, lw=2, color=color, label=f"{backend.upper()}-mean")

            print(f"  {backend}: RMSE = {result.get('rmse', 0.0):.6f}")

        except Exception as e:
            print(f"Error with {backend}: {e}")
            continue

    # Plot observations
    plt.scatter(mask, obs, marker='x', color='black', s=20, alpha=0.7, label='Obs')

    # Plot truth (optional)
    if args.show_truth:
        plt.plot(x, truth, 'k-', lw=1, alpha=0.8, label='Truth')

    # Formatting
    plt.xlabel("Grid index")
    plt.ylabel("Value")
    plt.title(f"Posterior comparison (n_obs={args.n_obs})")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()