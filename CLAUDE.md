# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for a paper demonstrating the mathematical equivalence between Matheron's rule and Ensemble Kalman Filter (EnKF) methods. The repository contains a single main Python script that implements both approaches and compares them on a 1D Gaussian process regression problem.

It also contains LaTeX documentation of the code in `main.tex`, with supporting macros in `preamble.tex`.

## Development Commands

The project uses `uv` as the Python package manager. Key commands:

- **Install dependencies**: `uv sync`
- **Run the main experiment**: `uv run python main.py`
- **Add new dependencies**: `uv add <package-name>`

Note: The `da-gp/` subdirectory also uses `uv` for package management. Use `uv run` to execute Python commands within the virtual environment instead of manually activating it.

## Code Architecture

The repository is structured as a single-file experiment (`main.py`) with the following sections:

1. **Problem Setup** (lines 7-19): Defines a 1D spatial grid with Gaussian process prior, generates N=100 prior samples, and creates m=25 noisy observations
2. **Matheron Update** (lines 21-30): Implements exact Bayesian update using Woodbury matrix identity for computational efficiency
3. **GP Baseline** (lines 32-40): Standard Gaussian process regression using scikit-learn for comparison
4. **Diagnostics** (lines 42-44): Validates that both methods produce identical posterior means
5. **Visualization** (lines 46-87): Creates publication-ready plots comparing sample paths from both methods

## Key Dependencies

- `filterpy`: Kalman filtering utilities
- `scikit-learn`: Gaussian process regression baseline
- `scipy`: Distance computations and linear algebra
- `numpy`: Core numerical operations
- `matplotlib`: Plotting
- `tueplots`: Publication-ready plot styling for ICLR 2024

## Important Constants

The experiment uses fixed parameters:
- `d=1000`: Grid resolution
- `N=100`: Number of ensemble members/prior samples
- `m=25`: Number of observations
- `ell=0.2`: Kernel length scale
- `sigma=1.0`: Kernel amplitude
- `tau=0.05`: Observation noise standard deviation

## Output

Running the script generates:
- Console output showing RMSE between methods (should be ≪ τ=0.05)
- `fig_enkf_vs_gp.pdf`: Comparison plot of sample trajectories