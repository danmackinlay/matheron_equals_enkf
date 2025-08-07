
# Data Assimilation vs Gaussian Process Benchmark

This repository implements a comparison between Data Assimilation (DA) methods and Gaussian Process regression, demonstrating their mathematical equivalence and performance characteristics. The repository contains both the code implementation and the LaTeX manuscript.

## Project Structure

The project is organized to separate the Python package, scripts, and publication artifacts.

```
.
├── da_gp/                     # Core Python package
│   ├── src/                   # Source code for backends and CLI
│   ├── scripts/               # Standalone scripts for analysis
│   │   ├── bench.py           # Performance benchmarking with internal timing
│   │   ├── plot_timing.py     # Dual-curve timing plots (fit vs predict)
│   │   └── plot_posterior.py  # Posterior comparison plots
│   └── tests/                 # Test suite
├── data/                      # Generated CSV files (gitignored)
├── figures/                   # Generated PDF plots (gitignored)
├── main.tex                   # LaTeX manuscript
└── README.md                  # This file
```

## Available Backends

This project compares three different inference methods:

- `sklearn`: The baseline implementation using `GaussianProcessRegressor` from scikit-learn. It represents the standard, exact GP regression approach.
- `dapper_enkf`: An implementation using the standard Ensemble Kalman Filter (EnKF) from the DAPPER library. This is the global, non-localized data assimilation method.
- `dapper_letkf`: An implementation using the Local Ensemble Transform Kalman Filter (LETKF) from DAPPER. This method applies localization, only updating state variables using nearby observations, which is analogous to sparse or localized GP methods.

## Installation

**IMPORTANT**: All commands should be run from the root directory of this repository.

```bash
# 1. Create virtual environment and install dependencies
uv venv
uv sync

# 2. Run commands using `uv run`
uv run pytest da_gp/tests/
```

## Quick Start

This provides a minimal, end-to-end workflow to generate a result.

```bash
# 1. Generate timing data with internal benchmarking (NEW: separate fit/predict times)
uv run python da_gp/scripts/bench.py \
    --n_obs_grid 100 500 --grid_size_fixed 1000 \
    --backends sklearn dapper_enkf dapper_letkf --csv data/timing_quick.csv

# 2. Create dual-curve timing plots showing fit vs predict times (NEW)
uv run python da_gp/scripts/plot_timing.py data/timing_quick.csv --output-dir figures

# 3. Generate a posterior plot showing all three backends
uv run python da_gp/scripts/plot_posterior.py --n_obs 50

# 4. Build the paper
latexmk -pdf main.tex
```


## Full Timing Benchmarking Workflow

This is the complete workflow to reproduce the timing figures for the paper using the new internal timing system.

```bash
# Step 1: Generate comprehensive timing data (observation scaling)
uv run python da_gp/scripts/bench.py \
    --n_obs_grid 100 500 1000 2000 5000 \
    --grid_size_fixed 2000 \
    --backends sklearn dapper_enkf dapper_letkf \
    --csv data/timing_obs.csv \
    --repeats 5

# Step 2: Generate dimension scaling timing data
uv run python da_gp/scripts/bench.py \
    --dim_grid 250 500 1000 2000 4000 \
    --n_obs_fixed 1000 \
    --backends sklearn dapper_enkf dapper_letkf \
    --csv data/timing_dim.csv \
    --repeats 5

# Step 3: Create dual-curve timing plots (fit vs predict times)
uv run python da_gp/scripts/plot_timing.py data/timing_obs.csv --output-dir figures
uv run python da_gp/scripts/plot_timing.py data/timing_dim.csv --output-dir figures

# Step 4: Generate the posterior comparison plot
uv run python da_gp/scripts/plot_posterior.py --n_obs 50

# Output files used by main.tex:
# - figures/timing_vs_observations.pdf  (NEW: fit + predict times vs # observations)  
# - figures/timing_vs_dimensions.pdf    (NEW: fit + predict times vs state dimension)
# - figures/posterior_samples.pdf       (posterior comparison)
```

## Key Improvements in Timing System

The new timing system provides several advantages:

1. **Separate fit and predict times**: Dual-curve plots show that GP training is O(m³) while EnKF prediction is effectively O(1) for fixed ensemble size
2. **Internal timing**: Uses `time.perf_counter()` to eliminate Python startup and I/O overhead  
3. **Statistical robustness**: Includes warm-up runs and reports median of 5 timing repeats
4. **Shared datasets**: All backends use identical synthetic data for fair comparison
5. **Hardened plotting**: Validates data points, uses unified JMLR styling, supports color-blind friendly palettes
6. **Flexible visualization**: CLI flags for fixed values, legend control, and explicit scale settings


## Testing

```bash
# Run all tests
uv run pytest da_gp/tests/ -v

# Run tests for a specific file
uv run pytest da_gp/tests/test_shapes.py -v
```

## CLI Reference

### Main CLI (`da-gp`)

Run a single experiment with internal timing. Available backends are `sklearn`, `dapper_enkf`, and `dapper_letkf`.

```bash
# Single experiment with detailed timing output
uv run da-gp --backend dapper_letkf --n_obs 1000 --grid_size 500 --verbose
```

The CLI now reports separate fit and predict times along with CSV output including both timings.

### Timing Benchmark Script (NEW)

Generate timing data with internal benchmarking and statistical robustness.

```bash
# Observation scaling with 5 timing repeats per configuration
uv run python da_gp/scripts/bench.py \
    --n_obs_grid 100 500 1000 --backends sklearn dapper_enkf dapper_letkf \
    --csv data/timing_results.csv --repeats 5

# Dimension scaling
uv run python da_gp/scripts/bench.py \
    --dim_grid 500 1000 2000 --n_obs_fixed 500 \
    --backends sklearn dapper_enkf --csv data/timing_dim.csv
```

Key features:
- **In-process timing**: Uses `time.perf_counter()` for precise measurement
- **Warm-up runs**: First iteration discarded to eliminate cold-start effects  
- **Statistical robustness**: Median of multiple timing repeats (default: 5)
- **Shared datasets**: Identical synthetic data across all backends for fair comparison

### Plot Generation Scripts

Generate publication-quality plots from benchmark data.

```bash
# NEW: Dual-curve timing plots with hardened validation and JMLR styling
uv run python da_gp/scripts/plot_timing.py data/timing_results.csv --output-dir figures --colorblind-friendly

# Create the posterior samples plot with unified styling
uv run python da_gp/scripts/plot_posterior.py --n_obs 50 --colorblind-friendly

# Advanced timing plot options
uv run python da_gp/scripts/plot_timing.py data/timing_results.csv \
    --fixed-n-obs 1000 --no-legend --output-dir figures
```

## Troubleshooting

### Shape/Broadcast Errors

**Problem**: You see errors like `ValueError: operands could not be broadcast together with shapes (4000,) (1000,)` during benchmarking.

**Cause**: This happens when the grid size is changed after a backend has been imported, due to Python's module import caching.

**Solution**: The codebase now handles this automatically, but if you encounter issues:

1. **Check import order**: Ensure `set_grid_size()` is called before importing any backend modules
2. **Use backend parameters**: Pass `grid_size` parameter directly to backend `run()` functions instead of relying on global state
3. **Restart Python**: If testing interactively, restart your Python session after changing grid sizes

**Example of correct usage**:
```python
from da_gp.src.gp_common import set_grid_size
rng = np.random.default_rng(42)

# Set grid size FIRST
set_grid_size(1000, rng)

# Then import and use backends
from da_gp.src.gp_sklearn import run
result = run(n_obs=100, grid_size=1000)  # Pass grid_size explicitly
```

### Timing Benchmark Failures

**Problem**: Benchmark runs return `inf` timing values or crash unexpectedly.

**Cause**: Usually indicates shape mismatches or backend configuration issues.

**Solution**: 
1. Run with verbose logging: `python -m da_gp.scripts.bench --backends sklearn --n_obs_grid 50 100 --csv test.csv` and check logs
2. Test individual backends first: `da-gp --backend sklearn --n_obs 100 --verbose`
3. For DAPPER backends, ensure proper environment setup

## Licensing

The code in this repository is licensed under the **MIT License**. See the `LICENSE` file for details.

The manuscript (`main.tex`) and associated figures are licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

## Citation

This code accompanies the paper "The Ensemble Kalman Update is an Empirical Matheron Update".