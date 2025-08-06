
# Data Assimilation vs Gaussian Process Benchmark

This repository implements a comparison between Data Assimilation (DA) methods and Gaussian Process regression, demonstrating their mathematical equivalence and performance characteristics. The repository contains both the code implementation and the LaTeX manuscript.

## Project Structure

The project is organized to separate the Python package, scripts, and publication artifacts.

```
.
├── da_gp/                     # Core Python package
│   ├── src/                   # Source code for backends and CLI
│   ├── scripts/               # Standalone scripts for analysis
│   │   ├── bench.py           # Performance benchmarking
│   │   ├── plot_perf_combined.py # Combined scaling plot generation
│   │   └── plot_posterior.py  # Posterior comparison plots
│   └── tests/                 # Test suite
├── data/                      # Generated CSV files (gitignored)
├── figures/                   # Generated PDF plots (gitignored)
├── main.tex                   # LaTeX manuscript
└── README.md                  # This file
```

## Installation

**IMPORTANT**: All commands should be run from the root directory of this repository.

```bash
# 1. Create the virtual environment
uv venv

# 2. Compile dependencies into a lock file
# This reads pyproject.toml, resolves all dependencies (including extras),
# and writes the exact versions to requirements.lock.
uv pip compile pyproject.toml --extra dev --extra dapper -o requirements.lock

# 3. Sync the environment with the lock file
uv pip sync requirements.lock

# 4. Run commands using `uv run`
uv run pytest da_gp/tests/
```

## Quick Start

This provides a minimal, end-to-end workflow to generate a result.

```bash
# 1. Generate a small dataset for observation scaling
uv run python da_gp/scripts/bench.py \
    --n_obs_grid 100 500 --grid_size_fixed 1000 \
    --backends sklearn dapper --csv data/quick_obs.csv[

# 2. Generate a small dataset for dimension scaling
uv run python da_gp/scripts/bench.py \
    --dim_grid 500 1000 --n_obs_fixed 500 \
    --backends sklearn dapper --csv data/quick_dim.csv

# 3. Create the combined performance plot from the quick data
uv run python da_gp/scripts/plot_perf_combined.py data/quick_obs.csv data/quick_dim.csv --out figures/quick_perf.pdf

# 4. Generate a posterior plot
uv run python da_gp/scripts/plot_posterior.py --n_draws 200

# 5. Build the paper
latexmk -pdf main.tex
```

## Full Benchmarking Workflow

This is the complete workflow to reproduce the figures for the paper.

```bash
# Step 1: Generate observation scaling data
uv run python da_gp/scripts/bench.py \
    --n_obs_grid 100 500 1000 2000 5000 \
    --grid_size_fixed 2000 \
    --backends sklearn dapper \
    --csv data/bench_obs.csv

# Step 2: Generate dimension scaling data
uv run python da_gp/scripts/bench.py \
    --dim_grid 500 1000 2000 4000 \
    --n_obs_fixed 2000 \
    --backends sklearn dapper \
    --csv data/bench_dim.csv

# Step 3: Create the final, combined scaling plot
uv run python da_gp/scripts/plot_perf_combined.py data/bench_obs.csv data/bench_dim.csv

# Step 4: Generate the final posterior comparison plot
uv run python da_gp/scripts/plot_posterior.py --backends sklearn dapper --n_obs 50

# Output files used by main.tex:
# - figures/perf_scaling.pdf
# - figures/posterior_samples.pdf
```

## Testing

```bash
# Run all tests
uv run pytest da_gp/tests/ -v

# Run tests for a specific file
uv run pytest da_gp/tests/test_shapes.py -v
```

## CLI Reference

### Main CLI (`da-gp`)

Run a single experiment.

```bash
uv run da-gp --backend sklearn --n_obs 1000 --grid_size 500```

### Benchmark Script

Generate performance data by sweeping over parameters.

```bash
uv run python da_gp/scripts/bench.py --n_obs_grid 100 500 --backends sklearn --csv data/results.csv
```

### Plot Generation Scripts

Generate plots from benchmark data.

```bash
# Create the combined, publication-quality performance plot
uv run python da_gp/scripts/plot_perf_combined.py data/bench_obs.csv data/bench_dim.csv

# Create the posterior samples plot
uv run python da_gp/scripts/plot_posterior.py --n_obs 50
```

## Citation

This code accompanies the paper "The Ensemble Kalman Update is an Empirical Matheron Update".