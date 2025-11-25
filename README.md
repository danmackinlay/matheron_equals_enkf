
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

This project compares four different inference methods:

- `sklearn`: The baseline implementation using `GaussianProcessRegressor` from scikit-learn. It represents the standard, exact GP regression approach.
- `dapper_enkf`: An implementation using the standard Ensemble Kalman Filter (EnKF) from the DAPPER library. This is the global, non-localized data assimilation method.
- `dapper_letkf`: An implementation using the Local Ensemble Transform Kalman Filter (LETKF) from DAPPER. This method applies localization, only updating state variables using nearby observations, which is analogous to sparse or localized GP methods.
- `matheron_ens`: hand rolled empirical Matheron update with random fourier features

## Problem Specification

All functions receive explicit configuration through an immutable `Problem` dataclass:
```python
from da_gp.src.gp_common import Problem

problem = Problem(
    grid_size=1000,     # State dimension
    n_obs=100,          # Number of observations
    noise_std=0.1,      # Observation noise
    rng=np.random.default_rng(42)  # Reproducible RNG
)
```

All backends follow the signature: `run(problem: Problem, **kwargs) -> dict`

## Installation

**Prerequisites**: You need Python ≥ 3.11, `uv` (or `pip`), `doit`, and `latexmk`.

**IMPORTANT**: All commands should be run from the root directory of this repository.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
uv venv
uv sync                    # Installs doit, numpy, matplotlib, latexmk deps, etc.

# 3. Test installation
uv run pytest da_gp/tests/
uv run doit list           # Show available build tasks
```

## Usage

### One-Command Build

Generate all benchmarks, figures, and the final PDF with a single command:

```bash
doit pdf   # Complete pipeline: benchmarks → figures → main.pdf
```

This automatically handles all dependencies and builds everything needed for the paper. Rerun `doit pdf` after making edits; only stale steps will rebuild thanks to intelligent dependency tracking.

Use `doit clean` to force a full rebuild from scratch.

### Incremental Development

For development and debugging, you can run individual parts:

```bash
# View all available tasks
doit list

# Generate just the timing data
doit timing_data

# Generate just the figures
doit figures

# Run tests
doit test

# Clean specific parts
doit clean_figures  # Remove generated plots
doit clean_data     # Remove CSV files
doit clean_latex    # Remove LaTeX aux files
```

## Benchmarks & Plots (doit)

- CSVs and figures are auto-regenerated when sweep params or backends change.
- We track these knobs: `OBS_SWEEP`, `DIM_SWEEP`, `BACKENDS`, `REPEATS`, and the fixed values (`n_obs_fixed`, `grid_size_fixed`).
- If a sweep has only one unique value, the corresponding plot is skipped by design.

### Typical workflow
```bash
uv run doit pdf             # build paper + generate/plot benchmarks as needed
```

### Clean vs force rebuild

```bash
uv run doit clean           # removes CSVs/figures/paper (tasks declare their own targets)
uv run doit pdf             # full rebuild from scratch
```



## CLI Reference

### Main CLI (`da-gp`)

Run a single experiment with internal timing. Available backends are `sklearn`, `dapper_enkf`, and `dapper_letkf`.

```bash
# Single experiment with detailed timing output
uv run da-gp --backend dapper_letkf --n_obs 1000 --grid_size 500 --verbose
```

The CLI now reports separate fit and predict times along with CSV output including both timings.

### Timing Benchmark Script

Generate timing data with internal benchmarking and statistical robustness.

```bash
# Observation scaling with 5 timing repeats per configuration
uv run python -m da_gp.scripts.bench \
    --n_obs_grid 100 500 1000 --backends sklearn dapper_enkf dapper_letkf \
    --csv data/timing_results.csv --repeats 5

# Dimension scaling
uv run python -m da_gp.scripts.bench \
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
uv run python -m da_gp.scripts.plot_timing data/timing_results.csv --output-dir figures --colorblind-friendly

# Create the posterior samples plot with unified styling
uv run python -m da_gp.scripts.plot_posterior --n_obs 50 --colorblind-friendly

# Advanced timing plot options
uv run python -m da_gp.scripts.plot_timing data/timing_results.csv \
    --fixed-n-obs 1000 --no-legend --output-dir figures
```

### Slides

Generate the RevealJS slide deck from `slides.qmd`:

```bash
doit slides              # build slides.html only
doit slides:serve=yes    # build + serve with live preview
doit slides_pdf          # convert slides.html to slides.pdf using decktape
```

**Note**: PDF generation requires [decktape](https://github.com/astefanutti/decktape) to be installed:
```bash
npm install -g decktape
```

## Logging

- CLIs accept logging flags:
  - `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` (default: WARNING)
  - `--log-json` for JSON-formatted logs
- Examples:
  ```bash
  uv run python -m da_gp.scripts.bench --log-level=INFO
  uv run python -m da_gp.scripts.plot_timing data/timing_obs.csv --log-level=DEBUG
  ```

From `doit`, logs default to WARNING. Set a different level temporarily:

```bash
LOG_LEVEL=INFO uv run doit pdf
```


## Testing

```bash
# Run all tests
uv run pytest da_gp/tests/ -v

# Run tests for a specific file
uv run pytest da_gp/tests/test_shapes.py -v
```


### Manual Workflow

<details>
<summary>Click to expand manual workflow</summary>

The following manual workflow still works but requires manual dependency tracking:

```bash
# Step 1: Generate comprehensive timing data (observation scaling)
uv run python -m da_gp.scripts.bench \
    --n_obs_grid 100 500 1000 2000 5000 \
    --grid_size_fixed 2000 \
    --backends sklearn dapper_enkf dapper_letkf matheron_ens \
    --csv data/timing_obs.csv \
    --repeats 5

# Step 2: Generate dimension scaling timing data
uv run python -m da_gp.scripts.bench \
    --dim_grid 250 500 1000 2000 4000 \
    --n_obs_fixed 1000 \
    --backends sklearn dapper_enkf dapper_letkf matheron_ens \
    --csv data/timing_dim.csv \
    --repeats 5

# Step 3: Create dual-curve timing plots (fit vs predict times)
uv run python -m da_gp.scripts.plot_timing data/timing_obs.csv --output-dir figures
uv run python -m da_gp.scripts.plot_timing data/timing_dim.csv --output-dir figures

# Step 4: Generate the posterior comparison plot
uv run python -m da_gp.scripts.plot_posterior --n_obs 50

# Step 5: Build the paper
latexmk -pdf main.tex
```

Output files used by main.tex:
- `figures/timing_vs_observations.pdf` (fit + predict times vs # observations)
- `figures/timing_vs_dimensions.pdf` (fit + predict times vs state dimension)
- `figures/posterior_samples.pdf` (posterior comparison across all methods)

</details>

## Gotchas

### Got only sklearn curves?

* Ensure DAPPER backends are installed and listed in `BACKENDS`.
* Ensure sweeps have ≥2 values (e.g., `OBS_SWEEP=[100, 500, 1000]`).
* You can smoke-test:

  ```bash
  uv run python -m da_gp.scripts.bench \
    --n_obs_grid 50 100 --grid_size_fixed 2000 \
    --backends sklearn dapper_enkf dapper_letkf matheron_ens \
    --csv /tmp/check.csv --repeats 1
  ```

  Then:

  ```bash
  uv run python -m da_gp.scripts.plot_timing /tmp/check.csv --output-dir figures
  ```

## Licensing

The code in this repository is licensed under the **MIT License**. See the `LICENSE` file for details.

The manuscript (`main.tex`) and associated figures are licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

## Citation

This code accompanies the paper "The Ensemble Kalman Update is an Empirical Matheron Update".