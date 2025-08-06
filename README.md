# Data Assimilation vs Gaussian Process Benchmark

This repository implements a comparison between Data Assimilation (DA) methods and Gaussian Process regression, demonstrating their mathematical equivalence and performance characteristics. The repository contains both the code implementation and the LaTeX manuscript.

## Project Structure

```
paper_matheron_equals_enkf/
├── da_gp/                     # Core implementation
│   ├── src/                   # Source code
│   │   ├── gp_common.py       # Shared utilities (grid, kernel, sampling)
│   │   ├── gp_dapper.py       # DAPPER EnKF backend  
│   │   ├── gp_pdaf.py         # pyPDAF parallel backend
│   │   ├── gp_sklearn.py      # scikit-learn GP baseline
│   │   └── cli.py             # Command-line interface
│   ├── scripts/               # Analysis scripts
│   │   ├── bench.py           # Performance benchmarking
│   │   ├── plot_perf.py       # Scaling plot generation
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
# Create environment with uv
uv venv && source .venv/bin/activate

# Install the project in editable mode with all dependencies
uv pip install -e .[dev,dapper,pdaf]
```

## Quick Start

```bash
# Run single experiment using the installed script
uv run da-gp --backend sklearn --n_obs 1000 --grid_size 500

# Generate performance data
uv run python da_gp/scripts/bench.py --n_obs_grid 100 500 --backends sklearn --csv data/quick_bench.csv

# Create scaling plots (you need both obs and dim data for combined plot)
uv run python da-gp/scripts/plot_perf.py data/quick_bench.csv --x obs

# Generate posterior plots  
uv run python da_gp/scripts/plot_posterior.py --n_obs 2000

# Build the paper
latexmk -pdf main.tex
```

## Performance Analysis Workflow

### Complete Benchmarking

```bash
# Step 1: Generate observation scaling data (varying observations, fixed dimensions)
uv run python da_gp/scripts/bench.py \
    --n_obs_grid 100 500 1000 2000 5000 \
    --grid_size_fixed 2000 \
    --backends sklearn dapper \
    --csv data/bench_obs.csv

# Step 2: Generate dimension scaling data (varying dimensions, fixed observations)  
uv run python da_gp/scripts/bench.py \
    --dim_grid 500 1000 2000 4000 \
    --n_obs_fixed 2000 \
    --backends sklearn dapper \
    --csv data/bench_dim.csv

# Step 3: Create the combined scaling plot
uv run python da_gp/scripts/plot_perf_combined.py data/bench_obs.csv data/bench_dim.csv

# Step 4: Generate posterior comparison
uv run python da_gp/scripts/plot_posterior.py --backends sklearn dapper --n_obs 2000

# Output files:
# - figures/perf_scaling.pdf (combined plot with both scaling analyses)
# - figures/posterior_samples.pdf
```

## Understanding the Output

The benchmark generates CSV files in `data/` with columns:
- `backend` - Method used (sklearn, dapper, pdaf)
- `n_obs` - Number of observations 
- `grid_size` - State dimension (d)
- `time_s` - Wall-clock time in seconds

The plotting scripts create log-log scaling plots showing:
- **Observation scaling** (`--x obs`): How time scales with observation count m
- **Dimension scaling** (`--x dim`): How time scales with state dimension d

## Key Features

- **Random Fourier Features**: Optional O(dm) sampling vs O(d³) exact methods
- **Configurable Grid Size**: Use `--grid_size` to test different state dimensions  
- **Dual-Axis Analysis**: Separate observation and dimension scaling studies
- **Multiple Backends**: Compare sklearn GP, DAPPER EnKF, and pyPDAF implementations

## Testing

```bash
# Run all tests
uv run pytest da_gp/tests/ -v

# Run specific test categories  
uv run pytest da_gp/tests/test_shapes.py -v  # Shape consistency
uv run pytest da_gp/tests/test_rmse.py -v    # RMSE validation

# Test with different grid sizes
uv run pytest da_gp/tests/test_shapes.py::test_draw_prior_shape_parametrized -v
```

## Performance Scaling

- **DA methods (DAPPER/pyPDAF)**: O(dm²) - linear in state dimension d
- **scikit-learn GP**: O(m³) - cubic in observation count m
- **Crossover point**: ~1000-5000 observations where DA becomes faster

## CLI Reference

### Main CLI
```bash
uv run da-gp --backend sklearn --n_obs 1000 --grid_size 500
```

### Benchmark Script
```bash
uv run python da_gp/scripts/bench.py --n_obs_grid 100 500 --backends sklearn --csv data/results.csv
```

### Plot Generation
```bash
# Individual plots
uv run python da_gp/scripts/plot_perf.py data/results.csv --x obs

# Combined publication-quality plot (recommended)
uv run python da_gp/scripts/plot_perf_combined.py data/bench_obs.csv data/bench_dim.csv
```

## Citation

This code accompanies the paper "The Ensemble Kalman Update is an Empirical Matheron Update" demonstrating the mathematical equivalence between EnKF and GP inference.