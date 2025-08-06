# Data Assimilation vs Gaussian Process Benchmark

This package implements a minimal comparison between Data Assimilation (DA) methods and Gaussian Process regression for large observation counts, demonstrating how DA methods can overcome the cubic scaling bottleneck of traditional GP inference.

## Three-Function API

Each backend (`dapper`, `pdaf`, `sklearn`) exposes exactly three functions for consistency:

### DAPPER Backend
- `init_state(n_ens)` - Initialize ensemble state matrix (n_ens × d)
- `obs_op(state, mask)` - Map state to observations via indexing
- `run(n_ens, n_obs)` - Execute EnKF assimilation and return posterior statistics

### pyPDAF Backend  
- `init_state_ens()` - Initialize single ensemble member state vector
- `obs_op(state)` - Map state to observations (uses global mask)
- `run(n_ens, n_obs)` - Execute parallel EnKF with MPI and return results

### scikit-learn Baseline
- `init_state()` - Generate synthetic truth (for API consistency)
- `obs_op(state, mask)` - Extract observations (for API consistency) 
- `run(n_obs)` - Fit GP to observations and return posterior mean/samples

## Installation

```bash
# Create environment with uv
uv venv && source .venv/bin/activate

# Basic installation (sklearn only)
uv pip install -e .

# With DAPPER backend
uv pip install -e .[dapper]

# With pyPDAF backend (requires MPI)
uv pip install -e .[pdaf]

# Development dependencies
uv pip install -e .[dev]
```

### Fallback Installation (if uv fails)

```bash
# Use conda for MPI-heavy dependencies
conda env create -f conda-env.yml
conda activate da-gp
pip install -e .
```

## Usage

```bash
# Run sklearn baseline with 1k observations
da-gp --backend sklearn --n_obs 1000 --verbose

# Run DAPPER backend with 5k observations  
da-gp --backend dapper --n_obs 5000 --n_ens 40

# Run pyPDAF with MPI (32 processes, 50k observations)
mpiexec -n 32 da-gp --backend pdaf --n_obs 50000 --n_ens 40

# Test with custom grid size (state dimension)
da-gp --backend sklearn --n_obs 1000 --grid_size 500
```

## Performance Scaling

- **DA methods (DAPPER/pyPDAF)**: O(dm²) - linear in state dimension d
- **scikit-learn GP**: O(m³) - cubic in observation count m
- **Crossover point**: ~1000-5000 observations where DA becomes faster

### Benchmarking

Generate dual-axis performance plots to analyze scaling behavior:

#### Complete Workflow

```bash
# Step 1: Generate observation scaling data (varying observations, fixed dimensions)
uv run python scripts/bench.py \
    --n_obs_grid 100 500 1000 2000 5000 \
    --grid_size_fixed 2000 \
    --backends sklearn dapper \
    --csv bench_obs.csv

# Step 2: Generate dimension scaling data (varying dimensions, fixed observations)  
uv run python scripts/bench.py \
    --dim_grid 500 1000 2000 4000 \
    --n_obs_fixed 2000 \
    --backends sklearn dapper \
    --csv bench_dim.csv

# Step 3: Create the plots
uv run python scripts/plot_perf.py bench_obs.csv --x obs
uv run python scripts/plot_perf.py bench_dim.csv --x dim

# Output: figures/perf_scaling_obs.pdf and figures/perf_scaling_dim.pdf
```

#### Quick Test Commands

```bash
# Test different backends at fixed size
uv run python -m src.cli --backend sklearn --n_obs 1000 --grid_size 2000
uv run python -m src.cli --backend dapper --n_obs 1000 --grid_size 2000

# Quick benchmark with small dataset
uv run python scripts/bench.py --n_obs_grid 100 200 --backends sklearn --csv quick.csv
uv run python scripts/plot_perf.py quick.csv --x obs
```

#### Understanding the Output

The benchmark generates CSV files with columns:
- `backend` - Method used (sklearn, dapper, pdaf)
- `n_obs` - Number of observations 
- `grid_size` - State dimension (d)
- `time_s` - Wall-clock time in seconds

The plotting script creates log-log scaling plots showing:
- **Observation scaling** (`--x obs`): How time scales with observation count m
- **Dimension scaling** (`--x dim`): How time scales with state dimension d

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories  
pytest tests/test_shapes.py -v  # Shape consistency
pytest tests/test_rmse.py -v    # RMSE validation

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
da-gp/
├── src/
│   ├── gp_common.py      # Shared utilities (grid, kernel, sampling)
│   ├── gp_dapper.py      # DAPPER EnKF backend  
│   ├── gp_pdaf.py        # pyPDAF parallel backend
│   ├── gp_sklearn.py     # scikit-learn GP baseline
│   └── cli.py            # Command-line interface
├── tests/
│   ├── test_shapes.py    # Shape consistency tests
│   └── test_rmse.py      # RMSE validation tests
└── .github/workflows/    # CI/CD pipeline
```

## Key Parameters

- `GRID_SIZE = 2000` - Default 1D spatial grid resolution (state dimension)
  - Can be changed via `--grid_size` CLI flag or `set_grid_size()` function
- `OBS_SITES = 5000` - Default number of observation locations
- `kernel = RBF(length_scale=10.0)` - Gaussian process kernel
- `n_ens = 40` - Default ensemble size for DA methods

### CLI Commands Reference

#### Main CLI (`src.cli`)
Run single experiments:
```bash
uv run python -m src.cli --backend sklearn --n_obs 1000 --grid_size 500
```

Options:
- `--backend {sklearn,dapper,pdaf}` - Backend to use
- `--n_obs N` - Number of observations (default: 5000)
- `--n_ens N` - Ensemble size for DA methods (default: 40)  
- `--grid_size N` - State dimension, overrides gp_common.GRID_SIZE
- `--verbose` - Print detailed results

#### Benchmark Script (`scripts/bench.py`)
Generate performance data:
```bash
uv run python scripts/bench.py --n_obs_grid 100 500 --backends sklearn
```

Options:
- `--n_obs_grid N N N` - List of observation counts to test
- `--dim_grid N N N` - List of grid dimensions to test
- `--grid_size_fixed N` - Fixed grid size when sweeping observations  
- `--n_obs_fixed N` - Fixed observation count when sweeping dimensions
- `--backends sklearn dapper pdaf` - Which backends to benchmark
- `--csv filename.csv` - Output CSV file (default: bench.csv)

#### Plot Script (`scripts/plot_perf.py`)
Generate scaling plots:
```bash
uv run python scripts/plot_perf.py data.csv --x obs --backends sklearn
```

Options:
- `filename.csv` - Input CSV from bench.py (required)
- `--x {obs,dim}` - X-axis variable: obs (observations) or dim (dimensions)
- `--backends sklearn dapper` - Which backends to include in plot
- `--out filename.pdf` - Output file (auto-generated by default)

## Citation

This code accompanies the paper "The Ensemble Kalman Update is an Empirical Matheron Update" demonstrating the mathematical equivalence between EnKF and GP inference.