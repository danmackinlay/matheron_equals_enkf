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
```

## Performance Scaling

- **DA methods (DAPPER/pyPDAF)**: O(dm²) - linear in state dimension d
- **scikit-learn GP**: O(m³) - cubic in observation count m
- **Crossover point**: ~1000-5000 observations where DA becomes faster

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

- `GRID_SIZE = 2000` - 1D spatial grid resolution (state dimension)
- `OBS_SITES = 5000` - Default number of observation locations
- `kernel = RBF(length_scale=10.0)` - Gaussian process kernel
- `n_ens = 40` - Default ensemble size for DA methods

## Citation

This code accompanies the paper "The Ensemble Kalman Update is an Empirical Matheron Update" demonstrating the mathematical equivalence between EnKF and GP inference.