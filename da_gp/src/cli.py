"""Command-line interface for da-gp benchmark."""

import argparse
import time
import sys
import numpy as np
from typing import Any, Dict, Tuple

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def _load_backend(backend: str, n_obs: int, n_ens: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load backend and return posterior statistics for plotting.
    
    Returns:
        mean: Posterior mean
        lower: Lower confidence bound (mean - 2*std)
        upper: Upper confidence bound (mean + 2*std)
        x: Grid coordinates
    """
    from .gp_common import X_grid
    
    if backend == "sklearn":
        from . import gp_sklearn as module
        result = module.run(n_obs=n_obs, n_ens=n_ens)
        mean = result['posterior_mean']
        std = result.get('posterior_std', np.ones_like(mean) * 0.1)
        
    elif backend == "dapper_enkf":
        from . import gp_dapper as module
        result = module.run_enkf(n_ens=n_ens, n_obs=n_obs)
        mean = result['posterior_mean']
        ensemble = result.get('posterior_ensemble', np.zeros((n_ens, len(mean))))
        std = np.std(ensemble, axis=0) if ensemble.size > 0 else np.ones_like(mean) * 0.1
        
    elif backend == "dapper_letkf":
        from . import gp_dapper as module
        result = module.run_letkf(n_ens=n_ens, n_obs=n_obs)
        mean = result['posterior_mean']
        ensemble = result.get('posterior_ensemble', np.zeros((n_ens, len(mean))))
        std = np.std(ensemble, axis=0) if ensemble.size > 0 else np.ones_like(mean) * 0.1
        
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    x = X_grid.flatten()
    lower = mean - 2 * std
    upper = mean + 2 * std
    
    return mean, lower, upper, x


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Assimilation vs Gaussian Process benchmark"
    )
    parser.add_argument(
        '--backend', 
        choices=['sklearn', 'dapper_enkf', 'dapper_letkf'],
        required=True,
        help="Backend to use for the experiment"
    )
    parser.add_argument(
        '--n_obs', 
        type=int, 
        default=5_000,
        help="Number of observations (default: 5000)"
    )
    parser.add_argument(
        '--n_ens',
        type=int,
        default=40,
        help="Number of ensemble members (default: 40)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Print detailed results"
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=None,
        help="State dimension d (overrides gp_common.GRID_SIZE)"
    )
    
    args = parser.parse_args()
    
    # Early resize: change grid size before any backend import
    if args.grid_size is not None:
        from . import gp_common as gpc
        gpc.set_grid_size(args.grid_size)
    
    # Check MPI rank for multi-process backends
    rank = 0
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.rank
    
    # Import and run backend
    start_time = time.perf_counter()
    
    try:
        if args.backend == 'sklearn':
            from . import gp_sklearn as backend
            result = backend.run(n_obs=args.n_obs, n_ens=args.n_ens)
        elif args.backend == 'dapper_enkf':
            from . import gp_dapper as backend
            result = backend.run_enkf(n_ens=args.n_ens, n_obs=args.n_obs)
        elif args.backend == 'dapper_letkf':
            from . import gp_dapper as backend
            result = backend.run_letkf(n_ens=args.n_ens, n_obs=args.n_obs)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
            
    except ImportError as e:
        if rank == 0:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if rank == 0:
            print(f"Error running {args.backend}: {e}", file=sys.stderr)
        sys.exit(1)
    
    elapsed_time = time.perf_counter() - start_time
    
    # Print results (only from rank 0 for MPI jobs)
    if rank == 0:
        print_results(args, result, elapsed_time)


def print_results(args: argparse.Namespace, result: Dict[str, Any], elapsed_time: float) -> None:
    """Print benchmark results."""
    from .gp_common import GRID_SIZE
    
    print(f"Backend: {args.backend}")
    print(f"Grid size: {GRID_SIZE:,}")
    print(f"Observations: {args.n_obs:,}")
    print(f"Ensemble size: {args.n_ens}")
    print(f"Elapsed time: {elapsed_time:.3f}s")
    
    if 'rmse' in result:
        print(f"RMSE: {result['rmse']:.6f}")
    
    if args.verbose:
        print("\nDetailed results:")
        for key, value in result.items():
            if key not in ['posterior_mean', 'posterior_ensemble', 'posterior_std']:
                print(f"  {key}: {value}")
    
    # CSV output for benchmarking
    print(f"\nCSV: {args.backend},{args.n_obs},{GRID_SIZE},{elapsed_time:.6f},{result.get('rmse', 0.0):.6f}")


if __name__ == '__main__':
    main()