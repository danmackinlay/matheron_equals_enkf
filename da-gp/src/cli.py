"""Command-line interface for da-gp benchmark."""

import argparse
import time
import sys
from typing import Any, Dict

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Assimilation vs Gaussian Process benchmark"
    )
    parser.add_argument(
        '--backend', 
        choices=['dapper', 'pdaf', 'sklearn'],
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
    
    args = parser.parse_args()
    
    # Check MPI rank for multi-process backends
    rank = 0
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.rank
    
    # Import and run backend
    start_time = time.perf_counter()
    
    try:
        if args.backend == 'dapper':
            from . import gp_dapper as backend
            result = backend.run(n_ens=args.n_ens, n_obs=args.n_obs)
        elif args.backend == 'pdaf':
            from . import gp_pdaf as backend
            result = backend.run(n_ens=args.n_ens, n_obs=args.n_obs)
        elif args.backend == 'sklearn':
            from . import gp_sklearn as backend
            result = backend.run(n_obs=args.n_obs, n_ens=args.n_ens)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
            
    except ImportError as e:
        if rank == 0:
            print(f"Error: {e}", file=sys.stderr)
            if args.backend == 'dapper':
                print("Install DAPPER with: uv pip install 'dapper>=1.0'", file=sys.stderr)
            elif args.backend == 'pdaf':
                print("Install pyPDAF with: uv pip install pyPDAF mpi4py", file=sys.stderr)
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
    print(f"Backend: {args.backend}")
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
    print(f"\nCSV: {args.backend},{args.n_obs},{elapsed_time:.6f},{result.get('rmse', 0.0):.6f}")


if __name__ == '__main__':
    main()