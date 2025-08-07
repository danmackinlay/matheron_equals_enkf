#!/usr/bin/env python3
# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark script for DA vs GP performance comparison with internal timing."""

import argparse
import csv
import time
import platform
import os
import numpy as np
import pandas as pd
from pathlib import Path
from statistics import median
from typing import Dict, Any

DEFAULT_BACKENDS = ["sklearn", "dapper_enkf", "dapper_letkf"]


def get_backend_runner(backend: str):
    """Get the appropriate backend runner function."""
    if backend == "sklearn":
        from da_gp.src.gp_sklearn import run
        return run
    elif backend == "dapper_enkf":
        from da_gp.src.gp_dapper import run_enkf
        return run_enkf
    elif backend == "dapper_letkf":
        from da_gp.src.gp_dapper import run_letkf
        return run_letkf
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_experiment_once(backend: str, n_obs: int, grid_size: int, shared_data: Dict[str, Any]) -> Dict[str, float]:
    """Run single experiment and return timing results."""
    # Get shared synthetic data for this (n_obs, grid_size) pair
    key = (n_obs, grid_size)
    truth, mask, obs = shared_data[key]
    
    # Get backend runner
    runner = get_backend_runner(backend)
    
    # Set grid size if needed
    from da_gp.src.gp_common import set_grid_size
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    set_grid_size(grid_size, rng)
    
    # Run experiment and extract timings (note: DAPPER backends don't accept rng parameter)
    if backend.startswith('dapper_'):
        result = runner(n_obs=n_obs, truth=truth, mask=mask, obs=obs, seed=42)
    else:
        result = runner(n_obs=n_obs, truth=truth, mask=mask, obs=obs, rng=rng)
    
    return {
        'fit_time': result['fit_time'],
        'predict_time': result['predict_time'],
        'total_time': result['total_time'],
        'rmse': result['rmse']
    }


def run_with_repeats(backend: str, n_obs: int, grid_size: int, shared_data: Dict[str, Any], n_repeats: int = 5) -> Dict[str, float]:
    """Run experiment multiple times and return median timings."""
    results = []
    
    # Warm-up run (discarded)
    try:
        run_experiment_once(backend, n_obs, grid_size, shared_data)
    except Exception as e:
        print(f"Warning: Warm-up failed for {backend}: {e}")
    
    # Collect timing results
    for _ in range(n_repeats):
        try:
            result = run_experiment_once(backend, n_obs, grid_size, shared_data)
            results.append(result)
        except Exception as e:
            print(f"Error running {backend} (n_obs={n_obs}, grid_size={grid_size}): {e}")
            # Return inf times to mark as failed
            return {
                'fit_time': float('inf'),
                'predict_time': float('inf'), 
                'total_time': float('inf'),
                'rmse': float('inf')
            }
    
    if not results:
        return {
            'fit_time': float('inf'),
            'predict_time': float('inf'),
            'total_time': float('inf'), 
            'rmse': float('inf')
        }
    
    # Return median times
    return {
        'fit_time': median([r['fit_time'] for r in results]),
        'predict_time': median([r['predict_time'] for r in results]),
        'total_time': median([r['total_time'] for r in results]),
        'rmse': median([r['rmse'] for r in results])
    }


def generate_shared_data(n_obs_list: list, grid_size_list: list) -> Dict[tuple, tuple]:
    """Generate shared synthetic datasets for all (n_obs, grid_size) combinations."""
    from da_gp.src.gp_common import set_grid_size, make_obs_mask, generate_truth, make_observations
    
    shared_data = {}
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    for grid_size in grid_size_list:
        set_grid_size(grid_size, rng)
        
        for n_obs in n_obs_list:
            # Generate single synthetic dataset for this configuration
            mask = make_obs_mask(n_obs, rng)
            truth = generate_truth(rng)
            obs = make_observations(truth, mask, 0.1, rng)
            
            shared_data[(n_obs, grid_size)] = (truth, mask, obs)
            
    return shared_data


def print_system_info():
    """Print system information for reproducibility."""
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Try to get BLAS info
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        config = np.show_config(mode='dicts')
        if 'blas_info' in config:
            print(f"BLAS: {config['blas_info'].get('name', 'unknown')}")
    except Exception:
        pass
        
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print()


def main():
    """Main benchmark runner with internal timing."""
    parser = argparse.ArgumentParser(description="Benchmark DA vs GP performance with internal timing")
    
    # Observation sweep parameters
    parser.add_argument(
        "--n_obs_grid", 
        type=int, 
        nargs="+",
        help="Grid of observation counts to test"
    )
    parser.add_argument(
        "--n_obs_fixed",
        type=int,
        default=2000,
        help="Fixed observation count when sweeping dimensions (default: 2000)"
    )
    
    # Dimension sweep parameters  
    parser.add_argument(
        "--dim_grid",
        type=int,
        nargs="+", 
        help="Grid of state dimensions (grid sizes) to test"
    )
    parser.add_argument(
        "--grid_size_fixed",
        type=int,
        default=2000,
        help="Fixed grid size when sweeping observations (default: 2000)"
    )
    
    parser.add_argument(
        "--csv", 
        default="data/bench_timing.csv",
        help="Output CSV file (default: data/bench_timing.csv)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["sklearn", "dapper_enkf", "dapper_letkf"],
        help="Backends to test (default: all)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repeats per configuration (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments - need either obs_grid or dim_grid
    if not args.n_obs_grid and not args.dim_grid:
        parser.error("Must specify either --n_obs_grid or --dim_grid (or both)")
    
    # Ensure output directory exists
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BENCHMARK WITH INTERNAL TIMING")
    print("="*60)
    print_system_info()
    
    print(f"Backends: {args.backends}")
    print(f"Timing repeats: {args.repeats}")
    if args.n_obs_grid:
        print(f"Observation counts: {args.n_obs_grid} (grid_size={args.grid_size_fixed})")
    if args.dim_grid:
        print(f"Grid sizes: {args.dim_grid} (n_obs={args.n_obs_fixed})")
    print(f"Output: {args.csv}")
    print()
    
    # Build complete parameter lists - ensure separate sweeps
    if args.n_obs_grid and not args.dim_grid:
        # Pure observation sweep
        n_obs_list = args.n_obs_grid
        grid_size_list = [args.grid_size_fixed]
    elif args.dim_grid and not args.n_obs_grid:
        # Pure dimension sweep
        n_obs_list = [args.n_obs_fixed]
        grid_size_list = args.dim_grid
    elif args.n_obs_grid and args.dim_grid:
        # Both sweeps specified - this creates a cross-product
        n_obs_list = args.n_obs_grid + [args.n_obs_fixed]
        grid_size_list = [args.grid_size_fixed] + args.dim_grid
    else:
        # Should not reach here due to earlier validation
        raise ValueError("Must specify either --n_obs_grid or --dim_grid")
    
    # Remove duplicates and sort
    n_obs_list = sorted(set(n_obs_list))
    grid_size_list = sorted(set(grid_size_list))
    
    # Fail-fast assertion to guarantee multiple data points in sweeps
    assert len(n_obs_list) > 1 or len(grid_size_list) > 1, \
        f"Sweep collapsed to single point: n_obs={n_obs_list}, grid_size={grid_size_list}. Need multiple values in at least one dimension."
    
    print("Generating shared synthetic datasets...")
    shared_data = generate_shared_data(n_obs_list, grid_size_list)
    print(f"Generated {len(shared_data)} datasets")
    print()
    
    # Collect results
    results = []
    total_runs = len(args.backends) * len(n_obs_list) * len(grid_size_list)
    current_run = 0
    
    for backend in args.backends:
        for grid_size in grid_size_list:
            for n_obs in n_obs_list:
                current_run += 1
                print(f"[{current_run}/{total_runs}] {backend} (n_obs={n_obs}, grid_size={grid_size})")
                
                timing_result = run_with_repeats(backend, n_obs, grid_size, shared_data, args.repeats)
                
                results.append({
                    'backend': backend,
                    'n_obs': n_obs,
                    'grid_size': grid_size,
                    'fit_time': timing_result['fit_time'],
                    'predict_time': timing_result['predict_time'],
                    'total_time': timing_result['total_time'],
                    'rmse': timing_result['rmse']
                })
                
                print(f"  fit: {timing_result['fit_time']:.3f}s, predict: {timing_result['predict_time']:.3f}s, total: {timing_result['total_time']:.3f}s")
                print(f"  RMSE: {timing_result['rmse']:.4f}")
                print()
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.csv, index=False)
    
    print("="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.csv}")
    print(f"Total experiments: {len(results)}")
    print()
    
    # Print summary statistics
    print("SUMMARY:")
    for backend in args.backends:
        backend_results = df[df['backend'] == backend]
        if len(backend_results) > 0:
            print(f"{backend}:")
            print(f"  Mean fit time: {backend_results['fit_time'].mean():.3f}s")
            print(f"  Mean predict time: {backend_results['predict_time'].mean():.3f}s")
            print(f"  Mean RMSE: {backend_results['rmse'].mean():.4f}")
            print()


if __name__ == "__main__":
    main()