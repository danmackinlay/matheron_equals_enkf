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

"""Test grid size changes and broadcast error prevention."""

import pytest
import numpy as np
import subprocess
import sys
import tempfile
from pathlib import Path

# Store original grid size to restore after tests
_ORIGINAL_GRID_SIZE = None

def setup_module():
    """Save original grid size before tests."""
    global _ORIGINAL_GRID_SIZE
    from da_gp.src.gp_common import GRID_SIZE
    _ORIGINAL_GRID_SIZE = GRID_SIZE

def teardown_module():
    """Restore original grid size after tests."""
    from da_gp.src.gp_common import set_grid_size
    rng = np.random.default_rng(42)
    set_grid_size(_ORIGINAL_GRID_SIZE, rng)


@pytest.mark.parametrize("grid_size", [500, 1000, 4000])
def test_benchmark_with_different_grid_sizes(grid_size):
    """Test that benchmark pipeline works with different grid sizes without broadcast errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / f"test_grid_{grid_size}.csv"
        
        # Run benchmark with specific grid size
        cmd = [
            sys.executable, "-m", "da_gp.scripts.bench",
            "--n_obs_grid", "50", "100",
            "--grid_size_fixed", str(grid_size),
            "--backends", "sklearn", 
            "--csv", str(csv_path),
            "--repeats", "1"  # Minimal for testing
        ]
        
        # This should not raise any exceptions
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode != 0:
            pytest.fail(f"Benchmark failed for grid_size={grid_size}:\n"
                       f"stdout: {result.stdout}\n"
                       f"stderr: {result.stderr}")
        
        # Verify CSV was created and has data
        assert csv_path.exists(), f"CSV file not created for grid_size={grid_size}"
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"CSV is empty for grid_size={grid_size}"
        assert (df['grid_size'] == grid_size).all(), f"Grid size mismatch in CSV"


def test_dimension_scaling_benchmark():
    """Test dimension scaling benchmark to ensure no broadcast errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_dim_scaling.csv"
        
        # Run dimension scaling benchmark
        cmd = [
            sys.executable, "-m", "da_gp.scripts.bench",
            "--dim_grid", "250", "500", "1000",
            "--n_obs_fixed", "100", 
            "--backends", "sklearn",
            "--csv", str(csv_path),
            "--repeats", "1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        if result.returncode != 0:
            pytest.fail(f"Dimension scaling benchmark failed:\n"
                       f"stdout: {result.stdout}\n"
                       f"stderr: {result.stderr}")
        
        # Verify results
        assert csv_path.exists()
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Should have 3 different grid sizes
        unique_sizes = df['grid_size'].unique()
        assert len(unique_sizes) == 3
        assert set(unique_sizes) == {250, 500, 1000}


def test_shape_mismatch_detection():
    """Test that shape mismatches are properly detected and reported."""
    from da_gp.src.gp_common import set_grid_size
    from da_gp.src.gp_sklearn import run
    
    # Create data with one grid size
    rng = np.random.default_rng(42)
    set_grid_size(1000, rng)
    
    from da_gp.src.gp_common import make_obs_mask, generate_truth, make_observations
    mask = make_obs_mask(50, rng)
    truth = generate_truth(rng) 
    obs = make_observations(truth, mask, 0.1, rng)
    
    # Change grid size after data generation (simulating the bug)
    set_grid_size(500, rng)  # Different size
    
    # This should now raise an IndexError (mask indices out of bounds) or AssertionError
    # The fix prevents this by either catching the index error or the shape mismatch
    with pytest.raises((IndexError, AssertionError)):
        run(n_obs=50, truth=truth, mask=mask, obs=obs, rng=rng)
        
    # When we pass explicit grid_size, it should work correctly
    result = run(n_obs=50, truth=truth, mask=mask, obs=obs, rng=rng, grid_size=1000)
    assert result['posterior_mean'].shape == truth.shape


def test_backend_import_order():
    """Test that backend import order doesn't affect results."""
    from da_gp.src.gp_common import set_grid_size
    
    rng = np.random.default_rng(42)
    
    # Test 1: Set grid size first, then import
    set_grid_size(500, rng)
    from da_gp.src.gp_sklearn import run as sklearn_run
    
    # Generate data
    from da_gp.src.gp_common import make_obs_mask, generate_truth, make_observations
    mask = make_obs_mask(50, rng)
    truth = generate_truth(rng)
    obs = make_observations(truth, mask, 0.1, rng)
    
    # This should work without errors
    result = sklearn_run(n_obs=50, truth=truth, mask=mask, obs=obs, rng=rng)
    assert 'rmse' in result
    assert result['posterior_mean'].shape == truth.shape


@pytest.mark.parametrize("backend", ["sklearn"])  # Add dapper backends when available
def test_single_backend_grid_changes(backend):
    """Test individual backend with grid size changes."""
    # This test ensures each backend handles grid size changes properly
    rng = np.random.default_rng(42)
    
    for grid_size in [500, 1000]:
        from da_gp.src.gp_common import set_grid_size, make_obs_mask, generate_truth, make_observations
        
        # Set grid size and generate matching data
        set_grid_size(grid_size, rng)
        mask = make_obs_mask(50, rng)
        truth = generate_truth(rng)
        obs = make_observations(truth, mask, 0.1, rng)
        
        # Import backend after grid size is set
        if backend == "sklearn":
            from da_gp.src.gp_sklearn import run
            result = run(n_obs=50, truth=truth, mask=mask, obs=obs, rng=rng)
        
        # Should work without shape mismatches
        assert result['posterior_mean'].shape == (grid_size,)
        assert result['posterior_mean'].shape == truth.shape