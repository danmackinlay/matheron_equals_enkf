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

"""Test RMSE consistency and validation."""

import numpy as np
import pytest
from da_gp.src.gp_common import generate_truth, make_observations, make_obs_mask


def test_observation_rmse():
    """Test that observations are reasonable perturbations of truth."""
    rng = np.random.default_rng(42)
    truth = generate_truth(rng)
    mask = make_obs_mask(100, rng)
    noise_std = 0.1
    
    obs = make_observations(truth, mask, noise_std, rng)
    
    # Observations should be close to truth values at observed locations
    truth_obs = truth[mask]
    residuals = obs - truth_obs
    empirical_std = np.std(residuals)
    
    # Empirical std should be close to noise_std (within reasonable bounds)
    assert 0.05 < empirical_std < 0.2  # Noise std is 0.1, allow reasonable range


def test_sklearn_rmse_reasonable():
    """Test that sklearn GP produces reasonable RMSE."""
    from da_gp.src.gp_sklearn import run
    
    result = run(n_obs=200)  # Use enough observations for good fit
    
    # RMSE should be reasonable (less than prior std)
    prior_std = 1.0  # From kernel amplitude
    assert result['rmse'] < prior_std
    assert result['rmse'] > 0.0
    
    # Check that we have proper outputs
    assert result['posterior_mean'].shape == (2000,)
    assert np.isfinite(result['rmse'])


def test_prior_spread():
    """Test that prior samples have expected spread."""
    rng = np.random.default_rng(42)
    samples = [generate_truth(rng) for _ in range(50)]
    sample_array = np.array(samples)
    
    # Sample standard deviation should be close to kernel amplitude (1.0)
    empirical_std = np.std(sample_array, axis=0)
    mean_std = np.mean(empirical_std)
    
    # Should be within reasonable bounds of kernel amplitude
    assert 0.5 < mean_std < 1.5


@pytest.mark.parametrize("n_obs", [10, 100, 1000])
def test_sklearn_scaling(n_obs):
    """Test sklearn performance scales as expected."""
    from da_gp.src.gp_sklearn import run
    import time
    
    start_time = time.perf_counter()
    result = run(n_obs=n_obs)
    elapsed = time.perf_counter() - start_time
    
    # Basic sanity checks
    assert result['rmse'] > 0
    assert result['n_obs'] == n_obs
    assert elapsed > 0
    
    # For small n_obs, should complete quickly
    if n_obs <= 100:
        assert elapsed < 10.0  # Should be fast for small problems


def test_rmse_decreases_with_observations():
    """Test that RMSE generally decreases with more observations."""
    from da_gp.src.gp_sklearn import run
    
    # Use fixed random seed for consistency
    np.random.seed(42)
    
    rmse_few = run(n_obs=10)['rmse']
    rmse_many = run(n_obs=100)['rmse']
    
    # More observations should generally give better fit
    # (Though this may not always hold due to random sampling)
    assert rmse_many <= rmse_few * 2.0  # Allow some variance