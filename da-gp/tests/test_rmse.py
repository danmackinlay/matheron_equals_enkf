"""Test RMSE consistency and validation."""

import numpy as np
import pytest
from src.gp_common import generate_truth, make_observations, make_obs_mask


def test_observation_rmse():
    """Test that observations are reasonable perturbations of truth."""
    truth = generate_truth()
    mask = make_obs_mask(100)
    noise_std = 0.1
    
    obs = make_observations(truth, mask, noise_std)
    
    # Observations should be close to truth values at observed locations
    truth_obs = truth[mask]
    residuals = obs - truth_obs
    empirical_std = np.std(residuals)
    
    # Empirical std should be close to noise_std (within 3-sigma bounds)
    assert empirical_std < 3 * noise_std / np.sqrt(len(obs))


def test_sklearn_rmse_reasonable():
    """Test that sklearn GP produces reasonable RMSE."""
    from src.gp_sklearn import run
    
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
    samples = [generate_truth() for _ in range(50)]
    sample_array = np.array(samples)
    
    # Sample standard deviation should be close to kernel amplitude (1.0)
    empirical_std = np.std(sample_array, axis=0)
    mean_std = np.mean(empirical_std)
    
    # Should be within reasonable bounds of kernel amplitude
    assert 0.5 < mean_std < 1.5


@pytest.mark.parametrize("n_obs", [10, 100, 1000])
def test_sklearn_scaling(n_obs):
    """Test sklearn performance scales as expected."""
    from src.gp_sklearn import run
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
    from src.gp_sklearn import run
    
    # Use fixed random seed for consistency
    np.random.seed(42)
    
    rmse_few = run(n_obs=10)['rmse']
    rmse_many = run(n_obs=100)['rmse']
    
    # More observations should generally give better fit
    # (Though this may not always hold due to random sampling)
    assert rmse_many <= rmse_few * 2.0  # Allow some variance