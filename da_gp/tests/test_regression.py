"""Regression tests for backend consistency."""

import numpy as np
import pytest


def test_backends_match():
    """Test that different backends produce consistent results."""
    from da_gp.src.cli import _load_backend
    
    # Test with moderate number of observations
    n_obs = 300
    
    try:
        mean_skl, *_ = _load_backend("sklearn", n_obs=n_obs)
        mean_dap, *_ = _load_backend("dapper", n_obs=n_obs)
        
        # RMSE between means should be less than prior std (1.0)
        rmse = np.sqrt(np.mean((mean_skl - mean_dap) ** 2))
        assert rmse < 1.0, f"Backend RMSE too large: {rmse:.6f}"
        
        # Should be much smaller than prior std in practice
        assert rmse < 0.1, f"Backend consistency poor: {rmse:.6f}"
        
    except ImportError:
        pytest.skip("DAPPER not available for comparison")


def test_backend_output_format():
    """Test that backends return required format."""
    from da_gp.src.gp_sklearn import run as run_sklearn
    
    result = run_sklearn(n_obs=100)
    
    # Check required keys are present
    required_keys = ["posterior_mean", "posterior_samples", "obs", "mask"]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"
    
    # Check shapes
    assert result["posterior_mean"].shape == (2000,), "Wrong posterior_mean shape"
    assert result["posterior_samples"].shape[1] == 2000, "Wrong posterior_samples shape"
    assert len(result["obs"]) == 100, "Wrong obs length"
    assert len(result["mask"]) == 100, "Wrong mask length"


def test_truth_and_mask_consistency():
    """Test that truth/mask generation is consistent with same function calls."""
    from da_gp.src.gp_common import get_truth_and_mask
    
    # Reset the global rng in gp_common
    import da_gp.src.gp_common
    da_gp.src.gp_common.rng = np.random.default_rng(42)
    truth1, mask1 = get_truth_and_mask(100)
    
    da_gp.src.gp_common.rng = np.random.default_rng(42)
    truth2, mask2 = get_truth_and_mask(100)
    
    assert np.allclose(truth1, truth2), "Truth generation not deterministic"
    assert np.array_equal(mask1, mask2), "Mask generation not deterministic"


def test_observations_generation():
    """Test observation generation consistency."""
    from da_gp.src.gp_common import get_truth_and_mask, get_observations
    
    truth, mask = get_truth_and_mask(50)
    obs1 = get_observations(truth, mask, noise_std=0.1)
    obs2 = get_observations(truth, mask, noise_std=0.1)
    
    # Different noise realizations should be different
    assert not np.allclose(obs1, obs2), "Observations should have different noise"
    
    # But should be close to truth values
    truth_obs = truth[mask]
    assert np.std(obs1 - truth_obs) < 0.2, "Observation noise too large"
    assert np.std(obs2 - truth_obs) < 0.2, "Observation noise too large"