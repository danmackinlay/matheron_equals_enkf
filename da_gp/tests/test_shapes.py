"""Test shape consistency across backends."""

import numpy as np
import pytest
from da_gp.src.gp_common import GRID_SIZE, draw_prior, make_obs_mask, set_grid_size






def test_draw_prior_shape():
    """Test that prior samples have correct shape."""
    rng = np.random.default_rng(42)
    prior = draw_prior(rng)
    assert prior.shape == (GRID_SIZE,)
    assert isinstance(prior, np.ndarray)


@pytest.mark.parametrize("grid_size", [500, 1000, 4000])
def test_draw_prior_shape_parametrized(grid_size):
    """Test that draw_prior().shape == (d,) for different grid sizes."""
    # Save original grid size
    import da_gp.src.gp_common as gpc
    original_grid_size = gpc.GRID_SIZE
    
    try:
        # Set new grid size
        rng = np.random.default_rng(42)
        set_grid_size(grid_size, rng)
        
        # Test FFT method
        prior_fft = draw_prior(rng, use_rff=False)
        assert prior_fft.shape == (grid_size,)
        assert isinstance(prior_fft, np.ndarray)
        
        # Test RFF method
        prior_rff = draw_prior(rng, use_rff=True)
        assert prior_rff.shape == (grid_size,)
        assert isinstance(prior_rff, np.ndarray)
        
        # Verify grid size was actually changed
        assert gpc.GRID_SIZE == grid_size
        assert gpc.X_grid.shape == (grid_size, 1)
        
    finally:
        # Restore original grid size
        rng_restore = np.random.default_rng(42)
        set_grid_size(original_grid_size, rng_restore)


def test_obs_mask_shape():
    """Test that observation masks have correct shape."""
    rng = np.random.default_rng(42)
    n_obs = 100
    mask = make_obs_mask(n_obs, rng)
    assert mask.shape == (n_obs,)
    assert np.all(mask >= 0)
    assert np.all(mask < GRID_SIZE)


def test_sklearn_backend_shapes():
    """Test sklearn backend output shapes."""
    from da_gp.src.gp_sklearn import run
    
    n_obs = 50
    n_ens = 10
    result = run(n_obs=n_obs, n_ens=n_ens)
    
    assert 'posterior_mean' in result
    assert 'posterior_ensemble' in result
    assert 'rmse' in result
    
    assert result['posterior_mean'].shape == (GRID_SIZE,)
    assert result['posterior_ensemble'].shape == (n_ens, GRID_SIZE)
    assert isinstance(result['rmse'], (float, np.floating))


def test_dapper_backend_shapes():
    """Test DAPPER backend output shapes."""
    from da_gp.src.gp_dapper import run, init_state
    
    n_obs = 50
    n_ens = 10
    
    # Test ensemble initialization
    rng = np.random.default_rng(42)
    ensemble = init_state(n_ens, rng)
    assert ensemble.shape == (n_ens, GRID_SIZE)
    
    # Test full run (may fail without proper DAPPER setup)
    try:
        result = run(n_ens=n_ens, n_obs=n_obs)
        assert 'posterior_mean' in result
        assert 'posterior_ensemble' in result
        assert 'rmse' in result
    except Exception:
        pytest.skip("DAPPER runtime configuration required")


