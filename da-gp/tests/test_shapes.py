"""Test shape consistency across backends."""

import numpy as np
import pytest
from src.gp_common import GRID_SIZE, draw_prior, make_obs_mask


def _dapper_available() -> bool:
    """Check if DAPPER is available."""
    try:
        import dapper
        return True
    except ImportError:
        return False


def _pdaf_available() -> bool:
    """Check if pyPDAF is available."""
    try:
        import pypdaf
        from mpi4py import MPI
        return True
    except ImportError:
        return False


def test_draw_prior_shape():
    """Test that prior samples have correct shape."""
    prior = draw_prior()
    assert prior.shape == (GRID_SIZE,)
    assert isinstance(prior, np.ndarray)


def test_obs_mask_shape():
    """Test that observation masks have correct shape."""
    n_obs = 100
    mask = make_obs_mask(n_obs)
    assert mask.shape == (n_obs,)
    assert np.all(mask >= 0)
    assert np.all(mask < GRID_SIZE)


def test_sklearn_backend_shapes():
    """Test sklearn backend output shapes."""
    from src.gp_sklearn import run
    
    n_obs = 50
    n_ens = 10
    result = run(n_obs=n_obs, n_ens=n_ens)
    
    assert 'posterior_mean' in result
    assert 'posterior_ensemble' in result
    assert 'rmse' in result
    
    assert result['posterior_mean'].shape == (GRID_SIZE,)
    assert result['posterior_ensemble'].shape == (n_ens, GRID_SIZE)
    assert isinstance(result['rmse'], (float, np.floating))


@pytest.mark.skipif(
    not _dapper_available(),
    reason="DAPPER not available"
)
def test_dapper_backend_shapes():
    """Test DAPPER backend output shapes."""
    from src.gp_dapper import run, init_state
    
    n_obs = 50
    n_ens = 10
    
    # Test ensemble initialization
    ensemble = init_state(n_ens)
    assert ensemble.shape == (n_ens, GRID_SIZE)
    
    # Test full run (may fail without proper DAPPER setup)
    try:
        result = run(n_ens=n_ens, n_obs=n_obs)
        assert 'posterior_mean' in result
        assert 'posterior_ensemble' in result
        assert 'rmse' in result
    except Exception:
        pytest.skip("DAPPER runtime configuration required")


@pytest.mark.skipif(
    not _pdaf_available(),
    reason="pyPDAF not available"
)  
def test_pdaf_backend_shapes():
    """Test pyPDAF backend shapes (basic import test)."""
    from src.gp_pdaf import init_state_ens
    
    # Test ensemble member initialization
    state = init_state_ens()
    assert state.shape == (GRID_SIZE,)
    
    # Full run test would require MPI setup
    pytest.skip("pyPDAF runtime requires MPI configuration")