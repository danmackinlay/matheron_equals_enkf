"""DAPPER backend for data assimilation."""

import numpy as np
from .gp_common import X_grid, draw_prior, make_obs_mask, GRID_SIZE

try:
    from dapper import da_methods, mods
    DAPPER_AVAILABLE = True
except ImportError:
    DAPPER_AVAILABLE = False


def init_state(n_ens: int) -> np.ndarray:
    """Initialize ensemble state."""
    return np.stack([draw_prior() for _ in range(n_ens)])


def obs_op(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Observation operator mapping state to observations."""
    return state[..., mask]  # DAPPER auto-vectors across ensemble


def run(n_ens: int = 40, n_obs: int = 5_000) -> dict:
    """Run DAPPER EnKF assimilation."""
    if not DAPPER_AVAILABLE:
        raise ImportError("DAPPER not available. Install with: uv pip install 'dapper>=1.0'")
    
    mask = make_obs_mask(n_obs)
    
    # Create Hidden Markov Model
    HMM = mods.HiddenMarkovModel(
        Dyn=mods.Identity(GRID_SIZE),
        Obs=mods.partial_direct(mask),
        t=mods.Chronology(1, dkObs=1, KObs=1, T=1),
        X0=mods.GaussRV(C=1.0, M0=np.zeros(GRID_SIZE)),
    )
    
    # Configure EnKF
    filter_method = da_methods.EnKF(N=n_ens, infl=1.0, rot=True)
    
    # Run assimilation
    stats = filter_method.assimilate(HMM)
    
    return {
        'posterior_mean': stats.mu.a[-1],  # Analysis mean at final time
        'posterior_ensemble': stats.E.a[-1],  # Analysis ensemble at final time
        'rmse': np.sqrt(np.mean((stats.mu.a[-1] - stats.mu.f[-1])**2)),
        'n_ens': n_ens,
        'n_obs': n_obs,
    }