"""DAPPER backend for data assimilation."""

import numpy as np
from .gp_common import X_grid, draw_prior, make_obs_mask, GRID_SIZE, generate_truth, make_observations

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


def run(n_ens: int = 40, n_obs: int = 5_000, truth: np.ndarray = None, mask: np.ndarray = None, obs: np.ndarray = None) -> dict:
    """Run DAPPER EnKF assimilation."""
    if not DAPPER_AVAILABLE:
        raise ImportError("DAPPER not available. Install with: uv pip install 'dapper>=1.0'")
    
    # Use provided data or generate synthetic experiment
    if truth is None or mask is None or obs is None:
        mask = make_obs_mask(n_obs)
        truth = generate_truth()
        obs = make_observations(truth, mask, noise_std=0.1)
    
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
    
    posterior_ensemble = stats.E.a[-1]  # Analysis ensemble at final time
    
    return {
        'posterior_mean': stats.mu.a[-1],  # Analysis mean at final time
        'posterior_ensemble': posterior_ensemble,
        'posterior_samples': posterior_ensemble,  # Required format (same as ensemble)
        'obs': obs,                               # Required format
        'mask': mask,                             # Required format
        'rmse': np.sqrt(np.mean((stats.mu.a[-1] - truth)**2)),
        'n_ens': n_ens,
        'n_obs': n_obs,
    }