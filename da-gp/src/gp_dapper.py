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
    
    # -- new single-step assimilation --
    obs_indices = np.where(mask)[0]  # Convert boolean mask to indices
    tseq = mods.Chronology(dt=1, dto=1, T=1)  # dt, dto, T triple from original plan
    Dyn = {'M': GRID_SIZE, 'model': mods.Id_op(), 'noise': 0}
    Obs = mods.partial_Id_Obs(GRID_SIZE, obs_indices)
    X0 = mods.GaussRV(mu=np.zeros(GRID_SIZE), C=1.0)
    HMM = mods.HiddenMarkovModel(Dyn, Obs, tseq=tseq, X0=X0)
    
    # Configure EnKF
    filter_method = da_methods.EnKF('Sqrt', N=n_ens, infl=1.0, rot=True)
    
    # Run assimilation - truth as 1D, obs must match observation operator output
    stats = filter_method.assimilate(HMM, xx=truth[None, :], yy=obs.reshape(1, -1))
    
    # Get analysis state (shape indicates only one time available)
    posterior_ensemble = stats.E.a[0]  # Analysis ensemble 
    posterior_mean = stats.mu.a[0]     # Analysis mean
    
    return {
        'posterior_mean': posterior_mean,
        'posterior_ensemble': posterior_ensemble,
        'posterior_samples': posterior_ensemble,  # Required format (same as ensemble)
        'obs': obs,                               # Required format
        'mask': mask,                             # Required format
        'rmse': np.sqrt(np.mean((posterior_mean - truth)**2)),
        'n_ens': n_ens,
        'n_obs': n_obs,
    }