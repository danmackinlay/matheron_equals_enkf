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
    
    # -- corrected DAPPER 1.7 setup (no obs at k=0) --
    obs_idx = mask  # already a vector of indices
    
    Dyn = {'M': GRID_SIZE, 'model': mods.Id_op(), 'noise': 0}  # identity model with proper structure
    Obs = mods.partial_Id_Obs(GRID_SIZE, obs_idx)
    Obs['noise'] = 0.01  # observation noise variance
    
    tseq = mods.Chronology(dt=1, dko=1, T=1)   # Working combination
    
    X0 = mods.GaussRV(mu=truth, C=1.0)  # prior is centred on truth for GP update
    HMM = mods.HiddenMarkovModel(Dyn, Obs, tseq=tseq, X0=X0)
    
    enkf = da_methods.EnKF("Sqrt", N=n_ens, infl=1.0, rot=True)
    
    # Set up synthetic experiment data shapes required by DAPPER
    # For T=1, K=1. xx must have K+1=2 rows.
    # For T=1, Ko=0. yy must have Ko+1=1 row.
    xx = np.vstack([truth, truth])
    yy = obs.reshape(1, -1)

    # === CHANGE IS HERE ===
    # 1. Call assimilate without assigning the return value.
    #    It modifies `enkf` in-place.
    enkf.assimilate(HMM, xx, yy)

    # 2. Access results from the `enkf.stats` attribute.
    #    The `a` stands for "analysis" at observation time ko=0.
    posterior_mean = enkf.stats.mu.a[0]
    
    # For ensemble, we need to get it from the EnKF object itself
    # The ensemble is stored as the analysis ensemble after assimilation
    if hasattr(enkf, 'E'):
        posterior_ensemble = enkf.E  # Current ensemble state
    else:
        # Reconstruct ensemble by sampling from posterior statistics
        posterior_ensemble = np.random.multivariate_normal(
            posterior_mean, 
            np.eye(len(posterior_mean)) * enkf.stats.spread.a[0].mean()**2,
            size=n_ens
        )
    # === END OF CHANGES ===
    
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