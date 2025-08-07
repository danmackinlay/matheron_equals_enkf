# FILE: da-gp/src/gp_dapper.py

import numpy as np
from .gp_common import GRID_SIZE, draw_prior, make_obs_mask, generate_truth, make_observations

from dapper import da_methods, mods
from dapper.tools.randvars import RV


def init_state(n_ens: int) -> np.ndarray:
    """Initialize a zero-mean ensemble with the correct spatial covariance."""
    return np.stack([draw_prior() for _ in range(n_ens)])


def _run(n_ens: int = 40, n_obs: int = 5_000, truth: np.ndarray = None, mask: np.ndarray = None, obs: np.ndarray = None, method: str = 'EnKF') -> dict:
    """Internal, generalized DAPPER runner."""
    if truth is None or mask is None or obs is None:
        mask = make_obs_mask(n_obs)
        truth = generate_truth()
        obs = make_observations(truth, mask, noise_std=0.1)

    # Prior: Zero-mean ensemble with kernel-derived covariance
    initial_ensemble = init_state(n_ens)
    X0 = RV(M=GRID_SIZE, func=lambda N: initial_ensemble)

    # HMM components: Use the minimal working timeline
    Dyn = {'M': GRID_SIZE, 'model': mods.Id_op(), 'noise': 0}
    Obs = mods.partial_Id_Obs(GRID_SIZE, mask)
    Obs['noise'] = 0.01  # Variance
    
    # Add localizer for LETKF if needed
    if method == 'LETKF':
        from dapper.tools.localization import nd_Id_localization
        Obs['localizer'] = nd_Id_localization((GRID_SIZE,), obs_inds=mask)
        
    tseq = mods.Chronology(dt=1, dko=1, T=1)  # Working chronology
    HMM = mods.HiddenMarkovModel(Dyn, Obs, tseq=tseq, X0=X0)

    # --- Select DA method based on argument ---
    if method == 'EnKF':
        da_method = da_methods.EnKF("Serial", N=n_ens, infl=1.0, rot=False)
    elif method == 'LETKF':
        da_method = da_methods.LETKF(N=n_ens, infl=1.05, loc_rad=50)
    else:
        raise ValueError(f"Unknown DAPPER method: {method}")

    # Data shapes matching the timeline
    xx = np.vstack([truth, truth])
    yy = obs.reshape(1, -1)

    # Assimilate. This modifies `da_method` in-place.
    da_method.assimilate(HMM, xx, yy)

    # Extract results - get mean from stats, ensemble from DA object or reconstruct
    posterior_mean = da_method.stats.mu.a[0]
    
    # Try to get ensemble from DA object, fallback to reconstruction
    if hasattr(da_method, 'E'):
        posterior_ensemble = da_method.E
    else:
        # Reconstruct ensemble by sampling from posterior statistics
        posterior_ensemble = np.random.multivariate_normal(
            posterior_mean, 
            np.eye(len(posterior_mean)) * da_method.stats.spread.a[0].mean()**2,
            size=n_ens
        )
    truth_at_analysis_time = xx[HMM.tseq.kko[0]]

    return {
        'posterior_mean': posterior_mean,
        'posterior_ensemble': posterior_ensemble,
        'posterior_samples': posterior_ensemble,
        'obs': obs,
        'mask': mask,
        'rmse': np.sqrt(np.mean((posterior_mean - truth_at_analysis_time)**2)),
        'n_ens': n_ens,
        'n_obs': n_obs,
    }


def run_enkf(**kwargs):
    """Public-facing runner for standard EnKF."""
    return _run(method='EnKF', **kwargs)


def run_letkf(**kwargs):
    """Public-facing runner for LETKF."""
    return _run(method='LETKF', **kwargs)


def run(**kwargs):
    """Backward compatibility: defaults to EnKF."""
    return run_enkf(**kwargs)