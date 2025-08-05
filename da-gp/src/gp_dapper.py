# FILE: da-gp/src/gp_dapper.py

import numpy as np
from .gp_common import GRID_SIZE, draw_prior, make_obs_mask, generate_truth, make_observations

try:
    from dapper import da_methods, mods
    from dapper.tools.randvars import RV
    DAPPER_AVAILABLE = True
except ImportError:
    DAPPER_AVAILABLE = False


def init_state(n_ens: int) -> np.ndarray:
    """Initialize a zero-mean ensemble with the correct spatial covariance."""
    return np.stack([draw_prior() for _ in range(n_ens)])


def run(n_ens: int = 40, n_obs: int = 5_000, truth: np.ndarray = None, mask: np.ndarray = None, obs: np.ndarray = None) -> dict:
    """Run DAPPER EnKF assimilation with an optimized, efficient setup."""
    if not DAPPER_AVAILABLE:
        raise ImportError("DAPPER not available. Install with: uv pip install 'dapper>=1.0'")

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
    tseq = mods.Chronology(dt=1, dko=1, T=1)  # Working chronology
    HMM = mods.HiddenMarkovModel(Dyn, Obs, tseq=tseq, X0=X0)

    # DA Method: Use the highly efficient "Serial" update
    enkf = da_methods.EnKF("Serial", N=n_ens, infl=1.0, rot=False)

    # Data shapes matching the timeline
    xx = np.vstack([truth, truth])
    yy = obs.reshape(1, -1)

    # Assimilate. This modifies `enkf` in-place.
    enkf.assimilate(HMM, xx, yy)

    # Extract results - get mean from stats, ensemble from EnKF object or reconstruct
    posterior_mean = enkf.stats.mu.a[0]
    
    # Try to get ensemble from EnKF object, fallback to reconstruction
    if hasattr(enkf, 'E'):
        posterior_ensemble = enkf.E
    else:
        # Reconstruct ensemble by sampling from posterior statistics
        posterior_ensemble = np.random.multivariate_normal(
            posterior_mean, 
            np.eye(len(posterior_mean)) * enkf.stats.spread.a[0].mean()**2,
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