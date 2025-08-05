# FILE: da-gp/src/gp_dapper.py

import numpy as np
from .gp_common import GRID_SIZE, draw_prior, make_obs_mask, generate_truth, make_observations

try:
    from dapper import da_methods, mods
    from dapper.tools.randvars import RV  # Use RV directly for clarity
    DAPPER_AVAILABLE = True
except ImportError:
    DAPPER_AVAILABLE = False


def init_state(n_ens: int) -> np.ndarray:
    """Initialize a zero-mean ensemble with the correct spatial covariance."""
    return np.stack([draw_prior() for _ in range(n_ens)])


def obs_op(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Observation operator mapping state to observations."""
    return state[..., mask]


def run(n_ens: int = 40, n_obs: int = 5_000, truth: np.ndarray = None, mask: np.ndarray = None, obs: np.ndarray = None) -> dict:
    """Run DAPPER EnKF assimilation with corrected prior and timeline."""
    if not DAPPER_AVAILABLE:
        raise ImportError("DAPPER not available. Install with: uv pip install 'dapper>=1.0'")

    # Use provided data or generate synthetic experiment
    if truth is None or mask is None or obs is None:
        mask = make_obs_mask(n_obs)
        truth = generate_truth()
        obs = make_observations(truth, mask, noise_std=0.1)

    # --- Corrected DAPPER Setup ---

    # 1. Define the initial ensemble with a zero mean and kernel-derived covariance
    # This ensemble has the correct statistics to match the GP prior.
    initial_ensemble = init_state(n_ens)

    # 2. Provide this ensemble to the HMM using the modern RV(func=...) API.
    # The developer's suggested `mods.Ensemble` class does not exist.
    # We wrap our specific ensemble matrix in a lambda to fit the API.
    X0 = RV(M=GRID_SIZE, func=lambda N: initial_ensemble)

    # 3. Define the dynamics, observations, and timeline.
    # The developer's timeline suggestion is correct.
    Dyn = {'M': GRID_SIZE, 'model': mods.Id_op(), 'noise': 0}
    Obs = mods.partial_Id_Obs(GRID_SIZE, mask)
    Obs['noise'] = 0.01  # obs error variance of 0.1*0.1
    tseq = mods.Chronology(dt=1, dko=1, T=1) # Working combination

    HMM = mods.HiddenMarkovModel(Dyn, Obs, tseq=tseq, X0=X0)

    # 4. Configure and run the EnKF
    enkf = da_methods.EnKF("Sqrt", N=n_ens, infl=1.0)

    # Truth (xx) needs K+1=2 rows. Obs (yy) needs Ko+1=1 row.
    xx = np.vstack([truth, truth])
    yy = obs.reshape(1, -1)

    # Assimilate. Results are stored in the enkf object.
    enkf.assimilate(HMM, xx, yy)

    # 5. Extract results from the analysis step
    posterior_mean = enkf.stats.mu.a[0]
    
    # For ensemble, we need to get it from the EnKF object itself or reconstruct
    if hasattr(enkf, 'E'):
        posterior_ensemble = enkf.E  # Current ensemble state
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