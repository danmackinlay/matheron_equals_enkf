"""pyPDAF backend for data assimilation."""

import numpy as np
from .gp_common import draw_prior, make_obs_mask, generate_truth, make_observations

try:
    import pypdaf as pf
    from mpi4py import MPI
    PDAF_AVAILABLE = True
except ImportError:
    PDAF_AVAILABLE = False

# Global state for callbacks
comm = MPI.COMM_WORLD if PDAF_AVAILABLE else None
mask = None
truth = None


def init_state_ens() -> np.ndarray:
    """Initialize ensemble member state."""
    return draw_prior()


def obs_op(state: np.ndarray) -> np.ndarray:
    """Observation operator mapping state to observations."""
    global mask
    if mask is None:
        raise RuntimeError("Observation mask not set")
    return state[mask]  # Vector index via NumPy


def prepost(step: int, ens_mean: np.ndarray, *_) -> None:
    """Pre/post processing callback."""
    global truth, comm
    if truth is not None and comm is not None and comm.rank == 0:
        rmse = np.sqrt(np.mean((ens_mean - truth)**2))
        print(f"Step {step}, RMSE={rmse:.3f}")


def run(n_ens: int = 40, n_obs: int = 5_000, truth_in: np.ndarray = None, mask_in: np.ndarray = None, obs_in: np.ndarray = None) -> dict:
    """Run pyPDAF EnKF assimilation."""
    if not PDAF_AVAILABLE:
        raise ImportError("pyPDAF not available. Install with: uv pip install pyPDAF mpi4py")
    
    global mask, truth
    
    # Use provided data or generate synthetic experiment
    if truth_in is None or mask_in is None or obs_in is None:
        mask = make_obs_mask(n_obs)
        truth = generate_truth()
        obs = make_observations(truth, mask)
    else:
        mask = mask_in
        truth = truth_in
        obs = obs_in
    
    # Initialize PDAF
    pf.init_filter(
        "enkf",
        init_state_ens,
        obs_op,
        state_size=len(draw_prior()),
        obs_size=n_obs,
        ensemble_size=n_ens,
        prepost=prepost
    )
    
    # Run assimilation
    result = pf.run()
    
    if comm is not None and comm.rank == 0:
        posterior_ensemble = result.get('posterior_ensemble', np.zeros((n_ens, len(truth))))
        return {
            'posterior_mean': result.get('posterior_mean', np.zeros_like(truth)),
            'posterior_ensemble': posterior_ensemble,
            'posterior_samples': posterior_ensemble,  # Required format
            'obs': obs,                               # Required format  
            'mask': mask,                             # Required format
            'rmse': np.sqrt(np.mean((result.get('posterior_mean', truth) - truth)**2)),
            'n_ens': n_ens,
            'n_obs': n_obs,
        }
    else:
        # Non-root ranks return empty result
        return {
            'posterior_mean': np.array([]),
            'posterior_ensemble': np.array([]),
            'posterior_samples': np.array([]),
            'obs': np.array([]),
            'mask': np.array([]),
            'rmse': 0.0,
            'n_ens': n_ens,
            'n_obs': n_obs,
        }