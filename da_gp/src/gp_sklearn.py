"""Scikit-learn baseline for Gaussian Process regression."""

import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from .gp_common import X_grid, kernel


def init_state(rng: np.random.Generator) -> np.ndarray:
    """Initialize GP state (not used, included for API consistency)."""
    from .gp_common import generate_truth
    return generate_truth(rng)


def obs_op(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Observation operator (not used, included for API consistency)."""
    return state[mask]


def run(n_obs: int = 5_000, truth: np.ndarray = None, mask: np.ndarray = None, obs: np.ndarray = None, rng: np.random.Generator = None, **kwargs) -> dict:
    """Run scikit-learn Gaussian Process regression."""
    # Handle default case
    if rng is None:
        rng = np.random.default_rng()
        
    # Use provided data or generate synthetic experiment
    if truth is None or mask is None or obs is None:
        from .gp_common import make_obs_mask, generate_truth, make_observations
        mask = make_obs_mask(n_obs, rng)
        truth = generate_truth(rng)
        obs = make_observations(truth, mask, 0.1, rng)
    
    # Create GP with fixed kernel (no optimization)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=0.01)
    
    # Fit GP to observations
    start_time = time.perf_counter()
    X_obs = X_grid[mask]
    gp.fit(X_obs, obs)
    fit_time = time.perf_counter() - start_time
    
    # Predict at all grid points
    start_time = time.perf_counter()
    posterior_mean, posterior_std = gp.predict(X_grid, return_std=True)
    predict_time = time.perf_counter() - start_time
    
    # Sample from posterior (approximate) using provided RNG
    n_samples = kwargs.get('n_ens', 40)
    posterior_samples = rng.standard_normal(size=(n_samples, len(posterior_mean)))
    posterior_samples = posterior_mean + posterior_samples * posterior_std
    
    rmse = np.sqrt(np.mean((posterior_mean - truth)**2))
    
    return {
        'posterior_mean': posterior_mean,
        'posterior_ensemble': posterior_samples,
        'posterior_samples': posterior_samples,  # Required format
        'posterior_std': posterior_std,
        'obs': obs,                              # Required format
        'mask': mask,                            # Required format
        'rmse': rmse,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': fit_time + predict_time,
        'n_obs': n_obs,
        'log_marginal_likelihood': gp.log_marginal_likelihood(),
    }