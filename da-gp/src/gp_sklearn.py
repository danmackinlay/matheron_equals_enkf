"""Scikit-learn baseline for Gaussian Process regression."""

import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from .gp_common import X_grid, kernel, make_obs_mask, generate_truth, make_observations


def init_state() -> np.ndarray:
    """Initialize GP state (not used, included for API consistency)."""
    return generate_truth()


def obs_op(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Observation operator (not used, included for API consistency)."""
    return state[mask]


def run(n_obs: int = 5_000, **kwargs) -> dict:
    """Run scikit-learn Gaussian Process regression."""
    # Generate synthetic experiment
    mask = make_obs_mask(n_obs)
    truth = generate_truth()
    obs = make_observations(truth, mask, noise_std=0.1)
    
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
    
    # Sample from posterior (approximate)
    n_samples = kwargs.get('n_ens', 40)
    posterior_samples = np.array([
        posterior_mean + posterior_std * np.random.standard_normal(len(posterior_mean))
        for _ in range(n_samples)
    ])
    
    rmse = np.sqrt(np.mean((posterior_mean - truth)**2))
    
    return {
        'posterior_mean': posterior_mean,
        'posterior_ensemble': posterior_samples,
        'posterior_std': posterior_std,
        'rmse': rmse,
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': fit_time + predict_time,
        'n_obs': n_obs,
        'log_marginal_likelihood': gp.log_marginal_likelihood(),
    }