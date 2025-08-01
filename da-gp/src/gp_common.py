"""Common utilities for GP and DA experiments."""

import numpy as np
from sklearn.gaussian_process.kernels import RBF

GRID_SIZE = 2_000          # state dimension
OBS_SITES = 5_000          # can CLI-override to 50_000+

rng = np.random.default_rng(42)
X_grid = np.arange(GRID_SIZE).reshape(-1, 1)
kernel = 1.0 * RBF(length_scale=10.0)


def draw_prior() -> np.ndarray:
    """Draw a sample from the GP prior."""
    K = kernel(X_grid)
    return rng.multivariate_normal(np.zeros(GRID_SIZE), K)


def make_obs_mask(n_obs: int = OBS_SITES) -> np.ndarray:
    """Create observation mask by randomly selecting grid points."""
    return rng.choice(GRID_SIZE, size=n_obs, replace=True)


def generate_truth() -> np.ndarray:
    """Generate synthetic truth for validation."""
    return draw_prior()


def make_observations(truth: np.ndarray, mask: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
    """Generate noisy observations from truth at masked locations."""
    obs = truth[mask]
    noise = rng.normal(0, noise_std, size=len(obs))
    return obs + noise


def get_truth_and_mask(n_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate consistent truth and observation mask for all backends."""
    truth = generate_truth()
    mask = make_obs_mask(n_obs)
    return truth, mask


def get_observations(truth: np.ndarray, mask: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
    """Generate observations from truth and mask."""
    return make_observations(truth, mask, noise_std)