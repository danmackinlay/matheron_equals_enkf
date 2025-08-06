"""Common utilities for GP and DA experiments."""

import numpy as np
from sklearn.gaussian_process.kernels import RBF

GRID_SIZE = 2_000          # state dimension
OBS_SITES = 5_000          # can CLI-override to 50_000+

rng = np.random.default_rng(42)
X_grid = np.arange(GRID_SIZE).reshape(-1, 1)
kernel = 1.0 * RBF(length_scale=10.0)


def draw_prior_fft() -> np.ndarray:
    """
    Draw an exact sample from the GP prior using the FFT-based
    circulant embedding method. This is highly efficient for stationary
    kernels on a regular grid.

    This method avoids the O(N^3) cost of Cholesky decomposition by
    leveraging the fact that the covariance matrix is Toeplitz. The
    multiplication of a Toeplitz matrix with a vector (which is what
    sampling requires) can be computed efficiently via convolution,
    which is implemented using the FFT.

    Returns:
        np.ndarray: A sample from the GP prior of shape (GRID_SIZE,).
    """
    # 1. Compute the first row of the covariance matrix. Due to stationarity,
    #    this row defines the entire Toeplitz matrix.
    cov_row = kernel(X_grid[:1], X_grid).flatten()

    # 2. Embed this into a larger circulant row for periodic convolution.
    #    The size must be at least 2*(GRID_SIZE - 1). Using 2*GRID_SIZE is safe.
    N_circ = 2 * GRID_SIZE
    circ_row = np.zeros(N_circ)
    circ_row[:GRID_SIZE] = cov_row
    circ_row[N_circ - GRID_SIZE + 1:] = cov_row[1:][::-1]

    # 3. The eigenvalues of the circulant matrix are the FFT of its first row.
    #    This should be real due to the symmetry of the kernel.
    lambda_ = np.fft.fft(circ_row).real
    # Ensure non-negativity due to potential floating point errors
    lambda_ = np.maximum(lambda_, 0)

    # 4. Generate complex Gaussian white noise in the Fourier domain.
    #    The FFT of a real signal has conjugate symmetry.
    noise_freq = rng.normal(size=N_circ) + 1j * rng.normal(size=N_circ)
    noise_freq[0] = rng.normal() # DC component is real
    if N_circ % 2 == 0:
        noise_freq[N_circ // 2] = rng.normal() # Nyquist component is real
    else:
        noise_freq[(N_circ+1)//2:] = np.conj(noise_freq[1:(N_circ+1)//2][::-1])


    # 5. Color the noise with the sqrt of the eigenvalues and transform back.
    #    This is equivalent to K^(1/2) @ noise in the spatial domain.
    #    The scaling factor of sqrt(N_circ) is required by the FFT definition.
    sample_freq = noise_freq * np.sqrt(lambda_)
    sample = np.fft.ifft(sample_freq).real * np.sqrt(N_circ)

    # 6. Truncate to the original grid size.
    return sample[:GRID_SIZE]


def draw_prior() -> np.ndarray:
    """Draw a sample from the GP prior."""
    # This now calls the highly efficient FFT-based version.
    return draw_prior_fft()


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