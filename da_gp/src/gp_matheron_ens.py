# FILE: da_gp/src/gp_matheron_ens.py
# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hand-rolled Gaussian process regression using a pathwise Matheron update.

This backend implements GP posterior sampling in a *state-space* representation
using random Fourier features (RFF) and the Woodbury identity. It never forms
the dense d×d covariance matrix over the state grid. All computations are
expressed in terms of:

  - feature dimension R (RFF_DIM),
  - number of observations m = problem.n_obs,
  - ensemble size n_ens,

with complexity O(R^2 m + R^3) for the update and O(d R n_ens) for drawing
posterior samples on a grid of size d.

Compared to the `gp_sklearn` backend this is:
  - approximate (due to the RFF kernel approximation),
  - entirely implemented in NumPy (no sklearn / DAPPER),
  - strictly sample based: the posterior mean is the ensemble mean.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from .gp_common import (
    Problem,
    RFF_DIM,
    generate_experiment_data,
    make_grid,
    make_rff_params,
)


def _phi(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Random Fourier feature map φ(x).

    Args:
        x: Input locations, shape (n, 1).
        W: Frequency matrix, shape (RFF_DIM, 1).
        b: Phase offsets, shape (RFF_DIM,).

    Returns:
        Feature matrix Φ(x) of shape (n, RFF_DIM).
    """
    # x @ W.T -> (n, RFF_DIM)
    return np.sqrt(2.0 / RFF_DIM) * np.cos(x @ W.T + b)


def _build_woodbury_system(
    Phi_obs: np.ndarray, noise_std: float
) -> Tuple[np.ndarray, float]:
    """Precompute matrices for applying Σ_yy^{-1} via Woodbury.

    Model:
        y = Φ_obs w + ε,     w ~ N(0, I_R),   ε ~ N(0, σ^2 I_m).

    Then
        Σ_yy = Φ_obs Φ_obs^T + σ^2 I_m.

    Woodbury:
        Σ_yy^{-1} = σ^{-2} I_m - σ^{-2} Φ_obs A^{-1} Φ_obs^T σ^{-2},

    with
        A = I_R + σ^{-2} Φ_obs^T Φ_obs   (R × R).

    We only ever apply Σ_yy^{-1} to vectors; Σ_yy itself is never formed.

    Args:
        Phi_obs: Feature matrix at observation locations, shape (m, RFF_DIM).
        noise_std: Observation noise standard deviation σ.

    Returns:
        (L, sigma2) where L is lower Cholesky factor of A (R × R) and
        sigma2 = σ^2.
    """
    sigma2 = float(noise_std) ** 2
    if sigma2 <= 0.0:
        # Guard against degenerate noise; add a tiny jitter.
        sigma2 = 1e-8

    # A = I_R + σ^{-2} Φ^T Φ  (R × R)
    gram = (Phi_obs.T @ Phi_obs) / sigma2
    A = gram + np.eye(gram.shape[0], dtype=gram.dtype)

    L = np.linalg.cholesky(A)
    return L, sigma2


def _apply_Syy_inv(
    r: np.ndarray, Phi_obs: np.ndarray, L: np.ndarray, sigma2: float
) -> np.ndarray:
    """Apply Σ_yy^{-1} to a vector using the Woodbury identity.

    Args:
        r: Right-hand side vector, shape (m,).
        Phi_obs: Feature matrix at observations, shape (m, RFF_DIM).
        L: Lower Cholesky factor of A = I_R + σ^{-2} Φ^T Φ, shape (RFF_DIM, RFF_DIM).
        sigma2: Observation noise variance σ^2.

    Returns:
        v = Σ_yy^{-1} r, shape (m,).
    """
    # α = Φ^T r   (RFF_DIM,)
    alpha = Phi_obs.T @ r

    # Solve A β = α via Cholesky: A = L L^T.
    v = np.linalg.solve(L, alpha)
    beta = np.linalg.solve(L.T, v)

    # Woodbury: Σ_yy^{-1} r = σ^{-2} r - σ^{-2} Φ β σ^{-2}
    return (r / sigma2) - (Phi_obs @ beta) / (sigma2**2)


def run(
    problem: Problem,
    *,
    truth: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    obs: np.ndarray | None = None,
    n_ens: int = 40,
) -> dict:
    """Run hand-rolled GP regression using a Matheron update in feature space.

    We represent the GP prior as

        f(x) = φ(x)^T w,    w ~ N(0, I_R),

    using RFF with R = RFF_DIM. Observations follow

        y = Φ_obs w + ε,    ε ~ N(0, σ^2 I_m),

    with Φ_obs the feature matrix at observed locations. The joint prior [w; y]
    is Gaussian, and the conditional is obtained pathwise by the Matheron rule

        w' = w + Cov(w, y) Σ_yy^{-1} (y* - y),

    where Cov(w, y) = Φ_obs^T and Σ_yy = Φ_obs Φ_obs^T + σ^2 I_m. Woodbury
    provides Σ_yy^{-1} without ever forming Σ_yy or any d×d covariance.

    Args:
        problem: Problem specification.
        truth: Optional true field (grid_size,). If None, generated internally.
        mask: Optional observation indices (n_obs,). If None, generated.
        obs: Optional observations (n_obs,). If None, generated.
        n_ens: Ensemble size / number of posterior draws.

    Returns:
        A result dict matching the other backends:

            {
                "posterior_mean": (grid_size,),
                "posterior_ensemble": (n_ens, grid_size),
                "posterior_samples": (n_ens, grid_size),
                "posterior_std": (grid_size,),
                "obs": (n_obs,),
                "mask": (n_obs,),
                "rmse": float,
                "fit_time": float,
                "predict_time": float,
                "total_time": float,
                "n_obs": int,
            }
    """
    # Use provided data or generate from problem specification.
    if truth is None or mask is None or obs is None:
        truth, mask, obs = generate_experiment_data(problem)

    grid_size = problem.grid_size
    n_obs = problem.n_obs

    # 1D grid and observed locations
    X_grid = make_grid(grid_size)  # (d, 1)
    X_obs = X_grid[mask]           # (m, 1)

    # ---------------------- STEP 1: feature + Woodbury (fit) ----------------
    fit_start = time.perf_counter()

    # RFF parameters from the kernel; use problem.rng for determinism.
    W, b = make_rff_params(grid_size, problem.rng)

    # Features at observations: Φ_obs (m × RFF_DIM)
    Phi_obs = _phi(X_obs, W, b)

    # Precompute Woodbury system for Σ_yy^{-1}
    L, sigma2 = _build_woodbury_system(Phi_obs, problem.noise_std)

    fit_time = time.perf_counter() - fit_start

    # --------------- STEP 2: Pathwise Matheron update on w (fit) ------------
    fit_start = time.perf_counter()

    R = Phi_obs.shape[1]
    rng = problem.rng

    # Posterior weights w' for each ensemble member: (n_ens × R)
    w_post = np.empty((n_ens, R), dtype=float)

    for j in range(n_ens):
        # Prior draws:
        #   w ~ N(0, I_R),
        #   ε ~ N(0, σ^2 I_m),
        # with y = Φ_obs w + ε.
        w_prior = rng.standard_normal(size=R)
        eps_prior = rng.normal(0.0, np.sqrt(sigma2), size=n_obs)
        y_prior = Phi_obs @ w_prior + eps_prior

        # Residual between actual observation and this prior draw.
        r = obs - y_prior  # (m,)

        # v = Σ_yy^{-1} r via Woodbury.
        v = _apply_Syy_inv(r, Phi_obs, L, sigma2)  # (m,)

        # Matheron update in feature space:
        #   w' = w + Φ_obs^T v
        w_post[j] = w_prior + Phi_obs.T @ v

    fit_time += time.perf_counter() - fit_start

    # -------- STEP 3: Project posterior weights to grid and summarise (pred) -
    predict_start = time.perf_counter()

    # Features on full grid: Φ_grid (d × R)
    Phi_grid = _phi(X_grid, W, b)

    # Posterior ensemble over state grid: f'(x) = Φ_grid w'
    # Shape: (n_ens, grid_size)
    posterior_ensemble = (Phi_grid @ w_post.T).T

    posterior_mean = posterior_ensemble.mean(axis=0)
    posterior_std = posterior_ensemble.std(axis=0, ddof=1)

    predict_time = time.perf_counter() - predict_start

    rmse = float(np.sqrt(np.mean((posterior_mean - truth) ** 2)))

    return {
        "posterior_mean": posterior_mean,
        "posterior_ensemble": posterior_ensemble,
        "posterior_samples": posterior_ensemble,
        "posterior_std": posterior_std,
        "obs": obs,
        "mask": mask,
        "rmse": rmse,
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time": fit_time + predict_time,
        "n_obs": n_obs,
    }
