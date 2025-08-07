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

# FILE: da-gp/src/gp_dapper.py

import time
import numpy as np
from .gp_common import GRID_SIZE

from dapper import da_methods, mods
from dapper.tools.randvars import RV
from dapper.tools.seeding import set_seed


def init_state(n_ens: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize a zero-mean ensemble with the correct spatial covariance."""
    from .gp_common import draw_prior
    return np.stack([draw_prior(rng) for _ in range(n_ens)])


def _run(n_ens: int = 40, n_obs: int = 5_000, truth: np.ndarray = None, mask: np.ndarray = None, obs: np.ndarray = None, method: str = 'EnKF', seed: int = None) -> dict:
    """Internal, generalized DAPPER runner."""
    if seed is not None:
        set_seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    if truth is None or mask is None or obs is None:
        from .gp_common import make_obs_mask, generate_truth, make_observations
        mask = make_obs_mask(n_obs, rng)
        truth = generate_truth(rng)
        obs = make_observations(truth, mask, 0.1, rng)

    # Prior: Zero-mean ensemble with kernel-derived covariance
    initial_ensemble = init_state(n_ens, rng)
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

    # Time the assimilation (fit) step
    t0 = time.perf_counter()
    da_method.assimilate(HMM, xx, yy)
    fit_time = time.perf_counter() - t0

    # Time the prediction/extraction (predict) step
    t0 = time.perf_counter()
    posterior_mean = da_method.stats.mu.a[0]
    
    # Try to get ensemble from DA object, fallback to reconstruction
    if hasattr(da_method, 'E'):
        posterior_ensemble = da_method.E
    else:
        # Reconstruct ensemble by sampling from posterior statistics using seeded RNG
        posterior_ensemble = rng.multivariate_normal(
            posterior_mean, 
            np.eye(len(posterior_mean)) * da_method.stats.spread.a[0].mean()**2,
            size=n_ens
        )
    truth_at_analysis_time = xx[HMM.tseq.kko[0]]
    predict_time = time.perf_counter() - t0

    return {
        'posterior_mean': posterior_mean,
        'posterior_ensemble': posterior_ensemble,
        'posterior_samples': posterior_ensemble,
        'obs': obs,
        'mask': mask,
        'rmse': np.sqrt(np.mean((posterior_mean - truth_at_analysis_time)**2)),
        'fit_time': fit_time,
        'predict_time': predict_time,
        'total_time': fit_time + predict_time,
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