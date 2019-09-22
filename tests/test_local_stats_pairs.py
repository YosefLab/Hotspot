import numpy as np
import pandas as pd
from hotspot import sim_data
from hotspot.knn import neighbors_and_weights
from hotspot.local_stats_pairs import (
    compute_moments_weights_pairs,
    compute_moments_weights_pairs_fast,
    compute_moments_weights_pairs_std
)
import math


def test_pairs_fast():
    """
    Output of compute_moments_weights_pairs should be equal to
    compute_moments_weights_pairs_fast
    """

    # Simulate some data
    N_CELLS = 100
    N_DIM = 10

    latent = sim_data.sim_latent(N_CELLS, N_DIM)
    latent = pd.DataFrame(latent)

    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3)

    neighbors = neighbors.values
    weights = weights.values

    # Just generate random muX, muY, x2, y2
    muX = np.random.rand(N_CELLS) * 2 + 2
    muY = np.random.rand(N_CELLS) * .5 + 1
    x2 = (np.random.rand(N_CELLS) * 2 + 1)**2 + muX**2
    y2 = (np.random.rand(N_CELLS) * 3 + .5)**2 + muY**2

    EG, EG2 = compute_moments_weights_pairs(
        muX, x2, muY, y2, neighbors, weights
    )
    EG_fast, EG2_fast = compute_moments_weights_pairs_fast(
        muX, x2, muY, y2, neighbors, weights
    )

    assert math.isclose(EG, EG_fast, rel_tol=1e-12)
    assert math.isclose(EG2, EG2_fast, rel_tol=1e-12)


def test_pairs_std():
    """
    Output of compute_moments_weights_pairs should be equal to
    compute_moments_weights_pairs_std when mu = 0 and x2 = 1
    """

    # Simulate some data
    N_CELLS = 100
    N_DIM = 10

    latent = sim_data.sim_latent(N_CELLS, N_DIM)
    latent = pd.DataFrame(latent)

    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3)

    neighbors = neighbors.values
    weights = weights.values

    # Just generate random muX, muY, x2, y2
    muX = np.zeros(N_CELLS)
    muY = np.zeros(N_CELLS)
    x2 = np.ones(N_CELLS)
    y2 = np.ones(N_CELLS)

    EG_fast, EG2_fast = compute_moments_weights_pairs(
        muX, x2, muY, y2, neighbors, weights
    )
    EG_std, EG2_std = compute_moments_weights_pairs_std(
        neighbors, weights
    )

    assert math.isclose(EG_std, EG_fast, rel_tol=1e-12)
    assert math.isclose(EG2_std, EG2_fast, rel_tol=1e-12)
