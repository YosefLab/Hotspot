import numpy as np
import pandas as pd
from hotspot import sim_data
from hotspot.knn import neighbors_and_weights, make_weights_non_redundant
from hotspot.gpu import _build_sparse_weight_matrix
from hotspot.local_stats import (
    _compute_moments_weights,
    _local_cov_weights,
    compute_moments_weights_slow
)
from .numba_reference import local_cov_weights
import math


def test_moments_fast():
    """
    Output of the batched _compute_moments_weights should be the same as
    compute_moments_weights_slow
    """

    # Simulate some data
    N_CELLS = 100
    N_DIM = 10

    latent = sim_data.sim_latent(N_CELLS, N_DIM)
    latent = pd.DataFrame(latent)

    def _compute(neighbors, weights, rel_tol=1e-12):
        neighbors = neighbors.values
        weights = weights.values

        weights = make_weights_non_redundant(neighbors, weights)

        # Just generate random muX, muY, x2, y2
        muX = np.random.rand(N_CELLS) * 2 + 2
        x2 = (np.random.rand(N_CELLS) * 2 + 1)**2 + muX**2

        EG, EG2 = compute_moments_weights_slow(
            muX, x2, neighbors, weights
        )
        shape = (N_CELLS, N_CELLS)
        W = _build_sparse_weight_matrix(neighbors, weights, shape)
        W_sq = _build_sparse_weight_matrix(neighbors, weights, shape, square=True)
        EG_fast, EG2_fast = _compute_moments_weights(muX[None], x2[None], W, W_sq)

        assert math.isclose(EG, EG_fast[0], rel_tol=rel_tol)
        assert math.isclose(EG2, EG2_fast[0], rel_tol=rel_tol)

        G = local_cov_weights(muX, neighbors, weights)
        assert math.isclose(G, _local_cov_weights(muX[None], W)[0], rel_tol=1e-12)

    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3, approx_neighbors=False)

    _compute(neighbors, weights)


    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3, approx_neighbors=True)

    _compute(neighbors, weights, rel_tol=0.005)
