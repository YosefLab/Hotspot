import numpy as np
import pandas as pd
from hotspot import sim_data
from hotspot.knn import neighbors_and_weights, make_weights_non_redundant
from hotspot.local_stats_pairs import (
    compute_moments_weights_pairs_slow,
    compute_moments_weights_pairs,
    compute_moments_weights_pairs_fast,
    compute_moments_weights_pairs_std,
    local_cov_pair
)
import math

from hotspot.local_stats import local_cov_weights


def test_pairs():
    """
    Output of compute_moments_weights_pairs should be equal to
    compute_moments_weights_pairs_slow
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

    weights = make_weights_non_redundant(neighbors, weights)

    # Just generate random muX, muY, x2, y2
    muX = np.random.rand(N_CELLS) * 2 + 2
    muY = np.random.rand(N_CELLS) * .5 + 1
    x2 = (np.random.rand(N_CELLS) * 2 + 1)**2 + muX**2
    y2 = (np.random.rand(N_CELLS) * 3 + .5)**2 + muY**2

    EG, EG2 = compute_moments_weights_pairs_slow(
        muX, x2, muY, y2, neighbors, weights
    )
    EG_fast, EG2_fast = compute_moments_weights_pairs(
        muX, x2, muY, y2, neighbors, weights
    )

    assert math.isclose(EG, EG_fast, rel_tol=1e-12)
    assert math.isclose(EG2, EG2_fast, rel_tol=1e-12)


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

    weights = make_weights_non_redundant(neighbors, weights)

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

    weights = make_weights_non_redundant(neighbors, weights)

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


def test_local_correlation():
    N_CELLS = 1000
    N_DIM = 10

    latent = sim_data.sim_latent(N_CELLS, N_DIM)
    latent = pd.DataFrame(latent)

    umi_counts = sim_data.sim_umi_counts(N_CELLS, 2000, 200)
    umi_counts = pd.Series(umi_counts)

    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3
    )
    neighbors = neighbors.values
    weights = weights.values
    weights = make_weights_non_redundant(neighbors, weights)

    counts_i = np.random.randn(N_CELLS)

    gxy = local_cov_pair(
        counts_i, counts_i, neighbors, weights
    )

    g = local_cov_weights(counts_i, neighbors, weights)

    assert math.isclose(
        g, gxy, rel_tol=1e-10
    ), "Pairwise covariance on (x, x) should be same as local covariance on (x)"
