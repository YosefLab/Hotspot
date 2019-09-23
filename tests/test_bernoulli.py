import numpy as np
import pandas as pd
import math
from hotspot import sim_data
from hotspot import local_stats
from hotspot.knn import neighbors_and_weights, make_weights_non_redundant
from hotspot import bernoulli_model
from hotspot.utils import center_values
from hotspot import local_stats_pairs


def test_local_autocorrelation():
    """
    Test if the expected moment calculation is correct
    """

    # Simulate some data
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

    # Simulate counts
    for gene_p in [.2, 1, 5, 10]:
        N_REPS = 10000

        Gs = []
        for i in range(N_REPS):

            counts_i = sim_data.sim_counts_bernoulli(
                N_CELLS, umi_counts.values, gene_p)

            g = local_stats.local_cov_weights(
                counts_i, neighbors, weights)

            Gs.append(g)
        Gs = np.array(Gs)

        D = np.sum(1-(1-gene_p/10000)**umi_counts.values)

        mu, var, x2 = bernoulli_model.fit_gene_model(
            D, umi_counts.values
        )

        EG, EG2 = local_stats.compute_moments_weights(
            mu, x2, neighbors, weights)

        EstdG = (EG2 - EG ** 2) ** 0.5

        Gmean = Gs.mean()
        Gstd = Gs.std()

        assert math.isclose(
            EG, Gmean, rel_tol=5e-2
        ), "EG is off for gene_p={}, Actual={:.2f}, Expected={:.2f}".format(
            gene_p, Gmean, EG
        )
        assert math.isclose(
            EstdG, Gstd, rel_tol=5e-2
        ), "stdG is off for gene_p={}, Actual={:.2f}, Expected={:.2f}".format(
            gene_p, Gstd, EstdG
        )


def test_local_autocorrelation_centered():
    """
    Test if the expected moment calculation is correct
    """

    # Simulate some data
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

    Wtot2 = (weights**2).sum()

    # Simulate counts
    for gene_p in [.2, 1, 5, 10]:
        N_REPS = 10000

        D = np.sum(1-(1-gene_p/10000)**umi_counts.values)

        mu, var, x2 = bernoulli_model.fit_gene_model(
            D, umi_counts.values
        )

        Gs = []
        for i in range(N_REPS):

            counts_i = sim_data.sim_counts_bernoulli(
                N_CELLS, umi_counts.values, gene_p)

            counts_i = center_values(counts_i, mu, var)

            g = local_stats.local_cov_weights(
                counts_i, neighbors, weights)
            Gs.append(g)

        Gs = np.array(Gs)

        EG = 0
        EG2 = Wtot2
        EstdG = (EG2 - EG ** 2) ** 0.5

        Gmean = Gs.mean()
        Gstd = Gs.std()

        assert math.isclose(
            0, abs(Gmean/Gstd), abs_tol=5e-2
        ), "EG is off for gene_p={}, Actual={:.2f}, Expected={:.2f}".format(
            gene_p, Gmean, EG
        )
        assert math.isclose(
            EstdG, Gstd, rel_tol=5e-2
        ), "stdG is off for gene_p={}, Actual={:.2f}, Expected={:.2f}".format(
            gene_p, Gstd, EstdG
        )


def test_local_correlation_centered():
    """
    Test if the expected moment calculation is correct
    For pair-wise function
    """

    # Simulate some data
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

    Wtot2 = (weights**2).sum()

    gene_p2 = 1

    # Simulate counts
    for gene_p in [.2, 1, 5, 10]:
        N_REPS = 10000

        D = np.sum(1-(1-gene_p/10000)**umi_counts.values)

        D_j = np.sum(1-(1-gene_p2/10000)**umi_counts.values)

        mu, var, x2 = bernoulli_model.fit_gene_model(
            D, umi_counts.values
        )

        mu_j, var_j, x2_j = bernoulli_model.fit_gene_model(
            D_j, umi_counts.values
        )

        Gs = []
        for i in range(N_REPS):

            counts_i = sim_data.sim_counts_bernoulli(
                N_CELLS, umi_counts.values, gene_p)

            counts_i = center_values(counts_i, mu, var)

            counts_j = sim_data.sim_counts_bernoulli(
                N_CELLS, umi_counts.values, gene_p2)

            counts_j = center_values(counts_j, mu_j, var_j)

            g = local_stats_pairs.local_cov_pair(
                counts_i, counts_j, neighbors, weights)
            Gs.append(g)

        Gs = np.array(Gs)

        EG = 0
        EG2 = Wtot2/2
        EstdG = (EG2 - EG ** 2) ** 0.5

        Gmean = Gs.mean()
        Gstd = Gs.std()

        assert math.isclose(
            0, abs(Gmean/Gstd), abs_tol=5e-2
        ), "EG is off for gene_p={}, Actual={:.2f}, Expected={:.2f}".format(
            gene_p, Gmean, EG
        )
        assert math.isclose(
            EstdG, Gstd, rel_tol=5e-2
        ), "stdG is off for gene_p={}, Actual={:.2f}, Expected={:.2f}".format(
            gene_p, Gstd, EstdG
        )
