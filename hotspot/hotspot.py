import numpy as np
import pandas as pd

from .knn import (
    neighbors_and_weights,
    neighbors_and_weights_from_distances,
    make_weights_non_redundant,
)
from .local_stats import compute_hs
from .local_stats_pairs import (compute_hs_pairs, compute_hs_pairs_centered)


class Hotspot:

    def __init__(self, counts, latent=None, distances=None, umi_counts=None):
        """
        Initialize a Hotspot object for analysis

        Either `latent` or `distances` is required.

        counts : pandas.DataFrame
            Count matrix (shape is genes x cells)
        latent : pandas.DataFrame, optional
            Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims)
        distances : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells)
        umi_counts : pandas.Series, optional
            Total umi count per cell.  Used as a size factor.  Optional,
            if omitted, the sum over genes in the counts matrix is used
        """

        assert counts.shape[1] == latent.shape[0]

        self.counts = counts
        self.latent = latent
        self.distances = distances

        if latent is None and distances is None:
            raise ValueError("Neither `latent` or `distance` arguments were supplied.  One of these is required")

        if latent is not None and distances is not None:
            raise ValueError("Both `latent` and `distances` provided - only one of these should be provided.")

        if umi_counts is None:
            umi_counts = counts.sum(axis=0)
        else:
            assert umi_counts.size == counts.shape[1]

        if not isinstance(umi_counts, pd.Series):
            umi_counts = pd.Series(umi_counts)

        self.umi_counts = umi_counts

        self.graph = None

    def create_knn_graph(
            self, weighted_graph=False, n_neighbors=30, neighborhood_factor=3):

        if self.latent is not None:
            neighbors, weights = neighbors_and_weights(
                self.latent, n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor)
        else:
            neighbors, weights = neighbors_and_weights_from_distances(
                self.distances, n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor)

        self.neighbors = neighbors

        if not weighted_graph:
            weights = pd.DataFrame(
                np.ones_like(weights.values),
                index=weights.index, columns=weights.columns
            )

        weights = make_weights_non_redundant(neighbors.values, weights.values)
        weights = pd.DataFrame(
            weights, index=neighbors.index, columns=neighbors.columns)

        self.weights = weights

    def compute_hotspot(self, model='danb', centered=False, jobs=1):

        results = compute_hs(
            self.counts, self.neighbors, self.weights,
            self.umi_counts, model, centered=centered, jobs=jobs)

        self.results = results

        return self.results

    def compute_modules(self, genes, model='danb', centered=False, jobs=1):
        """
        Returns:

            lc: pd.Dataframe (genes x genes)
                local covariance between genes

            lcz: pd.Dataframe (genes x genes)
                local covariance Z-scores between genes
        """

        if centered:
            self.modules = compute_hs_pairs_centered_cond(
                self.counts.loc[genes], self.neighbors, self.weights,
                self.umi_counts, model, jobs=jobs)

        else:
            self.modules = compute_hs_pairs(
                self.counts.loc[genes], self.neighbors, self.weights,
                self.umi_counts, model, centered=False, jobs=jobs)

        return self.modules
