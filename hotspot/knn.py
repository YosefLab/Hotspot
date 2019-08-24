from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from math import ceil
from numba import jit


def neighbors_and_weights(data, n_neighbors=30, neighborhood_factor=3):
    """
    Computes nearest neighbors and associated weights for data
    Uses euclidean distance between rows of `data`

    Parameters
    ==========
    data: pandas.Dataframe num_cells x num_features

    Returns
    =======
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """

    coords = data.values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1,
                            algorithm="ball_tree").fit(coords)
    dist, ind = nbrs.kneighbors(coords)

    dist = dist[:, 1:]  # Exclude 'self'
    ind = ind[:, 1:]

    weights = compute_weights(
        dist, neighborhood_factor=neighborhood_factor)

    ind = pd.DataFrame(ind, index=data.index)
    neighbors = ind
    weights = pd.DataFrame(weights, index=neighbors.index,
                           columns=neighbors.columns)

    return neighbors, weights


def compute_weights(distances, neighborhood_factor=3):
    """
    Computes weights on the nearest neighbors based on a
    gaussian kernel and their distances

    Kernel width is set to the num_neighbors / neighborhood_factor's distance

    distances:  cells x neighbors ndarray
    neighborhood_factor: float

    returns weights:  cells x neighbors ndarray

    """

    radius_ii = ceil(distances.shape[1] / neighborhood_factor)

    sigma = distances[:, [radius_ii-1]]

    weights = np.exp(-1 * distances**2 / sigma**2)

    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm

    return weights


@jit(nopython=True)
def compute_node_degree(neighbors, weights):

    D = np.zeros(neighbors.shape[0])

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            D[i] += w_ij
            D[j] += w_ij

    return D


@jit(nopython=True)
def make_weights_non_redundant(neighbors, weights):
    w_no_redundant = weights.copy()

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):
            j = neighbors[i, k]

            if j < i:
                continue

            for k2 in range(neighbors.shape[1]):
                if neighbors[j, k2] == i:
                    w_ji = w_no_redundant[j, k2]
                    w_no_redundant[j, k2] = 0
                    w_no_redundant[i, k] += w_ji

    return w_no_redundant
