import numpy as np
import pandas as pd
from math import ceil
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from ._version import __version__


def compute_gi_single(x, neighbors, weights):
    """
    Calculates the getis-ord of the variable x using sparse weights
    encoded in neighbors and weights.

    Paramters
    =========
    x:         numpy.ndarray of length num_cells
    neighbors: numpy.ndarray of neighbor indices num_cells x num_neighbors
    weights:   numpy.ndarray of neighbor weights num_cells x num_neighbors

    Returns
    =======
    G_i: numpy.ndarray of length num_cells
         The Getis-Ord coefficient for each cell

    """

    W_i = weights.sum(axis=1)
    S_1i = (weights**2).sum(axis=1)

    xbar = x.mean()
    s = x.std(ddof=1)

    n = weights.shape[0]

    denom = s * ((n * S_1i - W_i**2) / (n - 1))**(1 / 2)
    offset = xbar * W_i

    neighbor_xs = x[neighbors]
    num = (neighbor_xs * weights).sum(axis=1)

    G_i = (num - offset) / denom

    return G_i


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


def compute_gi_dataframe(x, neighbors, weights):
    """
    Calculates the getis-ord of the variable x using sparse weights
    encoded in neighbors and weights.

    Paramters
    =========
    x:         pandas.Dataframe of genes: num_genes x num_cells
    neighbors: pandas.Dataframe of neighbor indices num_cells x num_neighbors
    weights:   pandas.Dataframe of neighbor weights num_cells x num_neighbors

    Returns
    =======
    G_i: pandas.Dataframe of num_genes x num_cells
         The Getis-Ord coefficient for each cell/gene

    """

    genes = x.index
    cells = x.columns

    neighbors = neighbors.loc[x.columns].values
    weights = weights.loc[x.columns].values
    x = x.values

    W_i = weights.sum(axis=1, keepdims=True).T  # 1xcells
    S_1i = (weights**2).sum(axis=1, keepdims=True).T  # 1xcells

    xbar = x.mean(axis=1, keepdims=True)  # genesx1
    s = x.std(ddof=1, axis=1, keepdims=True)  # genesx1

    n = x.shape[1]

    # same for each gene,  1xcells
    denom = s * ((n * S_1i - W_i**2) / (n - 1))**(1 / 2)
    denom[denom == 0] = 1.0

    xbar_W_i = xbar.dot(W_i)  # genes x cells

    G_i = np.zeros_like(x)
    for ii in range(x.shape[0]):
        neighbor_xs = x[ii, :][neighbors]
        G_i[ii, :] = (neighbor_xs * weights).sum(axis=1)

    G_i -= xbar_W_i
    G_i /= denom

    G_i = pd.DataFrame(G_i, index=genes, columns=cells)

    return G_i


def gi_to_pval(G_i):
    """
    Computes the p-value associated with each getis-ord coefficient

    Paramters
    =========
    G_i: pandas.Dataframe of num_genes x num_cells
         The Getis-Ord coefficient for each cell/gene

    Returns
    =======
    pvals: pandas.Dataframe of num_genes x num_cells
         The p-value associated with each Getis-Ord coeffecient
    """

    G_i_pval = pd.DataFrame(
        norm.sf(G_i.abs().values)*2, index=G_i.index, columns=G_i.columns
    )

    return G_i_pval
