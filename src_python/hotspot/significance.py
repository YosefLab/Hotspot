"""
Functions used when computing the significance of Gi
values
"""
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, norm
from multiprocessing import Pool
from tqdm import tqdm


def get_zinn_params(data):
    """
    Get the mean, std, and p (probability of detection) modeling
    each gene as a zero-inflated normal

    Parameters
    ==========
    data: pandas.DataFrame or numpy.ndarray
        Expression matrix to get parameters
        Shape is genes x cells

    Returns
    =======
    mu: numpy.ndarray or pandas.Series
        Mean of zero-inflated normal
    std: numpy.ndarray or pandas.Series
        Std of zero-inflated normal
    p: numpy.ndarray or pandas.Series
        Probability of non-detection

    """
    nnz = (data > 0).sum(axis=1)
    nnz[nnz == 0] = 1

    mu = data.sum(axis=1) / nnz
    x2 = (data**2).sum(axis=1) / nnz
    var = x2 - mu**2
    std = var**(1/2)

    std[std < 0.01] = 0.01  # For stability

    p = 1 - nnz/data.shape[1]

    return mu, std, p


def dist_params(mu, sd, N):
    """
    What are the expected X and sd(X) where X is the sum
    of N items from a normal with mean=mu, sd=sd
    """

    if N == 0:
        return 0, 0

    ex = mu
    ex2 = sd**2 + mu**2

    e_mu = ex
    e_mu2 = (N*ex2 + (N**2-N)*ex**2) / N**2

    e_var_mu = e_mu2 - e_mu**2
    e_std_mu = e_var_mu ** (1/2)

    return e_mu*N, e_std_mu*N


def pvals_for_gene(vals, mu_i, std_i, p_i, n_neighbors):
    """
    Given an array of Gi values, and a model for the
    underlying gene (zero-inflated normal, 3 params),
    return the associated p-values
    """

    M = vals.size
    n = int(round((1 - p_i) * M))
    N = n_neighbors

    dist = hypergeom(M=M, N=N, n=n)
    pvals = np.zeros_like(vals.values)
    for k in range(0, N + 1):
        pk = dist.pmf(k)

        if k > 0:
            e_mu, e_sd = dist_params(mu_i, std_i, k)
            nonzero_dist = norm(e_mu, e_sd)

            pvals += nonzero_dist.sf(vals.values) * pk
        else:
            pvals += (vals == 0) * pk

    return pvals


def _pvals_for_gene_wrap(x):
    return pvals_for_gene(x[0], x[1], x[2], x[3], x[4])


def pvals_for_dataframe_binary(Gi, data, n_neighbors,
                               n_jobs=10):
    """
    Computes p-values for given Gi values.
    Assumes binary weights

    Gi: Gi values
    data: Underlying expression data
    N: Number of neighbors used
    n_jobs: Number of jobs to compute in parallel
    """

    mu, std, p = get_zinn_params(data)

    pool = Pool(n_jobs)
    mapfun = _pvals_for_gene_wrap

    vals = [(Gi.loc[i], mu[i], std[i], p[i], n_neighbors) for i in Gi.index]

    pvals = list(tqdm(
        pool.imap(mapfun, vals),
        total=len(vals)
    ))
    pool.close()

    pvals = np.vstack(pvals)
    pvals = pd.DataFrame(pvals, index=Gi.index, columns=Gi.columns)

    return pvals
