import numpy as np
from numba import jit
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import multiprocessing

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model

from .knn import compute_node_degree
from .utils import center_values


@jit(nopython=True)
def local_cov_weights(x, neighbors, weights):
    out = 0

    for i in range(len(x)):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xi = x[i]
            xj = x[j]
            if xi == 0 or xj == 0 or w_ij == 0:
                out += 0
            else:
                out += xi * xj * w_ij

    return out


@jit(nopython=True)
def compute_moments_weights_slow(mu, x2, neighbors, weights):
    """
    This version exaustively iterates over all |E|^2 terms
    to compute the expected moments exactly.  Used to test
    the more optimized formulations that follow
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0
    for i in range(N):

        EG2_i = 0

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            for x in range(N):
                for z in range(K):
                    y = neighbors[x, z]
                    wxy = weights[x, z]

                    s = wij * wxy
                    if s == 0:
                        continue

                    if i == x:
                        if j == y:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[y]
                    elif i == y:
                        if j == x:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[x]
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = mu[i] * x2[j] * mu[y]
                        elif j == y:
                            t1 = mu[i] * x2[j] * mu[x]
                        else:  # i and j are unique, no shared nodes
                            t1 = mu[i] * mu[j] * mu[x] * mu[y]

                    EG2_i += s * t1

        EG2 += EG2_i

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights(mu, x2, neighbors, weights):

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0

    #   Get the x^2*y*z terms
    t1 = np.zeros(N)
    t2 = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1[i] += wij * mu[j]
            t2[i] += wij**2 * mu[j] ** 2

            t1[j] += wij * mu[i]
            t2[j] += wij**2 * mu[i] ** 2

    t1 = t1**2

    for i in range(N):
        EG2 += (x2[i] - mu[i] ** 2) * (t1[i] - t2[i])

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2 * (x2[i] * x2[j] - (mu[i] ** 2) * (mu[j] ** 2))

    EG2 += EG**2

    return EG, EG2


@jit(nopython=True)
def compute_local_cov_max(node_degrees, vals):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2

def initializer(neighbors, weights, num_umi, model, centered, Wtot2, D):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    g_neighbors = neighbors
    g_weights = weights
    g_num_umi = num_umi
    g_model = model
    g_centered = centered
    g_Wtot2 = Wtot2
    g_D = D

def compute_hs(
    counts, neighbors, weights, num_umi, model, genes, centered=False, jobs=1,
    use_gpu=False
):

    if use_gpu:
        return _compute_hs_gpu(
            counts, neighbors, weights, num_umi, model, genes, centered
        )

    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    def data_iter():
        for i in range(counts.shape[0]):
            vals = counts[i]
            if issparse(vals):
                vals = vals.toarray().ravel()
            vals = vals.astype("double")
            yield vals

    if jobs > 1:
        with multiprocessing.Pool(
            processes=jobs,
            initializer=initializer,
            initargs=[neighbors, weights, num_umi, model, centered, Wtot2, D]
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        _map_fun_parallel,
                        data_iter()
                    ),
                    total=counts.shape[0]
                )
            )
    else:

        def _map_fun(vals):
            return _compute_hs_inner(
                vals, neighbors, weights, num_umi, model, centered, Wtot2, D
            )

        results = list(tqdm(map(_map_fun, data_iter()), total=counts.shape[0]))

    results = pd.DataFrame(results, index=genes, columns=["G", "EG", "stdG", "Z", "C"])

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]

    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"

    results = results[["C", "Z", "Pval", "FDR"]]  # Remove other columns

    return results


def _fit_gene(vals, model, num_umi):
    """Fit a gene model and return (vals, mu, var, x2).

    For the bernoulli model, vals is binarized before fitting.
    """
    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = bernoulli_model.fit_gene_model(vals, num_umi)
    elif model == "danb":
        mu, var, x2 = danb_model.fit_gene_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = normal_model.fit_gene_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = none_model.fit_gene_model(vals, num_umi)
    else:
        raise ValueError("Invalid Model: {}".format(model))
    return vals, mu, var, x2


def _compute_hs_inner(vals, neighbors, weights, num_umi, model, centered, Wtot2, D):
    """
    Note, since this is an inner function, for parallelization to work well
    none of the contents of the function can use MKL or OPENBLAS threads.
    Or else we open too many.  Because of this, some simple numpy operations
    are re-implemented using numba instead as it's difficult to control
    the number of threads in numpy after it's imported
    """

    vals, mu, var, x2 = _fit_gene(vals, model, num_umi)

    if centered:
        vals = center_values(vals, mu, var)

    G = local_cov_weights(vals, neighbors, weights)

    if centered:
        EG, EG2 = 0, Wtot2
    else:
        EG, EG2 = compute_moments_weights(mu, x2, neighbors, weights)

    stdG = (EG2 - EG * EG) ** 0.5

    Z = (G - EG) / stdG

    G_max = compute_local_cov_max(D, vals)
    C = (G - EG) / G_max

    return [G, EG, stdG, Z, C]


def _map_fun_parallel(vals):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    return _compute_hs_inner(
        vals, g_neighbors, g_weights, g_num_umi, g_model, g_centered, g_Wtot2, g_D
    )


def _local_cov_weights_gpu(vals_gpu, W):
    """GPU batch of local_cov_weights: G[g] = vals[g] . (W @ vals[g]) for all genes."""
    smoothed_T = W @ vals_gpu.T
    return (vals_gpu * smoothed_T.T).sum(axis=1)


def _compute_moments_weights_gpu(cp, mu_gpu, x2_gpu, W, W_sq):
    """GPU batch of compute_moments_weights for all genes at once."""
    # EG[g] = mu[g] . (W @ mu[g])
    EG = (mu_gpu * (W @ mu_gpu.T).T).sum(axis=1)

    # t1[g] = (W + W.T) @ mu[g],  t2[g] = (W_sq + W_sq.T) @ mu[g]^2
    W_sym = W + W.T
    W_sq_sym = W_sq + W_sq.T
    mu2_gpu = mu_gpu ** 2

    t1_T = W_sym @ mu_gpu.T
    t2_T = W_sq_sym @ mu2_gpu.T

    # Contribution 1: sum_i (x2[i] - mu[i]^2) * (t1[i]^2 - t2[i])
    diff_var = (x2_gpu - mu2_gpu).T
    eg2_c1 = (diff_var * (t1_T ** 2 - t2_T)).sum(axis=0)

    # Contribution 2: sum_{edges} w^2 * (x2[i]*x2[j] - mu[i]^2*mu[j]^2)
    eg2_c2 = (x2_gpu.T * (W_sq @ x2_gpu.T)).sum(axis=0)
    eg2_c2 -= (mu2_gpu.T * (W_sq @ mu2_gpu.T)).sum(axis=0)

    EG2 = eg2_c1 + eg2_c2 + EG ** 2
    return EG, EG2


def _compute_local_cov_max_gpu(D_gpu, vals_gpu):
    """GPU batch of compute_local_cov_max: G_max[g] = sum_i D[i]*vals[g,i]^2 / 2."""
    return (D_gpu * vals_gpu ** 2).sum(axis=1) / 2


def _compute_hs_gpu(counts, neighbors, weights, num_umi, model, genes, centered):
    """
    GPU-accelerated version of _compute_hs_inner, batched over all genes.
    All genes are processed in parallel via sparse matrix multiplication.
    """
    import cupy as cp
    from .gpu import _require_gpu, _build_sparse_weight_matrix, _build_sparse_weight_sq_matrix

    _require_gpu()

    neighbors_np = neighbors.values
    weights_np = weights.values
    num_umi_np = num_umi.values

    N_genes = counts.shape[0]
    N_cells = counts.shape[1]

    D = compute_node_degree(neighbors_np, weights_np)
    Wtot2 = (weights_np ** 2).sum()

    all_vals = np.zeros((N_genes, N_cells), dtype="double")
    all_mu = np.zeros((N_genes, N_cells), dtype="double")
    all_x2 = np.zeros((N_genes, N_cells), dtype="double")

    for i in range(N_genes):
        raw = counts[i]
        if issparse(raw):
            raw = raw.toarray().ravel()
        raw = np.asarray(raw).ravel().astype("double")

        vals, mu, var, x2 = _fit_gene(raw, model, num_umi_np)
        if centered:
            vals = center_values(vals, mu, var)
        all_vals[i] = vals
        all_mu[i] = mu
        all_x2[i] = x2

    vals_gpu = cp.asarray(all_vals)
    D_gpu = cp.asarray(D)
    W = _build_sparse_weight_matrix(neighbors_np, weights_np, shape=(N_cells, N_cells))

    G_stats = _local_cov_weights_gpu(vals_gpu, W)

    if centered:
        EG = cp.zeros(N_genes, dtype="double")
        EG2 = cp.full(N_genes, Wtot2, dtype="double")
    else:
        mu_gpu = cp.asarray(all_mu)
        x2_gpu = cp.asarray(all_x2)
        W_sq = _build_sparse_weight_sq_matrix(
            neighbors_np, weights_np, shape=(N_cells, N_cells)
        )
        EG, EG2 = _compute_moments_weights_gpu(cp, mu_gpu, x2_gpu, W, W_sq)

    stdG = (EG2 - EG * EG) ** 0.5
    Z = (G_stats - EG) / stdG

    G_max = _compute_local_cov_max_gpu(D_gpu, vals_gpu)
    C = (G_stats - EG) / G_max

    results = pd.DataFrame(
        {
            "G": cp.asnumpy(G_stats), "EG": cp.asnumpy(EG),
            "stdG": cp.asnumpy(stdG), "Z": cp.asnumpy(Z), "C": cp.asnumpy(C),
        },
        index=genes,
    )

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]
    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"
    results = results[["C", "Z", "Pval", "FDR"]]

    return results
