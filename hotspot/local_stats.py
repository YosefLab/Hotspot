import numpy as np
from numba import jit
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model

from .knn import compute_node_degree
from .utils import center_values
from .gpu import get_backend, asnumpy, _build_sparse_weight_matrix


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
def compute_local_cov_max(node_degrees, vals):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2

# Genes per block are chosen so each (genes x cells) dense array holds ~this many doubles
_BLOCK_ELEMS = 2**22


def compute_hs(
    counts, neighbors, weights, num_umi, model, genes, centered=False, jobs=1,
    use_gpu=False
):
    """
    Batched over genes via sparse matrix multiplication on CPU (numpy/scipy)
    or GPU (cupy, one block). CPU gene blocks run on `jobs` threads, BLAS
    limited to 1 thread each.
    """
    xp = get_backend(use_gpu)

    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    N_genes, N_cells = counts.shape

    D = xp.asarray(compute_node_degree(neighbors, weights))
    Wtot2 = (weights**2).sum()
    W = _build_sparse_weight_matrix(
        neighbors, weights, shape=(N_cells, N_cells), use_gpu=use_gpu
    )
    W_sq = None if centered else _build_sparse_weight_matrix(
        neighbors, weights, shape=(N_cells, N_cells), square=True, use_gpu=use_gpu
    )

    # GPU: all genes in one block, as before; CPU: small blocks spread over threads
    block = N_genes if use_gpu else max(1, _BLOCK_ELEMS // N_cells)

    def _block(i):
        return _compute_hs_block(
            counts[i:i + block], W, W_sq, D, Wtot2, num_umi, model, centered, xp
        )

    starts = range(0, N_genes, block)
    with threadpool_limits(1), ThreadPoolExecutor(jobs) as pool:
        results = list(tqdm(pool.map(_block, starts), total=len(starts)))

    results = pd.DataFrame(
        np.vstack(results), index=genes, columns=["G", "EG", "stdG", "Z", "C"]
    )

    return _postprocess_results(results)


def _postprocess_results(results):
    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]
    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"
    results = results[["C", "Z", "Pval", "FDR"]]
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


def _local_cov_weights(vals, W):
    """Batch of local_cov_weights: G[g] = vals[g] . (W @ vals[g]) for all genes."""
    smoothed_T = W @ vals.T
    return (vals * smoothed_T.T).sum(axis=1)


def _compute_moments_weights(mu, x2, W, W_sq):
    """Batch of compute_moments_weights for all genes at once."""
    # EG[g] = mu[g] . (W @ mu[g])
    EG = (mu * (W @ mu.T).T).sum(axis=1)

    # t1[g] = (W + W.T) @ mu[g],  t2[g] = (W_sq + W_sq.T) @ mu[g]^2
    W_sym = W + W.T
    W_sq_sym = W_sq + W_sq.T
    mu2 = mu ** 2

    t1_T = W_sym @ mu.T
    t2_T = W_sq_sym @ mu2.T

    # Contribution 1: sum_i (x2[i] - mu[i]^2) * (t1[i]^2 - t2[i])
    diff_var = (x2 - mu2).T
    eg2_c1 = (diff_var * (t1_T ** 2 - t2_T)).sum(axis=0)

    # Contribution 2: sum_{edges} w^2 * (x2[i]*x2[j] - mu[i]^2*mu[j]^2)
    eg2_c2 = (x2.T * (W_sq @ x2.T)).sum(axis=0)
    eg2_c2 -= (mu2.T * (W_sq @ mu2.T)).sum(axis=0)

    EG2 = eg2_c1 + eg2_c2 + EG ** 2
    return EG, EG2


def _compute_local_cov_max(D, vals):
    """Batch of compute_local_cov_max: G_max[g] = sum_i D[i]*vals[g,i]^2 / 2."""
    return (D * vals ** 2).sum(axis=1) / 2


def _compute_hs_block(counts, W, W_sq, D, Wtot2, num_umi, model, centered, xp):
    """Statistics for one block of genes, returned as a (genes x 5) numpy array."""
    if issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype="double")

    N_genes, N_cells = counts.shape
    all_vals = np.zeros((N_genes, N_cells), dtype="double")
    if not centered:
        all_mu = np.zeros((N_genes, N_cells), dtype="double")
        all_x2 = np.zeros((N_genes, N_cells), dtype="double")

    for i in range(N_genes):
        vals, mu, var, x2 = _fit_gene(counts[i], model, num_umi)
        if centered:
            vals = center_values(vals, mu, var)
        else:
            all_mu[i] = mu
            all_x2[i] = x2
        all_vals[i] = vals

    vals = xp.asarray(all_vals)

    G_stats = _local_cov_weights(vals, W)

    if centered:
        EG = xp.zeros(N_genes, dtype="double")
        EG2 = xp.full(N_genes, Wtot2, dtype="double")
    else:
        EG, EG2 = _compute_moments_weights(xp.asarray(all_mu), xp.asarray(all_x2), W, W_sq)

    stdG = (EG2 - EG * EG) ** 0.5
    Z = (G_stats - EG) / stdG

    G_max = _compute_local_cov_max(D, vals)
    C = (G_stats - EG) / G_max

    return asnumpy(xp.stack([G_stats, EG, stdG, Z, C], axis=1))
