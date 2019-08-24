import numpy as np
from numba import jit
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import multiprocessing

from . import danb_model
from . import bernoulli_model
from .knn import compute_node_degree


@jit(nopython=True)
def local_cov_weights(x, neighbors, weights):
    out = 0

    for i in range(len(x)):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xi = x[i]
            xj = x[j]

            out += xi*xj * w_ij

    return out


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

            EG += wij*mu[i]*mu[j]

    # Calculate E[G^2]
    EG2 = 0
    EG2 += (EG**2)

    #   Get the x^2*y*z terms
    t1 = np.zeros(N)
    t2 = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1[i] += wij*mu[j]
            t2[i] += wij**2*mu[j]**2

            t1[j] += wij*mu[i]
            t2[j] += wij**2*mu[i]**2

    t1 = t1**2

    for i in range(N):
        EG2 += (x2[i] - mu[i]**2)*(t1[i] - t2[i])

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2*(x2[i]*x2[j] - (mu[i]**2)*(mu[j]**2))

    return EG, EG2


@jit(nopython=True)
def compute_local_cov_max(node_degrees, vals):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i]*(vals[i]**2)

    return tot/2


@jit(nopython=True)
def center_values(vals, mu, var):
    out = np.zeros_like(vals)

    for i in range(len(vals)):
        std = var[i]**0.5
        if std == 0:
            out[i] = 0
        else:
            out[i] = (vals[i] - mu[i])/std

    return out


def compute_hs(counts, neighbors, weights, num_umi, model, centered=False, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    def data_iter():
        for i in range(counts.shape[0]):
            vals = counts[i].astype('double')
            yield vals

    def initializer():
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

    if jobs > 1:

        with multiprocessing.Pool(
                processes=jobs, initializer=initializer) as pool:

            results = list(
                tqdm(
                    pool.imap(_map_fun_parallel, data_iter()),
                    total=counts.shape[0]
                )
            )
    else:
        def _map_fun(vals):
            return _compute_hs_inner(
                vals, neighbors, weights, num_umi,
                model, centered, Wtot2, D
            )
        results = list(
            tqdm(
                map(_map_fun, data_iter()),
                total=counts.shape[0]
            )
        )

    results = pd.DataFrame(results,
                           index=genes,
                           columns=['G', 'EG', 'stdG', 'Z', 'C']
                           )

    results['Pval'] = norm.sf(results['Z'].values)
    results['FDR'] = multipletests(results['Pval'], method='fdr_bh')[1]

    return results


def _compute_hs_inner(vals, neighbors, weights, num_umi,
                      model, centered, Wtot2, D):
    """
    Note, since this is an inner function, for parallelization to work well
    none of the contents of the function can use MKL or OPENBLAS threads.
    Or else we open too many.  Because of this, some simple numpy operations
    are re-implemented using numba instead as it's difficult to control
    the number of threads in numpy after it's imported
    """

    if model == 'bernoulli':
        vals = (vals > 0).astype('double')
        mu, var, x2 = bernoulli_model.fit_gene_model(
            vals, num_umi)

    elif model == 'danb':
        mu, var, x2 = danb_model.fit_gene_model(
            vals, num_umi)
    else:
        raise Exception("Invalid Model: {}".format(model))

    if centered:
        vals = center_values(vals, mu, var)

    G = local_cov_weights(vals, neighbors, weights)

    if centered:
        EG, EG2 = 0, Wtot2
    else:
        EG, EG2 = compute_moments_weights(mu, x2, neighbors, weights)

    stdG = (EG2-EG*EG)**.5

    Z = (G-EG)/stdG

    G_max = compute_local_cov_max(D, vals)
    C = (G - EG)/G_max

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
        vals, g_neighbors, g_weights, g_num_umi,
        g_model, g_centered, g_Wtot2, g_D
    )


@jit(nopython=True)
def local_cov_pair(x, y, neighbors, weights):
    out = 0

    for i in range(len(x)):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xi = x[i]
            xj = x[j]

            yi = y[i]
            yj = y[j]

            out += w_ij*(xi*yj + yi*xj)

    return out


@jit(nopython=True)
def compute_moments_weights_pairs(
        muX, x2,
        muY, y2,
        neighbors, weights):

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij*(muX[i]*muY[j] + muY[i]*muX[j])

    # Calculate E[G^2]
    EG2 = 0
    EG2 += (EG**2)

    #   Get the x^2*y*z terms
    t1x = np.zeros(N)
    t2x = np.zeros(N)

    t1y = np.zeros(N)
    t2y = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1x[i] += wij*muX[j]
            t2x[i] += wij**2*muX[j]**2

            t1x[j] += wij*muX[i]
            t2x[j] += wij**2*muX[i]**2

            t1y[i] += wij*muY[j]
            t2y[i] += wij**2*muY[j]**2

            t1y[j] += wij*muY[i]
            t2y[j] += wij**2*muY[i]**2

    t1x = t1x**2
    t1y = t1y**2

    for i in range(N):
        EG2 += (y2[i] - muY[i]**2)*(t1x[i] - t2x[i])
        EG2 += (x2[i] - muX[i]**2)*(t1y[i] - t2y[i])

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2*(
                x2[i]*y2[j] - (muX[i]**2)*(muY[j]**2) +
                y2[i]*x2[j] - (muY[i]**2)*(muX[j]**2)
            )

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs_fast(
        muX, x2,
        muY, y2,
        neighbors, weights):
    """
    About 2x as fast as the above, but I haven't
    checked this for correctness yet...
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    #   Get the x^2*y*z terms
    t1x = np.zeros(N)
    t2x = np.zeros(N)

    t1y = np.zeros(N)
    t2y = np.zeros(N)

    EG = 0
    EG2 = 0

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]
            wij_2 = wij**2

            if wij == 0:
                continue

            muX_i2 = muX[i]**2
            muX_j2 = muX[j]**2
            muY_i2 = muY[i]**2
            muY_j2 = muY[j]**2

            EG += wij*(muX[i]*muY[j] + muY[i]*muX[j])

            t1x[i] += wij*muX[j]
            t2x[i] += wij_2*muX_j2

            t1x[j] += wij*muX[i]
            t2x[j] += wij_2*muX_i2

            t1y[i] += wij*muY[j]
            t2y[i] += wij_2*muY_j2

            t1y[j] += wij*muY[i]
            t2y[j] += wij_2*muY_i2

            EG2 += wij_2*(
                x2[i]*y2[j] - (muX_i2)*(muY_j2) +
                y2[i]*x2[j] - (muY_i2)*(muX_j2)
            )

    # Calculate E[G^2]
    t1x = t1x**2
    t1y = t1y**2

    for i in range(N):
        EG2 += (y2[i] - muY[i]**2)*(t1x[i] - t2x[i])
        EG2 += (x2[i] - muX[i]**2)*(t1y[i] - t2y[i])

    EG2 += (EG**2)

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs_std(neighbors, weights):
    """
    This version assumes variables are standardized,
    and so the moments are actually the same for all
    pairs of variables
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0

    # Calculate E[G^2]
    EG2 = 0

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            wij = weights[i, k]

            EG2 += wij**2*(2)

    return EG, EG2


def create_centered_counts(counts, model, num_umi):
    out = np.zeros_like(counts, dtype='double')

    for i in tqdm(range(out.shape[0])):

        vals_x = counts[i]

        if model == 'bernoulli':
            vals_x = (vals_x > 0).astype('double')
            mu_x, var_x, x2_x = bernoulli_model.fit_gene_model(
                vals_x, num_umi)

        elif model == 'danb':
            mu_x, var_x, x2_x = danb_model.fit_gene_model(
                vals_x, num_umi)
        else:
            raise Exception("Invalid Model: {}".format(model))

        var_x[var_x == 0] = 1
        vals_x = (vals_x-mu_x)/(var_x**0.5)
        vals_x[var_x == 0] = 0
        out[i] = vals_x

    return out


def _compute_hs_pairs_inner(row_i, counts, neighbors, weights, num_umi,
                            model, centered, Wtot2, D):

    vals_x = counts[row_i]

    lc_out = np.zeros(counts.shape[0])
    lc_z_out = np.zeros(counts.shape[0])

    if model == 'bernoulli':
        vals_x = (vals_x > 0).astype('double')
        mu_x, var_x, x2_x = bernoulli_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'danb':
        mu_x, var_x, x2_x = danb_model.fit_gene_model(
            vals_x, num_umi)
    else:
        raise Exception("Invalid Model: {}".format(model))

    if centered:
        vals_x = center_values(vals_x, mu_x, var_x)

    for row_j in range(counts.shape[0]):

        if row_j > row_i:
            continue

        vals_y = counts[row_j]

        if model == 'bernoulli':
            vals_y = (vals_y > 0).astype('double')
            mu_y, var_y, x2_y = bernoulli_model.fit_gene_model(
                vals_y, num_umi)

        elif model == 'danb':
            mu_y, var_y, x2_y = danb_model.fit_gene_model(
                vals_y, num_umi)
        else:
            raise Exception("Invalid Model: {}".format(model))

        if centered:
            vals_y = center_values(vals_y, mu_y, var_y)

        if centered:
            EG, EG2 = 0, 2*Wtot2
        else:
            EG, EG2 = compute_moments_weights_pairs_fast(mu_x, x2_x,
                                                         mu_y, x2_y,
                                                         neighbors, weights)

        lc = local_cov_pair(vals_x, vals_y,
                            neighbors, weights)

        stdG = (EG2 - EG**2)**.5

        Z = (lc - EG) / stdG

        lc_out[row_j] = lc
        lc_z_out[row_j] = Z

    return (lc_out, lc_z_out)


def _compute_hs_pairs_inner_centered(
        row_i, counts, neighbors, weights, Wtot2, D):
    """
    This version assumes that the counts have already been modeled
    and centered
    """

    vals_x = counts[row_i]

    lc_out = np.zeros(counts.shape[0])
    lc_z_out = np.zeros(counts.shape[0])

    for row_j in range(counts.shape[0]):

        if row_j > row_i:
            continue

        vals_y = counts[row_j]

        EG, EG2 = 0, 2*Wtot2

        lc = local_cov_pair(vals_x, vals_y,
                            neighbors, weights)

        stdG = (EG2 - EG**2)**.5

        Z = (lc - EG) / stdG

        lc_out[row_j] = lc
        lc_z_out[row_j] = Z

    return (lc_out, lc_z_out)


def compute_hs_pairs(counts, neighbors, weights,
                     num_umi, model, centered=False, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    def initializer():
        global g_neighbors
        global g_weights
        global g_num_umi
        global g_model
        global g_centered
        global g_Wtot2
        global g_D
        global g_counts
        g_counts = counts
        g_neighbors = neighbors
        g_weights = weights
        g_num_umi = num_umi
        g_model = model
        g_centered = centered
        g_Wtot2 = Wtot2
        g_D = D

    if jobs > 1:

        with multiprocessing.Pool(
                processes=jobs, initializer=initializer) as pool:

            results = list(
                tqdm(
                    pool.imap(_map_fun_parallel_pairs, range(counts.shape[0])),
                    total=counts.shape[0]
                )
            )
    else:
        def _map_fun(row_i):
            return _compute_hs_pairs_inner(
                row_i, counts, neighbors, weights, num_umi,
                model, centered, Wtot2, D)
        results = list(
            tqdm(
                map(_map_fun, range(counts.shape[0])),
                total=counts.shape[0]
            )
        )

    # Only have the lower triangle so we must rebuild the rest
    lcs = [x[0] for x in results]
    lc_zs = [x[1] for x in results]

    lcs = np.vstack(lcs)
    lc_zs = np.vstack(lc_zs)

    lcs = lcs + lcs.T
    lc_zs = lc_zs + lc_zs.T

    np.fill_diagonal(lcs, lcs.diagonal() / 2)
    np.fill_diagonal(lc_zs, lc_zs.diagonal() / 2)

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def compute_hs_pairs_centered(counts, neighbors, weights,
                              num_umi, model, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    counts = create_centered_counts(counts, model, num_umi)

    def initializer():
        global g_neighbors
        global g_weights
        global g_Wtot2
        global g_D
        global g_counts
        g_counts = counts
        g_neighbors = neighbors
        g_weights = weights
        g_Wtot2 = Wtot2
        g_D = D

    if jobs > 1:

        with multiprocessing.Pool(
                processes=jobs, initializer=initializer) as pool:

            results = list(
                tqdm(
                    pool.imap(_map_fun_parallel_pairs_centered,
                              range(counts.shape[0])), total=counts.shape[0]
                )
            )
    else:
        def _map_fun(row_i):
            return _compute_hs_pairs_inner_centered(
                row_i, counts, neighbors, weights, Wtot2, D)
        results = list(
            tqdm(
                map(_map_fun, range(counts.shape[0])),
                total=counts.shape[0]
            )
        )

    # Only have the lower triangle so we must rebuild the rest
    lcs = [x[0] for x in results]
    lc_zs = [x[1] for x in results]

    lcs = np.vstack(lcs)
    lc_zs = np.vstack(lc_zs)

    lcs = lcs + lcs.T
    lc_zs = lc_zs + lc_zs.T

    np.fill_diagonal(lcs, lcs.diagonal() / 2)
    np.fill_diagonal(lc_zs, lc_zs.diagonal() / 2)

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def _map_fun_parallel_pairs(row_i):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    global g_counts
    return _compute_hs_pairs_inner(
        row_i, g_counts, g_neighbors, g_weights, g_num_umi,
        g_model, g_centered, g_Wtot2, g_D)


def _map_fun_parallel_pairs_centered(row_i):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_Wtot2
    global g_D
    global g_counts
    return _compute_hs_pairs_inner_centered(
        row_i, g_counts, g_neighbors, g_weights, g_Wtot2, g_D)
