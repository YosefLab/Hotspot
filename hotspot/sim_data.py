import numpy as np
from numba import jit
from tqdm import tqdm


def sim_latent(N_CELLS, N_DIM):
    """
    Draws a latent space coordinate for each cell
    randomly from a unit gaussian
    """

    return np.random.normal(size=(N_CELLS, N_DIM))


def sim_umi_counts(N_CELLS, mid, scale):
    """
    Simulates counts from a log-normal distribution
    """

    mu = np.log(mid)
    sd = np.log(mid+scale)-mu

    vals = np.random.normal(loc=mu, scale=sd, size=N_CELLS)
    vals = np.floor(np.exp(vals))

    return vals


def sim_counts_bernoulli(N_CELLS, umi_counts, gene_p):
    """
    For a given transcript probability, simulates detection
    in N_CELLS with `umi_counts` umis per cell

    Note: gene_p is in transcripts / 10,000
    """
    gene_p = gene_p / 10000

    detect_p = 1-(1-gene_p)**umi_counts

    vals = (np.random.rand(N_CELLS) < detect_p).astype('double')

    return vals


def sim_counts_danb(N_CELLS, N_GENES, umi_counts, mean, size):
    """
    Simulates transcript counts under the DANB model
    Uses umi_counts to adjust the mean

    Returns N_GENES replicates -> result is N_GENES x N_CELLS
    """
    umi_scale = umi_counts / umi_counts.mean()

    vals = np.zeros((N_GENES, N_CELLS))

    for i in tqdm(range(N_CELLS)):
        mean_i = mean*umi_scale[i]
        var = mean_i*(1+mean_i/size)
        p = 1 - mean_i/var
        n = mean_i * (1-p) / p
        p = 1-p

        vals[:, i] = np.random.negative_binomial(n=n, p=p, size=N_GENES)

    return vals


def generate_permutation_null(observed_counts, N_REPS):

    out = []
    for i in range(N_REPS):
        out.append(np.random.permutation(observed_counts))

    return np.vstack(out)



np.random.negative_binomial(n=n, p=p, size=1000).mean()
np.random.negative_binomial(n=n, p=p, size=1000).var()


N_CELLS = 10000
N_REP = 1000
umi_counts = sim_umi_counts(N_CELLS, 5000, 1000)
detects = [
    sim_counts_bernoulli(N_CELLS, umi_counts, gene_p=1) for _ in range(N_REP)
]
latent = sim_latent(N_CELLS, 10)

umi_counts = pd.Series(umi_counts)
detects = pd.DataFrame(detects)
latent=pd.DataFrame(latent)

hs = hotspot.Hotspot(detects, latent, umi_counts)
hs.create_knn_graph()
res = hs.compute_hotspot(model='bernoulli', jobs=10, centered=True)


# Now for DANB

N_CELLS = 10000
N_REP = 1000
umi_counts = sim_umi_counts(N_CELLS, 5000, 1000)
counts = sim_counts_danb(N_CELLS, N_REP, np.ones(N_CELLS)*10000, mean=10, size=2)
latent = sim_latent(N_CELLS, 10)

umi_counts = pd.Series(umi_counts)
counts = pd.DataFrame(counts)
latent=pd.DataFrame(latent)

hs = hotspot.Hotspot(counts, latent, umi_counts)
hs.create_knn_graph()
res = hs.compute_hotspot(model='danb', jobs=1, centered=True)

from scipy.stats import nbinom

x = counts.values[5]

mu = x.mean()
var = x.var()
p = 1 - mu/var
n = mu * (1-p) / p
p = 1-p

tot = 0
for i in range(100):
    p_i = nbinom.pmf(i, n, p)
    mu_y = (mu*N_CELLS-i)/(N_CELLS-1)
    tot += i*p_i*mu_y

tot = 0
for i in range(100):
    p_i = nbinom.pmf(i, n, p)
    mu_y = (mu*N_CELLS-i)/(N_CELLS-1)
    var_y = ((var*N_CELLS) - i**2 - mu**2 + mu_y**2 +2*i*mu)/(N_CELLS-1)
    tot += i**2*p_i*(mu_y**2+var_y)


def log_lik(x, n, p):
    dist = nbinom(n, p)
    tot = dist.logpmf(x).sum()

    return tot

t1 = log_lik(x, n, p)

mu = 10
size = 2
var = mu*(1+mu/size)
p = 1 - mu/var
n = mu * (1-p) / p
p = 1-p

t2 = log_lik(x, n, p)

# Sample estimates
plt.figure()
mu_s = counts.mean(axis=1)
var_s = counts.var(axis=1)
size_s = mu_s/(1-var_s/mu_s)
plt.plot(mu_s, size_s, 'o', ms=2)
plt.show()

plt.scatter(mu_s, size_s, s=3, c=res.G)
plt.show()

x_std = (x - x.mean()) / x.std()

@jit(nopython=True)
def order2(x, ITER):
    tot = 0
    
    for i in range(ITER):
        nodes = np.random.choice(x.size, size=2, replace=False)
        tot += x[nodes[0]]*x[nodes[1]]

    return tot / ITER


@jit(nopython=True)
def order22(x, ITER):
    tot = 0
    
    for i in range(ITER):
        nodes = np.random.choice(x.size, size=2, replace=False)
        tot += x[nodes[0]]**2*x[nodes[1]]**2

    return tot / ITER


@jit(nopython=True)
def order3(x, ITER):
    tot = 0
    
    for i in range(ITER):
        nodes = np.random.choice(x.size, size=3, replace=False)
        tot += x[nodes[0]]**2*x[nodes[1]]*x[nodes[2]]

    return tot / ITER


@jit(nopython=True)
def order4(x, ITER):
    tot = 0
    
    for i in range(ITER):
        nodes = np.random.choice(x.size, size=4, replace=False)
        tot += x[nodes[0]]*x[nodes[1]]*x[nodes[2]]*x[nodes[3]]

    return tot / ITER


Wtot = hs.weights.values.sum()
Wtot2 = (hs.weights.values**2).sum()

order2(x_std, 1000000)
order22(x_std, 1000000) * Wtot2
order3(x_std, 1000000) * N_CELLS*30*30
order4(x_std, 100000000)


# What is the issue?? Is it correlation?
# Instead, test pairs (x, y) so that they are never correlated
# Well, kind of, terms like X^2Y1Y2 would still be possibly
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

N_CELLS = 10000
N_REP = 10000
#umi_counts = sim_umi_counts(N_CELLS, 5000, 1000)
umi_counts = np.ones(N_CELLS)*10000
counts_x = sim_counts_danb(N_CELLS, N_REP, umi_counts, mean=10, size=2)
counts_y = sim_counts_danb(N_CELLS, N_REP, umi_counts, mean=10, size=2)
latent = sim_latent(N_CELLS, 10)

umi_counts = pd.Series(umi_counts)
counts_x = pd.DataFrame(counts_x)
counts_y = pd.DataFrame(counts_y)
latent=pd.DataFrame(latent)


G = []
for i in tqdm(range(counts_x.shape[0])):
    cx = counts_x.values[i]
    cy = counts_y.values[i]
    cx = (cx - cx.mean())/cx.std()
    cy = (cy - cy.mean())/cy.std()
    G.append(
        local_cov_pair(cx, cy, hs.neighbors.values, hs.weights.values)
    )
G = pd.Series(G)

EG, EG2 = compute_moments_weights_pairs_std(
    hs.neighbors.values, hs.weights.values)

EG2**.5

from hotspot import local_stats


G = []
for i in tqdm(range(counts_x.shape[0])):
    cx = counts_x.values[i]
    cx = (cx - cx.mean())/cx.std()
    G.append(
        local_stats.local_cov_weights(cx, hs.neighbors.values, hs.weights.values)
    )
G = pd.Series(G)

Wtot2 = (hs.weights.values**2).sum()

hs = hotspot.Hotspot(counts_x, latent, umi_counts)
hs.create_knn_graph()
res_c = hs.compute_hotspot(model='danb', jobs=10, centered=True)
res = hs.compute_hotspot(model='danb', jobs=10, centered=False)

# Now it looks fine....
N_CELLS = 10000
N_REP = 10000
umi_counts = sim_umi_counts(N_CELLS, 5000, 1000)
counts_x = sim_counts_danb(N_CELLS, N_REP, umi_counts, mean=10, size=2)
umi_counts_flat = np.ones(N_CELLS)*10000
latent = sim_latent(N_CELLS, 10)

umi_counts = pd.Series(umi_counts)
umi_counts_falt = pd.Series(umi_counts_flat)
counts_x = pd.DataFrame(counts_x)
latent=pd.DataFrame(latent)

hs = hotspot.Hotspot(counts_x, latent, umi_counts_flat)
hs.create_knn_graph()
res_c = hs.compute_hotspot(model='danb', jobs=10, centered=True)
res = hs.compute_hotspot(model='danb', jobs=10, centered=False)

from hotspot import danb_model
mu, var, x2 = danb_model.fit_gene_model(counts_x.values[0], umi_counts)
mu, var, x2 = danb_model.fit_gene_model(counts_x.values[0], umi_counts_flat)


def log_lik(x, n, p):

    tot = 0

    for x_i, n_i, p_i in tqdm(zip(x, n, p)):
        tot += nbinom.logpmf(x_i, n=n_i, p=p_i)

    return tot


def log_lik2(x, mu, size):
    var = mu*(1+mu/size)

    return log_lik3(x, mu, var)


def log_lik3(x, mu, var):
    p = 1 - mu/var
    n = mu * (1-p) / p
    p = 1-p
    return log_lik(x, n, p)

mu, var, x2 = danb_model.fit_gene_model(counts_x.values[0], umi_counts)
a = log_lik3(counts_x.values[0], mu, var)

mu, var, x2 = danb_model.fit_gene_model(counts_x.values[0], umi_counts_flat)
b = log_lik3(counts_x.values[0], mu, var)

print(a-b)  # a is bigger, good!

hs1 = hotspot.Hotspot(counts_x, latent, umi_counts_flat)
hs1.create_knn_graph()
res_c1 = hs1.compute_hotspot(model='danb', jobs=10, centered=True)

hs2 = hotspot.Hotspot(counts_x, latent, umi_counts)
hs2.create_knn_graph()
res_c2 = hs2.compute_hotspot(model='danb', jobs=10, centered=True)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(res_c1.Z, res_c2.Z, 'o', ms=2)
plt.show()


# What about permutations?

N_CELLS = 100
N_REP = 10000
#umi_counts = sim_umi_counts(N_CELLS, 5000, 1000)
umi_counts = np.ones(N_CELLS)*10000
counts_x = sim_counts_danb(N_CELLS, 1, umi_counts, mean=10, size=2)
latent = sim_latent(N_CELLS, 10)
counts = generate_permutation_null(counts_x.ravel(), N_REP)

umi_counts = pd.Series(umi_counts)
counts = pd.DataFrame(counts)
latent=pd.DataFrame(latent)

hs = hotspot.Hotspot(counts, latent, umi_counts)
hs.create_knn_graph()
hs.weights[hs.weights > 1] = 1.0
res_c = hs.compute_hotspot(model='danb', jobs=10, centered=True)
res = hs.compute_hotspot(model='danb', jobs=10, centered=False)


from hotspot import danb_model
from hotspot.local_stats import local_cov_weights, compute_moments_weights
vals = counts.values[10]

mu, var, x2 = danb_model.fit_gene_model(
    vals, umi_counts.values)

G = local_cov_weights(vals, hs.neighbors.values, hs.weights.values)

EG, EG2 = compute_moments_weights(mu, x2, hs.neighbors.values, hs.weights.values)

stdG = (EG2-EG*EG)**.5

Z = (G-EG)/stdG


def enumerate_edges(neighbors, weights):

    edges = []
    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            wij = weights[i, k]

            if wij != 0:
                edges.append((i, j, wij))

    return edges


def compute_eg_slow(mu, x2, edges):

    tot = 0
    for i, j, wij in edges:

        tot += wij * mu[i] * mu[j]

    return tot


def compute_eg2_slow(mu, x2, edges):

    tot = 0
    fours = 0
    twos = 0
    threes = 0
    for i, j, wij in edges:
        for k, l, wkl in edges:

            un = len(set([i, j, k, l]))

            if un == 4:
                tot += mu[i]*mu[j]*mu[k]*mu[l]*wij*wkl
                fours += 1
                continue

            if un == 2:
                tot += x2[i]*x2[j]*wij*wkl
                twos += 1
                continue

            # un must equal 3, find the repeated term
            if un == 3:
                threes += 1
                if i == k:
                    tot += x2[i]*mu[j]*mu[l]*wij*wkl
                    continue
                if i == l:
                    tot += x2[i]*mu[j]*mu[k]*wij*wkl
                    continue
                if j == k:
                    tot += x2[j]*mu[i]*mu[l]*wij*wkl
                    continue
                if j == l:
                    tot += x2[j]*mu[i]*mu[k]*wij*wkl
                    continue

                raise Exception("Something Wrong")

            raise Exception("Bad UN value")

    return tot, twos, threes, fours


edges = enumerate_edges(hs.neighbors.values, hs.weights.values)

EG_s = compute_eg_slow(mu, x2, edges)
EG2_s, twos, threes, fours = compute_eg2_slow(mu, x2, edges)
stdG_s = (EG2_s-EG_s**2)**.5

four_val = mu[0]**4
three_val = x2[0]*mu[0]**2
two_val = x2[0]**2

three_ex = three_val - four_val
two_ex = two_val - four_val

# what it should be
EG2_emp = (res.G**2).mean()

# Difference
EG2_emp - four_val*len(edges)**2

# Relative contributions
four_val * fours / EG2_s  # 93%
three_val * threes / EG2_s  # 6.6%
two_val * twos / EG2_s  # .12%

# error
(EG2_s - EG2_emp)/EG2_s  # 3.4%

# This means that either:
#     - There's a small error (~4%) on the four-val terms
#     - There's  a large error (50%) on the three-val terms
#     - There's a HUGE error (2500%) on the two-val terms

# OR, there's an error in estimating the mean??
# Even a 1% error in the mean estimate results in a 4% error in the four-val terms
# Would also result in an error in the EG_s term

mu_e = np.ones_like(mu)*10
var_e = mu_e * (1+mu_e/2)
x2_e = mu_e**2 + var_e
EG_e = compute_eg_slow(mu_e, x2_e, edges)
EG2_e, twos, threes, fours = compute_eg2_slow(mu_e, x2_e, edges)
stdG_e = (EG2_e-EG_e**2)**.5



N_CELLS = 10000
N_REP = 1000
umi_counts = np.ones(N_CELLS)*10000
counts = sim_counts_danb(N_CELLS, N_REP, np.ones(N_CELLS)*10000, mean=10, size=2)
latent = sim_latent(N_CELLS, 10)

umi_counts = pd.Series(umi_counts)
counts = pd.DataFrame(counts)
latent=pd.DataFrame(latent)

hs = hotspot.Hotspot(counts, latent, umi_counts)
hs.create_knn_graph()
res = hs.compute_hotspot(model='danb', jobs=1, centered=False)

mu_e = np.ones(N_CELLS)*10
var_e = mu_e * (1+mu_e/2)
x2_e = mu_e**2 + var_e

EG, EG2 = compute_moments_weights(mu_e, x2_e, hs.neighbors.values, hs.weights.values)
stdG = (EG2-EG**2)**.5
print(stdG, res.G.std())

mu_e = counts.mean(axis=1)
var_e = counts.var(axis=1)
size_e = (var_e/mu_e - 1)**-1*mu_e

import matplotlib.pyplot as plt
plt.figure()
plt.plot(mu_e, size_e, 'o', ms=2)
plt.show()


mu = 10
mu_sigma = .077  # just calculated empirically

four_val = mu**4
four_val_corrected = mu**4+6*mu**2*mu_sigma**2+mu_sigma**4

(four_val_corrected - four_val) / four_val * 100

# Here the error is only .03 %

EG_e = res.G.mean()
EG2_e = (res.G**2).mean()

stdG_e = (res.G - res.EG).std()

# I think I'm no closer to figuring this out.

