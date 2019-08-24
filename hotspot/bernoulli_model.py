from numba import jit


@jit(nopython=True)
def find_gene_p(num_umi, D):
    """
    Finds gene_p such that sum of expected detects
    matches our data
    """

    low = 1e-12
    high = 1

    if D == 0:
        return 0

    for ITER in range(40):

        attempt = (high*low)**0.5
        tot = 0

        for i in range(len(num_umi)):
            tot = tot + 1-(1-attempt)**num_umi[i]

        if abs(tot-D)/D < 1e-3:
            break

        if tot > D:
            high = attempt
        else:
            low = attempt

    return (high*low)**0.5


def fit_gene_model(gene_detects, umi_counts):

    D = gene_detects.sum()

    gene_p = find_gene_p(umi_counts, D)

    detect_p = 1-(1-gene_p)**umi_counts

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2
