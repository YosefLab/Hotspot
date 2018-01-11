library(RANN)
library(matrixStats)

hotspot <- function(pca, expression, n_neighbors, neighborhood_factor){


    out <- nn2(data = pca, k = n_neighbors+1, treetype = "bd",
               searchtype = "standard")

    idx <- out$nn.idx
    dists <- out$nn.dists

    # Exclude self

    idx <- idx[, -1]
    dists <- dists[, -1]

    # Compute weights
    radius_ii <- ceiling(ncol(dists) / neighborhood_factor)
    sigma <- dists[, radius_ii]

    weights <- exp(-1 * dists ** 2 / sigma ** 2)
    wnorm <- rowSums(weights)
    wnorm[wnorm == 0] <- 1.0

    weights <- weights / wnorm

    # Load into a sparse matrix
    row_idxs <- as.vector(t(row(idx)))
    col_idxs <- as.vector(t(idx))
    values <- as.vector(t(weights))

    sparse_weights <- sparseMatrix(i = row_idxs, j = col_idxs, x = values)

    G_i <- tcrossprod(expression, sparse_weights)

    # Compute normalization factors
    W_i <- rowSums(weights)         # Length: N_Cells
    S_1i <- rowSums(weights ** 2)   # Length: N_Cells


    xbar <- rowMeans(expression)    # Length: N_Genes
    s <- rowSds(expression)         # Length: N_Genes
    n <- nrow(expression)

    offset <- matrix(xbar) %*% t(matrix(W_i))

    denom <- (n*S_1i - W_i**2) / (n-1)**(1/2)
    denom <- matrix(rep(denom, times = N_GENES), nrow = N_GENES)
    denom <- s * denom

    G_i <- (G_i - offset) / denom

    return(G_i)
}


n_neighbors <- 10
neighborhood_factor <- 3

# Generate some random test data
N_CELLS <- 20000
N_COMPONENTS <- 40
N_GENES <- 10000

pca <- matrix(runif(N_CELLS * N_COMPONENTS), nrow = N_CELLS)
expression <- matrix(runif(N_CELLS * N_GENES), nrow = N_GENES)

system.time(out <- hotspot(pca, expression, n_neighbors, neighborhood_factor))
