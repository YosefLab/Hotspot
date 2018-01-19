#library(RANN)
#library(matrixStats)
#library(Matrix)

#' Compute weights for neighbors
#'
#' Given a matrix of distances, computes appropriate 'weights'
#' which decay according to a Guassian Kernel with width equal
#' to the distance to the N_NEIGHBORS / <neighborhood_factor> neighbor
#' @import Matrix
#' @import matrixStats
#' @param distances distances to neighbors.  matrix of dimension
#' N_CELLS x N_NEIGHBORS
#' @param neighborhood_factor What proportion of neighborhood to use to
#' define the gaussian kernel width. Default is 3
#' @export
#' @return weights weights for each neighbor.  Same size/type as distances
#' input
computeWeights <- function(distances, neighborhood_factor=3){

    # Compute weights
    radius_ii <- ceiling(ncol(distances) / neighborhood_factor)
    sigma <- distances[, radius_ii]

    weights <- exp(-1 * distances ** 2 / sigma ** 2)
    wnorm <- rowSums(weights)
    wnorm[wnorm == 0] <- 1.0

    weights <- weights / wnorm

    return(weights)
}

#' Find Nearest Neighbors and Asociated Weights
#'
#' Using the input data matrix, for each cell (row) compute the
#' nearest n_neighbors and compute a weight for each neighbor.
#'
#' @importFrom RANN nn2
#' @param data Matrix to use to derive distances. It's recommended to u
#' se a PCA-reduced gene expression matrix here.  matrix of dimension 
#' N_CELLS x N_COMPONENTS
#' @param n_neighbors How many neighbors to use for each cell
#' @param neighborhood_factor What proportion of neighborhood to use to
#' define the gaussian kernel width. Default is 3
#' @export
#' @return out$neighbors n_nearest neighbors for each cell.  Matrix of
#' size N_CELLS x N_Neighbors.  Entries represent index of neighbors
#' @return out$weights weights for each neighbor.  Same size/type as out$neighbors
#' input
neighborsAndWeights <- function(data, n_neighbors=30, neighborhood_factor=3){

    nbrs <- nn2(data = data, k = n_neighbors+1, treetype = "bd",
               searchtype = "standard")

    idx <- nbrs$nn.idx
    dists <- nbrs$nn.dists

    # Exclude self
    idx <- idx[, -1]
    dists <- dists[, -1]

    weights <- computeWeights(dists, neighborhood_factor)

    out <- list(neighbors=idx, weights=weights)

    return(out)

}

#' Run Hotspot Analysis
#'
#' Compute the Getis-Ord coefficient for variables in expression using
#' the provided connectivity (neighbors) and associated edge weights
#'
#' @param expression Matrix on which to calculate G_i. 
#' G_i is calculated for each variable (row) separately.  matrix of dimension 
#' N_GENES x N_CELLS
#'
#' @param neighbors n_nearest neighbors for each cell.  Matrix of
#' size N_CELLS x N_Neighbors.  Entries represent index of neighbors
#' @param weights weights for each neighbor.  Same size/type as neighbors
#' input
#' @export
#' @return G_i Getis-ord values for each variable in G_i.  Matrix of size
#' N_GENES x N_CELLS
computeHotspot <- function(expression, neighbors, weights){

    if (is(expression, "data.frame")){
        expression <- as.matrix(expression)
    }

    if (is(neighbors, "data.frame")){
        neighbors <- as.matrix(neighbors)
    }

    if (is(weights, "data.frame")){
        weights <- as.matrix(weights)
    }

    N_GENES <- nrow(expression)


    # Load into a sparse matrix
    row_idxs <- as.vector(t(row(neighbors)))
    col_idxs <- as.vector(t(neighbors))
    values <- as.vector(t(weights))

    sparse_weights <- sparseMatrix(i = row_idxs, j = col_idxs, x = values)

    G_i <- tcrossprod(expression, sparse_weights)

    # Compute normalization factors
    W_i <- rowSums(weights)         # Length: N_Cells
    S_1i <- rowSums(weights ** 2)   # Length: N_Cells


    xbar <- rowMeans(expression)    # Length: N_Genes
    s <- rowSds(expression)         # Length: N_Genes
    n <- ncol(expression)

    offset <- matrix(xbar) %*% t(matrix(W_i))

    denom <- ((n*S_1i - W_i**2) / (n-1))**(1/2)
    denom <- matrix(rep(denom, times = N_GENES), nrow = N_GENES, byrow = TRUE)
    denom <- s * denom

    G_i <- (G_i - offset) / denom

    return(G_i)
}


# n_neighbors <- 10
# neighborhood_factor <- 3
# 
# # Generate some random test data
# N_CELLS <- 20000
# N_COMPONENTS <- 40
# N_GENES <- 10000
# 
# pca <- matrix(runif(N_CELLS * N_COMPONENTS), nrow = N_CELLS)
# expression <- matrix(runif(N_CELLS * N_GENES), nrow = N_GENES)
# 
# system.time(out <- hotspot(pca, expression, n_neighbors, neighborhood_factor))
