context("Test R Neghbors and Weights")

test_that("R computes same neighbors and weights as Python", {

    pca <- read.table("../data/pca.txt",
                             header = TRUE,
                             sep = "\t",
                             row.names = 1)

    r_out <- neighborsAndWeights(pca, n_neighbors = 9,
                                 neighborhood_factor = 3)

    r_neighbors <- r_out$neighbors
    r_weights <- r_out$weights


    expect_equal(nrow(r_neighbors), nrow(pca))
    expect_equal(ncol(r_neighbors), 9)
    expect_equal(nrow(r_neighbors), nrow(r_weights))
    expect_equal(ncol(r_neighbors), ncol(r_weights))


    py_neighbors <- read.table("../data/neighbors.txt",
                             header = TRUE,
                             sep = "\t",
                             row.names = 1)

    py_neighbors <- py_neighbors + 1 #  Python is 0-based, R is 1-based

    py_weights <- read.table("../data/weights.txt",
                             header = TRUE,
                             sep = "\t",
                             row.names = 1)

    expect_equal(nrow(r_neighbors), nrow(py_neighbors))
    expect_equal(ncol(r_neighbors), ncol(py_neighbors))
    expect_equal(nrow(r_weights), nrow(py_weights))
    expect_equal(ncol(r_weights), ncol(py_weights))

    expect_true(all(r_neighbors == py_neighbors))

    max_err <- max(abs(r_weights - py_weights))

    expect_true(max_err < 1e-10)

})
