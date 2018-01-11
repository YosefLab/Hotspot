context("Test R Hotspot against Python reference")

test_that("G_i values are equal for same expression/neighbors/weights", {
    expression <- read.table("../data/expression.txt",
                             header = TRUE,
                             sep = "\t",
                             row.names = 1)

    neighbors <- read.table("../data/neighbors.txt",
                             header = TRUE,
                             sep = "\t",
                             row.names = 1)

    neighbors <- neighbors + 1 #  Python is 0-based, R is 1-based

    weights <- read.table("../data/weights.txt",
                             header = TRUE,
                             sep = "\t",
                             row.names = 1)


    r_out <- computeHotspot(expression, neighbors, weights)

    # Read in python output

    python_out <- read.table("../data/python_gi.txt",
                             header = TRUE,
                             sep = "\t",
                             row.names = 1)

    python_out <- as.matrix(python_out)


    max_err <- max(abs(python_out - r_out))

    expect_true(max_err < 1e-10)

})
