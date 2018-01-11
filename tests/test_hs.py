import hotspot
import numpy as np
import pandas as pd
import os

EPS = 1e-10

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data")


def test_hotspot_gi():
    expression = pd.read_table(
        os.path.join(DATA_DIR, "expression.txt"),
        index_col=0)

    pca = pd.read_table(
        os.path.join(DATA_DIR, "pca.txt"),
        index_col=0)

    neighbors, weights = hotspot.neighbors_and_weights(
        pca, n_neighbors=9, neighborhood_factor=3)

    gi_df = hotspot.compute_gi_dataframe(expression, neighbors, weights)

    gi_single_rows = []
    for i in expression.index:
        gi_single = hotspot.compute_gi_single(
            expression.loc[i].values, neighbors.values,
            weights.values)
        gi_single_rows.append(gi_single)

    gi_single = np.vstack(gi_single_rows)

    max_error = np.abs(gi_df.values - gi_single).max()

    assert max_error < EPS


def generate_output_for_R():
    expression = pd.read_table(
        os.path.join(DATA_DIR, "expression.txt"),
        index_col=0)

    neighbors = pd.read_table(
        os.path.join(DATA_DIR, "neighbors.txt"),
        index_col=0)

    weights = pd.read_table(
        os.path.join(DATA_DIR, "weights.txt"),
        index_col=0)

    gi_df = hotspot.compute_gi_dataframe(expression, neighbors, weights)

    gi_df.to_csv(os.path.join(DATA_DIR, "python_gi.txt"),
                 sep="\t")


generate_output_for_R()
