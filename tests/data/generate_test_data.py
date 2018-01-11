"""
This file is used to generate some small, artificial, datasets
to be used to test the python and R versions of hotspot.
"""

import numpy as np
import pandas as pd
import hotspot
from sklearn.decomposition import PCA

N_CELLS = 100
N_GENES = 50
N_COMPONENTS = 10
N_NEIGHBORS = 5

expression = pd.DataFrame(np.random.rand(N_GENES, N_CELLS))
expression.columns = ['c' + str(x) for x in expression.columns]

model = PCA(n_components=N_COMPONENTS)

pca = pd.DataFrame(model.fit_transform(expression.T),
                   index=expression.columns)

expression.to_csv("expression.txt", sep="\t")
pca.to_csv("pca.txt", sep="\t")

# generate fake neighbor indices
neighbors, weights = hotspot.neighbors_and_weights(
        pca, n_neighbors=9, neighborhood_factor=3)

neighbors = pd.DataFrame(neighbors, index=expression.columns)
neighbors.to_csv("neighbors.txt", sep="\t")

weights = pd.DataFrame(weights, index=expression.columns)
weights.to_csv("weights.txt", sep="\t")
