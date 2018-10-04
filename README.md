# Hotspot

## R

### Installation

```bash
$ git clone git@github.com:Yoseflab/Hotspot
$ cd Hotspot
$ R CMD INSTALL .
```

### Usage

```R
library(Hotspot)

# pca is a numeric matrix of pca-reduced data (N_CELLS x N_COMPONENTS)
result <- neighborsAndWeights(pca)

# expression is a log-scale gene expression matrix (N_GENES x N_CELLS)
hs <- computeHotspot(expression, result$neighbors, result$weights)

# now hs is a numeric matrix (N_GENES x N_CELLS) of Gi* coefficients for each cell/gene
# Extreme high/low values represent regions of clustered high/low expression

rowMaxs <- apply(abs(hs), 1, max)
top100Genes <- names(sort(rowMaxs, decreasing=TRUE)[1:100])

```

## Python

### Installation

```bash
$ git clone git@github.com:Yoseflab/Hotspot
$ cd Hotspot
$ python setup.py install
```

### Usage


```python
import Hotspot

# pca is a pandas DataFrame of pca-reduced data (N_CELLS x N_COMPONENTS)
neighbors, weights = neighbors_and_weights(pca)

# expression is a log-scale gene expression DataFrame (N_GENES x N_CELLS)
hs = compute_gi_dataframe(expression, neighbors, weights)

# now hs is a numeric matrix (N_GENES x N_CELLS) of Gi* coefficients for each cell/gene
# Extreme high/low values represent regions of clustered high/low expression

rowMaxs = hs.abs.max(axis=1)
top100Genes rowMaxs.sort_values(ascending=False).index[:100]
```
