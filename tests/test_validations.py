import numpy as np
import pandas as pd
from hotspot import sim_data
from hotspot import Hotspot
import anndata
import pytest

from scipy.sparse import csc_matrix


def test_models():
    """
    Ensure each model runs
    """

    # Simulate some data
    N_CELLS = 100
    N_DIM = 10
    N_GENES = 10

    latent = sim_data.sim_latent(N_CELLS, N_DIM)
    latent = pd.DataFrame(
        latent, index=["Cell{}".format(i + 1) for i in range(N_CELLS)]
    )

    umi_counts = sim_data.sim_umi_counts(N_CELLS, 2000, 200)
    umi_counts = pd.Series(umi_counts)

    gene_exp = np.random.rand(N_GENES, N_CELLS)
    gene_exp = pd.DataFrame(
        gene_exp,
        index=["Gene{}".format(i + 1) for i in range(gene_exp.shape[0])],
        columns=latent.index,
    )

    adata = anndata.AnnData(gene_exp.transpose())
    adata.layers["sparse"] = csc_matrix(adata.X)
    adata.obsm["latent"] = latent.values
    adata.obs["umi_counts"] = umi_counts.values

    for model in ["danb", "bernoulli", "normal", "none"]:
        hs = Hotspot(
            adata,
            model=model,
            latent_obsm_key="latent",
            umi_counts_obs_key="umi_counts",
            layer_key="sparse",
        )
        hs.create_knn_graph(False, n_neighbors=30)
        hs.compute_autocorrelations(jobs=1)
        hs.compute_autocorrelations(jobs=2)

        assert isinstance(hs.results, pd.DataFrame)
        assert hs.results.shape[0] == N_GENES

        hs.compute_local_correlations(gene_exp.index, jobs=1)
        hs.compute_local_correlations(gene_exp.index, jobs=2)

        assert isinstance(hs.local_correlation_z, pd.DataFrame)
        assert hs.local_correlation_z.shape[0] == N_GENES
        assert hs.local_correlation_z.shape[1] == N_GENES

        hs.create_modules(min_gene_threshold=2, fdr_threshold=1)

        assert isinstance(hs.modules, pd.Series)
        assert (hs.modules.index & gene_exp.index).size == N_GENES

        assert isinstance(hs.linkage, np.ndarray)
        assert hs.linkage.shape == (N_GENES - 1, 4)

        hs.calculate_module_scores()

        assert isinstance(hs.module_scores, pd.DataFrame)
        assert (hs.module_scores.index == gene_exp.columns).all()


def test_filter_genes():
    """
    Ensure genes with no expression are pre-filtered
    """
    # Simulate some data
    N_CELLS = 100
    N_DIM = 10
    N_GENES = 10
    N_GENES_ZERO = 5

    latent = sim_data.sim_latent(N_CELLS, N_DIM)
    latent = pd.DataFrame(latent)

    umi_counts = sim_data.sim_umi_counts(N_CELLS, 2000, 200)
    umi_counts = pd.Series(umi_counts)

    gene_exp = np.random.rand(N_GENES + N_GENES_ZERO, N_CELLS)
    gene_exp[N_GENES:] = 0
    gene_exp = pd.DataFrame(
        gene_exp,
        index=["Gene{}".format(i + 1) for i in range(gene_exp.shape[0])],
        columns=latent.index,
    )

    adata = anndata.AnnData(gene_exp.transpose())
    adata.obsm["latent"] = latent.values
    adata.obs["umi_counts"] = umi_counts.values

    with pytest.raises(ValueError):
        hs = Hotspot(
            adata,
            model="normal",
            latent_obsm_key="latent",
            umi_counts_obs_key="umi_counts",
        )
