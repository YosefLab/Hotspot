"""
End-to-end regression test: the full pipeline on a pbmc3k subset must match
results frozen from master in data/pbmc3k_reference.h5ad.

The reference stores the cell/gene subset, latent, KNN graph and every model's
outputs. The KNN graph is injected so this tests Hotspot's statistics, not
pynndescent. Regenerate (only when a change in results is intended) by running
this file against the reference checkout:

    PYTHONPATH=<master checkout> python tests/test_e2e.py <master sha>
"""
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from hotspot import Hotspot

DATA = Path(__file__).parent / "data"
REF = DATA / "pbmc3k_reference.h5ad"
MODELS = ["danb", "bernoulli", "normal", "none"]
MODULE_KW = dict(min_gene_threshold=5, fdr_threshold=0.05)

sc.settings.datasetdir = DATA  # pbmc3k_raw.h5ad is cached here (gitignored)


def _run(adata, model, neighbors, weights):
    hs = Hotspot(adata, model=model, latent_obsm_key="latent",
                 umi_counts_obs_key="umi_counts")
    hs.neighbors = neighbors
    hs.weights = weights
    hs.compute_autocorrelations()
    hs.compute_local_correlations(adata.var_names)
    hs.create_modules(**MODULE_KW)
    hs.calculate_module_scores()
    return hs


def _input(ref):
    adata = sc.datasets.pbmc3k()[ref.obs_names, ref.var_names].copy()
    adata.obs["umi_counts"] = ref.obs["umi_counts"]
    adata.obsm["latent"] = ref.obsm["latent"]
    idx = ref.obs_names
    neighbors = pd.DataFrame(ref.obsm["neighbors"], index=idx)
    weights = pd.DataFrame(ref.obsm["weights"], index=idx)
    return adata, neighbors, weights


@pytest.fixture(scope="module")
def ref():
    return anndata.read_h5ad(REF)


@pytest.mark.parametrize("model", MODELS)
def test_matches_reference(ref, model):
    adata, neighbors, weights = _input(ref)
    hs = _run(adata, model, neighbors, weights)
    genes = ref.var_names

    for col in ["C", "Z", "Pval", "FDR"]:
        np.testing.assert_allclose(
            hs.results.loc[genes, col], ref.var[f"{model}_{col}"],
            rtol=1e-6, atol=1e-10, err_msg=col)
    np.testing.assert_allclose(
        hs.local_correlation_z.loc[genes, genes], ref.varp[f"{model}_lcz"],
        rtol=1e-6, atol=1e-10)
    np.testing.assert_array_equal(
        hs.modules.loc[genes], ref.var[f"{model}_module"])
    np.testing.assert_allclose(
        hs.module_scores.values, ref.obsm[f"{model}_module_scores"],
        rtol=1e-5, atol=1e-8)


def _make_reference(commit, n_cells=1000, n_genes=200):
    """Function used to generate the on-disk reference, run against: https://github.com/YosefLab/Hotspot/commit/f9855997b0a2e5edbfe8046747be88f556b5ba11"""
    full = sc.datasets.pbmc3k()[:n_cells].copy()
    full.obs["umi_counts"] = np.asarray(full.X.sum(1)).ravel()

    norm = full.copy()
    sc.pp.filter_genes(norm, min_cells=10)
    sc.pp.normalize_total(norm)
    sc.pp.log1p(norm)
    sc.pp.highly_variable_genes(norm, n_top_genes=n_genes)
    sc.pp.pca(norm, n_comps=20)

    ref = anndata.AnnData(
        obs=full.obs[["umi_counts"]],
        var=pd.DataFrame(index=norm.var_names[norm.var["highly_variable"]]))
    ref.obsm["latent"] = norm.obsm["X_pca"]

    adata = full[:, ref.var_names].copy()
    adata.obsm["latent"] = ref.obsm["latent"]
    hs = Hotspot(adata, model="danb", latent_obsm_key="latent",
                 umi_counts_obs_key="umi_counts")
    hs.create_knn_graph(weighted_graph=True, n_neighbors=30)
    ref.obsm["neighbors"] = hs.neighbors.values
    ref.obsm["weights"] = hs.weights.values

    for model in MODELS:
        hs = _run(adata, model, hs.neighbors, hs.weights)
        for col in ["C", "Z", "Pval", "FDR"]:
            ref.var[f"{model}_{col}"] = hs.results.loc[ref.var_names, col]
        ref.var[f"{model}_module"] = hs.modules.loc[ref.var_names]
        ref.varp[f"{model}_lcz"] = hs.local_correlation_z.loc[ref.var_names, ref.var_names].values
        ref.obsm[f"{model}_module_scores"] = hs.module_scores.values
        print(model, "modules:", hs.modules.value_counts().to_dict())

    ref.uns["hotspot_commit"] = commit
    ref.write_h5ad(REF, compression="gzip")
