import numpy as np
import pandas as pd

from .knn import (
    neighbors_and_weights,
    neighbors_and_weights_from_distances,
    tree_neighbors_and_weights,
    make_weights_non_redundant,
)
from .local_stats import compute_hs
from .local_stats_pairs import (
    compute_hs_pairs, compute_hs_pairs_centered_cond)

from . import modules
from .plots import local_correlation_plot
from tqdm import tqdm


class Hotspot:

    def __init__(
            self, counts, model='danb',
            latent=None, distances=None, tree=None,
            umi_counts=None):
        """Initialize a Hotspot object for analysis

        Either `latent` or `tree` or `distances` is required.

        Parameters
        ----------
        counts : pandas.DataFrame
            Count matrix (shape is genes x cells)
        model : string, optional
            Specifies the null model to use for gene expression.
            Valid choices are:

                - 'danb': Depth-Adjusted Negative Binomial
                - 'bernoulli': Models probability of detection
                - 'normal': Depth-Adjusted Normal
                - 'none': Assumes data has been pre-standardized

        latent : pandas.DataFrame, optional
            Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims)
        distances : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells)
        tree : ete3.coretype.tree.TreeNode
            Root tree node.  Can be created using ete3.Tree
        umi_counts : pandas.Series, optional
            Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used
        """

        if latent is None and distances is None and tree is None:
            raise ValueError("Neither `latent` or `tree` or `distance` arguments were supplied.  One of these is required")

        if latent is not None and distances is not None:
            raise ValueError("Both `latent` and `distances` provided - only one of these should be provided.")

        if latent is not None and tree is not None:
            raise ValueError("Both `latent` and `tree` provided - only one of these should be provided.")

        if distances is not None and tree is not None:
            raise ValueError("Both `distances` and `tree` provided - only one of these should be provided.")

        if latent is not None:
            if counts.shape[1] != latent.shape[0]:
                if counts.shape[0] == latent.shape[0]:
                    raise ValueError("`counts` input should be a Genes x Cells dataframe.  Maybe needs transpose?")
                raise ValueError("Size mismatch counts/latent. Columns of `counts` should match rows of `latent`.")

        if distances is not None:
            assert counts.shape[1] == distances.shape[0]
            assert counts.shape[1] == distances.shape[1]

        if tree is not None:
            try:
                all_leaves = []
                for x in tree:
                    if x.is_leaf():
                        all_leaves.append(x.name)
            except:
                raise ValueError("Can't parse supplied tree")

            if (
                len(all_leaves) != counts.shape[1] or
                len(set(all_leaves) & set(counts.columns)) != len(all_leaves)
               ):
                raise ValueError("Tree leaf labels don't match columns in supplied counts matrix")

        if umi_counts is None:
            umi_counts = counts.sum(axis=0)
        else:
            assert umi_counts.size == counts.shape[1]

        if not isinstance(umi_counts, pd.Series):
            umi_counts = pd.Series(umi_counts)

        valid_models = {'danb', 'bernoulli', 'normal', 'none'}
        if model not in valid_models:
            raise ValueError(
                'Input `model` should be one of {}'.format(valid_models)
            )

        valid_genes = counts.var(axis=1) > 0
        n_invalid = counts.shape[0] - valid_genes.sum()
        if n_invalid > 0:
            counts = counts.loc[valid_genes]
            print(
                "\nRemoving {} undetected/non-varying genes".format(n_invalid)
            )

        self.counts = counts
        self.latent = latent
        self.distances = distances
        self.tree = tree
        self.model = model

        self.umi_counts = umi_counts

        self.graph = None
        self.modules = None
        self.local_correlation_z = None
        self.linkage = None
        self.module_scores = None

    def create_knn_graph(
            self, weighted_graph=False, n_neighbors=30, neighborhood_factor=3):
        """Create's the KNN graph and graph weights

        The resulting matrices containing the neighbors and weights are
        stored in the object at `self.neighbors` and `self.weights`

        Parameters
        ----------
        weighted_graph: bool
            Whether or not to create a weighted graph
        n_neighbors: int
            Neighborhood size
        neighborhood_factor: float
            Used when creating a weighted graph.  Sets how quickly weights decay
            relative to the distances within the neighborhood.  The weight for
            a cell with a distance d will decay as exp(-d/D) where D is the distance
            to the `n_neighbors`/`neighborhood_factor`-th neighbor.
        """

        if self.latent is not None:
            neighbors, weights = neighbors_and_weights(
                self.latent, n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor)
        elif self.tree is not None:
            if weighted_graph:
                raise ValueError("When using `tree` as the metric space, `weighted_graph=True` is not supported")
            neighbors, weights = tree_neighbors_and_weights(
                self.tree, n_neighbors=n_neighbors, counts=self.counts)
        else:
            neighbors, weights = neighbors_and_weights_from_distances(
                self.distances, n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor)

        neighbors = neighbors.loc[self.counts.columns]
        weights = weights.loc[self.counts.columns]

        self.neighbors = neighbors

        if not weighted_graph:
            weights = pd.DataFrame(
                np.ones_like(weights.values),
                index=weights.index, columns=weights.columns
            )

        weights = make_weights_non_redundant(neighbors.values, weights.values)

        weights = pd.DataFrame(
            weights, index=neighbors.index, columns=neighbors.columns)

        self.weights = weights

    def compute_hotspot(self, jobs=1):
        """Perform feature selection using local autocorrelation

        In addition to returning output, this also stores the output
        in `self.results`.

        Alias for `self.compute_autocorrelations`

        Parameters
        ----------
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        results : pandas.DataFrame
            A dataframe with four columns:

              - C: Scaled -1:1 autocorrelation coeficients
              - Z: Z-score for autocorrelation
              - Pval:  P-values computed from Z-scores
              - FDR:  Q-values using the Benjamini-Hochberg procedure

            Gene ids are in the index

        """

        results = compute_hs(
            self.counts, self.neighbors, self.weights,
            self.umi_counts, self.model, centered=True, jobs=jobs)

        self.results = results

        return self.results

    def compute_autocorrelations(self, jobs=1):
        """Perform feature selection using local autocorrelation

        In addition to returning output, this also stores the output
        in `self.results`

        Parameters
        ----------
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        results : pandas.DataFrame
            A dataframe with four columns:

              - C: Scaled -1:1 autocorrelation coeficients
              - Z: Z-score for autocorrelation
              - Pval:  P-values computed from Z-scores
              - FDR:  Q-values using the Benjamini-Hochberg procedure

            Gene ids are in the index

        """
        return self.compute_hotspot(jobs)

    def compute_local_correlations(self, genes, jobs=1):
        """Define gene-gene relationships with pair-wise local correlations

        In addition to returning output, this method stores its result
        in `self.local_correlation_z`

        Parameters
        ----------
        genes: iterable of str
            gene identifies to compute local correlations on
            should be a smaller subset of all genes
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        local_correlation_z : pd.Dataframe
                local correlation Z-scores between genes
                shape is genes x genes
        """

        print(
            "Computing pair-wise local correlation on {} features..."
            .format(len(genes))
        )

        lc, lcz = compute_hs_pairs_centered_cond(
            self.counts.loc[genes], self.neighbors, self.weights,
            self.umi_counts, self.model, jobs=jobs)

        self.local_correlation_c = lc
        self.local_correlation_z = lcz

        return self.local_correlation_z

    def create_modules(
            self, min_gene_threshold=20, core_only=True, fdr_threshold=0.05
    ):
        """Groups genes into modules

        In addition to being returned, the results of this method are retained
        in the object at `self.modules`.  Additionally, the linkage matrix
        (in the same form as that of scipy.cluster.hierarchy.linkage) is saved
        in `self.linkage` for plotting or manual clustering.

        Parameters
        ----------
        min_gene_threshold: int
            Controls how small modules can be.  Increase if there are too many
            modules being formed.  Decrease if substructre is not being captured
        core_only: bool
            Whether or not to assign ambiguous genes to a module or leave unassigned
        fdr_threshold: float
            Correlation theshold at which to stop assigning genes to modules

        Returns
        -------
        modules: pandas.Series
            Maps gene to module number.  Unassigned genes are indicated with -1


        """

        gene_modules, Z = modules.compute_modules(
            self.local_correlation_z, min_gene_threshold=min_gene_threshold,
            fdr_threshold=fdr_threshold, core_only=core_only
        )

        self.modules = gene_modules
        self.linkage = Z

        return self.modules

    def calculate_module_scores(self):
        """Calculate Module Scores

        In addition to returning its result, this method stores
        its output in the object at `self.module_scores`

        Returns
        -------
        module_scores: pandas.DataFrame
            Scores for each module for each gene
            Dimensions are genes x modules

        """

        modules_to_compute = sorted(
            [x for x in self.modules.unique() if x != -1]
        )

        print(
            "Computing scores for {} modules..."
            .format(len(modules_to_compute))
        )

        module_scores = {}
        for module in tqdm(modules_to_compute):
            module_genes = self.modules.index[self.modules == module]

            scores = modules.compute_scores(
                self.counts.loc[module_genes].values,
                self.model, self.umi_counts.values,
                self.neighbors.values, self.weights.values
            )

            module_scores[module] = scores

        module_scores = pd.DataFrame(module_scores)
        module_scores.index = self.counts.columns

        self.module_scores = module_scores

        return self.module_scores

    def plot_local_correlations(
            self, mod_cmap='tab10', vmin=-8, vmax=8,
            z_cmap='RdBu_r', yticklabels=False
    ):
        """Plots a clustergrid of the local correlation values

        Parameters
        ----------
        mod_cmap: valid matplotlib colormap str or object
            discrete colormap for module assignments on the left side
        vmin: float
            minimum value for colorscale for Z-scores
        vmax: float
            maximum value for colorscale for Z-scores
        z_cmap: valid matplotlib colormap str or object
            continuous colormap for correlation Z-scores
        yticklabels: bool
            Whether or not to plot all gene labels
            Default is false as there are too many.  However
            if using this plot interactively you may with to set
            to true so you can zoom in and read gene names
        """

        return local_correlation_plot(
                    self.local_correlation_z, self.modules, self.linkage,
                    mod_cmap=mod_cmap, vmin=vmin, vmax=vmax,
                    z_cmap=z_cmap, yticklabels=yticklabels
        )
