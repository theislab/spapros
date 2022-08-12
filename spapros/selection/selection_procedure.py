import json
import os
import pickle
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
import spapros.evaluation.evaluation as ev
import spapros.selection.selection_methods as select
import spapros.util.util as util
from rich.console import RichCast
from scipy.sparse import issparse
from spapros.plotting import plot as pl
from spapros.util.util import dict_to_table
from spapros.util.util import filter_marker_dict_by_penalty
from spapros.util.util import filter_marker_dict_by_shared_genes


# from tqdm.autonotebook import tqdm


class ProbesetSelector:  # (object)
    """General class for probeset selection.

    Notes:
        The selector creates a probeset which identifies the celltypes of interest and captures transcriptomic variation
        beyond cell type labels.

        The Spapros selection pipeline combines basic feature selection builing blocks while optionally taking into
        account prior knowledge.

        **The main steps of the selection pipeline are:**

        1) PCA based selection of variation recovering genes.
        2) Selection of DE genes.
        3) Train decision trees on the DE genes (including an iterative optimization with additional DE tests).
        4) Train decision trees on the PCA genes (and optionally on pre-selected and prioritized genes).
        5) Enhancement of the PCA trees by adding beneficial DE genes.
        6) Rank genes, eventually add missing marker genes and compile probe set.

        The result of the selection is given in :attr:`.ProbesetSelector.probeset`.

        **Genes are ranked as follows (sorry it's a bit complicated):**

        * First the following groups are built
            1. preselected genes (optional, see parameter `preselected_genes`)
            2. genes that occur in the best decision trees of each cell type
            3. genes that are needed to achieve the minimal number of markers per cell type that occurs in
               :attr:`.ProbesetSelector.marker_list` but not in :attr:`.ProbesetSelector.adata_celltypes` (optional, see
               parameter `n_list_markers`). This group is separated from 3. because genes of 2. take care of
               classifying cell types in :attr:`.ProbesetSelector.adata_celltypes`.
            4. genes that are needed to achieve the minimal number of markers per cell type in
               :attr:`.ProbesetSelector.adata_celltypes`.  (optional, see parameter `n_min_markers`)
            5. all other genes
        * Afterwards within each "rank" group genes are further ranked by
            1. the **marker_rank**: first the best markers of celltypes, then 2nd best markers of celltypes, ..., then
               `n_min_markers` th best marker of celltypes, then genes that are not identified as required markers.
            2. the **tree_rank**: for each cell type the genes that occur in cell type classification trees with 2nd
               best performance, then 3rd best performance, and so on. Genes that don't occur in trees have the worst
               tree_rank.
            3. the **importance_score** from the best cell type classification tree of each gene. Genes that don't occur
               in any tree score worst.
            4. the **pca_score** which scores how much variation of the dataset each gene captures.


    Args:
        adata:
            Data with log normalised counts in ``adata.X``. The selection runs with an adata subsetted on fewer
            genes. It might be helpful though to keep all genes (when a marker_list and penalties are provided). The
            genes can be subsetted for selection via :attr:`genes_key`.
        celltype_key:
            Key in ``adata.obs`` with celltype annotations.
        genes_key:
            Key in ``adata.var`` for preselected genes (typically 'highly_variable_genes').
        n:
            Optionally set the number of finally selected genes. Note that when :attr:`n` is `None` we automatically
            infer :attr:`n` as the minimal number of recommended genes. This includes all preselected genes, genes in
            the best decision tree of each celltype, and the minimal number of identified and added markers defined by
            :attr:`n_min_markers` and :attr:`n_list_markers`. Als note that setting :attr:`n` might change the gene
            ranking since the final added list_markers are added based on the theoretically added genes without
            :attr:`list_markers`.
        preselected_genes:
            Pre selected genes (these will also have the highest ranking in the final list).
        prior_genes:
            Prioritized genes.
        n_pca_genes:
            Optionally set the number of preselected pca genes. If not set or set `<1`, this step will be skipped.
        min_mean_difference:
            Minimal difference of mean expression between at least one celltype and the background. In this test only
            cell types from :attr:`celltypes` are taken into account (also for the background). This minimal difference
            is applied as an additional binary penalty in pca_penalties, DE_penalties and m_penalties_adata_celltypes.
        n_min_markers:
            The minimal number of identified and added markers.
        celltypes:
            Cell types for which trees are trained.

            - The probeset is optimised to be able to distinguish each of these cell types from all other cells
              occuring in the dataset.
            - The pca selection is based on all cell types in the dataset (not only on :attr:`celltypes`).
            - The optionally provided marker list can include additional cell types not listed in :attr:`celltypes`
              (and ``adata.obs[celltype_key])``.

        marker_list: List of marker genes. Can either be a dictionary like this::

                {
                "celltype_1": ["S100A8", "S100A9", "LYZ", "BLVRB"],
                "celltype_2": ["BIRC3", "TMEM116"],
                "celltype_4": ["CD74", "CD79B", "MS4A1"],
                "celltype_3": ["C5AR1"],
                }

            Or the path to a csv-file containing the one column of markers for each celltype. The column names need to
            be the celltype identifiers used in ``adata.obs[celltype_key]``.

        n_list_markers:
            Minimal number of markers per celltype that are at least selected. Selected means either selecting genes
            from the marker list or having correlated genes in the already selected panel. (Set the correlation
            threshold with `marker_selection_hparams['penalty_threshold'])`. The correlation based check only applies to
            cell types that also occur in `adata.obs[celltype_key]` while for cell types that only occur in the
            `marker_list` the markers are just added.
            If you want to select a different number of markers for celltypes in adata and celltypes only in the marker
            list, set e.g.: ``n_list_markers = {'adata_celltypes':2,'list_celltypes':3}``.
        marker_corr_th:
            Minimal correlation to consider a gene as captured.
        pca_penalties:
            List of keys for columns in ``adata.var`` containing penalty factors that are multiplied with the scores for
            PCA based gene selection.
        DE_penalties:
            List of keys for columns in ``adata.var`` containing penalty factors that are multiplied with the scores for DE
            based gene selection.
        m_penalties_adata_celltypes:
            List of keys for columns in ``adata.var`` containing penalty factors to filter out marker genes if a gene's
            penalty < threshold for celltypes in adata.
        m_penalties_list_celltypes:
            List of keys for columns in ``adata.var`` containing penalty factors to filter out marker genes if a gene's
            penalty < threshold for celltypes not in adata.
        pca_selection_hparams:
            Dictionary with hyperparameters for the PCA based gene selection.
        DE_selection_hparams
            Dictionary with hyperparameters for the DE based gene selection.
        forest_hparams
            Dictionary with hyperparameters for the forest based gene selection.
        forest_DE_baseline_hparams:
            Dictionary with hyperparameters for adding DE genes to decision trees.
        add_forest_genes_hparams:
            Dictionary with hyperparameters for adding marker genes to decision trees.
        marker_selection_hparams:
            Dictionary with hyperparameters. So far only the threshold for the penalty filtering of marker genes if a
            gene's penalty < threshold.
        verbosity:
            Verbosity level.
        seed:
            Random number seed.
        save_dir: Directory path where all results are saved and loaded from if results already exist. Note for the case
            that results already exist:

               - if self.select_probeset() was fully run through and all results exist: then the initialization arguments
                 don't matter much
               - if only partial results were generated, make sure that the initialization arguments are the same as
                 before!
        n_jobs:
            Number of cpus for multi processing computations. Set to -1 to use all available cpus.


    Attributes:
        adata:
            Data with log normalised counts in ``adata.X``.
        ct_key:
            Key in ``adata.obs`` with celltype annotations.
        g_key:
            Key in ``adata.var`` for preselected genes (typically `'highly_variable_genes'`).
        n:
            Number of finally selected genes.
        genes:
            Pre selected genes (these will also have the highest ranking in the final list).
        selection:
            Dictionary with the final and several other gene set selections.
        n_pca_genes:
            The number of preselected pca genes. If `None` or `<1`, this step is skipped.
        min_mean_difference:
            Minimal difference of mean expression between at least one celltype and the background.
        n_min_markers:
            The minimal number of identified and added markers for cell types of `adata.obs[ct_key]`.
        celltypes:
            Cell types for which trees are trained.
        adata_celltypes:
            List of all celltypes occuring in ``adata.obs[ct_key]``.
        obs:
            Keys of ``adata.obs`` on which most of the selections are run.
        marker_list:
            Dictionary of the form ``{'celltype': list of markers of celltype}``.
        n_list_markers:
            Minimal number of markers from the `marker_list` that are at least selected per cell type. Note that for
            those cell types in the `marker_list` that also occur in `adata.obs[ct_key]` genes that are correlated with
            the markers might be selected (see :attr:`marker_corr_th`).
        marker_corr_th:
            Minimal correlation to consider a gene as captured.
        pca_penalties:
            List of keys for columns in ``adata.var`` containing penalty factors that are multiplied with the scores
            for PCA based gene selection.
        DE_penalties:
            List of keys for columns in ``adata.var`` containing penalty factors that are multiplied with the scores
            for DE based gene selection.
        m_penalties_adata_celltypes:
            List of keys for columns in ``adata.var`` containing penalty factors to filter out marker genes if a
            gene's penalty < threshold for celltypes in adata.
        m_penalties_list_celltypes:
            List of keys for columns in ``adata.var`` containing penalty factors to filter out marker genes if a
            gene's penalty < threshold for celltypes not in adata.
        pca_selection_hparams:
            Dictionary with hyperparameters for the PCA based gene selection.
        DE_selection_hparams:
            Dictionary with hyperparameters for the DE based gene selection.
        forest_hparams:
            Dictionary with hyperparameters for the forest based gene selection.
        forest_DE_baseline_hparams:
            Dictionary with hyperparameters for adding DE genes to decision trees.
        add_forest_genes_hparams:
            Dictionary with hyperparameters for adding marker genes to decision trees.
        m_selection_hparams:
            Dictionary with hyperparameters. So far only the threshold for the penalty filtering of marker genes if a
            gene's penalty < threshold.
        verbosity:
            Verbosity level.
        seed:
            Random number seed.
        save_dir:
            Directory path where all results are saved and loaded from if results already exist.
        n_jobs:
            Number of cpus for multi processing computations. Set to `-1` to use all available cpus.
        forest_results:
            Forest results.
        forest_clfs
            Forest classifier.
        min_test_n:
            Minimal number of samples in each celltype's test set
        loaded_attributes:
            List of which results were loaded from disc.
        disable_pbars.
            Disable progress bars.
        probeset:
            The final probeset list. Available only after calling :func:`~ProbesetSelector.select_probeset`. The table
            contains the following columns:
                * **index**
                    Gene symbol.
                * **gene_nr**
                    Integer assigned to each gene.
                * **selection**
                    Wether a gene was selected.
                * **rank**
                    Gene ranking as describes in Notes above.
                * **marker_rank**
                    Rank of the required markers per cell type. The best marker per cell type has marker_rank 1, the
                    second best 2, and so on. Required markers are ranked till :attr:`~ProbesetSelector.n_min_markers`
                    or :attr:`~ProbesetSelector.n_list_markers` depending on the cell type.
                * **tree_rank**
                    Ranking of the best tree the gene occured in. Per cell type multiple decision trees are trained and
                    the best one is selected. To extend the ranking of genes in the probeset list, the 2nd, 3rd, ...
                    best performing trees are considered.
                * **importance_score**
                    Highest importance score of a gene in the highest ranked trees that the gene occured in. (see TODO:
                    reference tree training fct and there the description of the output)
                * **pca_score**
                    Score from PCA-based selection (see TODO: document pca based selection and reference procedure
                    here). Genes with high scores capture high amounts of general transcriptomic variation.
                * **pre_selected**
                    Whether a gene was in the list of pre-selected genes.
                * **prior_selected**
                    Whether a gene was in the list of prioritized genes.
                * **pca_selected**
                    Whether a gene was in the list of `n_pca_genes` of PCA selected genes.
                * **celltypes_DE_1vsall**
                    Cell type in which a given gene is up-regulated (compared to all other cell types as background,
                    identified via differential expression tests during the selection).
                * **celltypes_DE_specific**
                    Like **celltypes_DE_1vsall** but for DE tests that use a subset of the background (typically genes
                    that distinguish similar cell types).
                * **celltypes_DE**
                    **celltypes_DE_1vsall** and **celltypes_DE_specific** combined.
                * **celltypes_marker**
                    **celltypes_DE_1vsall** combined with **celltypes_DE_specific** and the cell type of
                    :attr:`~ProbesetSelector.marker_list` if the gene was listed as a marker there.
                * **list_only_ct_marker**
                    Whether a gene is listed as a marker in :attr:`~ProbesetSelector.marker_list`.
                * **required_marker**
                    Whether a gene was required to reach the minimal number of markers per cell type
                    (:attr:`~ProbesetSelector.n_min_markers`, :attr:`~ProbesetSelector.n_list_markers`).
                * **required_list_marker**
                    Whether a gene was required to reach the minimal number of markers for cell types that only occur in
                    :attr:`~ProbesetSelector.marker_list` but not in :attr:`~ProbesetSelector.adata_celltypes`.
        genes_of_primary_trees:
            The genes of the best tree of each cell type. Available only after calling
            :func:`~ProbesetSelector.select_probeset`. The table contains the following columns:
                * **gene**
                    Gene symbol.
                * **celltype**
                    Cell type in which the tree occurs.
                * **importance**
                    Importance score of the gene for the given cell type.
                * **nr_of_celltypes**
                    Number of primary trees i.e. cell types in which the gene occurs.


    """

    # TODO:
    #  add parameter or remove docstring:
    #  marker_penalties: str, list or dict of strs
    #  proofread docstrings

    def __init__(
        self,
        adata: sc.AnnData,
        celltype_key: str,
        genes_key: str = "highly_variable",
        n: Optional[int] = None,
        preselected_genes: List[str] = [],
        prior_genes: List[str] = [],
        n_pca_genes: int = 100,
        min_mean_difference: float = None,
        n_min_markers: int = 2,
        celltypes: Union[List[str], str] = "all",
        marker_list: Union[str, Dict[str, List[str]]] = None,
        n_list_markers: Union[int, Dict[str, int]] = 2,
        marker_corr_th: float = 0.5,
        pca_penalties: list = [],
        DE_penalties: list = [],
        m_penalties_adata_celltypes: list = [],  # reasonable choice: positive lower and upper threshold
        m_penalties_list_celltypes: list = [],  # reasonable choice: only upper threshold
        pca_selection_hparams: Dict[str, Any] = {},
        DE_selection_hparams: Dict[str, Any] = {"n": 3, "per_group": True},
        forest_hparams: Dict[str, Any] = {"n_trees": 50, "subsample": 1000, "test_subsample": 3000},
        forest_DE_baseline_hparams: Dict[str, Any] = {
            "n_DE": 1,
            "min_score": 0.9,
            "n_stds": 1.0,
            "max_step": 3,
            "min_outlier_dif": 0.02,
            "n_terminal_repeats": 3,
        },
        add_forest_genes_hparams: Dict[str, Any] = {"n_max_per_it": 5, "performance_th": 0.02, "importance_th": 0},
        marker_selection_hparams: Dict[str, Any] = {"penalty_threshold": 1},
        verbosity: int = 2,
        seed: int = 0,
        save_dir: Optional[str] = None,
        n_jobs: int = -1,
    ):
        # TODO
        #  - Should we add a "copy_adata" option? If False we can use less memory, if True adata will be changed at the
        #    end
        #  - add this feature, it means we have at least 3 genes per celltype that we consider as celltype
        #    marker and this time we define markers based on DE & list and min_mean_diff. pca that also occur in DE list
        #    are also markers and the other pca genes not? Might be difficult than to achieve the number
        #    Priorities: best tree genes (pca included), fill up n_min_markers (from 2nd trees), fill up from further trees
        #    without n_min_markers condiseration.
        #    I still think it would be nice to have a better marker criteria, but idk what

        self.adata = adata.copy()
        self.ct_key = celltype_key
        self.g_key = genes_key
        self.n = n
        # To easily access genes on which we run selections define self.genes
        self.genes = self.adata[:, self.adata.var[self.g_key]].var_names if self.g_key else self.adata.var_names
        if prior_genes:
            prior_genes = [g for g in prior_genes if g not in preselected_genes]
        if preselected_genes:
            self.genes.append(
                pd.Index([g for g in preselected_genes if (g not in self.genes) and (g in self.adata.var_names)])
            )
        if prior_genes:
            self.genes.append(
                pd.Index([g for g in prior_genes if (not (g in self.genes)) and (g in self.adata.var_names)])
            )
        self.selection: Dict[str, Any] = {
            "final": None,
            "pre": preselected_genes,
            "prior": prior_genes,
            "pca": None,
            "DE": None,
            "forest_DEs": None,
            "DE_baseline_forest": None,
            "forest": None,
            "marker": None,
        }
        self.n_pca_genes = n_pca_genes
        self.min_mean_difference = min_mean_difference
        self.n_min_markers = n_min_markers
        self.celltypes = self.adata.obs[self.ct_key].unique().tolist() if (celltypes == "all") else celltypes
        # All celltypes in adata
        self.adata_celltypes = self.adata.obs[self.ct_key].unique().tolist()
        # To easily access obs on which we run most selections (all except of pca) define self.obs
        self.obs = self.adata[self.adata.obs[self.ct_key].isin(self.celltypes)].obs_names
        self.marker_list = marker_list
        self.n_list_markers = n_list_markers
        self.marker_corr_th = marker_corr_th
        self.pca_penalties = pca_penalties
        self.DE_penalties = DE_penalties
        self.m_penalties_adata_celltypes = m_penalties_adata_celltypes
        self.m_penalties_list_celltypes = m_penalties_list_celltypes
        # Set hyper parameters of selection steps
        self.pca_selection_hparams = self._get_hparams(pca_selection_hparams, subject="pca_selection")
        self.DE_selection_hparams = self._get_hparams(DE_selection_hparams, subject="DE_selection")
        self.forest_hparams = self._get_hparams(forest_hparams, subject="forest")
        self.forest_DE_baseline_hparams = self._get_hparams(forest_DE_baseline_hparams, subject="forest_DE_basline")
        self.add_forest_genes_hparams = self._get_hparams(add_forest_genes_hparams, subject="add_forest_genes")
        self.m_selection_hparams = self._get_hparams(marker_selection_hparams, subject="marker_selection")

        self.verbosity = verbosity
        self.seed = 0
        # TODO: there are probably a lot of places where the seed needs to be provided that are not captured yet.
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        if "n_jobs" not in self.forest_hparams:
            self.forest_hparams["n_jobs"] = self.n_jobs

        self.forest_results: Dict[
            str,
            Union[
                Union[list, Tuple[list, dict], None, Tuple[Any, Dict[str, Any], Dict[str, Any]]],
                Optional[List[Union[int, Dict[str, pd.DataFrame]]]],
                Union[list, Tuple[list, dict], None],
                Optional[list],
            ],
        ] = {
            "DE_prior_forest": None,
            "DE_baseline_forest": None,
            "pca_prior_forest": None,
            "forest": None,
        }

        self.forest_clfs: Dict[str, Union[dict, list, None]] = {"DE_baseline_forest": None, "forest": None}

        if not (self.n_pca_genes and (self.n_pca_genes > 0)) and isinstance(self.n, int):
            print(
                f"Note: No PCA selection will be performed since n_pca_genes = {self.n_pca_genes}. The selected genes "
                + f"will only be based on the DE forests. In that case it can happen that fewer than n = {self.n} "
                + f"genes are selected. To get n = {self.n} exclude the selected genes of the first run from adata, "
                + "rerun the method, and combine the results of the two runs.\n"
            )

        self.min_test_n = 20

        cts_below_min_test_size, counts_below_min_test_size = ev.get_celltypes_with_too_small_test_sets(
            self.adata[self.obs], self.ct_key, min_test_n=self.min_test_n, split_kwargs={"seed": self.seed, "split": 4}
        )
        if cts_below_min_test_size:
            print(
                "Note: The following celltypes' test set sizes for forest training are below min_test_n "
                + f"(={self.min_test_n}):"
            )
            max_length = max([len(ct) for ct in cts_below_min_test_size])  # TODO: bug fix: type(ct) != str doesnt work.
            for i, ct in enumerate(cts_below_min_test_size):
                print(f"\t {ct:<{max_length}} : {counts_below_min_test_size[i]}")
            print(
                "The genes selected for those cell types potentially don't generalize well. Find the genes for each of "
                + "those cell types in self.genes_of_primary_trees after running self.select_probeset()."
            )

        self.loaded_attributes: list = []
        if self.save_dir:
            self._initialize_file_paths()
            if self.verbosity > 0:
                print(f"Searching for previous results in {self.save_dir}")
            self._load_from_disk()
            self._save_preselected_and_prior_genes()

        # Checks:
        # - check if adata is normalised+log1p

        # Check if genes were preselected to a smaller number (which is recommended)
        if self.adata[:, self.genes].n_vars > 12000:
            tmp_str = "adata[:,adata.var[genes_key]]" if self.g_key else "adata"
            print(f"{tmp_str} contains many genes, consider reducing the number to fewer highly variable genes.")

        # Mean difference constraint
        self._prepare_mean_diff_constraint()

        # Marker list
        self._prepare_marker_list()

        # show progress bars for verbosity levels > 0
        # self.disable_pbars = self.verbosity < 1
        self.progress = util.NestedProgress()  # redirect_stdout=False

    def select_probeset(self) -> None:
        """Run full selection procedure.

        This is the central method of the Spapros selection pipeline. After initializing an instance of the
        :class:`ProbesetEvaluator`, invoke this method to start the selection.

        Examples:
            Minimal example::

                # create an instance of the ProbesetSelector class
                selector = se.ProbesetSelector(adata,
                                            n=50,
                                            celltype_key="celltype",
                                            verbosity=1)
                selector.select_probeset()

        Notes:
            For further examples, see our tutorials: https://spapros.readthedocs.io/en/latest/tutorials.html

        """
        assert isinstance(self.progress, RichCast)
        with self.progress:

            if self.verbosity > 0:
                selection_task = self.progress.add_task(
                    description="SPAPROS PROBESET SELECTION:", only_text=True, header=True, total=0
                )

            # PCA based pre selection
            if self.n_pca_genes and (self.n_pca_genes > 0):
                self._pca_selection()

            # DE forests
            self._forest_DE_baseline_selection()

            # PCA forests (including optimization based on DE forests), or just DE forests if no PCA genes were selected
            if self.n_pca_genes and (self.n_pca_genes > 0):
                self._forest_selection()
            else:
                self._set_DE_baseline_forest_to_final_forest()

            # Add markers from curated list
            if self.marker_list:
                self._marker_selection()

            # Compile probe set
            self.probeset = self._compile_probeset_list()
            self.selection["final"] = self.probeset

            # Save attribute genes_of_primary_trees
            self.genes_of_primary_trees = self._get_genes_of_primary_trees()

            if self.verbosity > 0:
                self.progress.advance(selection_task)
                self.progress.add_task(description="FINISHED\n", footer=True, only_text=True, total=0)

            if self.save_dir:
                self.probeset.to_csv(self.probeset_path)
            # TODO: we haven't included the checks to load the probeset if it already exists

    def _pca_selection(self) -> None:
        """Select genes based on pca loadings."""
        if self.selection["pca"] is None:
            self.selection["pca"] = select.select_pca_genes(
                self.adata[:, self.genes],
                self.n_pca_genes,
                penalty_keys=self.pca_penalties,
                inplace=False,
                progress=self.progress,
                level=1,
                verbosity=self.verbosity,
                **self.pca_selection_hparams,
            )
            assert self.selection["pca"] is not None
            self.selection["pca"] = self.selection["pca"].sort_values("selection_ranking")

            if self.save_dir:
                self.selection["pca"].to_csv(self.selections_paths["pca"])
        else:
            if self.progress and 2 * self.verbosity > 0:
                self.progress.add_task("PCA genes already selected...", only_text=True, level=1)

    def _forest_DE_baseline_selection(self) -> None:
        """Select genes based on forests and differentially expressed genes."""

        if self.progress and self.verbosity > 0:
            baseline_task = self.progress.add_task("Train baseline forest based on DE genes...", total=4, level=1)

        if not isinstance(self.selection["DE"], pd.DataFrame):
            # if self.verbosity > 1:
            #     print("\t Select differentially expressed genes...")
            self.selection["DE"] = select.select_DE_genes(
                self.adata[:, self.genes],
                obs_key=self.ct_key,
                **self.DE_selection_hparams,
                penalty_keys=self.DE_penalties,
                groups=self.celltypes,
                reference="rest",
                rankby_abs=False,
                inplace=False,
                progress=self.progress,
                verbosity=self.verbosity,
                level=2,
            )
            # if self.verbosity > 1:
            #     print("\t\t ...finished.")
            if self.save_dir:
                self.selection["DE"].to_csv(self.selections_paths["DE"])
        else:
            if self.progress and 2 * self.verbosity > 1:
                self.progress.add_task("Differentially expressed genes already selected...", only_text=True, level=2)

        if self.progress and self.verbosity > 0:
            self.progress.advance(baseline_task)

        if not self.forest_results["DE_prior_forest"]:
            # if self.verbosity > 1:
            #     print("\t Train trees on DE selected genes as prior forest for the DE_baseline forest...")
            DE_prior_forest_genes = (
                self.selection["pre"]
                + self.selection["prior"]
                + self.selection["DE"].loc[self.selection["DE"]["selection"]].index.tolist()
            )
            DE_prior_forest_genes = [g for g in np.unique(DE_prior_forest_genes).tolist() if g in self.genes]

            save_DE_prior_forest: Union[str, bool] = (
                self.forest_results_paths["DE_prior_forest"] if self.save_dir else False
            )
            self.forest_results["DE_prior_forest"] = ev.forest_classifications(
                self.adata[:, self.genes],
                DE_prior_forest_genes,
                celltypes=self.celltypes,
                ref_celltypes="all",
                ct_key=self.ct_key,
                save=save_DE_prior_forest,
                seed=0,
                verbosity=self.verbosity,
                progress=self.progress,
                task="Train prior forest for DE_baseline forest...",
                level=2,
                **self.forest_hparams,
            )
        else:
            if self.progress and 2 * self.verbosity > 1:
                self.progress.add_task(
                    "Prior forest for DE_baseline forest already trained...", only_text=True, level=2
                )

        if self.progress and self.verbosity > 0:
            self.progress.advance(baseline_task)

        if not (
            self.forest_results["DE_baseline_forest"]
            and self.forest_clfs["DE_baseline_forest"]
            and isinstance(self.selection["forest_DEs"], pd.DataFrame)
        ):
            save_DE_baseline_forest: Union[str, bool] = (
                self.forest_results_paths["DE_baseline_forest"] if self.save_dir else False
            )
            assert isinstance(self.forest_results["DE_prior_forest"], list)
            de_forest_results = select.add_DE_genes_to_trees(
                self.adata[:, self.genes],
                self.forest_results["DE_prior_forest"],
                ct_key=self.ct_key,
                penalty_keys=self.DE_penalties,
                tree_clf_kwargs=self.forest_hparams,
                verbosity=self.verbosity,
                save=save_DE_baseline_forest,
                return_clfs=True,
                progress=self.progress,
                level=2,
                **self.forest_DE_baseline_hparams,
                baseline_task=baseline_task if (self.progress and self.verbosity > 0) else None,
            )
            assert isinstance(de_forest_results, tuple)
            self.forest_results["DE_baseline_forest"] = de_forest_results[0]
            self.forest_clfs["DE_baseline_forest"] = de_forest_results[1]
            assert len(de_forest_results) == 3
            de_forest_results = cast(Tuple[list, pd.DataFrame, pd.DataFrame], de_forest_results)
            self.selection["forest_DEs"] = de_forest_results[2]

            if self.save_dir:
                self.selection["forest_DEs"].to_csv(self.selections_paths["forest_DEs"])
                with open(self.forest_clfs_paths["DE_baseline_forest"], "wb") as f:
                    pickle.dump(self.forest_clfs["DE_baseline_forest"], f)
        else:
            if self.progress and 2 * self.verbosity > 1:
                self.progress.add_task("DE genes already added...", only_text=True, level=2)
                self.progress.add_task("DE_baseline forest already trained...", only_text=True, level=2)
                if self.verbosity > 0 and baseline_task:
                    self.progress.advance(baseline_task)

        if self.progress and self.verbosity > 0:
            self.progress.advance(baseline_task)

        # might be interesting for utility plots:
        if not isinstance(self.selection["DE_baseline_forest"], pd.DataFrame):
            assert isinstance(self.forest_results["DE_baseline_forest"], list)
            assert isinstance(self.forest_results["DE_baseline_forest"][2], dict)
            self.selection["DE_baseline_forest"] = ev.forest_rank_table(self.forest_results["DE_baseline_forest"][2])
            if self.save_dir:
                self.selection["DE_baseline_forest"].to_csv(self.selections_paths["DE_baseline_forest"])

    def _forest_selection(self) -> None:
        """Select genes based on forests and differentially expressed genes."""
        # TODO
        #  - eventually put this "test set size"-test somewhere (ideally at __init__)

        if self.progress and self.verbosity > 0:
            final_forest_task = self.progress.add_task("Train final forests...", total=3, level=1)

        if not self.forest_results["pca_prior_forest"]:
            pca_prior_forest_genes = (
                self.selection["pre"]
                + self.selection["prior"]
                + self.selection["pca"].loc[self.selection["pca"]["selection"]].index.tolist()
            )
            pca_prior_forest_genes = [g for g in np.unique(pca_prior_forest_genes).tolist() if g in self.genes]
            save_pca_prior_forest: Union[str, bool] = (
                self.forest_results_paths["pca_prior_forest"] if self.save_dir else False
            )
            self.forest_results["pca_prior_forest"] = ev.forest_classifications(
                self.adata[:, self.genes],
                pca_prior_forest_genes,
                celltypes=self.celltypes,
                ref_celltypes="all",
                ct_key=self.ct_key,
                save=save_pca_prior_forest,
                seed=self.seed,  # TODO!!! same seeds for all forests!
                verbosity=self.verbosity,
                progress=self.progress,
                level=2,
                task="Train forest on pre/prior/pca selected genes...",
                **self.forest_hparams,
            )
            # if self.verbosity > 2:
            #     print("\t\t ...finished.")
        else:
            if self.progress and 2 * self.verbosity >= 2:
                self.progress.add_task(
                    "Forest on pre/prior/pca selected genes already trained...", only_text=True, level=2
                )

        if self.progress and self.verbosity > 0:
            self.progress.advance(final_forest_task)
        # if self.verbosity > 1:
        #     print("\t Iteratively add genes from DE_baseline_forest...")
        if (not self.forest_results["forest"]) or (not self.forest_clfs["forest"]):
            save_forest: Union[str, bool] = self.forest_results_paths["forest"] if self.save_dir else False
            assert isinstance(self.forest_results["pca_prior_forest"], list)
            assert isinstance(self.forest_results["DE_baseline_forest"], list)
            forest_results = select.add_tree_genes_from_reference_trees(
                self.adata[:, self.genes],
                self.forest_results["pca_prior_forest"],
                self.forest_results["DE_baseline_forest"],
                ct_key=self.ct_key,
                ref_celltypes="all",
                **self.add_forest_genes_hparams,
                tree_clf_kwargs=self.forest_hparams,
                verbosity=self.verbosity,
                save=save_forest,
                return_clfs=True,
                final_forest_task=final_forest_task if (self.progress and self.verbosity > 0) else None,
                progress=self.progress,
                level=2,
                task="Iteratively add genes from DE_baseline_forest...",
            )
            assert isinstance(forest_results[0], list)
            # assert isinstance(forest_results[1], dict)
            assert len(forest_results) == 2
            self.forest_results["forest"] = forest_results[0]
            self.forest_clfs["forest"] = forest_results[1]
            if self.save_dir:
                with open(self.forest_clfs_paths["forest"], "wb") as f:
                    pickle.dump(self.forest_clfs["forest"], f)
            # if self.verbosity > 2:
            #     print("\t\t ...finished.")
        else:
            if self.progress and 2 * self.verbosity >= 2:
                self.progress.add_task("Genes from DE_baseline_forest were already added...", only_text=True, level=2)
                self.progress.add_task("Final forest was alread trained...", only_text=True, level=2)
                self.progress.advance(final_forest_task)

        if self.progress and self.verbosity > 0:
            self.progress.advance(final_forest_task)

        if not isinstance(self.selection["forest"], pd.DataFrame):
            assert isinstance(self.forest_results["forest"], list)
            assert len(self.forest_results["forest"]) == 3
            assert isinstance(self.forest_results["forest"][2], dict)
            self.selection["forest"] = ev.forest_rank_table(self.forest_results["forest"][2])
            if self.save_dir:
                self.selection["forest"].to_csv(self.selections_paths["forest"])

    def _set_DE_baseline_forest_to_final_forest(self):
        """Set the results of the DE forests as the final forest results.

        This method is applied when no PCA genes were selected and therefore no forests were trained on PCA genes.
        """

        self.forest_results["forest"] = self.forest_results["DE_baseline_forest"]
        self.forest_clfs["forest"] = self.forest_clfs["DE_baseline_forest"]
        self.selection["forest"] = self.selection["DE_baseline_forest"]

    def _save_preselected_and_prior_genes(self):
        """Save pre selected and prior genes to files."""
        for s in ["pre", "prior"]:
            if self.selection[s] and not (f"selection_{s}" in self.loaded_attributes):
                with open(self.selections_paths[s], "w") as file:
                    json.dump(self.selection[s], file)

    def _prepare_marker_list(self) -> None:
        """Process marker list if not loaded and save to file."""
        # TODO Eventually reformat n_list_markers
        if self.marker_list and isinstance(self.n_list_markers, int):
            self.n_list_markers = {"adata_celltypes": self.n_list_markers, "list_celltypes": self.n_list_markers}

        # Check and process marker list
        if self.marker_list and not ("marker_list" in self.loaded_attributes):
            # Eventually load marker list
            if isinstance(self.marker_list, str):
                self.marker_list = pd.read_csv(self.marker_list, index_col=0)
                self.marker_list = dict_to_table(self.marker_list, genes_as_index=False, reverse=True)

            # Filter out genes from marker list
            self._filter_and_save_marker_list()

            # Check number of markers per celltype
            if self.verbosity > 1:
                self._check_number_of_markers_per_celltype()

    def _filter_and_save_marker_list(self) -> None:
        """Get marker list in shape if necessary.

        The following steps are applied to the marker list:

        - Filter out markers that occur multiple times
        - Filter out genes based on penalties (self.marker_penalties)
        - Warn the user if markers don't occur in adata and can't be tested on penalties

        Returns:
            Modifies self.marker_list
        """
        # Filter genes that occur multiple times # TODO: if a genes occurs multiple times for the same celltype we
        # should actually keep it!...
        assert isinstance(self.marker_list, dict)
        self.marker_list = filter_marker_dict_by_shared_genes(self.marker_list, verbose=(self.verbosity > 1))

        # filter genes based on penalties
        marker_list_adata_cts = {ct: genes for ct, genes in self.marker_list.items() if ct in self.adata_celltypes}
        marker_list_other_cts = {
            ct: genes for ct, genes in self.marker_list.items() if not (ct in self.adata_celltypes)
        }
        m_list_adata_cts_dropped: dict = {}
        m_list_other_cts_dropped: dict = {}
        if self.m_penalties_adata_celltypes:
            if marker_list_adata_cts:
                if self.verbosity > 0:
                    print("Filter marker dict by penalties for markers of celltypes in adata")
                filtered_marker_tuple = filter_marker_dict_by_penalty(
                    marker_list_adata_cts,
                    self.adata,
                    self.m_penalties_adata_celltypes,
                    threshold=self.m_selection_hparams["penalty_threshold"],
                    verbose=(self.verbosity > 1),
                    return_filtered=True,
                )
                assert isinstance(filtered_marker_tuple, tuple)
                marker_list_adata_cts, m_list_adata_cts_dropped = filtered_marker_tuple
        if self.m_penalties_list_celltypes:
            if marker_list_other_cts:
                if self.verbosity > 0:
                    print("Filter marker dict by penalties for markers of celltypes not in adata")
                filtered_other_marker_tuple = filter_marker_dict_by_penalty(
                    marker_list_other_cts,
                    self.adata,
                    self.m_penalties_list_celltypes,
                    threshold=self.m_selection_hparams["penalty_threshold"],
                    verbose=(self.verbosity > 1),
                    return_filtered=True,
                )
                assert isinstance(filtered_other_marker_tuple, tuple)
                marker_list_other_cts, m_list_other_cts_dropped = filtered_other_marker_tuple

        # Combine differently penalized markers to filtered marker dict and out filtered marker dict
        self.marker_list = dict(marker_list_adata_cts, **marker_list_other_cts)
        self.marker_list_filtered_out = dict(m_list_adata_cts_dropped, **m_list_other_cts_dropped)

        # Save marker list
        if self.save_dir:
            with open(self.marker_list_path, "w") as file:
                json.dump([self.marker_list, self.marker_list_filtered_out], file)

    def _check_number_of_markers_per_celltype(self) -> None:
        """Check if given markers per celltype are below the number of marker we want to add eventually."""
        low_gene_numbers = {}
        assert isinstance(self.n_list_markers, dict)
        assert isinstance(self.marker_list, dict)
        for ct, genes in self.marker_list.items():
            if (ct in self.adata_celltypes) and (len(genes) < self.n_list_markers["adata_celltypes"]):
                low_gene_numbers[ct] = [len(genes), self.n_list_markers["adata_celltypes"]]
            elif len(genes) < self.n_list_markers["list_celltypes"]:
                low_gene_numbers[ct] = [len(genes), self.n_list_markers["list_celltypes"]]
        if low_gene_numbers:
            max_length = max([len(ct) for ct in low_gene_numbers])
            print(
                "The following celltypes in the marker list have lower numbers of markers than given by the minimal "
                "selection criteria:"
            )
            for ct in low_gene_numbers:
                print(f"\t {ct:<{max_length}}: {low_gene_numbers[ct][0]} (expected: {low_gene_numbers[ct][1]})")

    def _marker_selection(self) -> None:
        """Select genes from marker list based on correlations with already selected genes."""
        if self.progress and self.verbosity > 0 and self.marker_list:
            marker_task = self.progress.add_task("Marker selection...", total=len(self.marker_list), level=1)
        pre_pros = self._compile_probeset_list(with_markers_from_list=False)

        # Check which genes are already selected
        selected_genes = pre_pros.loc[pre_pros["selection"]].index.tolist()
        assert isinstance(self.marker_list, dict)
        marker_list_genes = [g for _, genes in self.marker_list.items() for g in genes]

        # Compute correlation matrix for potentially selected markers and genes in the selection
        genes = np.unique(marker_list_genes + selected_genes).tolist()
        cor_mat = util.correlation_matrix(
            self.adata, genes=genes, absolute=False, diag_zero=True, unknown_genes_to_zero=True
        )

        # Initialize marker selection dataframe
        self.selection["marker"] = pd.DataFrame(
            index=pre_pros.index, data={"selection": False, "celltype": ""}
        )  # ,'marker_rank':np.nan})

        # Select genes
        for ct, genes in self.marker_list.items():
            if ct in self.celltypes:
                # selected_markers = [g for g in selected_genes if (ct in probeset.loc[g,'celltypes_DE'].split(','))]
                # max_rank_of_ct = prepros.loc[selected_genes,'marker_rank']
                # Get correlated genes (they are considered as markers) and the necessary missing numbers of markers
                assert isinstance(self.n_list_markers, dict)
                marker, cor_genes = select.get_markers_and_correlated_genes(
                    cor_mat, genes, selected_genes, n_min=self.n_list_markers["adata_celltypes"], th=self.marker_corr_th
                )
                self.selection["marker"].loc[marker + cor_genes, ["selection", "celltype"]] = [True, ct]
                # print("MARKER SELECTION: ",ct,marker+cor_genes)
                # add_genes = [g for g in marker if g in selected_genes] + cor_genes + [g for g in marker if not (g in\
                # selected_genes)]
                # for i,g in enumerate(add_genes):
                #    #marker rank ist der vorhandene und ansonsten der maximal vorhandene + i
                #    #self.selection['marker'].loc[g,['selection','celltype','marker_rank']] = [True,ct,i+1]
            else:
                # just add the necessary number
                marker = [g for g in selected_genes if g in genes] + [g for g in genes if not (g in selected_genes)]
                assert isinstance(self.n_list_markers, dict)
                marker = genes[: min([len(genes), self.n_list_markers["list_celltypes"]])]
                self.selection["marker"].loc[marker, ["selection", "celltype"]] = [True, ct]
                # print("MARKER SELECTION (list only): ",ct,marker)
                # for i,g in enumerate(marker):
                #    self.selection['marker'].loc[g,['selection','celltype','marker_rank']] = [True,ct,i+1]
            if self.progress and self.verbosity > 0:
                self.progress.advance(marker_task)

    def _compile_probeset_list(self, with_markers_from_list: bool = True) -> pd.DataFrame:
        """Compile the probeset list.

        Args:
            with_markers_from_list:
                Indikate all markers from the marker list.

        Returns:
            pd.DataFrame:
                compiled probeset list

        """

        # TODO
        #  How/where to add genes from marker_list that are not in adata?
        #  --> Oh btw, same with pre and prior selected genes.

        if self.progress and self.verbosity > 0 and with_markers_from_list:
            list_task = self.progress.add_task("Compile probeset list...", total=1, level=1)

        # Initialize probeset table
        index = self.genes.tolist()
        # Add marker genes that are not in adata
        if self.marker_list:
            assert isinstance(self.marker_list, dict)
            index = np.unique(index + [g for ct, genes in self.marker_list.items() for g in genes]).tolist()
        # Add pre selected genes that are not in adata
        if self.selection["pre"]:
            index = np.unique(index + self.selection["pre"]).tolist()
        # Init table
        probeset = pd.DataFrame(index=index)
        # Get rank and importance score from last trained forest
        probeset = pd.concat([probeset, self.selection["forest"][["rank", "importance_score"]]], axis=1)
        probeset.rename(columns={"rank": "tree_rank"}, inplace=True)
        # Indicator if gene is within pre selected set
        probeset["pre_selected"] = False
        probeset.loc[self.selection["pre"], "pre_selected"] = True
        # Indicator if gene is within selected set of prior genes
        probeset["prior_selected"] = False
        probeset.loc[self.selection["prior"], "prior_selected"] = True
        # Indicator if gene was in the prior selected pca set and all genes' pca scores
        probeset["pca_selected"] = False
        probeset["pca_score"] = 0
        if self.n_pca_genes and (self.n_pca_genes > 0):
            probeset.loc[self.selection["pca"][self.selection["pca"]["selection"]].index, "pca_selected"] = True
            probeset.loc[self.selection["pca"].index, "pca_score"] = self.selection["pca"]["selection_score"]

        # get celltypes of the 1-vs-all DE tests
        tmp_cts = [ct for ct in self.celltypes if ct in self.selection["DE"].columns]
        df_tmp = self.selection["DE"].loc[self.genes, tmp_cts].copy()
        # df_tmp = self.selection['DE'][,tmp_cts].copy() # TODO: use this one later, atm I have an old versioned file
        # that I load with a longer index
        df_tmp["celltypes_DE_1vsall"] = df_tmp.apply(
            lambda row: ",".join([tmp_cts[i] for i, v in enumerate(row) if v > 0]), axis=1
        )
        probeset["celltypes_DE_1vsall"] = ""
        probeset.loc[df_tmp.index, "celltypes_DE_1vsall"] = df_tmp["celltypes_DE_1vsall"]

        # get celltypes of the 1-vs-hard_to_distinguish_references DE tests (used when building the DE_baseline_forest)
        tmp_cts = [ct for ct in self.celltypes if ct in self.selection["forest_DEs"].columns]
        df_tmp = self.selection["forest_DEs"][tmp_cts].copy()
        df_tmp["celltypes_DE_specific"] = df_tmp.apply(
            lambda row: ",".join([tmp_cts[i] for i, v in enumerate(row) if v > 0]), axis=1
        )
        probeset["celltypes_DE_specific"] = ""
        probeset.loc[df_tmp.index, "celltypes_DE_specific"] = df_tmp["celltypes_DE_specific"]

        # The following works since genes are either in group 1vsall or specific (or none of them)
        probeset["celltypes_DE"] = probeset["celltypes_DE_1vsall"] + probeset["celltypes_DE_specific"]

        # Indicate all marker from the marker_list
        if with_markers_from_list:

            def str_combiner(a, b):
                return (a or "") + ("," if (a and b) else "") + (b or "")

            probeset["celltypes_marker"] = ""
            if isinstance(self.selection["marker"], pd.DataFrame) or self.selection["marker"]:  # TODO check this
                probeset["celltypes_marker"] = probeset["celltypes_DE"].combine(
                    self.selection["marker"]["celltype"], str_combiner
                )
            else:
                probeset["celltypes_marker"] = probeset["celltypes_DE"]

        probeset = probeset.sort_values(
            ["pre_selected", "tree_rank", "importance_score"], ascending=[False, True, False]
        )

        # rank = preselected, tree rank1, missing markers list cts, mms list&data cts, mms data cts, other tree ranks

        # Now add the number of markers that is required:
        # Celltypes are divided in three groups
        # 1. Celltypes in marker list only
        # 2. Celltypes in dataset and marker list
        # 3. Celltypes in dataset only
        # (We consider the priority order to be (1.), (2. & 3.) for the min marker requirements)

        # The final ranking will be as follows:
        # 1. preselected genes (optional)
        # 2. tree_rank = 1 genes
        # 3. min markers for list only celltypes
        # 4. min markers for other celltypes

        if self.marker_list:
            adata_only_cts = [ct for ct in self.celltypes if ct not in self.marker_list]
            list_only_cts = [ct for ct in self.marker_list if ct not in self.celltypes]
            shared_cts = [ct for ct in self.celltypes if ct not in adata_only_cts]
        else:
            adata_only_cts = [ct for ct in self.celltypes]
            list_only_cts = []
            shared_cts = []

        probeset["list_only_ct_marker"] = False
        probeset["required_marker"] = False
        probeset["required_list_marker"] = False
        probeset["marker_rank"] = np.nan
        if with_markers_from_list:
            # display(self.selection['marker'])
            # marker_selection_ordered = self.selection['marker'].loc[probeset.index]
            # display(marker_selection_ordered)
            # display(probeset)
            for ct in list_only_cts:
                gene_idxs = [
                    g
                    for g, is_marker in (
                        (self.selection["marker"]["celltype"] == ct) & (self.selection["marker"]["selection"])
                    ).items()
                    if is_marker
                ]
                gene_idxs += [g for g, cts in probeset["celltypes_marker"].str.split(",").items() if ct in cts]
                assert isinstance(self.n_list_markers, dict)
                n_min_markers = max([self.n_min_markers, self.n_list_markers["list_celltypes"]])
                gene_idxs = gene_idxs[: min([len(gene_idxs), n_min_markers])]
                # print("COMPILE LIST: ",ct, probeset.loc[gene_idxs])
                # display(self.selection['marker'].loc[gene_idxs])
                for i, g in enumerate(gene_idxs):
                    tmp_rank = np.nanmin([i + 1, probeset.loc[g]["marker_rank"]])
                    probeset.loc[g, ["list_only_ct_marker", "required_marker", "marker_rank"]] = [True, True, tmp_rank]
            for ct in self.celltypes:
                if isinstance(self.selection["marker"], pd.DataFrame) or self.selection["marker"]:  # TODO check this
                    gene_idxs = [
                        g
                        for g, is_marker in (
                            (self.selection["marker"]["celltype"] == ct) & (self.selection["marker"]["selection"])
                        ).items()
                        if is_marker
                    ]
                else:
                    gene_idxs = []
                gene_idxs += [g for g, cts in probeset["celltypes_marker"].str.split(",").items() if ct in cts]
                if ct in shared_cts:
                    assert isinstance(self.n_list_markers, dict)
                    n_min_markers = max([self.n_min_markers, self.n_list_markers["adata_celltypes"]])
                else:
                    assert isinstance(self.n_min_markers, int)
                    n_min_markers = self.n_min_markers
                gene_idxs = gene_idxs[: min([len(gene_idxs), n_min_markers])]
                for i, g in enumerate(gene_idxs):
                    tmp_rank = np.nanmin([i + 1, probeset.loc[g]["marker_rank"]])
                    probeset.loc[g, ["required_marker", "marker_rank"]] = [True, tmp_rank]
        else:
            # Annotate genes that are required to fulfill the n_min_marker constraint
            # we also assign a marker_rank: if a gene is a marker for several celltypes is gets the lower rank
            for ct in self.celltypes:
                # For the given cell type get all genes that are markers
                gene_idxs = [g for g, cts in probeset["celltypes_DE"].str.split(",").items() if ct in cts]
                assert isinstance(self.n_list_markers, dict)
                if ct in shared_cts:
                    assert isinstance(self.n_list_markers, dict)
                    n_min_markers = max([self.n_min_markers, self.n_list_markers["adata_celltypes"]])
                else:
                    assert isinstance(self.n_min_markers, int)
                    n_min_markers = self.n_min_markers
                gene_idxs = gene_idxs[: min([len(gene_idxs), n_min_markers])]
                for i, g in enumerate(gene_idxs):
                    # Assign the rank according index of gene in gene_idxs in case there isn't already a lower rank
                    # assigned
                    tmp_rank = np.nanmin([i + 1, probeset.loc[g]["marker_rank"]])
                    probeset.loc[g, ["required_marker", "marker_rank"]] = [True, tmp_rank]

        # final ordering
        probeset["rank"] = np.nan
        probeset.loc[probeset["pre_selected"], "rank"] = 1
        probeset.loc[(probeset["tree_rank"] == 1) & probeset["rank"].isnull(), "rank"] = 2
        probeset.loc[probeset["list_only_ct_marker"] & probeset["rank"].isnull(), "rank"] = 3
        required_markers = probeset["required_marker"] & probeset["rank"].isnull()
        probeset.loc[required_markers, "rank"] = probeset.loc[required_markers, "marker_rank"] + 3
        probeset["rank"] = probeset["rank"].rank(method="dense")
        max_rank = probeset["rank"].max()
        other_tree_genes = probeset["rank"].isnull() & ~probeset["tree_rank"].isnull()
        probeset.loc[other_tree_genes, "rank"] = (
            probeset.loc[other_tree_genes, "tree_rank"] + max_rank - 1
        )  # missing genes have tree_rank >= 2
        probeset = probeset.sort_values(
            ["rank", "marker_rank", "importance_score", "pca_score"], ascending=[True, True, False, False]
        )
        probeset["rank"] = probeset["rank"].rank(method="dense")
        probeset["gene_nr"] = [i for i in range(1, len(probeset) + 1)]
        probeset["selection"] = False
        if not self.n:
            probeset.loc[probeset["rank"] <= max_rank, "selection"] = True
        else:
            probeset.loc[probeset["gene_nr"] <= self.n, "selection"] = True

        # reorder columns
        first_cols = ["gene_nr", "selection", "rank", "marker_rank", "tree_rank", "importance_score", "pca_score"]
        other_cols = [col for col in probeset.columns if col not in first_cols]
        cols = first_cols + other_cols

        # Throw out genes that dont have any scoring or ranking but were still selected (this can only happen if no PCA
        # selection was performed.)
        no_rank_genes = probeset["rank"].isnull() & (probeset["pca_score"] == 0)
        probeset.loc[no_rank_genes, "selection"] = False

        if self.progress and self.verbosity > 0 and with_markers_from_list:
            self.progress.advance(list_task)
            self.progress.advance(list_task)

        return probeset[cols].copy()

    def _get_genes_of_primary_trees(self) -> pd.DataFrame:
        """Get genes of the best trees of each cell type

        Returns
            pd.DataFrame

        """

        genes = []
        cts = []
        importances = []
        for ct in self.forest_results["forest"][2].keys():
            tmp = self.forest_results["forest"][2][ct]["0"]
            tmp = tmp.loc[tmp > 0].sort_values(ascending=False)
            genes += tmp.index.to_list()
            cts += [ct for _ in range(len(tmp))]
            importances += tmp.to_list()

        df = pd.DataFrame(data={"gene": genes, "celltype": cts, "importance": importances})
        nr_of_cts = df["gene"].value_counts().to_dict()
        df["nr_of_celltypes"] = df["gene"].apply(lambda g: nr_of_cts[g])

        return df

    def _prepare_mean_diff_constraint(self) -> None:
        """Compute if mean difference constraint is fullfilled."""
        if self.min_mean_difference:
            # Check if already loaded
            if not ("mean_diff_constraint" in self.loaded_attributes):  # not 'mean_diff_constraint' in self.__dict__:
                if self.verbosity > 0:
                    print(
                        f"Compute mean difference constraint (mean of celltype - mean of background > "
                        f"{self.min_mean_difference})"
                    )
                self.mean_diff_constraint = pd.DataFrame(
                    index=self.adata.var_names, columns=self.celltypes, dtype="bool"
                )
                for ct in self.celltypes:  # self._tqdm(self.celltypes):
                    mean_diffs = (
                        util.marker_mean_difference(self.adata[self.obs], ct, ct_key=self.ct_key, genes="all")
                        > self.min_mean_difference
                    )
                    self.mean_diff_constraint[ct] = mean_diffs
                if self.save_dir:
                    self.mean_diff_constraint.to_csv(self.mean_diff_constraint_path)

        # In case constraint was loaded or recomputed add a new penalty to adata
        if ("mean_diff_constraint" in self.loaded_attributes) or self.min_mean_difference:
            self.adata.var["mean_diff_constraint"] = self.mean_diff_constraint.any(axis=1)
            self.adata.var["mean_diff_constraint"] = self.adata.var["mean_diff_constraint"].astype(int)

            if self.verbosity > 0:
                print(
                    "Add mean difference constraint to penalties: pca_penalties, DE_penalties, and "
                    "m_penalties_adata_celltypes."
                )
            self.pca_penalties += ["mean_diff_constraint"]
            self.DE_penalties += ["mean_diff_constraint"]
            self.m_penalties_adata_celltypes += ["mean_diff_constraint"]

    def _get_hparams(self, new_params: dict, subject: str = "DE_selection") -> dict:
        """Add missing default parameters to dictionary in case they are not given.

        Example:
            The ``forest_hparams`` are given in the class init definition as ``{"n_trees": 50, "subsample": 1000,
            "test_subsample": 3000}``. If the class is called with ``forest_hparams={"n_trees": 100}`` we would actually
            like to have ``{"n_trees": 100, "subsample": 1000, "test_subsample": 3000}``.
            The last two are added by this functions.

        Args:
            new_params:
                Dictionary with custom hyperparameters for the selection method :attr:`subject`.
            subject:
                Type of hyper parameters of interest.

        Returns:
            dict:
                Custom hyperparameters updated with default values for the remaining parameters.
        """
        params = new_params.copy()

        # Here we define default values
        if subject == "pca_selection":
            defaults: Dict[str, Any] = {}
        elif subject == "DE_selection":
            defaults = {"n": 3, "per_group": True}
        elif subject == "forest":
            defaults = {"n_trees": 50, "subsample": 1000, "test_subsample": 3000}
        elif subject == "forest_DE_basline":
            defaults = {
                "n_DE": 1,
                "min_score": 0.9,
                "n_stds": 1.0,
                "max_step": 3,
                "min_outlier_dif": 0.02,
                "n_terminal_repeats": 3,
            }
        elif subject == "add_forest_genes":
            defaults = {"n_max_per_it": 5, "performance_th": 0.02, "importance_th": 0}
        elif subject == "marker_selection":
            defaults = {"penalty_threshold": 1}

        # Set parameters with default values if they weren't provided
        params.update({k: v for k, v in defaults.items() if k not in params})

        # Force some parameters to certain defaults
        if subject == "forest":
            params["return_clfs"] = False  # this only applies to intermediate forests. Final forest clfs are returned.

        return params

    def _initialize_file_paths(self) -> None:
        """Initialize path variables and set up folder hierarchy.

        Call this function in the initialization to define all file names that are eventually saved.
        This function also aims to have all possibly generated file names organised in one place.
        """
        # Create base directory
        assert self.save_dir is not None
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # marker list
        marker_list_dir = os.path.join(self.save_dir, "marker_list")
        Path(marker_list_dir).mkdir(parents=True, exist_ok=True)
        self.marker_list_path = os.path.join(marker_list_dir, "marker_list.txt")

        # mean difference constraint
        mean_diff_constraint_dir = os.path.join(self.save_dir, "mean_diff_constraint")
        Path(mean_diff_constraint_dir).mkdir(parents=True, exist_ok=True)
        self.mean_diff_constraint_path = os.path.join(mean_diff_constraint_dir, "mean_diff_constraint.csv")

        # selections
        selections_dir = os.path.join(self.save_dir, "selections")
        Path(selections_dir).mkdir(parents=True, exist_ok=True)
        self.selections_paths = {}
        self.selections_paths["pre"] = os.path.join(selections_dir, "pre_selected.txt")
        self.selections_paths["prior"] = os.path.join(selections_dir, "prior_genes.txt")
        self.selections_paths["pca"] = os.path.join(selections_dir, "pca_selected.csv")
        self.selections_paths["DE"] = os.path.join(selections_dir, "DE_selected.csv")
        self.selections_paths["forest_DEs"] = os.path.join(selections_dir, "forest_DEs_selected.csv")
        self.selections_paths["DE_baseline_forest"] = os.path.join(selections_dir, "DE_baseline_forest_selected.csv")
        self.selections_paths["forest"] = os.path.join(selections_dir, "forest_selected.csv")
        self.selections_paths["marker"] = os.path.join(selections_dir, "marker_list_selection.csv")
        self.selections_paths["final"] = os.path.join(self.save_dir, "probeset.csv")

        # forest results, and for final forest: sklearn tree class instances
        forest_dir = os.path.join(self.save_dir, "trees")
        Path(forest_dir).mkdir(parents=True, exist_ok=True)
        self.forest_results_paths = {}
        self.forest_results_paths["DE_prior_forest"] = os.path.join(forest_dir, "DE_prior_forest_results.pkl")
        self.forest_results_paths["DE_baseline_forest"] = os.path.join(forest_dir, "DE_baseline_forest_results.pkl")
        self.forest_results_paths["pca_prior_forest"] = os.path.join(forest_dir, "pca_prior_forest_results.pkl")
        self.forest_results_paths["forest"] = os.path.join(forest_dir, "forest_results.pkl")
        # tree class objects of final forest
        self.forest_clfs_paths = {}
        self.forest_clfs_paths["DE_baseline_forest"] = os.path.join(forest_dir, "DE_baseline_forest_clfs.pkl")
        self.forest_clfs_paths["forest"] = os.path.join(forest_dir, "forest_clfs.pkl")

        # Final probeset result
        self.probeset_path = os.path.join(self.save_dir, "probeset.csv")

    def _load_from_disk(self) -> None:
        """Load existing files into variables."""

        # marker list
        if os.path.exists(self.marker_list_path):
            with open(self.marker_list_path, "r") as file:
                self.marker_list, self.marker_list_filtered_out = json.load(file)
            if self.verbosity > 1:
                print(f"\t Found and load {os.path.basename(self.marker_list_path)} (filtered marker list).")
            self.loaded_attributes.append("marker_list")

        # mean difference constraint
        if os.path.exists(self.mean_diff_constraint_path):
            self.mean_diff_constraint = pd.read_csv(self.mean_diff_constraint_path, index_col=0)
            if self.verbosity > 1:
                print(
                    f"\t Found and load {os.path.basename(self.mean_diff_constraint_path)} (mean difference "
                    "constraint table)."
                )
            self.loaded_attributes.append("mean_diff_constraint")

        # selections
        for s in ["pre", "prior"]:
            if os.path.exists(self.selections_paths[s]):
                with open(self.selections_paths[s], "r") as file:
                    self.selection[s] = json.load(file)
                if self.verbosity > 1:
                    print(f"\t Found and load {os.path.basename(self.selections_paths[s])} (selection results).")
                self.loaded_attributes.append(f"selection_{s}")
        for s in ["pca", "DE", "forest_DEs", "DE_baseline_forest", "forest", "marker", "final"]:
            if os.path.exists(self.selections_paths[s]):
                self.selection[s] = pd.read_csv(self.selections_paths[s], index_col=0)
                if self.verbosity > 1:
                    print(f"\t Found and load {os.path.basename(self.selections_paths[s])} (selection results).")
                self.loaded_attributes.append(f"selection_{s}")

        # forest
        for f in ["DE_prior_forest", "DE_baseline_forest", "pca_prior_forest", "forest"]:
            if os.path.exists(self.forest_results_paths[f]):
                self.forest_results[f] = pickle.load(open(self.forest_results_paths[f], "rb"))
                if self.verbosity > 1:
                    print(
                        f"\t Found and load {os.path.basename(self.forest_results_paths[f])} (forest training results)."
                    )
                self.loaded_attributes.append(f"forest_results_{f}")
        for f in ["DE_baseline_forest", "forest"]:
            if os.path.exists(self.forest_clfs_paths[f]):
                self.forest_clfs[f] = pickle.load(open(self.forest_clfs_paths[f], "rb"))
                if self.verbosity > 1:
                    print(
                        f"\t Found and load {os.path.basename(self.forest_clfs_paths[f])} (forest classifier objects)."
                    )
                self.loaded_attributes.append(f"forest_clfs_{f}")

    # def _tqdm(self, iterator):
    #     """Wrapper for tqdm with verbose condition."""
    #     return tqdm(iterator) if self.verbosity >= 1 else iterator

    ############
    # plotting #
    ############

    def plot_histogram(
        self,
        x_axis_keys: Dict[str, str] = None,
        selections: List[Literal["pca", "DE", "marker"]] = None,
        penalty_keys: Dict = None,
        unapplied_penalty_keys: Dict = None,
        background_key: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        """Plot histograms of (basic) selections under given penalties.

        The full selection procedure consists of steps partially based on basic score based selection procedures.
        This is an interactive plotting function to investigate if the constructed penalty kernels are well chosen.

        Note:
            The green line shows a linear interpolation of the penalty scores which is only an approximation of the
            penalty function.

        Args:
            x_axis_keys:
                Dictionary with the values from ``penaly_keys`` as keys and the column name of ``adata.var`` that was
                used to calcuate the respective penalty factors.
            selections:
                Plot the histograms of selections based on

                    - `'pca'` : pca loadings based selection of prior genes
                    - `'DE'` : genes selected based on differential expressed which are used as the
                      `'forest_DE_baseline'`
                    - `'marker'` : genes from the marker list

            penalty_keys:
                Dictionary with the selection method name as key and the name of the column for ``adata.var`` containing
                the penalty factors to plot. Selections that are not in ``penalty_keys`` are plottet without penalty.
                If `None` (default), we use the columns defined in the selector for each selection. The respective
                attributes of the selector are:

                    - `'pca'`: :attr:`pca_penalties`
                    - `'DE'`: :attr:`DE_penalties`
                    - `'marker'`: :attr:`m_penalties_adata_celltypes`

            unapplied_penalty_keys:
                Same as ``penalty_keys`` but for penalties that were not applied to the selection.
            background_key:
                Key in ``adata.var`` for preselected genes (typically `'highly_variable_genes'`) to plot as background
                histogram . If `True` (default), :attr:`g_key` is used. If `None` no background is plottet. If `'all'`, all genes
                are used as background.
            kwargs:
                Further arguments for :func:`.selection_histogram`.

        Returns:
            Figure can be showed (default `True`) and stored to path (default `None`).
            Change this with `show` and `save` in ``kwargs``.
        """
        # TODO:
        #  - Describe in docstring where the penalties are defined
        #  - Atm I think I won't include the penalt kernels into the class therefore this plotting function simply plots
        #    histograms - a little boring but better than nothing)

        SELECTIONS = ["pca", "DE", "marker"]
        PENALTY_KEYS = {
            "pca": self.pca_penalties,
            "DE": self.DE_penalties,
            "marker": self.m_penalties_adata_celltypes + self.m_penalties_list_celltypes,
        }
        UNAPPLIED_PENALTY_KEYS = {p_key: ["expression_penalty"] for p_key in PENALTY_KEYS}
        X_AXIS_KEYS = {
            "expression_penalty": "quantile_0.99",
            "expression_penalty_upper": "quantile_0.99",
            "expression_penalty_lower": "quantile_0.9 expr > 0",
            "marker": "quantile_0.99",
        }

        if selections is None:
            selections = SELECTIONS
        if penalty_keys is None:
            penalty_keys = PENALTY_KEYS
        if unapplied_penalty_keys is None:
            unapplied_penalty_keys = UNAPPLIED_PENALTY_KEYS
        if x_axis_keys is None:
            x_axis_keys = X_AXIS_KEYS

        selections_dict = {}
        penalty_labels = {}
        assert isinstance(selections, list)
        for selection in selections:

            # check selection:
            if selection not in self.selection:  # or selection not in SELECTIONS:
                raise ValueError(f"{selection} selection can't be plottet because no results were found.")

            selections_dict[selection] = self.selection[selection]["selection"]

            # plot without penalties:
            if selection not in penalty_keys:
                penalty_keys[selection] = []

            # default penalty keys:
            elif penalty_keys[selection] is None and selection in PENALTY_KEYS:
                penalty_keys[selection] = PENALTY_KEYS[selection]

            # no unapplied penalties
            if selection not in unapplied_penalty_keys:
                unapplied_penalty_keys[selection] = []

            # default unapplied penalties:
            elif unapplied_penalty_keys[selection] is None and selection in UNAPPLIED_PENALTY_KEYS:
                unapplied_penalty_keys[selection] = UNAPPLIED_PENALTY_KEYS[selection]

            penalty_labels[selection] = {
                **{
                    p_name: "partially applied" if selection == "marker" else "penalty"
                    for p_name in penalty_keys[selection]
                },
                **{p_name: "unapplied penal." for p_name in unapplied_penalty_keys[selection]},
            }

            # plot with penalties:
            penalty_keys[selection] = penalty_keys[selection] + unapplied_penalty_keys[selection]
            for penalty_key in penalty_keys[selection]:
                # check penalty key:
                if penalty_key not in x_axis_keys:
                    raise ValueError(f"Can't plot penalties because {penalty_key} was not found in x_axis_keys.")
                # check x-axis values:
                x_axis_key = x_axis_keys[penalty_key]
                if x_axis_key not in self.adata.var:
                    raise ValueError(f"Can't plot histogram because {x_axis_key} was not found.")

        pl.selection_histogram(
            adata=self.adata,
            selections_dict=selections_dict,
            background_key=self.g_key if background_key is True else background_key,
            penalty_keys=penalty_keys,
            penalty_labels=penalty_labels,
            x_axis_keys=x_axis_keys,
            **kwargs,
        )

    def plot_coexpression(
        self,
        selections: List[str] = ["final", "pca", "DE", "marker", "pre", "prior"],
        **kwargs,
    ) -> None:
        """Plot correlation matrix of selected genes


        Args:
            selections: Plot the coexpression of

                - 'final' : all selected genes (see :attr:`.ProbesetSelector.probeset`)
                - 'pca' : selected genes that also occured in the pca based selection
                - 'DE' : selected genes that also occured in the 1-vs-all DE based selection
                - 'marker' : selected genes from the marker list
                - 'pre' : selected genes that were given as pre selected genes
                - 'prior' : selected genes that were given as prioritized genes

            kwargs:
                Any keyword argument from :func:`.correlation_matrix`.


        Example:

            (Takes a few minutes to calculate)

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selector = sp.se.ProbesetSelector(adata, "celltype", n=30, verbosity=0)
                selector.select_probeset()

                selector.plot_coexpression()

            .. image:: ../../docs/plot_examples/Selector_plot_coexpression.png


        """

        # Supported selections for plotting
        SELECTIONS = ["final", "pca", "DE", "marker", "pre", "prior"]

        if selections is None:
            selections = SELECTIONS

        # Check selections
        for selection in selections:
            if selection not in self.selection:
                raise ValueError(f"{selection} selection can't be plottet because no results were found.")
            elif selection not in SELECTIONS:
                raise ValueError(f"{selection} not in supported selections for plotting.")
            elif (self.selection[selection] is None) or (len(self.selection[selection]) == 0):
                print(f"No genes were selected for selection {selection}.")

        # Throw out selections for which no genes were selected
        selections = [s for s in selections if (not (self.selection[s] is None)) and (len(self.selection[s]) > 0)]

        cor_matrices = {}
        assert isinstance(selections, list)
        for selection in selections:

            # Get data of selected genes (overlap between adata and self.selection[selection] and
            # self.probeset["selection"]):
            a = self.adata.copy()
            if isinstance(self.selection[selection], list):
                selected_genes = self.selection[selection]
            else:
                selected_genes = self.selection[selection].index[self.selection[selection]["selection"]]
            selection_mask = a.var_names.isin(selected_genes)
            probeset = self.probeset.index[self.probeset["selection"]]
            probeset_mask = a.var_names.isin(probeset)
            a = a[:, probeset_mask & selection_mask]

            if a.shape[1] < 2:
                print(f"No plot is drawn for {selection} because it contains less than 2 genes. ")
                continue

            # Create correlation matrix
            if issparse(a.X):
                cor_mat = pd.DataFrame(
                    index=a.var.index, columns=a.var.index, data=np.corrcoef(a.X.toarray(), rowvar=False)
                )
            else:
                cor_mat = pd.DataFrame(index=a.var.index, columns=a.var.index, data=np.corrcoef(a.X, rowvar=False))

            cor_mat = util.cluster_corr(cor_mat)
            cor_matrices[selection] = cor_mat

        # Add number of genes in title
        selections_n_genes = []
        for selection in selections:
            n_genes = f" ({cor_matrices[selection].shape[0]} genes)"
            selections_n_genes.append(selection + n_genes)
            cor_matrices[selection + n_genes] = cor_matrices[selection]
            del cor_matrices[selection]

        pl.correlation_matrix(set_ids=selections_n_genes, cor_matrices=cor_matrices, **kwargs)

    def plot_clf_genes(
        self,
        basis: int = "X_umap",
        celltypes: Optional[List[str]] = None,
        till_rank: Optional[int] = 1,
        importance_th: Optional[float] = None,
        add_marker_genes: bool = True,
        neighbors_params: dict = {},
        umap_params: dict = {},
        **kwargs,
    ):
        """Plot umaps of selected genes needed for cell type classification of each cell type.

        Args:
            basis:
                Name of the ``obsm`` embedding to use.
            celltypes:
                Subset of cell types for which to plot decision genes. If `None`, :attr:`celltypes` is used.
            till_rank:
                Plot decision genes only up to the given tree rank of the probeset list.
            importance_th:
                Only plot genes with a tree feature importance above the given threshold.
            add_marker_genes:
                Whether to add subplots for marker genes from :attr:`marker_list` for each celltype.
                TODO: what about cell types that only occur in the marker list?
            neighbors_params:
                Parameters for :meth:`sc.pp.neighbors`. Only applicable if ``adata.obsm[basis]`` does not exist.
                TODO: do we rly need that parameter? Would be fine to always expect a pre calculated embedding!
            umap_params:
                Parameters for :meth:`sc.tl.umap`. Only applicable if ``adata.obsm[basis]`` does not exist.
                TODO: do we rly need that parameter? Would be fine to always expect a pre calculated embedding!
            kwargs:
                Keyword arguments of :func:`.selection_histogram`.


        Example:

            (Takes a few minutes to calculate)

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selector = sp.se.ProbesetSelector(adata, "celltype", n=30, verbosity=0)
                selector.select_probeset()

                selector.plot_clf_genes(n_cols=4,celltypes=["FCGR3A+ Monocytes","Megakaryocytes"])

            .. image:: ../../docs/plot_examples/Selector_plot_clf_genes.png


        TODO: this function and pl.clf_genes_umaps need to be tested on all argument combinations + can be optimized.

        """
        adata = self.adata.copy()  # TODO: why copy?? should be avoided here... ah okay, because of umap recalc...

        # filter rank and importance:
        if till_rank is None:
            till_rank = max(self.selection["forest"]["rank"])
        if importance_th is None:
            importance_th = min(self.selection["forest"]["importance_score"])
        df = (
            self.selection["forest"]
            .loc[
                (self.selection["forest"]["rank"] <= till_rank)
                & (self.selection["forest"]["importance_score"] > importance_th)
            ]
            .copy()
        )
        selected_genes = [g for g in df.index if g in self.probeset[self.probeset["selection"]].index]
        df = df.loc[selected_genes]
        if len(df) < 1:
            raise ValueError("Filtering for rank and importance score left no genes. Set lower thresholds.")

        # prepare df
        if celltypes is None:
            celltypes = self.celltypes
        df["decision_celltypes"] = df[celltypes].apply(lambda row: list(row[row == True].index), axis=1)
        if add_marker_genes and (self.selection["marker"] is not None):
            df["marker_celltypes"] = [self.selection["marker"]["celltype"][gene] for gene in df.index]

        # check if embedding, neighbors, pca already in adata
        redo_umap = (adata.obsm is None) or (basis not in adata.obsm)
        ###try:
        ###    # check params
        ###    for param, value in adata.uns[basis]["params"].items():
        ###        if value != umap_params[param]:
        ###            redo_umap = True
        ###except KeyError:
        ###    redo_umap = True

        if redo_umap:
            redo_neighbors = False
            try:
                # check params
                for param, value in adata.uns["neighbors"]["params"].items():
                    if value != neighbors_params[param]:
                        redo_neighbors = True
            except KeyError:
                redo_neighbors = True

            if redo_neighbors:
                sc.pp.neighbors(adata, **neighbors_params)

            sc.tl.umap(adata, **umap_params)

        df = df.sort_values(by=["rank", "importance_score"], ascending=[True, False])

        pl.clf_genes_umaps(adata, df, **kwargs)

    # def plot_tree_performances(self) -> None:
    #     """Plot histograms of tree performances of DE baseline and final forests.
    #
    #     This function is important as a sanity check to see if the tree performances show proper statistical
    #     behavior.
    #     """
    #     # TODO:
    #     #  - think about when we can plot this: after tree
    #     #  - The function is only supported if self.save_dir != None... meh, actually that's not necessary. the
    #     #    memory usage is not too high for this...

    def plot_gene_overlap(
        self,
        origins: List[
            Literal["pre_selected", "prior_selected", "pca", "DE", "DE_1vsall", "DE_specific", "marker_list"]
        ] = None,
        **kwargs,
    ) -> None:
        """Plot the overlap of origins for the selected genes

        Args:
           origins:
                Origin groups to investigate. Supported are

                    - "pre_selected"   : User defined pre selected genes
                    - "prior_selected" : User defined prior selected genes
                    - "pca"            : Genes that originate from the prior pca based selection
                    - "DE"             : Genes that occur in the DE test when building the reference DE trees
                    - "DE_1vsall"      : Subset of "DE" from tests of single cell types vs background
                    - "DE_specific"    : Subset of "DE" from tests of single cell types vs subset of background
                    - "marker_list"    : Genes that occur in the user defined marker list

           **kwargs:
               Any keyword argument from :func:`.gene_overlap`.


        Example:

            (Takes a few minutes to calculate)

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selector = sp.se.ProbesetSelector(adata, "celltype", n=30, verbosity=0, marker_list={"celltypeX": ["PF4"]})
                selector.select_probeset()

                selector.plot_gene_overlap()

            .. image:: ../../docs/plot_examples/Selector_plot_gene_overlap.png


        """
        ORIGINS: List[
            Literal["pre_selected", "prior_selected", "pca", "DE", "DE_1vsall", "DE_specific", "marker_list"]
        ] = ["pre_selected", "prior_selected", "pca", "DE", "DE_1vsall", "DE_specific", "marker_list"]

        ORIGIN_TO_PROBESET_COLNAME = {
            "pre_selected": "pre_selected",
            "prior_selected": "prior_selected",
            "pca": "pca_selected",
            "DE": "celltypes_DE",
            "DE_1vsall": "celltypes_DE_1vsall",
            "DE_specific": "celltypes_DE_specific",
        }

        if not origins:
            origins = ORIGINS
            origins.remove("DE")
        assert isinstance(origins, list)

        probeset_selection = self.probeset[self.probeset["selection"]]
        selection_df = pd.DataFrame()
        for key in origins:
            if key in ORIGIN_TO_PROBESET_COLNAME:
                if probeset_selection[ORIGIN_TO_PROBESET_COLNAME[key]] is None:
                    continue
                selection_df[key] = probeset_selection[ORIGIN_TO_PROBESET_COLNAME[key]].astype("bool")
            elif key == "marker_list":
                if self.marker_list is None:
                    continue
                assert isinstance(self.marker_list, dict)
                marker_list = [x for y in self.marker_list.values() for x in y]
                mask = probeset_selection.index.isin(marker_list)
                selection_df["marker_list"] = False
                selection_df.loc[mask, "marker_list"] = True
            else:
                raise ValueError(f"Can't plot {key} since no results are found.")

        pl.gene_overlap(selection_df=selection_df, **kwargs)

    def plot_explore_constraint(
        self,
        selection_method: str = "pca_selection",
        selection_params: dict = None,
        background_key: str = "highly_variable",
        x_axis_key: str = "quantile_0.99",
        penalty_kernels: List[Callable] = None,
        factors: List[float] = None,
        upper: float = 1,
        lower: float = 0,
        **kwargs,
    ):
        """Plot histogram of quantiles for selected genes for different penalty kernels.

        Args:
            selection_method:
                Selection method to explore. Available are:

                    - "pca_selection"
                    - "DE_selection"
                    - "marker"

            selection_params:
                Hyperparameter for the respective gene selection.
            background_key:
                Key of column in ``adata.var`` containing the values to be plottet as background.
            x_axis_key:
                Key of column in ``adata.var`` containing the values to be plottet.
            penalty_kernels:
                Any penalty kernel. If None, :func:`.plateau_penalty_kernel` is used to create three gaussian
                plateau penalty kernels with different variances. Only then, ``factors,``, ``lower`` and ``upper`` are
                used.
            factors:
                Factors for the variance for creating a gaussian penalty kernel.
            lower:
                Lower border above which the kernel is 1.
            upper:
                Upper boder below which the kernel is 1.
            **kwargs:
                Further arguments for :func:`.explore_constraint`.

        Returns:
            Figure can be shown (default `True`) and stored to path (default `None`).
            Change this with `show` and `save` in ``kwargs``.
        """

        # TODO:
        #  1) Fix explore_constraint plot. The following circular import is causing problems atm:
        #     DONE: moved parts of this method to selector.plot_expore_constraint --> this solves the problem
        #  2) How to generalize the plotting function, support:
        #     - any selection method with defined hyperparameters --> DONE --> add more
        #     - any penalty kernel --> DONE --> test
        #     - any key to be plotted (not only quantiles) --> DONE --> test

        SELECTION_METHODS: Dict[str, Callable] = {
            "pca_selection": select.select_pca_genes,
            "DE_selection": select.select_DE_genes,
        }

        if selection_method not in SELECTION_METHODS:
            raise ValueError(f"Selection with method {selection_method} is not available.")

        if selection_params is None:
            selection_params = {}

        selection_params = self._get_hparams(new_params=selection_params, subject=selection_method)

        if factors is None:
            factors = [10, 1, 0.1]

        if penalty_kernels is None:
            penalty_kernels = [
                util.plateau_penalty_kernel(
                    var=[factor * 0.1, factor * 0.5], x_min=np.array(lower), x_max=np.array(upper)
                )
                for factor in factors
            ]

        a = []
        selections_tmp = []
        for i, factor in enumerate(factors):

            if background_key not in self.adata.var:
                raise ValueError(f"Can't plot background histogram because {background_key} was not found.")
            a.append(self.adata[:, self.adata.var[background_key]].copy())

            if x_axis_key not in a[i].var:
                raise ValueError(f"No column {x_axis_key} in adata.var found.")

            a[i].var["penalty_expression"] = penalty_kernels[i](a[i].var[x_axis_key])
            selections_tmp.append(
                SELECTION_METHODS[selection_method](
                    a[i],
                    n=100,
                    **selection_params,
                    penalty_keys=["penalty_expression"],
                    inplace=False,
                    verbosity=self.verbosity,
                )
            )
            print(f"N genes selected: {np.sum(selections_tmp[i]['selection'])}")

        pl.explore_constraint(
            a,
            selections_tmp,
            penalty_kernels,
            factors=factors,
            x_axis_key=x_axis_key,
            upper=upper,
            lower=lower,
            **kwargs,
        )

    def info(self) -> None:
        """Print info."""
        print("No info yet")


def select_reference_probesets(
    adata: sc.AnnData,
    n: int,
    genes_key: str = "highly_variable",
    methods: Union[List[str], Dict[str, Dict]] = ["PCA", "DE", "HVG", "random"],
    seeds: List[int] = [0],
    verbosity: int = 2,
    save_dir: Union[str, None] = None,
) -> Dict[str, pd.DataFrame]:
    """Select reference probesets with basic selection methods.

    Args:
        adata:
            Data with log normalised counts in adata.X.
        n:
            Number of selected genes.
        genes_key:
            adata.var key for subset of preselected genes to run the selections on (typically 'highly_variable_genes').
        methods:
            Methods used for selections. Supported methods and default are `['PCA', 'DE', 'HVG', 'random']`. To specify
            hyperparameters of the methods provide a dictionary, e.g.::

                {
                    'DE':{},
                    'PCA':{'n_pcs':30},
                    'HVG':{},
                    'random':{},
                }

        seeds:
            List of random seeds. For each seed, one random gene set is selected if `'random'` in `methods`.
        verbosity:
            Verbosity level.
        save_dir:
            Directory path where all results are saved.

    Returns:
        Dictionary with one entry for each method. The key is the selection method name and the value is
        a DataFrame with the same index as adata.var and at least one boolean column called 'selection' representing
        the selected probeset. For some methods, additional information is provided in other columns.
    """

    # Supported selection functions
    selection_fcts: Dict[str, Callable] = {
        "PCA": select.select_pca_genes,
        "DE": select.select_DE_genes,
        "random": select.random_selection,
        "HVG": select.select_highly_variable_features,
    }

    # Reshape methods to dict with empty hyperparams if given as a list
    if isinstance(methods, list):
        methods = {method: {} for method in methods}
    assert isinstance(methods, dict)

    # Filter unsupported methods
    for method in methods:
        if method not in selection_fcts:
            print(f"Method {method} is not available. Supported methods are {[key for key in selection_fcts]}.")
            del methods[method]
    methods = {m: methods[m] for m in methods if m in selection_fcts}

    # Create list of planed selections
    selections: List[dict] = []
    for method in methods:
        if method == "random":
            for seed in seeds:
                seed_str = "" if len(seeds) == 0 else f" (seed={seed})"
                selections.append(
                    {"method": method, "name": f"{method}{seed_str}", "params": dict(methods[method], **{"seed": seed})}
                )
        else:
            selections.append({"method": method, "name": method, "params": methods[method]})

    # Run selections
    progress = util.NestedProgress(disable=(verbosity == 0))
    probesets = {}

    with progress:
        ref_task = progress.add_task("Reference probeset selection...", total=len(selections), level=1)

        for s in selections:

            if verbosity > 1:
                sel_task = progress.add_task(f"Selecting {s['name']} genes...", total=1, level=2)

            probesets[s["name"]] = selection_fcts[s["method"]](
                adata[:, adata.var[genes_key]], n, inplace=False, **s["params"]
            )

            if save_dir:
                probesets[s["name"]].to_csv(os.path.join(save_dir, s["name"]))

            if verbosity > 0:
                progress.advance(ref_task)

            if verbosity > 1:
                progress.advance(sel_task)

        if verbosity > 0:
            progress.add_task("Finished", total=1, footer=True, only_text=True)

    return probesets
