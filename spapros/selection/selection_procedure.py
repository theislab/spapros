import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import spapros.evaluation.evaluation as ev
import spapros.selection.selection_methods as select
import spapros.util.util as util
from spapros.util.util import dict_to_table
from spapros.util.util import filter_marker_dict_by_penalty
from spapros.util.util import filter_marker_dict_by_shared_genes
from tqdm.autonotebook import tqdm


class ProbesetSelector:  # (object)
    """Class for probeset selection

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(
        self,
        adata,
        celltype_key,
        genes_key="highly_variable",
        n=None,
        preselected_genes=[],
        prior_genes=[],
        n_pca_genes=100,
        min_mean_difference=None,
        n_min_markers=2,
        # TODO TODO: add this feature, it means we have at least 3 genes per celltype that we consider as celltype
        # marker and this time we define markers based on DE & list and min_mean_diff. pca that also occur in DE list
        # are also markers and the other pca genes not? Might be difficult than to achieve the number
        # Priorities: best tree genes (pca included), fill up n_min_markers (from 2nd trees), fill up from further trees
        # without n_min_markers condiseration.
        # I still think it would be nice to have a better marker criteria, but idk what
        celltypes="all",
        marker_list=None,
        n_list_markers=2,
        marker_corr_th=0.5,
        pca_penalties=[],
        DE_penalties=[],
        m_penalties_adata_celltypes=[],  # reasonable choice: positive lower and upper threshold
        m_penalties_list_celltypes=[],  # reasonable choice: only upper threshold
        pca_selection_hparams={},
        DE_selection_hparams={"n": 3, "per_group": True},
        forest_hparams={"n_trees": 50, "subsample": 1000, "test_subsample": 3000},
        forest_DE_baseline_hparams={
            "n_DE": 1,
            "min_score": 0.9,
            "n_stds": 1.0,
            "max_step": 3,
            "min_outlier_dif": 0.02,
            "n_terminal_repeats": 3,
        },
        add_forest_genes_hparams={"n_max_per_it": 5, "performance_th": 0.02, "importance_th": 0},
        marker_selection_hparams={"penalty_threshold": 1},
        verbosity=2,
        seed=0,
        save_dir=None,
        n_jobs=-1,
        reference_selections={},
    ):
        # Should we add a "copy_adata" option? If False we can use less memory, if True adata will be changed at the end
        """

        Parameters
        ----------
        adata: AnnData
            Data with log normalised counts in adata.X. The selection runs with an adata subsetted on fewer genes. It
            might be helpful though to keep all genes (when a marker_list and penalties are provided). The genes can be
            subsetted for selection via `genes_key`.
        celltype_key: str
            adata.obs key with celltypes annotations.
        genes_key: str
            adata.var key for preselected genes (typically 'highly_variable_genes')
        n: integer (default = None)
            Optionally set the number of finally selected genes. Note that when `n is None` we automatically infer `n`
            as the minimal number of recommended genes. This includes all preselected genes, genes in the best decision
            tree of each celltype, and the minimal number of identified and added markers defined by `n_min_markers` and
            `n_list_markers`. Als note that setting `n` might change the gene ranking since the final added list_markers
            are added based on the theoretically added genes without list_markers.
        preselected_genes: list of strs
            Pre selected genes (these will also have the highest ranking in the final list).
        min_mean_difference: float
            Minimal difference of mean expression between at least one celltype and the background. In this test only
            cell types from `celltypes` are taken into account (also for the background). This minimal difference is
            applied as an additional binary penalty in pca_penalties, DE_penalties and m_penalties_adata_celltypes.
        celltypes: str or list of strs
            Cell types for which trees are trained.
            - The probeset is optimised to be able to distinguish each of these cell types from all other cells occuring
              in the dataset
            - The pca selection is based on all cell types in the dataset (not only on `celltypes`)
            - The optionally provided marker list can include additional cell types not listed in `celltypes` (and
              adata.obs[celltype_key])
        n_list_markers: int or dict
            Minimal number of markers per celltype that are at least selected. Selected means either selecting genes
            from the marker list or having correlated genes in the already selected panel. (Set the correlation
            threshold with marker_selection_hparams['penalty_threshold']).
            If you want to select a different number of markers for celltypes in adata and celltypes only in the marker
            list, set e.g.: n_list_markers = {'adata_celltypes':2,'list_celltypes':3}
        marker_penalties: str, list or dict of strs
        reference_selections: dict(str)
            A dictionary where the keys are basic selecion methods, that should be used to create reference probesets.
            Available are `pca_selection`, `DE_selection`, `random_selection` and `hvg_selection`
            The values are dictionaries of parameters for a metric or {}.
        save_dir: str
            Directory path where all results are saved and loaded from if results already exist.
            Note for the case that results already exist:
            - if self.select_probeset() was fully run through and all results exist: then the initialization arguments
              don't matter much
            - if only partial results were generated, make sure that the initialization arguments are the same as
              before!


        """
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
        self.selection = {
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

        # Reference gene sets
        self.reference_selections = reference_selections

        self.forest_results = {
            "DE_prior_forest": None,
            "DE_baseline_forest": None,
            "pca_prior_forest": None,
            "forest": None,
        }
        self.forest_clfs = {"DE_baseline_forest": None, "forest": None}

        self.min_test_n = 20

        cts_below_min_test_size, counts_below_min_test_size = ev.get_celltypes_with_too_small_test_sets(
            self.adata[self.obs], self.ct_key, min_test_n=self.min_test_n, split_kwargs={"seed": self.seed, "split": 4}
        )
        if cts_below_min_test_size:
            print(
                "The following celltypes' test set sizes for forest training are below min_test_n "
                + f"(={self.min_test_n}):"
            )
            max_length = max([len(ct) for ct in cts_below_min_test_size])  # TODO: bug fix: type(ct) != str doesnt work.
            for i, ct in enumerate(cts_below_min_test_size):
                print(f"\t {ct:<{max_length}} : {counts_below_min_test_size[i]}")

        self.loaded_attributes = []
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

        # Check that reference selections are available, check their parameters, add keys to selection dict
        available_ref_selections = ["hvg_selection", "random_selection", "DE_selection", "pca_selection"]
        for selection_name in self.reference_selections:
            if selection_name not in available_ref_selections:
                del self.reference_selections[selection_name]
                print(
                    f"Selecting {selection_name} genes as reference is not available. Options are 'hvg' and 'random'."
                )
            else:
                self.selection[f"ref_{selection_name}"] = None
                self.reference_selections[selection_name] = self._get_hparams(self.reference_selections[selection_name], subject=selection_name)

        # Mean difference constraint
        self._prepare_mean_diff_constraint()

        # Marker list
        self._prepare_marker_list()

    def select_probeset(self):
        """Run full selection procedure"""

        if self.n_pca_genes and (self.n_pca_genes > 0):
            self._pca_selection()
        self._forest_DE_baseline_selection()
        self._forest_selection()
        if self.marker_list:
            self._marker_selection()

        # select reference sets
        if "hvg_selection" in self.reference_selections:
            self._ref_wrapper(select.select_highly_variable_features, "ref_hvg_selection", **self.reference_selections["hvg_selection"])
        if "random_selection" in self.reference_selections:
            self._ref_wrapper(select.random_selection, "ref_random_selection",  **self.reference_selections["random_selection"])
        if "pca_selection" in self.reference_selections:
            self._ref_wrapper(select.select_pca_genes, "ref_pca_selection", **self.reference_selections["pca_selection"])
        if "DE_selection" in self.reference_selections:
            if "n" in self.reference_selections["DE_selection"]:
                del self.reference_selections["DE_selection"]["n"]
                self.reference_selections["DE_selection"]["obs_key"] = self.ct_key
            self._ref_wrapper(select.select_DE_genes, "ref_DE_selection", **self.reference_selections["DE_selection"])

        self.probeset = self._compile_probeset_list()
        if self.save_dir:
            self.probeset.to_csv(self.probeset_path)
        # TODO: we haven't included the checks to load the probeset if it already exists

    def _ref_wrapper(self, selection_fun, selection_name, **kwargs):
        """Select highly variable genes"""
        if self.selection[selection_name] is None:
            if self.verbosity > 0:
                print(f"Select {selection_name} genes...")
            self.selection[selection_name] = selection_fun(self.adata[:, self.genes], self.n, inplace=False, **kwargs)
            if self.verbosity > 1:
                print("\t ...finished.")
            if self.save_dir:
                self.selection[selection_name].to_csv(self.selections_paths[selection_name])
        else:
            if self.verbosity > 0:
                print(f"{selection_name} genes already selected...")

    def _pca_selection(self):
        """Select genes based on pca loadings"""
        if self.selection["pca"] is None:
            if self.verbosity > 0:
                print("Select pca genes...")
            self.selection["pca"] = select.select_pca_genes(
                self.adata[:, self.genes],
                self.n_pca_genes,
                penalty_keys=self.pca_penalties,
                inplace=False,
                **self.pca_selection_hparams,
            )
            self.selection["pca"] = self.selection["pca"].sort_values("selection_ranking")
            if self.verbosity > 1:
                print("\t ...finished.")
            if self.save_dir:
                self.selection["pca"].to_csv(self.selections_paths["pca"])
        else:
            if self.verbosity > 0:
                print("PCA genes already selected...")

    def _forest_DE_baseline_selection(self):
        """Select genes based on forests and differentially expressed genes"""
        if self.verbosity > 0:
            print("Select genes based on differential expression and forests as baseline for the final forests...")

        if not isinstance(self.selection["DE"], pd.DataFrame):
            if self.verbosity > 1:
                print("\t Select differentially expressed genes...")
            self.selection["DE"] = select.select_DE_genes(
                self.adata[:, self.genes],
                obs_key=self.ct_key,
                **self.DE_selection_hparams,
                penalty_keys=self.DE_penalties,
                groups=self.celltypes,
                reference="rest",
                rankby_abs=False,
                inplace=False,
            )
            if self.verbosity > 1:
                print("\t\t ...finished.")
            if self.save_dir:
                self.selection["DE"].to_csv(self.selections_paths["DE"])
        else:
            if self.verbosity > 1:
                print("\t Differentially expressed genes already selected...")

        if not self.forest_results["DE_prior_forest"]:
            if self.verbosity > 1:
                print("\t Train trees on DE selected genes as prior forest for the DE_baseline forest...")
            DE_prior_forest_genes = (
                self.selection["pre"]
                + self.selection["prior"]
                + self.selection["DE"].loc[self.selection["DE"]["selection"]].index.tolist()
            )
            DE_prior_forest_genes = [g for g in np.unique(DE_prior_forest_genes).tolist() if g in self.genes]

            save_DE_prior_forest = self.forest_results_paths["DE_prior_forest"] if self.save_dir else False
            self.forest_results["DE_prior_forest"] = ev.forest_classifications(
                self.adata[:, self.genes],
                DE_prior_forest_genes,
                celltypes=self.celltypes,
                ref_celltypes="all",
                ct_key=self.ct_key,
                save=save_DE_prior_forest,
                seed=0,
                verbosity=self.verbosity,
                **self.forest_hparams,
            )

        if not (
            self.forest_results["DE_baseline_forest"]
            and self.forest_clfs["DE_baseline_forest"]
            and isinstance(self.selection["forest_DEs"], pd.DataFrame)
        ):
            if self.verbosity > 1:
                print(
                    "\t Train DE_baseline forest by iteratively selecting specific differentially expressed genes for "
                    "celltypes that are hard to distinguish..."
                )
            save_DE_baseline_forest = self.forest_results_paths["DE_baseline_forest"] if self.save_dir else False
            (
                self.forest_results["DE_baseline_forest"],
                self.forest_clfs["DE_baseline_forest"],
                self.selection["forest_DEs"],
            ) = select.add_DE_genes_to_trees(
                self.adata[:, self.genes],
                self.forest_results["DE_prior_forest"],
                ct_key=self.ct_key,
                penalty_keys=self.DE_penalties,
                tree_clf_kwargs=self.forest_hparams,
                verbosity=self.verbosity,
                save=save_DE_baseline_forest,
                return_clfs=True,
                **self.forest_DE_baseline_hparams,
            )
            if self.save_dir:
                self.selection["forest_DEs"].to_csv(self.selections_paths["forest_DEs"])
                with open(self.forest_clfs_paths["DE_baseline_forest"], "wb") as f:
                    pickle.dump(self.forest_clfs["DE_baseline_forest"], f)

        # might be interesting for utility plots:
        if not isinstance(self.selection["DE_baseline_forest"], pd.DataFrame):
            self.selection["DE_baseline_forest"] = ev.forest_rank_table(self.forest_results["DE_baseline_forest"][2])
            if self.save_dir:
                self.selection["DE_baseline_forest"].to_csv(self.selections_paths["DE_baseline_forest"])

    def _forest_selection(self):
        """ """
        # - eventually put this "test set size"-test somewhere (ideally at __init__)

        if self.verbosity > 0:
            print(
                "Train final forests by adding genes from the DE_baseline forest for celltypes with low performance..."
            )

        if self.n_pca_genes and (self.n_pca_genes > 0):
            if self.verbosity > 1:
                print("\t Train forest on pre/prior/pca selected genes...")
            if not self.forest_results["pca_prior_forest"]:
                pca_prior_forest_genes = (
                    self.selection["pre"]
                    + self.selection["prior"]
                    + self.selection["pca"].loc[self.selection["pca"]["selection"]].index.tolist()
                )
                pca_prior_forest_genes = [g for g in np.unique(pca_prior_forest_genes).tolist() if g in self.genes]
                save_pca_prior_forest = self.forest_results_paths["pca_prior_forest"] if self.save_dir else False
                self.forest_results["pca_prior_forest"] = ev.forest_classifications(
                    self.adata[:, self.genes],
                    pca_prior_forest_genes,
                    celltypes=self.celltypes,
                    ref_celltypes="all",
                    ct_key=self.ct_key,
                    save=save_pca_prior_forest,
                    seed=self.seed,  # TODO!!! same seeds for all forests!
                    verbosity=self.verbosity,
                    **self.forest_hparams,
                )
                if self.verbosity > 2:
                    print("\t\t ...finished.")
            else:
                if self.verbosity > 2:
                    print("\t\t ...was already trained.")

            if self.verbosity > 1:
                print("\t Iteratively add genes from DE_baseline_forest...")
            if (not self.forest_results["forest"]) or (not self.forest_clfs["forest"]):
                save_forest = self.forest_results_paths["forest"] if self.save_dir else False
                self.forest_results["forest"], self.forest_clfs["forest"] = select.add_tree_genes_from_reference_trees(
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
                )
                if self.save_dir:
                    with open(self.forest_clfs_paths["forest"], "wb") as f:
                        pickle.dump(self.forest_clfs["forest"], f)
                if self.verbosity > 2:
                    print("\t\t ...finished.")
            else:
                if self.verbosity > 2:
                    print("\t\t ...were already added.")

            if not isinstance(self.selection["forest"], pd.DataFrame):
                self.selection["forest"] = ev.forest_rank_table(self.forest_results["forest"][2])
                if self.save_dir:
                    self.selection["forest"].to_csv(self.selections_paths["forest"])

    def _save_preselected_and_prior_genes(self):
        """Save pre selected and prior genes to files"""
        for s in ["pre", "prior"]:
            if self.selection[s] and not (f"selection_{s}" in self.loaded_attributes):
                with open(self.selections_paths[s], "w") as file:
                    json.dump(self.selection[s], file)

    def _prepare_marker_list(self):
        """Process marker list if not loaded and save to file"""
        # Eventually reformat n_list_markers
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

    def _filter_and_save_marker_list(self):
        """Get marker list in shape if necessary

        The following steps are applied to the marker list:
        - Filter out markers that occur multiple times
        - Filter out genes based on penalties (self.marker_penalties)
            - Warn the user if markers don't occur in adata and can't be tested on penalties

        Returns
        -------
        Modifies self.marker_list
        """
        # Filter genes that occur multiple times # TODO: if a genes occurs multiple times for the same celltype we
        # should actually keep it!...
        if self.verbosity > 0:
            print("Filter out genes in marker dict that occur multiple times.")
        self.marker_list = filter_marker_dict_by_shared_genes(self.marker_list, verbose=(self.verbosity > 1))

        # filter genes based on penalties
        marker_list_adata_cts = {ct: genes for ct, genes in self.marker_list.items() if ct in self.adata_celltypes}
        marker_list_other_cts = {
            ct: genes for ct, genes in self.marker_list.items() if not (ct in self.adata_celltypes)
        }
        m_list_adata_cts_dropped = {}
        m_list_other_cts_dropped = {}
        if self.m_penalties_adata_celltypes:
            if marker_list_adata_cts:
                if self.verbosity > 0:
                    print("Filter marker dict by penalties for markers of celltypes in adata")
                marker_list_adata_cts, m_list_adata_cts_dropped = filter_marker_dict_by_penalty(
                    marker_list_adata_cts,
                    self.adata,
                    self.m_penalties_adata_celltypes,
                    threshold=self.m_selection_hparams["penalty_threshold"],
                    verbose=(self.verbosity > 1),
                    return_filtered=True,
                )
        if self.m_penalties_list_celltypes:
            if marker_list_other_cts:
                if self.verbosity > 0:
                    print("Filter marker dict by penalties for markers of celltypes not in adata")
                marker_list_other_cts, m_list_other_cts_dropped = filter_marker_dict_by_penalty(
                    marker_list_other_cts,
                    self.adata,
                    self.m_penalties_list_celltypes,
                    threshold=self.m_selection_hparams["penalty_threshold"],
                    verbose=(self.verbosity > 1),
                    return_filtered=True,
                )

        # Combine differently penalized markers to filtered marker dict and out filtered marker dict
        self.marker_list = dict(marker_list_adata_cts, **marker_list_other_cts)
        self.marker_list_filtered_out = dict(m_list_adata_cts_dropped, **m_list_other_cts_dropped)

        # Save marker list
        if self.save_dir:
            with open(self.marker_list_path, "w") as file:
                json.dump([self.marker_list, self.marker_list_filtered_out], file)

    def _check_number_of_markers_per_celltype(self):
        """Check if given markers per celltype are below the number of marker we want to add eventually"""
        low_gene_numbers = {}
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

    def _marker_selection(self):
        """Select genes from marker list based on correlations with already selected genes"""
        pre_pros = self._compile_probeset_list(with_markers_from_list=False)

        # Check which genes are already selected
        selected_genes = pre_pros.loc[pre_pros["selection"]].index.tolist()
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
                marker = genes[: min([len(genes), self.n_list_markers["list_celltypes"]])]
                self.selection["marker"].loc[marker, ["selection", "celltype"]] = [True, ct]
                # print("MARKER SELECTION (list only): ",ct,marker)
                # for i,g in enumerate(marker):
                #    self.selection['marker'].loc[g,['selection','celltype','marker_rank']] = [True,ct,i+1]

    def _compile_probeset_list(self, with_markers_from_list=True):

        # How/where to add genes from marker_list that are not in adata? --> Oh btw, same with pre and prior selected
        # genes.

        # Initialize probeset table
        index = self.genes.tolist()
        # Add marker genes that are not in adata
        if self.marker_list:
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
        probeset.loc[self.selection["pca"][self.selection["pca"]["selection"]].index, "pca_selected"] = True
        probeset.loc[self.selection["pca"].index, "pca_score"] = self.selection["pca"]["selection_score"]

        # Reference selections
        for selection_name in self.reference_selections:
            ref_selection_name = f"ref_{selection_name}"
            probeset[ref_selection_name] = False
            probeset.loc[
                self.selection[ref_selection_name][self.selection[ref_selection_name]["selection"]].index,
                ref_selection_name,
            ] = True

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
                n_min_markers = (
                    max([self.n_min_markers, self.n_list_markers["adata_celltypes"]])
                    if (ct in shared_cts)
                    else self.n_min_markers
                )
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
                n_min_markers = (
                    max([self.n_min_markers, self.n_list_markers["adata_celltypes"]])
                    if (ct in shared_cts)
                    else self.n_min_markers
                )
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

        return probeset[cols].copy()

    def _prepare_mean_diff_constraint(self):
        """Compute if mean difference constraint is fullfilled"""
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

    def _get_hparams(self, new_params, subject="DE_selection"):
        """Add missing default parameters to dictionary in case they are not given

        Example: forest_hparams are given in the class init definition as {"n_trees": 50, "subsample": 1000,
        "test_subsample": 3000}. If the class is called with forest_hparams={"n_trees": 100} we would actually like
        to have {"n_trees": 100, "subsample": 1000, "test_subsample": 3000}. The last two are added by this functions

        # TODO we should have a test that checks if these default params are the same as in the __init__ method.
        Why we keep parameters in init is that you directly see values that can be set. The laternative would be to
        provide empty dicts in __init__.

        Arguments
        ---------
        new_params: dict
        subject: str
            type of hyper parameters of interest.

        Returns
        -------
        dict
        """
        params = new_params.copy()
        if subject == "pca_selection":
            defaults = {}
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
        elif subject == "hvg_selection":
            defaults = {"flavor": "cell_ranger"}
        elif subject == "random_selection":
            defaults = {"seed": 0}
        params.update({k: v for k, v in defaults.items() if k not in params})
        return params

    def _initialize_file_paths(self):
        """Initialize path variables and set up folder hierarchy

        Call this function in the initialization to define all file names that are eventually saved.
        This function also aims to have all possibly generated file names organised in one place.
        """
        # Create base directory
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

        # reference selections
        for selection_name in self.reference_selections:
            ref_selection_name = f"ref_{selection_name}"
            self.selections_paths[ref_selection_name] = os.path.join(selections_dir, ref_selection_name)

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

    def _load_from_disk(self):
        """Load existing files into variables"""

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

    def _tqdm(self, iterator):
        """Wrapper for tqdm with verbose condition"""
        return tqdm(iterator) if self.verbosity > 1 else iterator

    def plot_histogram(self, x_axis_key="quantile_0.99", selections=["pca", "DE", "marker"]):
        """Plot histograms of (basic) selections under given penalties

        The full selection procedure consists of steps partially based on basic score based selection procedures.
        This is an interactive plotting function to investigate if the constructed penalty kernels are well chosen.
        TODO: Describe where the penalties are defined
        (Atm I think I won't include the penalt kernels into the class therefore this plotting function simply plots
        histograms - a little boring but better than nothing)

        Parameters
        ----------
        x_axis_key: str
            adata.obs key that is used for the x axis of the plotted histograms
        selections: str
            Plot the histograms of selections based on
            - 'pca' : pca loadings based selection of prior genes
            - 'DE' : genes selected based on differential expressed which are used as the "forest_DE_baseline"
            - 'marker' : genes from the marker list
        """

    def plot_coexpression(self, selections=["pca", "DE", "marker"]):
        """Plot gene correlations of basic selections


        # When plotting for some selection: Print where these genes are used
        # - pca -> first genes on which the forests are trained
        # - DE -> first genes on which the baseline forests are trained
        # - marker -> possible genes that are selected from the marker list
        #   for marker you could use max nr of markers (2?) from each celltype
        """

    def plot_decision_genes(self, add_markers=True, tree_levels=[1]):
        """Plot umaps of genes that are used for celltype classification

        Parameters
        ----------
        add_markers: bool
            Also plot genes originating from selection from marker list
        tree_levels: int
            In case that genes were added not only based on the best performing tree we can plot more genes
            TODO: maybe find a better way than this variable to control this
        """

    def plot_tree_performances(self):
        """Plot histograms of tree performances of DE baseline and final forests

        This function is important as a sanity check to see if the tree performances show proper statistical
        behavior.
        # The function is only supported if self.save_dir != None... meh, actually that's not necessary. the
        # memory usage is not too high for this...

        TODO: think about when we can plot this: after tree

        """

    def info(self):
        print("No info yet")
