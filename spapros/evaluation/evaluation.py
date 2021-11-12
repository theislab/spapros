import os
import pickle
import warnings
from pathlib import Path
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import spapros.plotting as pl
from sklearn import tree
from sklearn.metrics import classification_report
from spapros.evaluation.metrics import get_metric_default_parameters
from spapros.evaluation.metrics import metric_computations
from spapros.evaluation.metrics import metric_pre_computations
from spapros.evaluation.metrics import metric_shared_computations
from spapros.evaluation.metrics import metric_summary
from spapros.util.mp_util import _get_n_cores
from spapros.util.mp_util import parallelize
from spapros.util.mp_util import Signal


class ProbesetEvaluator:
    """General class for probe set evaluation, comparison, plotting

    The evaluator works on one given dataset and calculates metrics/analyses with respect to that dataset.

    The calculation steps of the metrics can be divided into:
    1. calculations that need to be run one time for the given dataset (not all metrics have this step)
    2. calculations that need to be run for each probe set
        2.1 calculations independent of 1.
        2.2 calculations dependent on 1. (if 1. existed for a given metric)
    3. Summarize results into summary statistics

    ###################
    # Run evaluations #
    ###################

    Evaluate a single probeset:
        evaluator = ProbesetEvaluator(adata)
        evaluator.evaluate_probeset(gene_set)

    In a pipeline to evaluate multiple probesets you would run
        - sequential setup:
            evaluator = ProbesetEvaluator(adata)
            for i,gene_set in enumerate(sets):
                evaluator.evaluate_probeset(gene_set,set_id=f"set_{i}")
        - parallelised setup:
            evaluator = ProbesetEvaluator(adata)
            # 1. step:
            evaluator.compute_or_load_shared_results()
            # 2. step: parallelised processes
            evaluator.evaluate_probeset(gene_set,set_id,update_summary=False,pre=True) # parallelised over set_ids
            # 3. step: parallelised processes (needs 1. to be finished)
            evaluator.evaluate_probeset(gene_set,set_id,update_summary=False) # parallelised over set_ids
            # 4. step: (needs 3. to be finished)
            evaluator.summary_statistics()

    #########################
    # Reference evaluations #
    #########################

    In practice the evaluations are meaningful when having reference evaluations to compare to.

    A simple way to get reference probe sets:
        reference_sets = spapros.selection.select_reference_probesets(adata)
    Evaluate them (we also provide ids to keep track of the probesets):
        evaluator = ProbesetEvaluator(adata)
        for set_id, gene_set in reference_sets.items():
            evaluator.evaluate_probeset(gene_set,set_id=set_id)
        evaluator.plot_summary()

    ######################
    # Evaluation schemes #
    ######################

    Some metrics take very long to compute, we prepared different metric sets for a quick or a full evaluation.
    You can also specify the list of metrics yourself by setting scheme="custom". Note that in any scheme it might
    still be reasonable to adjust `metrics_params`.

    #####################
    # Saving of results #
    #####################

    If `results_dir` is not None we save the results in files.
    Why:
    - some computations are time demanding, especially when you evaluate multiple sets it's reasonable to keep results.
    - load previous results when initializing a ProbesetEvaluator. Makes it very easy to access and compare old results.

    Two saving directories need to be distinguished:
    1. `results_dir`: each probeset's evaluation results are saved here
    2. `reference_dir`: for shared reference dataset results (default is reference_dir = results_dir + reference_name)

    In which files the results are saved:
    - Shared computations are saved as
        reference_dir # (default: results_dir+"references")
        └── {reference_name}_{metric}.csv # shared computations for given reference dataset
    - The final probeset specific results are saved as
        results_dir
        ├── {metric} # one folder for each metric
        │   ├── {reference_name}_{set_id}_pre.csv # pre results file for given set_id, reference dataset, and metric
        │   │                                     # (only for some metrics)
        │   └── {reference_name}_{set_id}.csv # result file for given set_id, reference dataset, and metric
        └── {reference_name}_summary.csv # summary statistics

    ############
    # Plotting #
    ############

    Plot a summary metrics table to get an overall performance overview:
        evaluator.plot_summary()

    For each evaluation we provide a detailed plot, e.g.:
    - forest_clfs: heatmap of normalised confusion matrix
    - gene_corr: heatmap of ordered correlation matrix

    Create detailed plots with:
        evaluator.plot_evaluations()

    """

    def __init__(
        self,
        adata,
        celltype_key="celltype",
        results_dir="./probeset_evaluation/",
        scheme="quick",
        metrics=None,
        metrics_params={},
        marker_list=None,
        reference_name="adata1",
        reference_dir=None,
        verbosity=1,
        n_jobs=-1,
    ) -> None:
        """
        adata: AnnData
            Already preprocessed. Typically we use log normalised data.
        celltype_key: str or list of strs
            adata.obs key for cell type annotations. Provide a list of keys to calculate the according metrics on
            multiple keys.
        results_dir: str
            Directory where probeset results are saved. Set to `None` if you don't want to save results. When
            initializing the class we also check for existing results

            Note if
            TODO: Decide: saving results is nice since we don't need to keep them in memory. On the other hand the stuff
                          doesn't need that much memory I think. Even if you have 100 probesets, it's not that much
        scheme: str
            Defines which metrics are calculated
            - "quick" : knn, forest classification, marker correlation (if marker list given), gene correlation
            - "full" : nmi, knn, forest classification, marker correlation (if marker list given), gene correlation
            - "custom": define metrics of intereset in `metrics`
        metrics: list of strs
            Define which metrics are calculated. This is set automatically if `scheme != "custom"`. Supported are
            - "nmi"
            - "knn"
            - "forest_clfs"
            - "marker_corr"
            - "gene_corr"
        metrics_params: dict of dicts
            Provide parameters for the calculation of each metric. E.g.
            metrics_params = {
                "nmi":{
                    "ns": [5,20],
                    "AUC_borders": [[7, 14], [15, 20]],
                    }
                }
            This overwrites the arguments `ns` and `AUC_borders` of the nmi metric. See
            spapros.evaluation.get_metric_default_parameters() for the default values of each metric
        reference_name: str
            Name of reference dataset. This is chosen automatically if `None` is given.
        reference_dir: str
            Directory where reference results are saved. If `None` is given `reference_dir` is set to
            `results_dir+"reference/"`
        n_jobs: int
            Number of cpus for multi processing computations. Set to -1 to use all available cpus.


        metrics_params: list of dicts
        """
        self.adata = adata
        self.celltype_key = celltype_key
        self.dir = results_dir
        self.scheme = scheme
        self.marker_list = marker_list
        self.metrics_params = self._prepare_metrics_params(metrics_params)
        self.metrics = metrics if (scheme == "custom") else self._get_metrics_of_scheme()
        self.ref_name = reference_name
        self.ref_dir = reference_dir if (reference_dir is not None) else self._default_reference_dir()
        self.verbosity = verbosity
        self.n_jobs = n_jobs

        self.shared_results: Dict[str, Any] = {}
        self.pre_results: Dict[str, Any] = {metric: {} for metric in self.metrics}
        self.results: Dict[str, Any] = {metric: {} for metric in self.metrics}
        self.summary_results = None

        self._shared_res_file = lambda metric: os.path.join(self.ref_dir, f"{self.ref_name}_{metric}.csv")
        self._summary_file = os.path.join(self.dir, f"{self.ref_name}_summary.csv") if self.dir else None

        # TODO:
        # For the user it could be important to get some warning when reinitializing the Evaluator with new
        # params but still having the old directory. The problem is then that old results are loaded that are
        # calculated with old parameters. Don't know exactly how to do this to make it still pipeline friendly

    def compute_or_load_shared_results(
        self,
    ):
        """Compute results that are potentially reused for evaluations of different probesets"""

        for metric in self.metrics:
            if self.ref_dir and os.path.isfile(self._shared_res_file(metric)):
                self.shared_results[metric] = pd.read_csv(self._shared_res_file(metric), index_col=0)
            else:
                self.shared_results[metric] = metric_shared_computations(
                    self.adata,
                    metric=metric,
                    parameters=self.metrics_params[metric],
                )
                if self.ref_dir and (self.shared_results[metric] is not None):
                    Path(self.ref_dir).mkdir(parents=True, exist_ok=True)
                    self.shared_results[metric].to_csv(self._shared_res_file(metric))

    def evaluate_probeset(self, genes, set_id="probeset1", update_summary=True, pre_only=False):
        """Compute probe set specific evaluations

        For some metrics the computations are split up in pre computations (independent of shared results) and the
        computations where the shared results are used.

        probeset_name: str
            Name of probeset. This is chosen automatically if `None` is given.
        update_summary: bool
            Whether to compute summary statistics, update the summary table and also update the summary csv file if
            self.dir is not None. This option is interesting when using ProbesetEvaluator in a distributed pipeline
            since multiple processes would access the same file in parallel.
        pre_only: bool
            For some metrics there are computationally expensive calculations that can be started independent of
            the shared results being finished. This is interesting for a parallelised pipeline.
            If `pre_only` is set to True only these pre calculations are computed.
        """
        if not pre_only:
            self.compute_or_load_shared_results()

        # Probeset specific pre computation (shared results are not needed for these)
        for metric in self.metrics:
            if (self.dir is None) or (not os.path.isfile(self._res_file(metric, set_id, pre=True))):
                self.pre_results[metric][set_id] = metric_pre_computations(
                    genes,
                    adata=self.adata,
                    metric=metric,
                    parameters=self.metrics_params[metric],
                )
                if self.dir and (self.pre_results[metric][set_id] is not None):
                    Path(os.path.dirname(self._res_file(metric, set_id, pre=True))).mkdir(parents=True, exist_ok=True)
                    self.pre_results[metric][set_id].to_csv(self._res_file(metric, set_id, pre=True))
            elif os.path.isfile(self._res_file(metric, set_id, pre=True)):
                self.pre_results[metric][set_id] = pd.read_csv(self._res_file(metric, set_id, pre=True), index_col=0)

        # Probeset specific computation (shared results are needed)
        if not pre_only:
            for metric in self.metrics:
                if (self.dir is None) or (not os.path.isfile(self._res_file(metric, set_id))):
                    self.results[metric][set_id] = metric_computations(
                        genes,
                        adata=self.adata,
                        metric=metric,
                        shared_results=self.shared_results[metric],
                        pre_results=self.pre_results[metric][set_id],
                        parameters=self.metrics_params[metric],
                        n_jobs=self.n_jobs,
                    )
                    if self.dir:
                        Path(os.path.dirname(self._res_file(metric, set_id))).mkdir(parents=True, exist_ok=True)
                        self.results[metric][set_id].to_csv(self._res_file(metric, set_id))

            if update_summary:
                self.summary_statistics(set_ids=[set_id])

    def evaluate_probeset_pipeline(
        self, genes, set_id: str, shared_pre_results_path: list, step_specific_results: list
    ):
        """Pipeline specific adaption of evaluate_probeset.

        Computes probeset specific evaluations. The parameters for this function are adapted for the spapros-pipeline

        Args:
            genes: Genes by the 'get_genes' function.
            set_id: ID of the current probeset
            shared_pre_results_path: Path to the shared results
            step_specific_results: List of paths to the specific results
        """
        # Load shared and pre results
        for metric in self.metrics:
            matches = list(filter(lambda result: metric in result, shared_pre_results_path))
            self.shared_results[metric] = pd.read_csv(matches[0], index_col=0)
            if step_specific_results is not None:
                matches = list(filter(lambda result: metric in result, step_specific_results))
                self.pre_results[metric][set_id] = pd.read_csv(matches[0], index_col=0)
            else:
                self.pre_results[metric][set_id] = metric_pre_computations(
                    genes,
                    adata=self.adata,
                    metric=metric,
                    parameters=self.metrics_params[metric],
                )

            # evaluate probeset
            self.results[metric][set_id] = metric_computations(
                genes,
                adata=self.adata,
                metric=metric,
                shared_results=self.shared_results[metric],
                pre_results=self.pre_results[metric][set_id],
                parameters=self.metrics_params[metric],
                n_jobs=self.n_jobs,
            )
            if self.dir:
                Path(os.path.dirname(self._res_file(metric, set_id))).mkdir(parents=True, exist_ok=True)
                self.results[metric][set_id].to_csv(self._res_file(metric, set_id))

    def summary_statistics(self, set_ids):
        """Compute summary statistics and update summary csv (if self.results_dir is not None)"""
        df = self._init_summary_table(set_ids)

        for set_id in set_ids:
            for metric in self.metrics:
                if (set_id in self.results[metric]) and (self.results[metric][set_id] is not None):
                    results = self.results[metric][set_id]
                elif self.dir and os.path.isfile(self._res_file(metric, set_id)):
                    results = pd.read_csv(self._res_file(metric, set_id), index_col=0)
                summary = metric_summary(
                    adata=self.adata, results=results, metric=metric, parameters=self.metrics_params[metric]
                )
                for key in summary:
                    df.loc[set_id, key] = summary[key]
        if self.dir:
            df.to_csv(self._summary_file)

        self.summary_results = df

    def pipeline_summary_statistics(self, result_files: list, probeset_ids: str) -> None:
        """Adaptation of the function summary_statistics for the spapros-pipeline.

        Takes the input files directly to calculate the summary statistics.

        Args:
            result_files: Probeset evaluation result file paths
            probeset_ids: Probeset ids as a single string in the format: probe_id1,probe_id2,probe_id3
        """
        df = self._init_summary_table(probeset_ids)

        # Example file name: gene_corr_small_data_genesets_1_1.csv
        for result_file in result_files:
            # Assumption for the set ID: last 3 words minus file extension split by _
            set_id = "_".join(result_file[:-4].split("_")[-3:])
            # Assumption for the metric: first 2 words split by _
            metric = "_".join(result_file[:-4].split("_")[:2])

            if (set_id in self.results[metric]) and (self.results[metric][set_id] is not None):
                results = self.results[metric][set_id]
            else:
                results = pd.read_csv(result_file, index_col=0)

            summary = metric_summary(
                adata=self.adata, results=results, metric=metric, parameters=self.metrics_params[metric]
            )
            for key in summary:
                df.loc[set_id, key] = summary[key]

        if self.dir:
            from pathlib import Path

            output_dir = Path(self.dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            df.to_csv(self._summary_file)

        self.summary_results = df

    def _prepare_metrics_params(self, new_params):
        """Set metric parameters to default values and overwrite defaults in case user defined param is given"""
        params = get_metric_default_parameters()
        for metric in params:
            if metric in new_params:
                for param in params[metric]:
                    if param in new_params[metric]:
                        params[metric][param] = new_params[metric][param]
        if self.marker_list is not None:
            params["marker_corr"]["marker_list"] = self.marker_list
        if self.celltype_key is not None:
            params["forest_clfs"]["ct_key"] = self.celltype_key
        return params

    def _get_metrics_of_scheme(
        self,
    ):
        """ """

        if self.scheme == "quick":
            metrics = ["knn_overlap", "forest_clfs", "gene_corr"]
        elif self.scheme == "full":
            metrics = ["cluster_similarity", "knn_overlap", "forest_clfs", "gene_corr"]

        # Add marker correlation metric if a marker list is provided
        if ("marker_corr" in self.metrics_params) and ("marker_list" in self.metrics_params["marker_corr"]):
            if self.metrics_params["marker_corr"]["marker_list"]:
                metrics.append("marker_corr")

        return metrics

    def _init_summary_table(self, set_ids):
        """Initialize or load table with summary results

        Note that column names for the summary metrics are not initialized here (except of the ones that already exist
        in the csv).

        set_ids: list of strs
            Initialize dataframe with set_ids as index. We also keep all set_ids that already exist in the summary csv.

        Returns
        -------
        pd.DataFrame with set_ids as index
        """
        if self.dir:
            if os.path.isfile(self._summary_file):
                df = pd.read_csv(self._summary_file, index_col=0)
                sets_tmp = [s for s in set_ids if (s not in df.index)]
                return pd.concat([df, pd.DataFrame(index=sets_tmp)])
        return pd.DataFrame(index=set_ids)

    def _res_file(
        self,
        metric,
        set_id,
        pre=False,
    ):
        """ """
        pre_str = "_pre" if pre else ""
        return os.path.join(self.dir, f"{metric}/{metric}_{self.ref_name}_{set_id}{pre_str}.csv")

    def _default_reference_dir(
        self,
    ):
        """ """
        if self.dir:
            return os.path.join(self.dir, "references")
        else:
            return None

    def plot_summary(
        self,
        set_ids="all",
        **plot_kwargs,
    ):
        """Plot heatmap of summary metrics

        set_ids: "all" or list of strs
        """
        if (self.summary_results is None) and self.dir:
            self.summary_results = pd.read_csv((self._summary_file), index_col=0)
        if set_ids == "all":
            set_ids = self.summary_results.index.tolist()
        table = self.summary_results.loc[set_ids]
        pl.summary_table(table, **plot_kwargs)

    def plot_evaluations(
        self,
        set_ids="all",
        metrics="all",
        show=True,
        save=False,
        plt_kwargs={},
    ):
        """Plot detailed results plots for specified metrics

        Note: not all plots can be supported here. If we want to plot a penalty kernel for example we're missing the
              kernel info. For such plots we need separate functions.

        set_ids: "all" or list of strs
            Check out self.summary_results for available sets.
        metrics: "all" or list of strs
            Check out

        """

        if set_ids == "all":
            if self.dir:
                self.summary_results = pd.read_csv(self._summary_file, index_col=0)
            set_ids = self.summary_results.index.tolist()

        if metrics == "all":
            metrics = self.metrics

        for metric in metrics:
            if metric not in self.results:
                self.results[metric] = {}
            for set_id in set_ids:
                if (set_id not in self.results[metric]) or (self.results[metric][set_id] is None):
                    try:
                        self.results[metric][set_id] = pd.read_csv(self._res_file(metric, set_id), index_col=0)
                    except FileNotFoundError:
                        print(f"No results file found for set {set_id} for metric {metric}.")

        if "forest_clfs" in metrics:
            conf_plt_kwargs = plt_kwargs["forest_clfs"] if ("forest_clfs" in plt_kwargs) else {}
            pl.confusion_heatmap(set_ids, self.results["forest_clfs"], **conf_plt_kwargs)

        if "gene_corr" in metrics:
            corr_plt_kwargs = plt_kwargs["gene_corr"] if ("gene_corr" in plt_kwargs) else {}
            pl.correlation_matrix(set_ids, self.results["gene_corr"], **corr_plt_kwargs)


########################################################################################################
########################################################################################################
########################################################################################################
# evaluation.py will be reserved for the ProbesetEvaluator class only at the end. Therefore
# the following functions will be moved or removed (quite a few dependencies need to be adjusted first though)


def plot_gene_expressions(adata, f_idxs, fig_title=None, save_to=None):
    a = adata.copy()  # TODO Think we can get rid of this copy and just work with the views
    gene_obs = []
    for _, f_idx in enumerate(f_idxs):
        gene_name = a.var.index[f_idx]
        a.obs[gene_name] = a.X[:, f_idx]
        gene_obs.append(gene_name)

    fig = sc.pl.umap(a, color=gene_obs, ncols=4, return_fig=True)
    fig.suptitle(fig_title)
    if save_to is not None:
        fig.savefig(save_to)
    plt.show()


def plot_nmis(results_path, cols=None, colors=None, labels=None, legend=None):
    """Custom legend: e.g. legend = [custom_lines,line_names]

    custom_lines = [Line2D([0], [0], color='red',    lw=linewidth),
                    Line2D([0], [0], color='orange', lw=linewidth),
                    Line2D([0], [0], color='green',  lw=linewidth),
                    Line2D([0], [0], color='blue',   lw=linewidth),
                    Line2D([0], [0], color='cyan',   lw=linewidth),
                    Line2D([0], [0], color='black',  lw=linewidth),
                    ]
    line_names = ["dropout", "dropout 1 donor", "pca", "marker", "random"]
    """
    df = pd.read_csv(results_path, index_col=0)
    fig = plt.figure(figsize=(10, 6))
    if cols is not None:
        for col in df.columns:
            plt.plot(df.index, df[col], label=col)
    else:
        for i, col in enumerate(cols):
            if labels is None:
                label = col
            else:
                label = labels[i]
            if colors is None:
                plt.plot(df.index, df[col], label=label)
            else:
                plt.plot(df.index, df[col], label=label, c=colors[i])
    if legend is None:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(
            legend[0],
            legend[1],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
    plt.xlabel("n clusters")
    plt.ylabel("NMI")
    plt.show()
    return fig


##########################################################
############### tree_classifications() ###################
##########################################################
# Note that the single tree classifications are too noisy
# we go with random forest classifications now, picking the best tree


def split_train_test_sets(adata, split=4, seed=2020, verbose=True, obs_key=None):
    """Split data to train and test set

    obs_key: str
        Provide a column name of adata.obs. If an obs_key is provided each group is split with the defined ratio.
    """
    if not obs_key:
        n_train = (adata.n_obs // (split + 1)) * split
        np.random.seed(seed=seed)
        train_obs = np.random.choice(adata.n_obs, n_train, replace=False)
        test_obs = np.array([True for i in range(adata.n_obs)])
        test_obs[train_obs] = False
        train_obs = np.invert(test_obs)
        if verbose:
            print(f"Split data to ratios {split}:1 (train:test)")
            print(f"datapoints: {adata.n_obs}")
            print(f"train data: {np.sum(train_obs)}")
            print(f"test data: {np.sum(test_obs)}")
        adata.obs["train_set"] = train_obs
        adata.obs["test_set"] = test_obs
    else:
        adata.obs["train_set"] = False
        adata.obs["test_set"] = False
        for group in adata.obs[obs_key].unique():
            df = adata.obs.loc[adata.obs[obs_key] == group]
            n_obs = len(df)
            n_train = (n_obs // (split + 1)) * split
            np.random.seed(seed=seed)
            train_obs = np.random.choice(n_obs, n_train, replace=False)
            test_obs = np.array([True for i in range(n_obs)])
            test_obs[train_obs] = False
            train_obs = np.invert(test_obs)
            if verbose:
                print(f"Split data for group {group}")
                print(f"to ratios {split}:1 (train:test)")
                print(f"datapoints: {n_obs}")
                print(f"train data: {np.sum(train_obs)}")
                print(f"test data: {np.sum(test_obs)}")
            adata.obs.loc[df.index, "train_set"] = train_obs
            adata.obs.loc[df.index, "test_set"] = test_obs


############## FOREST CLASSIFICATIONS ##############


def get_celltypes_with_too_small_test_sets(adata, ct_key, min_test_n=20, split_kwargs={"seed": 0, "split": 4}):
    """Get celltypes whith test set sizes below `min_test_n`

    We split the observations in adata into train and test sets for forest training. Check if the resulting
    test sets have at least `min_test_n` samples.

    Arguments
    ---------
    adata: AnnData
    ct_key: str
        adata.obs key for cell types
    min_test_n: int
        Minimal number of samples in each celltype's test set
    split_kwargs: dict
        Keyword arguments for ev.split_train_test_sets()

    Returns
    -------
    cts_below_min: list
        celltypes with too small test sets
    counts_below_min: list
        test set sample count for each celltype in celltype_list

    """
    a = adata.copy()
    split_train_test_sets(a, verbose=False, obs_key=ct_key, **split_kwargs)
    a = a[a.obs["test_set"], :]

    counts = a.obs.groupby(ct_key)["test_set"].value_counts()
    below_min = counts < min_test_n
    cts_below_min = [idx[0] for idx in below_min[below_min].index]
    counts_below_min = counts[below_min].tolist()
    return cts_below_min, counts_below_min


def uniform_samples(adata, ct_key, set_key="train_set", subsample=500, seed=2020, celltypes="all"):
    """Subsample `subsample` cells per celltype

    If the number of cells of a celltype is lower we're oversampling that celltype.
    """
    a = adata[adata.obs[set_key], :]
    if celltypes == "all":
        celltypes = list(a.obs[ct_key].unique())
    # Get subsample for each celltype
    all_obs = []
    for ct in celltypes:
        df = a.obs.loc[a.obs[ct_key] == ct]
        n_obs = len(df)
        np.random.seed(seed=seed)
        if n_obs > subsample:
            obs = np.random.choice(n_obs, subsample, replace=False)
            all_obs += list(df.iloc[obs].index.values)
        else:
            obs = np.random.choice(n_obs, subsample, replace=True)
            all_obs += list(df.iloc[obs].index.values)

    if scipy.sparse.issparse(a.X):
        X = a[all_obs, :].X.toarray()
    else:
        X = a[all_obs, :].X.copy()
    y = {}
    for ct in celltypes:
        y[ct] = np.where(a[all_obs, :].obs[ct_key] == ct, ct, "other")

    cts = a[all_obs].obs[ct_key].values

    return X, y, cts


def save_forest(results, path):
    """Save forest results to file

    results: list
        Output from forest_classifications()
    path: str
        Path to save file
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_forest(path):
    """Load forest results from path

    path: str
        Path to file
    """
    return pickle.load(open(path, "rb"))


def get_reference_masks(cts, ct_to_ref):
    """Get celltype specific boolean masks over celltype annotation vector

    cts: list
        celltype annotations.
    ct_to_ref: dict
        Each celltype's list of reference celltypes e.g.:
        {'AT1':['AT1','AT2','Club'],'Pericytes':['Pericytes','Smooth muscle']}
    """
    masks = {}
    for ct, ref in ct_to_ref.items():
        masks[ct] = np.in1d(cts, ref)
    return masks


def train_ct_tree_helper(celltypes, X_train, y_train, seed, max_depth=3, masks=None, queue=None):
    """Train decision trees parallelized over celltypes

    TODO: Write docstring
    """
    ct_trees = {}
    for ct in celltypes:
        ct_trees[ct] = tree.DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=max_depth, random_state=seed, class_weight="balanced"  # 3,
        )
        if masks is None:
            ct_trees[ct] = ct_trees[ct].fit(X_train, y_train[ct])
        elif np.sum(masks[ct]) > 0:
            ct_trees[ct] = ct_trees[ct].fit(X_train[masks[ct], :], y_train[ct][masks[ct]])

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return ct_trees


def pool_train_ct_tree_helper(ct_trees_dicts):
    """Combine list of dictonaries to one dict

    TODO: Write docstring
    """
    tmp = [ct for trees_dict in ct_trees_dicts for ct in trees_dict]
    if len(set(tmp)) < len(tmp):
        raise ValueError("Multiple trees for the same celltype are in the results of the parallelized execution")
    ct_trees = {ct: tree for ct_trees_dict in ct_trees_dicts for ct, tree in ct_trees_dict.items()}
    return ct_trees


def eval_ct_tree_helper(ixs, celltypes, ref_celltypes, ct_trees, X_test, y_test, cts_test, masks=None, queue=None):
    """
    TODO: Write docstring

    Returns
    -------
    f1_scores:
    """
    tree_names = [str(i) for i in ixs]
    summary_metric = pd.DataFrame(index=celltypes, columns=tree_names, dtype="float64")
    specificities = {ct: pd.DataFrame(index=ref_celltypes, columns=tree_names, dtype="float64") for ct in celltypes}
    for i in ixs:
        for ct in celltypes:
            if (masks is None) or (np.sum(masks[ct]) > 0):
                X = X_test if (masks is None) else X_test[masks[ct], :]
                y = y_test[ct] if (masks is None) else y_test[ct][masks[ct]]
                prediction = ct_trees[ct][i].predict(X)
                report = classification_report(y, prediction, output_dict=True)
                summary_metric.loc[ct, str(i)] = report["macro avg"]["f1-score"]
                for ct2 in [c for c in ref_celltypes if not (c == ct)]:
                    idxs = (cts_test == ct2) if (masks is None) else (cts_test[masks[ct]] == ct2)
                    if np.sum(idxs) > 0:
                        specificity = np.sum(prediction[idxs] == y[idxs]) / np.sum(idxs)
                        specificities[ct].loc[ct2, str(i)] = specificity

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return summary_metric, specificities


def pool_eval_ct_tree_helper(f1_and_specificities):
    """
    TODO: Write docstring

    We parallelize over n_trees
    """
    summary_metric_dfs = [val[0] for val in f1_and_specificities]
    specificity_df_dicts = [val[1] for val in f1_and_specificities]
    summary_metric = pd.concat(summary_metric_dfs, axis=1)
    cts = list(specificity_df_dicts[0])
    specificities = {
        ct: pd.concat([specificity_df_dicts[i][ct] for i in range(len(specificity_df_dicts))], axis=1) for ct in cts
    }
    return summary_metric, specificities


def single_forest_classifications(
    adata,
    selection,
    celltypes="all",
    ref_celltypes="all",
    ct_key="Celltypes",
    ct_spec_ref=None,
    save=False,
    seed=0,
    n_trees=50,
    max_depth=3,
    subsample=1000,
    test_subsample=3000,
    sort_by_tree_performance=True,
    verbose=False,
    return_clfs=False,
    n_jobs=1,
    backend="loky",
):
    """Compute or load decision tree classification results

    TODO: This doc string is partially from an older version. Update it! (Descripiton and Return is already up to date)
    TODO: Add progress bars to trees, and maybe change verbose to verbosity levels

    As metrics we use:
    macro f1 score as summary statistic - it's a uniformly weighted statistic wrt celltype groups in 'others' since
    we sample uniformly.
    For the reference celltype specific metric we use specificity = TN/(FP+TN) (also because FN and TP are not feasible
    in the given setting)

    Parameters
    ----------
    adata: AnnData
    selection: list or pd.DataFrame
        Trees are trained on genes of the list or genes defined in the bool column selection['selection'].
    celltypes: 'all' or list
        Trees are trained on the given celltypes
    ct_key: str
        Column name of adata.obs with celltype infos
    ct_spec_ref: dict of lists
        Celltype specific references (e.g.: {'AT1':['AT1','AT2','Club'],'Pericytes':['Pericytes','Smooth muscle']}).
        This argument was introduced to train secondary trees.
    save: str or False
        If not False load results if the given file exists, otherwise save results after computation.
    max_depth: str
        max_depth argument of DecisionTreeClassifier.
    subsample: int
        For each trained tree we use samples of maximal size=`subsample` for each celltype. If fewer cells
        are present for a given celltype all cells are used.
    sort_by_tree_performance: str
        Wether to sort results and trees by tree performance (best first) per celltype
    return_clfs: str
        Wether to return the sklearn tree classifier objects. (if `return_clfs` and `save_load` we still on
        save the results tables, if you want to save the classifiers this needs to be done separately).
    n_jobs: int
        Multiprocessing number of processes.
    backend: str
        Which backend to use for multiprocessing. See class `joblib.Parallel` for valid options.

    Returns
    -------
    Note in all output files trees are ordered according macro f1 performance.

    summary_metric: pd.DataFrame
        macro f1 scores for each celltype's trees (Ordered according best performing trees)
    ct_specific_metric: dict of pd.DataFrame
        For each celltype's tree: specificity (= TN / (FP+TN)) wrt each other celltype's test sample
    importances: dict of pd.DataFrame
        Gene's feature importances for each tree.

    if return_clfs:
        return [summary_metric,ct_specific_metric,importances], forests

    """

    if verbose:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm_notebook as tqdm
    else:
        tqdm = None

    n_jobs = _get_n_cores(n_jobs)

    if isinstance(selection, (list, np.ndarray, np.generic)):
        genes = selection
    elif isinstance(selection, pd.Index):
        genes = selection.values
    elif isinstance(selection, pd.DataFrame):
        genes = list(selection.loc[selection["selection"]].index)
    a = adata[:, genes].copy()

    # apply a train test set split one time just to get test set numbers for each celltype to eventually filter out celltypes
    split_train_test_sets(a, split=4, seed=2020, verbose=False, obs_key=ct_key)
    if celltypes == "all":
        celltypes = np.unique(a.obs[ct_key].values)
    if ref_celltypes == "all":
        ref_celltypes = np.unique(a.obs[ct_key].values)
    celltypes_tmp = [
        ct
        for ct in celltypes
        if (ct in a.obs.loc[a.obs["train_set"], ct_key].values) and (ct in a.obs.loc[a.obs["test_set"], ct_key].values)
    ]
    ref_celltypes_tmp = [
        ct
        for ct in ref_celltypes
        if (ct in a.obs.loc[a.obs["train_set"], ct_key].values) and (ct in a.obs.loc[a.obs["test_set"], ct_key].values)
    ]
    for c in [c for c in celltypes if c not in celltypes_tmp]:
        warnings.warn(f"Zero cells of celltype {c} in train or test set. No tree is calculated for celltype {c}.")
    for c in [c for c in ref_celltypes if c not in ref_celltypes_tmp]:
        warnings.warn(
            f"Zero cells of celltype {c} in train or test set. Celltype {c} is not included as reference celltype."
        )
    celltypes = celltypes_tmp
    ref_celltypes = ref_celltypes_tmp
    cts_not_in_ref = [ct for ct in celltypes if not (ct in ref_celltypes)]
    if cts_not_in_ref:
        warnings.warn(
            f"For celltypes {cts_not_in_ref} trees are computed, they are not listed in reference celltypes though. Added them..."
        )
        ref_celltypes += cts_not_in_ref

    # Reset subsample and test_subsample if dataset is actually too small.
    max_counts_train = a.obs.loc[a.obs["train_set"], ct_key].value_counts().loc[ref_celltypes].max()
    max_counts_test = a.obs.loc[a.obs["test_set"], ct_key].value_counts().loc[ref_celltypes].max()
    if subsample > max_counts_train:
        subsample = max_counts_train
    if test_subsample > max_counts_test:
        test_subsample = max_counts_test

    X_test, y_test, cts_test = uniform_samples(
        a, ct_key, set_key="test_set", subsample=test_subsample, seed=seed, celltypes=ref_celltypes
    )
    if ct_spec_ref is not None:
        masks_test = get_reference_masks(cts_test, ct_spec_ref)
    else:
        masks_test = None

    ct_trees = {ct: [] for ct in celltypes}
    np.random.seed(seed=seed)
    seeds = np.random.choice(100000, n_trees, replace=False)
    # Compute trees (for each tree index we parallelize over celltypes)
    for i in tqdm(range(n_trees), desc="Train trees") if tqdm else range(n_trees):
        X_train, y_train, cts_train = uniform_samples(
            a, ct_key, set_key="train_set", subsample=subsample, seed=seeds[i], celltypes=ref_celltypes
        )
        if ct_spec_ref is not None:
            masks = get_reference_masks(cts_train, ct_spec_ref)
        else:
            masks = None
        ct_trees_i = parallelize(
            callback=train_ct_tree_helper,
            collection=celltypes,
            n_jobs=n_jobs,
            backend=backend,
            extractor=pool_train_ct_tree_helper,
            show_progress_bar=False,  # verbose,
        )(X_train=X_train, y_train=y_train, seed=seeds[i], max_depth=max_depth, masks=masks)
        for ct in celltypes:
            ct_trees[ct].append(ct_trees_i[ct])
    # Get feature importances
    importances = {
        ct: pd.DataFrame(index=a.var.index, columns=[str(i) for i in range(n_trees)], dtype="float64")
        for ct in celltypes
    }
    for i in range(n_trees):
        for ct in celltypes:
            if (masks_test is None) or (np.sum(masks_test[ct]) > 0):
                importances[ct][str(i)] = ct_trees[ct][i].feature_importances_
    # Evaluate trees (we parallelize over tree indices)
    summary_metric, ct_specific_metric = parallelize(
        callback=eval_ct_tree_helper,
        collection=[i for i in range(n_trees)],
        n_jobs=n_jobs,
        backend=backend,
        extractor=pool_eval_ct_tree_helper,
        show_progress_bar=verbose,
        desc="Evaluate trees",
    )(
        celltypes=celltypes,
        ref_celltypes=ref_celltypes,
        ct_trees=ct_trees,
        X_test=X_test,
        y_test=y_test,
        cts_test=cts_test,
        masks=masks_test,
    )

    # Sort results
    if sort_by_tree_performance:
        for ct in summary_metric.index:
            if (masks_test is None) or (np.sum(masks_test[ct]) > 0):
                order = summary_metric.loc[ct].sort_values(ascending=False).index.values.copy()
                order_int = [summary_metric.loc[ct].index.get_loc(idx) for idx in order]
                summary_metric.loc[ct] = summary_metric.loc[ct, order].values
                ct_specific_metric[ct].columns = order.copy()
                importances[ct].columns = order.copy()
                ct_specific_metric[ct] = ct_specific_metric[ct].reindex(
                    sorted(ct_specific_metric[ct].columns, key=int), axis=1
                )
                importances[ct] = importances[ct].reindex(sorted(importances[ct].columns, key=int), axis=1)
                ct_trees[ct] = [ct_trees[ct][i] for i in order_int]

    # We change the f1 summary metric now, since we can't summarize this value anymore when including secondary trees.
    # When creating the results for secondary trees we take the specificities according reference celltypes of each tree.
    # Our new metric is just the mean of these specificities. Btw we still keep the ordering to be based on f1 scores.
    # Think that makes more sense since it's the best balanced result.
    # TODO TODO TODO: Change misleading variable names in other functions (where the old "f1_table" is used)
    summary_metric = summarize_specs(ct_specific_metric)

    if save:
        save_forest([summary_metric, ct_specific_metric, importances], save)

    if return_clfs:
        return [summary_metric, ct_specific_metric, importances], ct_trees
    else:
        return summary_metric, ct_specific_metric, importances


def summarize_specs(specs):
    """Summarize specificities to summary metrics per celltype"""
    cts = [ct for ct in specs]
    df = pd.DataFrame(index=cts, columns=specs[cts[0]].columns)
    for ct in specs:
        df.loc[ct] = specs[ct].mean()
    return df


def combine_tree_results(primary, secondary, with_clfs=False):
    """Combine results of primary and secondary trees

    There are three parts in the forest results:
    1. f1_table
    2. classification specificities
    3. feature_importances

    The output for 2. and 3. will be in the same form as the input.
    Specificities are taken from secondary where existend, otherwise from primary.
    Feature_importances are summed up (reasoning: distinguishing celltypes that are
    hard to distinguish is very important and therefore good to rank respective genes high).
    The f1 tables are just aggregated to a list

    primary: dict of pd.DataFrames
    secondary: dict of pd.DataFrames
    with_clfs: bool
        Whether primary, secondary and the output each contain a list of forest results and the forest classifiers
        or only the results.

    """
    expected_len = 2 if with_clfs else 3
    if (len(primary) != expected_len) or (len(secondary) != expected_len):
        raise ValueError(
            f"inputs primary and secondary are expected to be lists of length == {expected_len}, not {len(primary)},"
            f"{len(secondary)}"
        )

    if with_clfs:
        primary, primary_clfs = primary
        secondary, secondary_clfs = secondary

    combined = [0, {}, {}]
    ## f1 (exchanged by summary stats below)
    # for f1_table in [primary[0],secondary[0]]:
    #    if isinstance(f1_table,list):
    #        combined[0] += f1_table
    #    else:
    #        combined[0].append(f1_table)
    # specificities
    celltypes = [key for key in secondary[1]]
    combined[1] = {ct: df.copy() for ct, df in primary[1].items()}
    for ct in celltypes:
        filt = ~secondary[1][ct].isnull().all(axis=1)
        combined[1][ct].loc[filt] = secondary[1][ct].loc[filt]
    # summary stats
    combined[0] = summarize_specs(combined[1])
    # feature importance
    combined[2] = {ct: df.copy() for ct, df in primary[2].items()}
    for ct in celltypes:
        combined[2][ct] += secondary[2][ct].fillna(0)
        combined[2][ct] = combined[2][ct].div(combined[2][ct].sum(axis=0), axis=1)

    if with_clfs:
        combined_clfs = primary_clfs
        for ct in combined_clfs:
            if ct in secondary_clfs:
                combined_clfs[ct] += secondary_clfs[ct]
        return combined, combined_clfs
    else:
        return combined


def outlier_mask(df, n_stds=1, min_outlier_dif=0.02, min_score=0.9):
    """Get mask over df.index based on values in df columns"""
    crit1 = df < (df.mean(axis=0) - (n_stds * df.std(axis=0))).values[np.newaxis, :]
    crit2 = df < (df.mean(axis=0) - min_outlier_dif).values[np.newaxis, :]
    crit3 = df < min_score
    return (crit1 & crit2) | crit3


def get_outlier_reference_celltypes(specs, n_stds=1, min_outlier_dif=0.02, min_score=0.9):
    """For each celltype's best tree get reference celltypes with low performance

    specs: dict of pd.DataFrames
        Each celltype's specificities of reference celltypes.
    """
    outliers = {}
    for ct, df in specs.items():
        outliers[ct] = df.loc[outlier_mask(df[["0"]], n_stds, min_outlier_dif, min_score).values].index.tolist() + [ct]
        if len(outliers[ct]) == 1:
            outliers[ct] = []
    return outliers


def forest_classifications(
    adata, selection, max_n_forests=3, verbosity=1, save=False, outlier_kwargs={}, **forest_kwargs
):
    """Train best trees including secondary trees

    max_n_forests: int
        Number of best trees considered as a tree group. Including the primary tree.

    """

    if verbosity > 0:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm_notebook as tqdm
    else:
        tqdm = None

    ct_spec_ref = None
    res = None
    with_clfs = "return_clfs" in forest_kwargs

    for _ in tqdm(range(max_n_forests), desc="Train hierarchical trees") if tqdm else range(max_n_forests):
        new_res = single_forest_classifications(
            adata, selection, ct_spec_ref=ct_spec_ref, verbose=verbosity > 1, save=False, **forest_kwargs
        )
        res = new_res if (res is None) else combine_tree_results(res, new_res, with_clfs=with_clfs)
        specs = res[0][1] if with_clfs else res[1]
        ct_spec_ref = get_outlier_reference_celltypes(specs, **outlier_kwargs)

    if save:
        if with_clfs:
            save_forest(res[0], save)
        else:
            save_forest(res, save)

    return res


def forest_rank_table(importances, celltypes="all", return_ct_specific_rankings=False):
    """Rank genes according importances of the celltypes' forests

    celltypes: str or list of strs
        If 'all' create ranking based on all celltypes in importances. Otherwise base the ranking only on
        the trees of the subset `celltypes`.
    importances: dict of pd.DataFrame
        Output from `forest_classifications()`. DataFrame for each celltype's forest.
        Each column refers to genes of one tree. The columns are sorted according performance (best first)

    Returns
    -------
    pd.DataFrame
        index: Genes found in `importances`
        columns:
            - 'rank': integer, tree rank in which a gene occured the first time (best over all celltypes)
            - 'importance_score': max feature importance of gene over celltypes where the gene occured at `rank`
            - ct in celltypes:
    """
    im = (
        importances.copy()
        if (celltypes == "all")
        else {ct: df.copy() for ct, df in importances.items() if ct in celltypes}
    )

    # ranking per celltype
    worst_rank = max([len(im[ct].columns) + 1 for ct in im])
    for ct in im:
        im[ct] = im[ct].reindex(columns=im[ct].columns.tolist() + ["tree", "rank", "importance_score"])
        rank = 1
        for tree_idx, col in enumerate(im[ct].columns):
            filt = im[ct]["rank"].isnull() & (im[ct][col] > 0)
            if filt.sum() > 0:
                im[ct].loc[filt, ["tree", "rank"]] = [tree_idx, rank]
                im[ct].loc[filt, "importance_score"] = im[ct].loc[filt, col]
                rank += 1
        filt = im[ct]["rank"].isnull()
        im[ct].loc[filt, ["rank", "importance_score"]] = [worst_rank, 0]

    # Save celltype specific rankings in current form if later returned
    if return_ct_specific_rankings:
        im_ = {ct: df.copy() for ct, df in im.items()}

    # collapse to general ranking.
    for ct in im:
        im[ct].columns = [f"{ct}_{col}" for col in im[ct]]
    tab = pd.concat([df for _, df in im.items()], axis=1)
    tab["rank"] = tab[[f"{ct}_rank" for ct in im]].min(axis=1)
    tab["importance_score"] = 0
    tab = tab.reindex(columns=tab.columns.tolist() + [ct for ct in im])
    tab[[ct for ct in im]] = False
    for gene in tab.index:
        tmp_cts = [ct for ct in im if (tab.loc[gene, f"{ct}_rank"] == tab.loc[gene, "rank"])]
        tmp_scores = [
            tab.loc[gene, f"{ct}_importance_score"]
            for ct in im
            if (tab.loc[gene, f"{ct}_rank"] == tab.loc[gene, "rank"])
        ]
        tab.loc[gene, "importance_score"] = max(tmp_scores) if tmp_scores else 0
        if tab.loc[gene, "rank"] != worst_rank:
            tab.loc[gene, tmp_cts] = True

    tab = tab[["rank", "importance_score"] + [ct for ct in im]]
    tab = tab.sort_values(["rank", "importance_score"], ascending=[True, False])
    tab["rank"] = tab["rank"].rank(axis=0, method="dense", na_option="keep", ascending=True)

    if return_ct_specific_rankings:
        im_ = {ct: df.sort_values(["rank", "importance_score"], ascending=[True, False]) for ct, df in im_.items()}
        return tab, im_
    else:
        return tab
