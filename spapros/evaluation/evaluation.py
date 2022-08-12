import os
import pickle
import warnings
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import spapros.plotting as pl
from rich.progress import Progress
from sklearn import tree
from sklearn.metrics import classification_report
from spapros.evaluation.metrics import get_metric_default_parameters
from spapros.evaluation.metrics import get_metric_names
from spapros.evaluation.metrics import metric_computations
from spapros.evaluation.metrics import metric_pre_computations
from spapros.evaluation.metrics import metric_shared_computations
from spapros.evaluation.metrics import metric_summary
from spapros.util.mp_util import _get_n_cores
from spapros.util.mp_util import parallelize
from spapros.util.mp_util import Signal
from spapros.util.mp_util import SigQueue
from spapros.util.util import init_progress
from spapros.util.util import NestedProgress


# helper for type checking:


class Empty(Enum):
    token = None


_empty = Empty.token


class ProbesetEvaluator:
    """General class for probe set evaluation, comparison, plotting.

    Notes:
        The evaluator works on one given dataset and calculates metrics/analyses with respect to that dataset.

        The calculation steps of the metrics can be divided into:

        1. calculations that need to be run one time for the given dataset (not all metrics have this step)
        2. calculations that need to be run for each probe set

           a. calculations independent of 1.
           b. calculations dependent on 1. (if 1. existed for a given metric)

        3. Summarize results into summary statistics

        **Run evaluations**

            Evaluate a single probeset::

                evaluator = ProbesetEvaluator(adata)
                evaluator.evaluate_probeset(gene_set)

            In a pipeline to evaluate multiple probesets you would run

                - sequential setup::

                    evaluator = ProbesetEvaluator(adata)
                    for i, gene_set in enumerate(sets):
                        evaluator.evaluate_probeset(gene_set, set_id=f"set_{i}")

                - parallelised setup::

                    evaluator = ProbesetEvaluator(adata)
                    # 1. step:
                    evaluator.compute_or_load_shared_results()
                    # 2. step: parallelised processes
                    evaluator.evaluate_probeset(gene_set, set_id, update_summary=False, pre=True) # parallelised over set_ids
                    # 3. step: parallelised processes (needs 1. to be finished)
                    evaluator.evaluate_probeset(gene_set, set_id, update_summary=False) # parallelised over set_ids
                    # 4. step: (needs 3. to be finished)
                    evaluator.summary_statistics()

        **Reference evaluations**

        In practice the evaluations are meaningful when having reference evaluations to compare to.

        A simple way to get reference probe sets::

            reference_sets = spapros.selection.select_reference_probesets(adata)

        Evaluate them (we also provide ids to keep track of the probesets)::

            evaluator = ProbesetEvaluator(adata)
            for set_id, gene_set in reference_sets.items():
                evaluator.evaluate_probeset(gene_set, set_id=set_id)
            evaluator.plot_summary()

        **Evaluation schemes**

        Some metrics take very long to compute, we prepared different metric sets for a quick or a full evaluation.
        You can also specify the list of metrics yourself by setting ``scheme="custom"``.
        Note that in any scheme it might still be reasonable to adjust :attr:`metrics_params`.

        **Saving of results**

        If ``results_dir`` is not None we save the results in files.

        Why:

        - some computations are time demanding, especially when you evaluate multiple sets it's reasonable to keep results.
        - load previous results when initializing a :class:`ProbesetEvaluator`. Makes it very easy to access and compare
          old results.

        Two saving directories need to be distinguished:

        1. ``results_dir``: each probeset's evaluation results are saved here
        2. ``reference_dir``: for shared reference dataset results (default is ``reference_dir = results_dir + reference_name``)

        In which files the results are saved:

        - Shared computations are saved as::

            reference_dir # (default: results_dir+"references")
            └── {reference_name}_{metric}.csv # shared computations for given reference dataset

        - The final probeset specific results are saved as::

            results_dir
            ├── {metric} # one folder for each metric
            │   ├── {reference_name}_{set_id}_pre.csv # pre results file for given set_id, reference dataset, and metric
            │   │                                     # (only for some metrics)
            │   └── {reference_name}_{set_id}.csv # result file for given set_id, reference dataset, and metric
            └── {reference_name}_summary.csv # summary statistics

        **Plotting**

        Plot a summary metrics table to get an overall performance overview::

            evaluator.plot_summary()

        For each evaluation we provide a detailed plot, e.g.:

        - `forest_clfs`: heatmap of normalised confusion matrix
        - `gene_corr`: heatmap of ordered correlation matrix

        Create detailed plots with::

            evaluator.plot_evaluations()

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        celltype_key:
            The adata.obs key for cell type annotations. Provide a list of keys to calculate the according metrics on
            multiple keys.
        results_dir:
            Directory where probeset results are saved. Defaults to `./probeset_evaluation/`. Set to `None` if you don't
            want to save results. When initializing the class we also check for existing results.
            Note if
        scheme:
            Defines which metrics are calculated

                - `'quick'` : knn, forest classification, marker correlation (if marker list given), gene correlation
                - `'full'` : nmi, knn, forest classification, marker correlation (if marker list given), gene correlation
                - `'custom'`: define metrics of intereset in :attr:`metrics`

        metrics: Define which metrics are calculated. This is set automatically if :attr:`scheme != "custom"`. Supported are:

            - `'cluster_similarity'`
            - `'knn_overlap'`
            - `'forest_clfs'`
            - `'marker_corr'`
            - `'gene_corr'`

        metrics_params:
            Provide parameters for the calculation of each metric. E.g.::

                metrics_params = {
                    "nmi":{
                        "ns": [5,20],
                        "AUC_borders": [[7, 14], [15, 20]],
                    }
                }

            This overwrites the arguments ``ns`` and ``AUC_borders`` of the nmi metric. See
            :func:`.get_metric_default_parameters()` for the default values of each metric
        marker_list:
            Dictionary containing celltypes as keys and the respective markers as a list as values.
        reference_name:
            Name of reference dataset. This is chosen automatically if `None` is given.
        reference_dir:
            Directory where reference results are saved. If `None` is given ``reference_dir`` is set to
            ``results_dir+'reference/'``.
        verbosity:
            Verbosity level.
        n_jobs:
            Number of CPUs for multi processing computations. Set to `-1` to use all available CPUs.

    Attributes:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        celltype_key:
            The ``adata.obs`` key for cell type annotations or list of keys.
        dir:
            Directory where probeset results are saved.
        scheme:
            Defines which metrics are calculated
        marker_list:
            Celltypes and the respective markers.
        metrics_params:
            Parameters for the calculation of each metric. Either default or user specified.
        metrics:
            The metrics to be calculated. Either custom or defined according to :attr:`scheme`.
        ref_name:
            Name of reference dataset.
        ref_dir:
            Directory where reference results are saved.
        verbosity:
            Verbosity level.
        n_jobs:
            Number of CPUs for multi processing computations. Set to `-1` to use all available CPUs.
            Verbosity level.
        shared_results:
            Results of shared metric computations.
        pre_results:
            Results of metric pre computations.
        results:
            Results of probe set specific metric computations.
        summary_results:
            Table of summary statistics.

    """

    # TODO:
    #  Decide: saving results is nice since we don't need to keep them in memory. On the other hand the
    #  stuff doesn't need that much memory I think. Even if you have 100 probesets, it's not that much

    def __init__(
        self,
        adata: sc.AnnData,
        celltype_key: Union[str, List[str]] = "celltype",
        results_dir: Union[str, None] = "./probeset_evaluation/",
        scheme: str = "quick",
        metrics: Optional[List[str]] = None,
        metrics_params: Dict[str, Dict] = {},
        marker_list: Union[str, Dict[str, List[str]]] = None,
        reference_name: str = "adata1",
        reference_dir: str = None,
        verbosity: int = 1,
        n_jobs: int = -1,
    ) -> None:

        self.adata = adata
        self.celltype_key = celltype_key
        self.dir: Union[str, None] = results_dir
        self.scheme = scheme
        self.marker_list = marker_list
        self.metrics_params = self._prepare_metrics_params(metrics_params)
        self.metrics: List[str] = metrics if (scheme == "custom") else self._get_metrics_of_scheme()
        self.ref_name = reference_name
        self.ref_dir = reference_dir if (reference_dir is not None) else self._default_reference_dir()
        self.verbosity = verbosity
        self.n_jobs = n_jobs

        self.shared_results: Dict[str, Any] = {}
        self.pre_results: Dict[str, Any] = {metric: {} for metric in self.metrics}
        self.results: Dict[str, Any] = {metric: {} for metric in self.metrics}
        self.summary_results: pd.DataFrame

        self._shared_res_file = lambda metric: os.path.join(self.ref_dir, f"{self.ref_name}_{metric}.csv")
        self._summary_file: Union[str, Empty] = (
            os.path.join(self.dir, f"{self.ref_name}_summary.csv") if self.dir else _empty
        )

        self.progress = None
        self.started = False

        # TODO: (kind of solved with the progress bars: they say either calculating xy or loading xy
        # For the user it could be important to get some warning when reinitializing the Evaluator with new
        # params but still having the old directory. The problem is then that old results are loaded that are
        # calculated with old parameters. Don't know exactly how to do this to make it still pipeline friendly

    def compute_or_load_shared_results(
        self,
    ) -> None:
        """Compute results that are potentially reused for evaluations of different probesets."""

        if self.progress and self.verbosity > 0:
            task_shared = self.progress.add_task("Shared metric computations...", total=len(self.metrics), level=1)

        for metric in self.metrics:
            if self.ref_dir and os.path.isfile(self._shared_res_file(metric)):

                if self.progress and self.verbosity > 1:
                    task_load = self.progress.add_task(
                        "Loading shared computations for " + metric + "...", total=1, level=2
                    )

                self.shared_results[metric] = pd.read_csv(self._shared_res_file(metric), index_col=0)

                if self.progress and self.verbosity > 1:
                    self.progress.advance(task_load)

            else:
                self.shared_results[metric] = metric_shared_computations(
                    self.adata,
                    metric=metric,
                    parameters=self.metrics_params[metric],
                    progress=self.progress if self.verbosity > 1 else None,
                    level=2,
                    verbosity=self.verbosity,
                )
                if self.ref_dir and (self.shared_results[metric] is not None):
                    Path(self.ref_dir).mkdir(parents=True, exist_ok=True)
                    self.shared_results[metric].to_csv(self._shared_res_file(metric))

            if self.progress and self.verbosity > 0:
                self.progress.advance(task_shared)

    def evaluate_probeset(
        self, genes: List, set_id: str = "probeset1", update_summary: bool = True, pre_only: bool = False
    ) -> None:
        """Compute probe set specific evaluations.

        Notes:
            For some metrics, the computations are split up in pre computations (independent of shared results) and the
            computations where the shared results are used.

        Args:
            genes:
                The selected genes.
            set_id:
                ID of the current probeset. This is chosen automatically if `None` is given.
            update_summary:
                Whether to compute summary statistics, update the summary table and also update the summary csv file if
                :attr:`.dir` is not None. This option is interesting when using
                :class:`~evaluation.ProbesetEvaluator` in a distributed pipeline since multiple processes would access
                the same file in parallel.
            pre_only:
                For some metrics there are computationally expensive calculations that can be started independent of
                the shared results being finished. This is interesting for a parallelised pipeline. If :attr:`pre_only`
                is set to `True` only these pre calculations are computed.
        """

        try:
            self.progress, self.started = init_progress(None, verbosity=self.verbosity, level=1)
            if self.progress and self.verbosity > 0:
                evaluation_task = self.progress.add_task(
                    description="SPAPROS PROBESET EVALUATION:", only_text=True, header=True, total=0
                )

            if not pre_only:
                self.compute_or_load_shared_results()

            # Probeset specific pre computation (shared results are not needed for these)

            if self.progress and self.verbosity > 0:
                task_pre = self.progress.add_task(
                    "Probeset specific pre computations...", total=len(self.metrics), level=1
                )

            for metric in self.metrics:
                if self.dir:
                    pre_res_file: str = self._res_file(metric, set_id, pre=True)
                    pre_res_file_isfile = os.path.isfile(pre_res_file)
                else:
                    pre_res_file_isfile = False
                if (self.dir is None) or (not pre_res_file_isfile):
                    self.pre_results[metric][set_id] = metric_pre_computations(
                        genes,
                        adata=self.adata,
                        metric=metric,
                        parameters=self.metrics_params[metric],
                        progress=self.progress if self.verbosity > 1 else None,
                        level=2,
                        verbosity=self.verbosity,
                    )
                    if self.dir and (self.pre_results[metric][set_id] is not None):
                        Path(os.path.dirname(self._res_file(metric, set_id, pre=True))).mkdir(
                            parents=True, exist_ok=True
                        )
                        self.pre_results[metric][set_id].to_csv(self._res_file(metric, set_id, pre=True))
                elif os.path.isfile(self._res_file(metric, set_id, pre=True)):

                    if self.progress and self.verbosity > 1:
                        task_pre_load = self.progress.add_task(
                            "Loading pre computations for " + metric + "...", total=1, level=2
                        )

                    self.pre_results[metric][set_id] = pd.read_csv(
                        self._res_file(metric, set_id, pre=True), index_col=0
                    )

                    if self.progress and self.verbosity > 1:
                        self.progress.advance(task_pre_load)

                if self.progress and self.verbosity > 0:
                    self.progress.advance(task_pre)

            # Probeset specific computation (shared results are needed)

            if self.progress and self.verbosity > 0:
                task_final = self.progress.add_task(
                    "Final probeset specific computations...", total=len(self.metrics), level=1
                )

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
                            progress=self.progress if self.verbosity > 1 else None,
                            level=2,
                            verbosity=self.verbosity,
                        )
                        if self.dir:
                            Path(os.path.dirname(self._res_file(metric, set_id))).mkdir(parents=True, exist_ok=True)
                            self.results[metric][set_id].to_csv(self._res_file(metric, set_id))

                    elif os.path.isfile(self._res_file(metric, set_id, pre=False)):

                        if self.progress and self.verbosity > 1:
                            task_final_load = self.progress.add_task(
                                "Loading final computations for " + metric + "...", total=1, level=2
                            )

                        self.results[metric][set_id] = pd.read_csv(
                            self._res_file(metric, set_id, pre=False), index_col=0
                        )

                        if self.progress and self.verbosity > 1:
                            self.progress.advance(task_final_load)

                    if self.progress and self.verbosity > 0:
                        self.progress.advance(task_final)

                if update_summary:
                    # self.summary_statistics(set_ids=[set_id])
                    self.summary_statistics(set_ids=list(set(self._get_set_ids_with_results() + [set_id])))

            if self.progress and self.verbosity > 0:
                self.progress.advance(evaluation_task)
                self.progress.add_task(description="FINISHED\n", footer=True, only_text=True, total=0)

            if self.progress and self.started:
                self.progress.stop()

        except Exception as error:
            if self.progress:
                self.progress.stop()
            raise error

    def evaluate_probeset_pipeline(
        self, genes: List, set_id: str, shared_pre_results_path: List, step_specific_results: List
    ) -> None:
        """Pipeline specific adaption of evaluate_probeset.

        Computes probeset specific evaluations. The parameters for this function are adapted for the spapros-pipeline.

        Args:
            genes:
                Genes by the :func:`.get_genes` function.
            set_id:
                ID of the current probeset.
            shared_pre_results_path:
                Path to the shared results
            step_specific_results:
                List of paths to the specific results
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

    def summary_statistics(self, set_ids: List[str]) -> None:
        """Compute summary statistics and update summary csv.
        (if :attr:`.results_dir` is not None)

        Args:
            set_ids:
                IDs of the current probesets.
        """
        df = self._init_summary_table(set_ids)

        for set_id in set_ids:
            for metric in self.metrics:
                if (set_id in self.results[metric]) and (self.results[metric][set_id] is not None):
                    results = self.results[metric][set_id]
                elif self.dir and os.path.isfile(self._res_file(metric, set_id)):
                    results = pd.read_csv(self._res_file(metric, set_id), index_col=0)
                summary = metric_summary(results=results, metric=metric, parameters=self.metrics_params[metric])
                for key in summary:
                    df.loc[set_id, key] = summary[key]
        if self.dir:
            df.to_csv(self._summary_file)

        self.summary_results = df

    def pipeline_summary_statistics(self, result_files: List, probeset_ids: List[str]) -> None:
        """Adaptation of the function summary_statistics for the spapros-pipeline.

        Takes the input files directly to calculate the summary statistics.

        Args:
            result_files:
                Probeset evaluation result file paths
            probeset_ids:
                IDs of the current probesets.
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

            summary = metric_summary(results=results, metric=metric, parameters=self.metrics_params[metric])
            for key in summary:
                df.loc[set_id, key] = summary[key]

        if self.dir:
            from pathlib import Path

            output_dir = Path(self.dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            df.to_csv(self._summary_file)

        self.summary_results = df

    def _get_set_ids_with_results(
        self,
    ):
        """Get list of set ids that currently have results for any metric in the Evaluator"""

        set_ids = []
        for metric in self.metrics:
            set_ids += list(self.results[metric].keys())

        return list(set(set_ids))

    def _prepare_metrics_params(self, new_params: Dict[str, Dict]) -> Dict[str, Dict]:
        """Set metric parameters to default values and overwrite defaults in case user defined param is given.

        Args:
            new_params:
                User specified parameters for the calculation of the metrics.
        """
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
    ) -> List[str]:
        """Get the metrics according to the chosen scheme."""

        if self.scheme == "quick":
            metrics = ["knn_overlap", "forest_clfs", "gene_corr"]
        elif self.scheme == "full":
            metrics = ["cluster_similarity", "knn_overlap", "forest_clfs", "gene_corr"]

        # Add marker correlation metric if a marker list is provided
        if ("marker_corr" in self.metrics_params) and ("marker_list" in self.metrics_params["marker_corr"]):
            if self.metrics_params["marker_corr"]["marker_list"]:
                metrics.append("marker_corr")

        return metrics

    def _init_summary_table(self, set_ids: List[str]) -> pd.DataFrame:
        """Initialize or load table with summary results.

        Note that column names for the summary metrics are not initialized here (except of the ones that already exist
        in the csv).

        Args:
            set_ids:
                IDs of the current probesets. Initialize dataframe with set_ids as index. We also keep all set_ids that
                already exist in the summary csv.

        Returns:
            pd.DataFrame with set_ids as index
        """
        if self.dir:
            assert isinstance(self._summary_file, str)
            if os.path.isfile(self._summary_file):
                df = pd.read_csv(self._summary_file, index_col=0)
                sets_tmp = [s for s in set_ids if (s not in df.index)]
                return pd.concat([df, pd.DataFrame(index=sets_tmp)])
        return pd.DataFrame(index=set_ids)

    def _res_file(
        self,
        metric: str,
        set_id: str,
        pre: bool = False,
        dir: Optional[str] = None,
    ) -> str:
        """Get the default name for a result file.

        Args:
            metric:
                The calculated evaluation metric, for which the results will be stored.
            set_id:
                ID of the current probeset.
            pre:
                Whether the file will should pre calculations or probeset specific metric calculations.
            dir:
                Alternative results directory (instead of self.dir)
        """
        pre_str = "_pre" if pre else ""
        if dir is None:
            assert self.dir is not None
            return os.path.join(self.dir, f"{metric}/{metric}_{self.ref_name}_{set_id}{pre_str}.csv")
        else:
            return os.path.join(dir, f"{metric}/{metric}_{self.ref_name}_{set_id}{pre_str}.csv")

    def _default_reference_dir(
        self,
    ):
        """Get the default reference directory."""
        if self.dir:
            return os.path.join(self.dir, "references")
        else:
            return None

    def load_results(
        self,
        directories: Optional[Union[str, List[str]]] = None,
        reference_dir: Optional[str] = None,
        steps: List[str] = ["shared", "pre", "main", "summary"],
        set_ids: Optional[List[str]] = None,
        verbosity: int = 1,
    ) -> pd.DataFrame:
        """Load existing results from files of one or multiple evaluation output directories

        In case of multiple directories we assume that the different evaluations were done with the same parameters. You
        can control which metrics are loaded by setting :attr:`.ProbesetEvaluator.metrics`.

        Args:
            directories:
                Directory or list of directories of previous evaluations. If `None` is given it's set to
                :attr:`.ProbesetEvaluator.dir`.
            reference_dir:
                Directory with reference results. If `None` is given it's set to :attr:`.ProbesetEvaluator.ref_dir`.
            steps:
                The results steps that are loaded. These include
                    * `'shared'` - computations on the reference gene set
                    * `'pre'` - computations on the selected gene set independent of the results on the reference gene set
                    * `'main'` - computations on the selected gene set taking into account the reference gene set results
                    * `'summary'` - summary metrics
            set_ids:
                Optionally only load the results for a subset of set ids.
            verbosity:
                Verbosity level.

        Returns:
            pd.DataFrame
                A boolean table that indicates which results were loaded for each set_id. Note that some metrics don't
                have result files for certain steps.

        Examples:

            Load results from a previous evaluation

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selections = sp.se.select_reference_probesets(adata, methods=["DE", "HVG"], n=30, verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(adata, verbosity=0, results_dir="eval_results")
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)
                del evaluator

                evaluator = sp.ev.ProbesetEvaluator(adata, verbosity=0, results_dir="eval_results")
                df_info = evaluator.load_results()

            Load results from previous evaluations that were distributed in two directories

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()

                selections = sp.se.select_reference_probesets(
                    adata, methods=["DE", "HVG"], n=30, verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(
                    adata, verbosity=0, results_dir="eval_results1")
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                selections = sp.se.select_reference_probesets(
                    adata, methods=["PCA", "random"], n=30, verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(
                    adata, verbosity=0, results_dir="eval_results2")
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                evaluator = sp.ev.ProbesetEvaluator(adata, verbosity=2)
                df_info = evaluator.load_results(
                    directories=['./eval_results1/', './eval_results2/'], reference_dir="./eval_results1/references")

        """

        # Check if given steps are sound
        supported_steps = set(["shared", "pre", "main", "summary"])
        assert set(steps) <= supported_steps, f"Unsupported results steps: {set(steps)-supported_steps}"

        # Set directories to list
        if directories is None:
            if self.dir is None:
                raise ValueError("Neither `directories` are given nor `ProbesetEvaluator.dir`.")
            directories = [self.dir]
        elif isinstance(directories, str):
            directories = [directories]

        # Eventually update self.ref_dir if argument reference_dir is given
        if (reference_dir is not None) and (reference_dir != self.ref_dir):
            if verbosity > 0:
                print(f"Update self.ref_dir to {reference_dir}")
            self.ref_dir = reference_dir
        else:
            if not os.path.isdir(self.ref_dir):
                raise ValueError(
                    f"Directory with expected reference results does not exist: {self.ref_dir}. Set the "
                    "correct path with argument `reference_dir`."
                )

        self._shared_res_file = lambda metric: os.path.join(self.ref_dir, f"{self.ref_name}_{metric}.csv")

        # Get metrics for which reference files do exist
        metrics_with_ref_files = [metric for metric in self.metrics if os.path.isfile(self._shared_res_file(metric))]
        # Check if any metric was found. If not: check if results exist for another self.ref_name
        if (len(metrics_with_ref_files) == 0) and (len(os.listdir(self.ref_dir)) > 0):
            # note that all metric names have an underscore (file name e.g.: <ref_name>_gene_corr.csv)
            tmp_ref_names = list(set([file.rsplit("_", 2)[0] for file in os.listdir(self.ref_dir)]))
            if len(tmp_ref_names) == 1:
                self.ref_name = tmp_ref_names[0]
                if verbosity > 0:
                    print(f"Update self.ref_name to {self.ref_name}")
                metrics_with_ref_files = [
                    metric for metric in self.metrics if os.path.isfile(self._shared_res_file(metric))
                ]
            else:
                raise ValueError(
                    "No result files found for the given ref_name but for more than one other reference name."
                )

        # Check which set_ids and result files exist (note: "shared" results are no set_id specific and handled above)
        results_found = []
        summary_columns = []
        for dir in directories:
            for metric in self.metrics:
                files = os.listdir(os.path.join(dir, metric))
                for step in steps:
                    if step == "pre":
                        pre_files = [f for f in files if f.endswith("_pre.csv")]
                        set_ids_tmp = [f.rsplit("_pre.csv")[0].rsplit(f"{self.ref_name}_")[-1] for f in pre_files]
                        for set_id in set_ids_tmp:
                            results_found.append([dir, metric, step, set_id])
                    elif step == "main":
                        main_files = [f for f in files if f.endswith(".csv") and not f.endswith("_pre.csv")]
                        set_ids_tmp = [f.rsplit(".csv")[0].rsplit(f"{self.ref_name}_")[-1] for f in main_files]
                        for set_id in set_ids_tmp:
                            results_found.append([dir, metric, step, set_id])
            if "summary" in steps:
                tmp_summary_file = os.path.join(dir, f"{self.ref_name}_summary.csv")
                tmp_summary = pd.read_csv(tmp_summary_file, index_col=0)
                set_ids_tmp = tmp_summary.index.to_list()
                columns_subset = []
                for metric in get_metric_names():
                    if (metric in self.metrics) and np.any([metric in col for col in tmp_summary.columns]):
                        columns_subset += [col for col in tmp_summary.columns if metric in col]
                        for set_id in set_ids_tmp:
                            results_found.append([dir, metric, step, set_id])
                summary_columns.append(columns_subset)
        if len(summary_columns) > 1:
            if not np.all([set(summary_columns[i]) == set(summary_columns[0]) for i in range(1, len(summary_columns))]):
                raise ValueError("The column names in summary files of different directories are not identical.")

        # Reduce found results to set ids of interest if given
        if set_ids is not None:
            results_found = [r for r in results_found if r[3] in set_ids]

        # Create table with infos which result files were found
        df = pd.DataFrame(columns=["dir", "metric", "step", "set_id"], data=results_found)

        # Test if set_ids occur multiple times in different dirs
        set_id_occurence_per_dir = pd.crosstab(df["dir"], df["set_id"])
        set_id_in_multiple_dir = set_id_occurence_per_dir.sum() > set_id_occurence_per_dir.max()
        if set_id_in_multiple_dir.any():
            tmp_ids = set_id_in_multiple_dir.loc[set_id_in_multiple_dir]
            raise ValueError(f"Found results for same set_ids in multiple directories, ids: {tmp_ids.index.to_list()}")

        # Get all set ids with results if not set by user
        if set_ids is None:
            set_ids = df["set_id"].unique().tolist()

        # Initialize boolean table of found results
        df_bool = pd.DataFrame(
            index=[f"{metric}_{step}" for step in steps for metric in self.metrics], columns=set_ids, data=False
        )

        # Load shared results
        if "shared" in steps:
            for metric in metrics_with_ref_files:
                self.shared_results[metric] = pd.read_csv(self._shared_res_file(metric), index_col=0)
                df_bool.loc[f"{metric}_shared"] = True

        # Load pre results
        if "pre" in steps:
            for dir in df["dir"].unique():
                for metric in df.loc[(df["dir"] == dir) & (df["step"] == "pre"), "metric"].unique():
                    df_tmp = df.loc[(df["dir"] == dir) & (df["metric"] == metric) & (df["step"] == "pre")]
                    for set_id in df_tmp["set_id"]:
                        self.pre_results[metric][set_id] = pd.read_csv(
                            self._res_file(metric, set_id, pre=True, dir=dir), index_col=0
                        )
                        df_bool.loc[f"{metric}_pre", set_id] = True

        # Load main results
        if "main" in steps:
            for dir in df["dir"].unique():
                for metric in df.loc[(df["dir"] == dir) & (df["step"] == "main"), "metric"].unique():
                    df_tmp = df.loc[(df["dir"] == dir) & (df["metric"] == metric) & (df["step"] == "main")]
                    for set_id in df_tmp["set_id"]:
                        self.results[metric][set_id] = pd.read_csv(
                            self._res_file(metric, set_id, pre=False, dir=dir), index_col=0
                        )
                        df_bool.loc[f"{metric}_main", set_id] = True

        # Load summary results
        if "summary" in steps:
            summaries = []
            for dir in df["dir"].unique():
                summary_tmp = pd.read_csv(os.path.join(dir, f"{self.ref_name}_summary.csv"), index_col=0)
                summary_tmp = summary_tmp.loc[df.loc[(df["dir"] == dir) & (df["step"] == "summary"), "set_id"].unique()]
                summary_tmp = summary_tmp[summary_columns[0]]
                summaries.append(summary_tmp)
            if len(summaries) > 1:
                self.summary_results = pd.concat(summaries)
            else:
                self.summary_results = summaries[0]
            for _, row in df.loc[df["step"] == "summary"].iterrows():
                df_bool.loc[f"{row['metric']}_summary", row["set_id"]] = True

        return df_bool

    ############################
    ##    EVALUATION PLOTS    ##
    ############################

    def plot_summary(
        self,
        set_ids: Union[str, List[str]] = "all",
        **plot_kwargs,
    ) -> None:
        """Plot heatmap of summary metrics

        See our basic evaluation tutorial for descriptions of each metric.

        Args:
            set_ids:
                IDs of the current probesets or "all". Check out :attr:`.ProbesetEvaluator.summary_results` for
                available sets.
            **plot_kwargs:
                Keyword arguments for :func:`.summary_table`.

        Example:

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selections = sp.se.select_reference_probesets(
                    adata, methods=["PCA", "DE", "HVG", "random"], n=30, verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(adata, verbosity=0, results_dir=None)
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                evaluator.plot_summary(set_ids=["PCA", "DE", "HVG", "random (seed=0)"])

            .. image:: ../../docs/plot_examples/Evaluator_plot_summary.png


        """
        if self.summary_results is _empty:
            if self.dir:
                self.summary_results = pd.read_csv(self._summary_file, index_col=0)
            else:
                raise ValueError("No summaries found.")
        else:
            if set_ids == "all":
                set_ids = self.summary_results.index.tolist()
            table = self.summary_results.loc[set_ids]
            pl.summary_table(table, **plot_kwargs)

    def plot_cluster_similarity(
        self, set_ids: List[str] = None, selections_info: Optional[pd.DataFrame] = None, **kwargs
    ) -> None:
        """Plot cluster similarity as NMI over number of clusters

        Args:
            set_ids:
                List of probeset IDs. Check out :attr:`.ProbesetEvaluator.summary_results` for available sets.
            selections_info:
                Information on selections for plotting. The dataframe includes:

                    - selection ids or alternative names as index
                    - column: `path` (mandatory if ``data=None``): path to results csv of each selection (which contains
                      number of clusters (as index) and one column containing the data to plot)
                    - optional columns:

                        - `color`: matplotlib color
                        - `linewidth`: matplotlib linewidth
                        - `linestyle`: matplotlib linestyle
                        - `<groupby>`: some annotation that can be used to group the legend

                Note that the legend order will follow the row order in :attr:`selections_info`.
            **kwargs:
                Any keyword argument from :func:`.cluster_similarity`.


        Example:

            (Takes a few minutes to calculate)

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selections = sp.se.select_reference_probesets(
                    adata, methods=["PCA", "DE", "HVG", "random"], n=30, seeds=[0, 777], verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(
                    adata, verbosity=0, results_dir=None, scheme="custom", metrics=["cluster_similarity"])
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                evaluator.plot_cluster_similarity()

            .. image:: ../../docs/plot_examples/Evaluator_plot_cluster_similarity.png


        """

        if "cluster_similarity" not in self.results:
            raise ValueError("Can't plot cluster similarities since no results are found.")

        if selections_info is None:
            selections_info = pd.DataFrame(index=list(self.results["cluster_similarity"].keys()))

        if set_ids:
            selections_info = selections_info.loc[set_ids].copy()

        pl.cluster_similarity(
            selections_info,
            data=self.results["cluster_similarity"],
            **kwargs,
        )

    def plot_knn_overlap(
        self, set_ids: List[str] = None, selections_info: Optional[pd.DataFrame] = None, **kwargs
    ) -> None:
        """Plot mean knn overlap over k

        Args:
            set_ids:
                List of probeset IDs. Check out :attr:`.ProbesetEvaluator.summary_results` for available sets.
            selections_info:
                Information on selections for plotting. The dataframe includes:

                    - selection ids or alternative names as index
                    - column: `path` (mandatory if ``data=None``): path to results csv of each selection (which contains
                      number of clusters (as index) and one column containing the data to plot)
                    - optional columns:

                        - `color`: matplotlib color
                        - `linewidth`: matplotlib linewidth
                        - `linestyle`: matplotlib linestyle
                        - `<groupby>`: some annotation that can be used to group the legend

                Note that the legend order will follow the row order in :attr:`selections_info`.
            **kwargs:
                Any keyword argument from :func:`.knn_overlap`.


        Example:

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selections = sp.se.select_reference_probesets(
                    adata, methods=["PCA", "DE", "HVG", "random"], n=30, seeds=[0, 777], verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(
                    adata, verbosity=0, results_dir=None, scheme="custom", metrics=["knn_overlap"])
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                evaluator.plot_knn_overlap()

            .. image:: ../../docs/plot_examples/Evaluator_plot_knn_overlap.png


        """

        if "knn_overlap" not in self.results:
            raise ValueError("Can't plot KNN overlap since no results are found.")

        if selections_info is None:
            selections_info = pd.DataFrame(index=list(self.results["knn_overlap"].keys()))

        if set_ids:
            selections_info = selections_info.loc[set_ids].copy()

        pl.knn_overlap(
            selections_info,
            data=self.results["knn_overlap"],
            **kwargs,
        )

    def plot_confusion_matrix(self, set_ids: List[str] = None, **kwargs) -> None:
        """Plot heatmaps of cell type classification confusion matrices

        Args:
            set_ids:
                List of probeset IDs. Check out :attr:`.ProbesetEvaluator.summary_results` for available sets.
            **kwargs:
                Any keyword argument from :func:`.confusion_matrix`.


        Example:

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selections = sp.se.select_reference_probesets(adata,methods=["DE","HVG","random"],n=30,verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(adata, verbosity=0, results_dir=None, scheme="custom", metrics=["forest_clfs"])
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                evaluator.plot_confusion_matrix()

            .. image:: ../../docs/plot_examples/Evaluator_plot_confusion_matrix.png


        """

        if "forest_clfs" not in self.results:
            raise ValueError("Can't plot cell type classification since no results are found.")

        # if no set_ids specified, use all
        if not set_ids:
            set_ids = list(self.results["forest_clfs"].keys())

        assert isinstance(set_ids, list)

        pl.confusion_matrix(set_ids, self.results["forest_clfs"], **kwargs)

    def plot_coexpression(
        self,
        set_ids: List[str] = None,
        **kwargs,
    ) -> None:
        """Plot heatmaps of gene correlation matrices

        Args:
            set_ids:
                List of probeset IDs. Check out :attr:`.ProbesetEvaluator.summary_results` for available sets.
            **kwargs:
                Any keyword argument from :func:`.correlation_matrix`.


        Example:

            .. code-block:: python

                import spapros as sp
                adata = sp.ut.get_processed_pbmc_data()
                selections = sp.se.select_reference_probesets(adata, methods=["PCA","DE", "HVG", "random"], n=30, verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(adata, verbosity=0, results_dir=None, scheme="custom", metrics=["gene_corr"])
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                evaluator.plot_coexpression(n_cols=4)

            .. image:: ../../docs/plot_examples/Evaluator_plot_coexpression.png


        """

        if "gene_corr" not in self.results:
            raise ValueError("Can't plot gene correlation since no results are found.")

        # if no set_ids specified, use all
        if not set_ids:
            set_ids = list(self.results["gene_corr"].keys())

        assert isinstance(set_ids, list)

        pl.correlation_matrix(set_ids, self.results["gene_corr"], **kwargs)

    def plot_marker_corr(self, **kwargs):
        """Plot maximal correlations with marker genes

        Args:
            **kwargs:
                Any keyword argument from :func:`.marker_correlation`.


        Example:

            .. code-block:: python

                import spapros as sp
                marker_list ={
                    'B cells': ['EAF2', 'MS4A1', 'HVCN1', 'TCL1A', 'LINC00926', 'CD79A', 'IGLL5'],
                    'NK cells': ['XCL2', 'CLIC3', 'AKR1C3'],
                    'CD8 T cells': ['GZMK'],
                    'Dendritic cells': ['FCER1A', 'CLEC10A'],
                    'Megakaryocytes': ['RGS18','C2orf88','SDPR','TMEM40','GP9','MFSD1','PF4','PPBP'],
                }
                adata = sp.ut.get_processed_pbmc_data()
                selections = sp.se.select_reference_probesets(
                    adata, methods=["PCA", "DE", "HVG", "random"], n=30, seeds=range(7), verbosity=0)
                evaluator = sp.ev.ProbesetEvaluator(
                    adata, verbosity=0, results_dir=None, scheme="custom", metrics=["marker_corr"], marker_list=marker_list)
                for set_id, df in selections.items():
                    gene_set = df[df["selection"]].index.to_list()
                    evaluator.evaluate_probeset(gene_set, set_id=set_id)

                evaluator.plot_marker_corr()

            .. image:: ../../docs/plot_examples/Evaluator_plot_marker_corr.png


        """

        if "marker_corr" not in self.results:
            raise ValueError("Can't plot marker correlations since no results are found.")

        pl.marker_correlation(marker_corr=self.results["marker_corr"], **kwargs)

    # TODO remove this function (instead, we now have individual plot_'metric'() functions)
    def plot_evaluations(
        self,
        set_ids: Union[str, List[str]] = "all",
        metrics: Union[str, List[str]] = "all",
        show: bool = True,
        save: Union[str, bool] = False,
        plt_kwargs={},
    ) -> None:
        """Plot detailed results plots for specified metrics.

        Note:
            Not all plots can be supported here. If we want to plot a penalty kernel for example we're missing the
            kernel info. For such plots we need separate functions.

        Args:
            set_ids:
                ID of the current probeset or "all". Check out :attr:`.ProbesetEvaluator.summary_results` for available
                sets.
            metrics:
                List of calculated metrics or "all". Check out :attr:`.ProbesetEvaluator.metrics` for available metrics.
            save:
                If `True` or a `str`, save the figure.
            show:
                Show the figure.
            plt_kwargs:
                Keyword Args for the selection method specific plotting function.
                The used plotting functions are:

                .. list-table::
                    :header-rows: 1

                    * - selection metric
                      - plotting function
                    * - forest_clfs
                      - :func:`.confusion_matrix`
                    * - gene_corr
                      - :func:`.correlation_matrix`
        """

        if set_ids == "all":
            if self.dir:
                self.summary_results = pd.read_csv(self._summary_file, index_col=0)
            set_ids = self.summary_results.index.tolist()
        assert isinstance(set_ids, list)

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
            pl.confusion_matrix(set_ids, self.results["forest_clfs"], show=show, save=save, **conf_plt_kwargs)

        if "gene_corr" in metrics:
            corr_plt_kwargs = plt_kwargs["gene_corr"] if ("gene_corr" in plt_kwargs) else {}
            pl.correlation_matrix(set_ids, self.results["gene_corr"], show=show, save=save, **corr_plt_kwargs)


########################################################################################################
########################################################################################################
########################################################################################################
# evaluation.py will be reserved for the ProbesetEvaluator class only at the end. Therefore
# the following functions will be moved or removed (quite a few dependencies need to be adjusted first though)


def plot_gene_expressions(
    adata: sc.AnnData, f_idxs: Iterable[Any], fig_title: str = None, show: bool = True, save: Union[str, bool] = False
) -> None:
    """Plot a UMAP of the gene expression.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        f_idxs:
            Indices of the genes to plot.
        fig_title:
            Figure title.
        show:
            Show the figure.
        save:
            If `True` or a `str`, save the figure.
    """
    a = adata.copy()  # TODO Think we can get rid of this copy and just work with the views
    gene_obs = []
    for _, f_idx in enumerate(f_idxs):
        gene_name = a.var.index[f_idx]
        a.obs[gene_name] = a.X[:, f_idx]
        gene_obs.append(gene_name)

    fig = sc.pl.umap(a, color=gene_obs, ncols=4, return_fig=True)
    fig.suptitle(fig_title)
    if save:
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()


def plot_nmis(
    results_path: str,
    cols: Iterable = None,
    colors: List[str] = None,
    labels: List[str] = None,
    legend: tuple = None,
    show: bool = True,
    save: Union[bool, str] = False,
) -> plt.Figure:
    """Plot the distribution of NMI values.

    Notes:

        Custom legend: e.g. ``legend = [custom_lines,line_names]``

        Custom lines: eg::

            custom_lines = [Line2D([0], [0], color='red',    lw=linewidth),
                            Line2D([0], [0], color='orange', lw=linewidth),
                            Line2D([0], [0], color='green',  lw=linewidth),
                            Line2D([0], [0], color='blue',   lw=linewidth),
                            Line2D([0], [0], color='cyan',   lw=linewidth),
                            Line2D([0], [0], color='black',  lw=linewidth),
                        ]
            line_names = ["dropout", "dropout 1 donor", "pca", "marker", "random"]

    Args:
        results_path:
            Path where NMI results are stored.
        cols:
            Which columns of the table stored at the `results_path` should be plotted.
        colors:
            List of a color for each line.
        labels:
            List with a label for each line.
        legend:
            Handles and Labels of figure legend.
        show:
            Show the figure.
        save:
            If `True` or a `str`, save the figure.
    """
    df = pd.read_csv(results_path, index_col=0)
    fig = plt.figure(figsize=(10, 6))
    if cols is None:
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
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


##########################################################
############### tree_classifications() ###################
##########################################################
# Note that the single tree classifications are too noisy
# we go with random forest classifications now, picking the best tree


def split_train_test_sets(
    adata: sc.AnnData, split: int = 4, seed: int = 2020, verbose: bool = True, obs_key: str = None
) -> None:
    """Split data to train and test set.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        split:
            Number of splits.
        seed:
            Random number seed.
        verbose:
            Verbosity level > 1.
        obs_key:
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


def get_celltypes_with_too_small_test_sets(
    adata: sc.AnnData, ct_key: str, min_test_n: int = 20, split_kwargs: Dict[str, int] = {"seed": 0, "split": 4}
) -> Tuple[List[str], List[int]]:
    """Get celltypes whith test set sizes below `min_test_n`.

    We split the observations in adata into train and test sets for forest training. Check if the resulting
    test sets have at least `min_test_n` samples.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        ct_key:
            Column of `adata.obs` with cell type annotation.
        min_test_n:
            Minimal number of samples in each celltype's test set.
        split_kwargs:
            Keyword arguments for ev.split_train_test_sets().

    Returns:
        cts_below_min: list:
            celltypes with too small test sets
        counts_below_min: list:
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


def uniform_samples(
    adata: sc.AnnData,
    ct_key: str,
    set_key: str = "train_set",
    subsample: int = 500,
    seed: int = 2020,
    celltypes: Union[List[str], str] = "all",
) -> Tuple[np.ndarray, Dict[str, np.ndarray], list]:
    """Subsample `subsample` cells per celltype.

    If the number of cells of a celltype is lower we're oversampling that celltype.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        ct_key:
            Column of `adata.obs` with cell type annotation.
        set_key:
            Column of `adata.obs` with indicating the train set.
        subsample:
            Number of random choices.
        seed:
            Random number seed.
        celltypes:
            List of celltypes to consider or `all`.
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


def save_forest(results: list, path: str) -> None:
    """Save forest results to file.

    Args:
        results:
            Output from forest_classifications().
        path:
            Path to save file.
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_forest(path: str) -> Any:
    """Load forest results from path.

    Args:
        path:
            Path to file.
    """
    return pickle.load(open(path, "rb"))


def get_reference_masks(cts: list, ct_to_ref: Dict[str, list]) -> Dict[str, bool]:
    """Get celltype specific boolean masks over celltype annotation vector.

    Args:
        cts:
            celltype annotations.
        ct_to_ref:
            Each celltype's list of reference celltypes e.g.::

                {'AT1':['AT1','AT2','Club'],'Pericytes':['Pericytes','Smooth muscle']}

    """
    masks = {}
    for ct, ref in ct_to_ref.items():
        masks[ct] = np.in1d(cts, ref)
    return masks


def train_ct_tree_helper(
    celltypes: List[str],
    X_train: Any,
    y_train: Any,
    seed: int,
    max_depth: int = 3,
    masks: Dict[str, bool] = None,
    queue: SigQueue = None,
):
    """Train decision trees parallelized over celltypes.

    Args:
        celltypes:
            List of celltypes to consider.
        X_train:
        y_train:
        seed:
            Random number seed.
        max_depth:
        masks:
        queue:

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


def pool_train_ct_tree_helper(ct_trees_dicts: Sequence[Any]) -> Any:
    """Combine list of dictonaries to one dict.

    Args:
        ct_trees_dicts:

    TODO: Write docstring
    """
    tmp = [ct for trees_dict in ct_trees_dicts for ct in trees_dict]
    if len(set(tmp)) < len(tmp):
        raise ValueError("Multiple trees for the same celltype are in the results of the parallelized execution")
    ct_trees = {ct: tree for ct_trees_dict in ct_trees_dicts for ct, tree in ct_trees_dict.items()}
    return ct_trees


def eval_ct_tree_helper(
    ixs: List[int],
    celltypes: List[str],
    ref_celltypes: List[str],
    ct_trees: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cts_test: List[str],
    masks: Dict[str, bool] = None,
    queue: SigQueue = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    TODO: Write docstring

    Args:
        ixs:
        celltypes:
        ref_celltypes:
        ct_trees:
        X_test:
        y_test:
        cts_test:
        masks:
        queue:

    Returns
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


def pool_eval_ct_tree_helper(f1_and_specificities: Iterable) -> Tuple[pd.DataFrame, Any]:
    """
    TODO: Write docstring

    Notes:
        We parallelize over n_trees.

    Args:
        f1_and_specificities:
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
    adata: sc.AnnData,
    selection: Union[list, pd.DataFrame],
    celltypes: Union[str, list] = "all",
    ref_celltypes: Union[str, list] = "all",
    ct_key: str = "Celltypes",
    ct_spec_ref: Dict[str, List[str]] = None,
    save: Union[str, bool] = False,
    seed: int = 0,
    n_trees: int = 50,
    max_depth: int = 3,
    subsample: int = 1000,
    test_subsample: int = 3000,
    sort_by_tree_performance: bool = True,
    verbose: bool = False,
    return_clfs: bool = False,
    n_jobs: int = 1,
    backend: str = "loky",
    progress: Optional[Progress] = None,
    level: int = 3,
    task: str = "Train trees...",
) -> Union[
    Tuple[List[Union[Union[pd.DataFrame, dict], Any]], dict],  # return_clfs = True
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]],
]:  # return_clfs = False
    """Compute or load decision tree classification results.

    Notes:
        As metrics we use:
        macro f1 score as summary statistic - it's a uniformly weighted statistic wrt celltype groups in 'others' since
        we sample uniformly.
        For the reference celltype specific metric we use specificity = TN/(FP+TN) (also because FN and TP are not
        feasible in the given setting)

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        selection:
            Trees are trained on genes of the list or genes defined in the bool column ``selection['selection']``.
        celltypes:
            Trees are trained on the given celltypes
        ref_celltypes:
            List of celltypes used as reference or ``'all'``.
        ct_key: str
            Column name of adata.obs with celltype infos
        ct_spec_ref:
            Celltype specific references (e.g.:
            ``{'AT1':['AT1','AT2','Club'],'Pericytes':['Pericytes','Smooth muscle']}``). This argument was introduced to
            train secondary trees.
        save:
            If not False load results if the given file exists, otherwise save results after computation.
        n_trees:
            Number of trees to train.
        seed:
            Random seed.
        max_depth:
            max_depth argument of DecisionTreeClassifier.
        subsample: int
            For each trained tree we use samples of maximal size=`subsample` for each celltype. If fewer cells
            are present for a given celltype all cells are used.
        test_subsample:
            Number of random choices for drawing test sets.
        sort_by_tree_performance:
            Wether to sort results and trees by tree performance (best first) per celltype
        verbose:
            Verbosity level > 1.
        return_clfs:
            Wether to return the sklearn tree classifier objects. (if `return_clfs` and `save_load` we still on
            save the results tables, if you want to save the classifiers this needs to be done separately).
        n_jobs:
            Multiprocessing number of processes.
        backend:
            Which backend to use for multiprocessing. See class `joblib.Parallel` for valid options.
        progress:
            ``rich.Progress`` object if progress bars should be shown.
        level:
            Progress bar level.
        task:
            Description of progress task.


    Returns:

        tuple: tuple containing:

            - summary_metric: pd.DataFrame
                macro f1 scores for each celltype's trees (Ordered according best performing trees)
            - ct_specific_metric: dict of pd.DataFrame
                For each celltype's tree: specificity (= TN / (FP+TN)) wrt each other celltype's test sample
            - importances: dict of pd.DataFrame
                Gene's feature importances for each tree.
            - forests: dict
                only returned if ``return_clfs=True``. Then the other three return values will be packed in a list:
                ``[summary_metric,ct_specific_metric,importances], forests``.

    Note:
        In all output files trees are ordered according macro f1 performance.

    """
    # TODO: This doc string is partially from an older version. Update it! (Descripiton and Return is already up to
    #  date)
    # TODO: Add progress bars to trees, and maybe change verbose to verbosity levels

    # if verbose:
    #     try:
    #         from tqdm.notebook import tqdm
    #     except ImportError:
    #         from tqdm import tqdm_notebook as tqdm
    # else:
    #     tqdm = None

    n_jobs = _get_n_cores(n_jobs)

    if isinstance(selection, (list, np.ndarray, np.generic)):
        genes = selection
    elif isinstance(selection, pd.Index):
        genes = selection.values
    elif isinstance(selection, pd.DataFrame):
        genes = list(selection.loc[selection["selection"]].index)
    a = adata[:, genes].copy()

    # apply a train test set split one time just to get test set numbers for each celltype to eventually filter out
    # celltypes
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
            f"For celltypes {cts_not_in_ref} trees are computed, they are not listed in reference celltypes though. "
            f"Added them..."
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
        masks_test: Union[Dict[str, bool], None] = get_reference_masks(cts_test, ct_spec_ref)
    else:
        masks_test = None

    ct_trees: Dict[str, list] = {ct: [] for ct in celltypes}
    np.random.seed(seed=seed)
    seeds = np.random.choice(100000, n_trees, replace=False)
    # Compute trees (for each tree index we parallelize over celltypes)
    # for i in tqdm(range(n_trees), desc="Train trees") if tqdm else range(n_trees):
    if progress and verbose:
        forest_task = progress.add_task(task, total=n_trees, level=level)
    for i in range(n_trees):
        X_train, y_train, cts_train = uniform_samples(
            a, ct_key, set_key="train_set", subsample=subsample, seed=seeds[i], celltypes=ref_celltypes
        )
        if ct_spec_ref is not None:
            masks: Union[Dict[str, bool], None] = get_reference_masks(cts_train, ct_spec_ref)
        else:
            masks = None
        ct_trees_i = parallelize(
            callback=train_ct_tree_helper,
            collection=celltypes,
            n_jobs=n_jobs,
            backend=backend,
            extractor=pool_train_ct_tree_helper,
            show_progress_bar=False,  # verbose, # False
        )(X_train=X_train, y_train=y_train, seed=seeds[i], max_depth=max_depth, masks=masks)
        for ct in celltypes:
            ct_trees[ct].append(ct_trees_i[ct])
        if progress and verbose:
            progress.advance(forest_task)
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
        show_progress_bar=False,  # =verbose,
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
    # When creating the results for secondary trees we take the specificities according reference celltypes of each
    # tree.
    # Our new metric is just the mean of these specificities. Btw we still keep the ordering to be based on f1 scores.
    # Think that makes more sense since it's the best balanced result.
    # TODO TODO TODO: Change misleading variable names in other functions (where the old "f1_table" is used)
    summary_metric = summarize_specs(ct_specific_metric)

    if save:
        assert isinstance(save, str)
        save_forest([summary_metric, ct_specific_metric, importances], save)

    if return_clfs:
        return [summary_metric, ct_specific_metric, importances], ct_trees
    else:
        return summary_metric, ct_specific_metric, importances


def summarize_specs(specs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize specificities to summary metrics per celltype.

    Args:
        specs:
            Each celltype's specificities of reference celltypes.

    """
    cts = [ct for ct in specs]
    df = pd.DataFrame(index=cts, columns=specs[cts[0]].columns)
    for ct in specs:
        df.loc[ct] = specs[ct].mean()
    return df


def combine_tree_results(
    primary: Union[list, Tuple[list, dict]],  # with_clfs=False  # with_clfs=True
    secondary: Union[list, Tuple[list, dict]],  # with_clfs=False  # with_clfs=True
    with_clfs: bool = False,
) -> Union[list, Tuple[list, dict]]:  # with_clfs=False  # with_clfs=True
    """Combine results of primary and secondary trees.

    Notes:

        There are three parts in the forest results:
        1. f1_table
        2. classification specificities
        3. feature_importances

        The output for 2. and 3. will be in the same form as the input.
        Specificities are taken from secondary where existend, otherwise from primary.
        Feature_importances are summed up (reasoning: distinguishing celltypes that are
        hard to distinguish is very important and therefore good to rank respective genes high).
        The f1 tables are just aggregated to a list

    Args:

        primary:
            with_clfs_True:
                - Results: list:
                    - pd.DataFrame
                    - Dict[celltype: str, pd.DataFrame]
                    - Dict[celltype: str, pd.DataFrame]
                - Classifiers: Dict[celltype: str, DecisionTreeClassifier]

        secondary:
            data structure like primary

        with_clfs:
            Whether primary, secondary and the output each contain a list of forest results and the forest classifiers
            or only the results.

    Returns:
        with_clfs_True:
            - Results: list:
                - pd.DataFrame
                - Dict[celltype: str, pd.DataFrame]
                - Dict[celltype: str, pd.DataFrame]
            - Classifiers: Dict[celltype: str, DecisionTreeClassifier]

    """
    expected_len = 2 if with_clfs else 3
    if (len(primary) != expected_len) or (len(secondary) != expected_len):
        raise ValueError(
            f"inputs primary and secondary are expected to be lists of length == {expected_len}, not {len(primary)},"
            f"{len(secondary)}"
        )

    if with_clfs:
        primary_res, primary_clfs = primary
        secondary_res, secondary_clfs = secondary
    else:
        primary_res = primary
        secondary_res = secondary

    combined: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = [pd.DataFrame(), {}, {}]
    ## f1 (exchanged by summary stats below)
    # for f1_table in [primary[0],secondary[0]]:
    #    if isinstance(f1_table,list):
    #        combined[0] += f1_table
    #    else:
    #        combined[0].append(f1_table)
    # specificities
    celltypes = [key for key in secondary_res[1]]
    combined[1] = {ct: df.copy() for ct, df in primary_res[1].items()}
    assert isinstance(combined[1], dict)
    for ct in celltypes:
        filt = ~secondary_res[1][ct].isnull().all(axis=1)
        combined[1][ct].loc[filt] = secondary_res[1][ct].loc[filt]
    # summary stats
    assert isinstance(combined[0], pd.DataFrame)
    combined[0] = summarize_specs(combined[1])
    # feature importance
    combined[2] = {ct: df.copy() for ct, df in primary_res[2].items()}
    assert isinstance(combined[2], dict)
    for ct in celltypes:
        combined[2][ct] += secondary_res[2][ct].fillna(0)
        combined[2][ct] = combined[2][ct].div(combined[2][ct].sum(axis=0), axis=1)

    if with_clfs:
        combined_clfs = primary_clfs
        for ct in combined_clfs:
            if ct in secondary_clfs:
                combined_clfs[ct] += secondary_clfs[ct]
        return combined, combined_clfs
    else:
        return combined


def outlier_mask(
    df: pd.DataFrame, n_stds: int = 1, min_outlier_dif: float = 0.02, min_score: float = 0.9
) -> pd.DataFrame:
    """Get mask over df.index based on values in df columns.

    Args:
        df:
        n_stds:
        min_outlier_dif:
        min_score:
    """
    # TODO write docstring

    crit1 = df < (df.mean(axis=0) - (n_stds * df.std(axis=0))).values[np.newaxis, :]
    crit2 = df < (df.mean(axis=0) - min_outlier_dif).values[np.newaxis, :]
    crit3 = df < min_score
    return (crit1 & crit2) | crit3


def get_outlier_reference_celltypes(
    specs: Dict[str, pd.DataFrame], n_stds: int = 1, min_outlier_dif: float = 0.02, min_score: float = 0.9
) -> Dict[str, pd.DataFrame]:
    """For each celltype's best tree get reference celltypes with low performance.

    specs:
        Each celltype's specificities of reference celltypes.
    n_stds:

    min_outlier_dif:

    min_score:

    """
    outliers = {}
    for ct, df in specs.items():
        outliers[ct] = df.loc[outlier_mask(df[["0"]], n_stds, min_outlier_dif, min_score).values].index.tolist() + [ct]
        if len(outliers[ct]) == 1:
            outliers[ct] = []
    return outliers


def forest_classifications(
    adata: sc.AnnData,
    selection: Union[list, pd.DataFrame],
    max_n_forests: int = 3,
    verbosity: int = 1,
    save: Union[str, bool] = False,
    outlier_kwargs: dict = {},
    progress: Optional[Progress] = None,
    task: str = "Train hierarchical trees...",
    level: int = 2,
    **forest_kwargs,
) -> Union[
    list,  # with_clfs=False
    Tuple[list, dict],  # with_clfs=True
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]],
]:  # from single_forest_classifications()
    """Train best trees including secondary trees.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        selection:
            Trees are trained on genes of the list or genes defined in the bool column ``selection[‘selection’]``.
        max_n_forests:
            Number of best trees considered as a tree group. Including the primary tree.
        verbosity:
            Verbosity level.
        save:
            If not False load results if the given file exists, otherwise save results after computation.
        outlier_kwargs:
            Parameters for :func:`.get_outlier_reference_celltypes`.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        task:
            Description of progress task.
        level:
            Progress bar level.
        **forest_kwargs:
            Parameters for :func:`.single_forest_classifications`.

    """

    # TODO write docstring

    # if verbosity > 0:
    #     try:
    #         from tqdm.notebook import tqdm
    #     except ImportError:
    #         from tqdm import tqdm_notebook as tqdm
    # else:
    #     tqdm = None

    ct_spec_ref = None
    res = None
    with_clfs = "return_clfs" in forest_kwargs and forest_kwargs["return_clfs"]
    stop_progress = False
    if not progress and 2 * verbosity >= level:
        progress = NestedProgress()
        progress.start()
        stop_progress = True
    if progress and 2 * verbosity >= level:
        forest_task = progress.add_task(task, total=max_n_forests, level=level)
    # for _ in tqdm(range(max_n_forests), desc="Train hierarchical trees") if tqdm else range(max_n_forests):
    for _ in range(max_n_forests):
        new_res = single_forest_classifications(
            adata,
            selection,
            ct_spec_ref=ct_spec_ref,
            verbose=verbosity > 1,
            save=False,
            progress=progress,
            level=level + 1,
            **forest_kwargs,
        )
        assert isinstance(new_res, tuple)
        if res is None:
            res = new_res
        else:
            res = combine_tree_results(res, new_res, with_clfs=with_clfs)
        specs = res[0][1] if with_clfs else res[1]
        ct_spec_ref = get_outlier_reference_celltypes(specs, **outlier_kwargs)
        if progress and 2 * verbosity >= level:
            progress.advance(forest_task)

    if save:
        assert isinstance(save, str)
        if with_clfs:
            assert isinstance(res, tuple)
            save_forest(res[0], save)
        else:
            assert isinstance(res, list)
            save_forest(res, save)

    if progress and 2 * verbosity >= level and stop_progress:
        progress.stop()
    assert res is not None
    return res


def forest_rank_table(
    importances: Dict[str, pd.DataFrame],
    celltypes: Union[str, List[str]] = "all",
    return_ct_specific_rankings: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Rank genes according importances of the celltypes' forests.

    # TODO complete docstring

    Args:
        importances:

        celltypes: str or list of strs
            If 'all' create ranking based on all celltypes in importances. Otherwise base the ranking only on
            the trees of the subset `celltypes`.
        importances: dict of pd.DataFrame
            Output from `forest_classifications()`. DataFrame for each celltype's forest.
            Each column refers to genes of one tree. The columns are sorted according performance (best first)
        return_ct_specific_rankings:


    Returns:
        pd.DataFrame
            index:
                Genes found in :attr:`importances`
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
