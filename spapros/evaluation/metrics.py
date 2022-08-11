from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
from rich import print
from rich.progress import Progress as Progress
from scipy.sparse import issparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from spapros.util.util import clean_adata
from spapros.util.util import cluster_corr
from spapros.util.util import dict_to_table
from spapros.util.util import gene_means
from spapros.util.util import init_progress
from xgboost import XGBClassifier

METRICS_PARAMETERS: Dict[str, Dict] = {
    "cluster_similarity": {
        "ns": [5, 60],
        "AUC_borders": [[5, 20], [21, 60]],
    },
    "knn_overlap": {
        "ks": [5, 10, 15, 20, 25, 30],
    },
    "forest_clfs": {
        "ct_key": "celltype",
        "threshold": 0.8,
    },
    "marker_corr": {
        "marker_list": None,
        "per_celltype": True,
        "per_marker": True,
        "per_celltype_min_mean": None,
        "per_marker_min_mean": 0.025,
    },
    "gene_corr": {"threshold": 0.8},
}


# utility functions to get general infos about metrics #
def get_metric_names() -> List[str]:
    """Get a list of all available metric names."""
    return [metric for metric in METRICS_PARAMETERS]


def get_metric_parameter_names() -> Dict[str, List[str]]:
    """Get all parameter names for each available metric."""
    names = {metric: [param for param in METRICS_PARAMETERS[metric]] for metric in METRICS_PARAMETERS}
    return names


def get_metric_default_parameters() -> Dict[str, Dict]:
    """Get the default metric parameters."""
    return METRICS_PARAMETERS


############################
# General metric functions #
############################
# We define theses to systematically seperate all metrics in three different parts
# 1. shared computations: only need to compute one time for a given dataset
# 2. specific computations: Need to be computed for each probe set
# 3. summary: Simple final computations (per probe set) that aggregate evaluations to summary metrics


def metric_shared_computations(
    adata: sc.AnnData,
    metric: str,
    parameters: Dict = {},
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
) -> pd.DataFrame:
    """Calculate the metric compuations that can be shared between probe sets.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        metric:
            The metric to be calculated.
        parameters:
            Parameters for the calculation of the metric.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
    """
    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    progress, started = init_progress(progress, verbosity, level)

    description = "Computing shared compuations for " + metric + "..."

    if metric == "cluster_similarity":
        return leiden_clusterings(
            adata, parameters["ns"], progress=progress, level=level, verbosity=verbosity, description=description
        )

    elif metric == "knn_overlap":
        return knns(
            adata,
            genes="all",
            ks=parameters["ks"],
            progress=progress,
            level=level,
            verbosity=verbosity,
            description=description,
        )
    elif metric == "forest_clfs":
        pass
    elif metric == "marker_corr":
        return marker_correlation_matrix(
            adata,
            marker_list=parameters["marker_list"],
            progress=progress,
            level=level,
            verbosity=verbosity,
            description=description,
        )
    elif metric == "gene_corr":
        return correlation_matrix(adata, progress=progress, level=level, verbosity=verbosity, description=description)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if progress and started:
        progress.stop()


def metric_pre_computations(
    genes: List,
    adata: sc.AnnData,
    metric: str = None,
    parameters: Dict = {},
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
) -> Union[pd.DataFrame, None]:
    """Calculate the metric computations that are independent of the shared results.

    Note:
        If there are no shared results needed at all to calculate a metric, the computations are put in
        `metric_computations`, this is the case for e.g. forest_clfs.

    Args:
        genes:
            The selected genes.
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        metric:
            The metric to be calculated.
        parameters:
            Parameters for the calculation of the metric.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
    """
    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    progress, started = init_progress(progress, verbosity, level)

    description = "Computing pre compuations for " + metric + ".."

    if metric == "cluster_similarity":
        return leiden_clusterings(
            adata[:, genes],
            parameters["ns"],
            progress=progress,
            level=level,
            verbosity=verbosity,
            description=description,
        )
    elif metric == "knn_overlap":
        return knns(
            adata,
            genes=genes,
            ks=parameters["ks"],
            progress=progress,
            level=level,
            verbosity=verbosity,
            description=description,
        )
    elif metric == "forest_clfs":
        return None
    elif metric == "marker_corr":
        return None
    elif metric == "gene_corr":
        return None
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if progress and started:
        progress.stop()


def metric_computations(
    genes: list,
    adata: sc.AnnData = None,
    metric: str = None,
    shared_results: pd.DataFrame = None,
    pre_results: pd.DataFrame = None,
    parameters: Dict = {},
    n_jobs: int = -1,
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
) -> pd.DataFrame:
    """Compute the probe set specific evaluation metrics.

    Args:
        genes:
            The selected genes.
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        metric:
            The metric to be calculated.
        shared_results:
            The results of the metric calculations, that are not probe set specific.
        pre_results:
            The results of the metric calculations, that are independent of the shared calculations.
        parameters:
            Parameters for the calculation of the metric.
        n_jobs:
            Number of cpus for multi processing computations. Set to -1 to use all available cpus.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
    """

    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    progress, started = init_progress(progress, verbosity, level)

    description = "Computing final compuations for " + metric + "..."

    if metric == "cluster_similarity":
        ann = pre_results
        ref_ann = shared_results
        nmis = clustering_nmis(
            ann, ref_ann, parameters["ns"], progress=progress, level=level, verbosity=verbosity, description=description
        )
        return nmis
    elif metric == "knn_overlap":
        knn_df = pre_results
        ref_knn_df = shared_results
        return mean_overlaps(
            knn_df,
            ref_knn_df,
            parameters["ks"],
            progress=progress,
            level=level,
            verbosity=verbosity,
            description=description,
        )
    elif metric == "forest_clfs":
        results = xgboost_forest_classification(
            adata,
            genes,
            celltypes="all",
            ct_key=parameters["ct_key"],
            n_jobs=n_jobs,
            progress=progress,
            level=level,
            verbosity=verbosity,
            description=description,
        )
        conf_mat = results[0]
        return conf_mat
    elif metric == "marker_corr":
        marker_cor = shared_results
        params = {k: v for k, v in parameters.items() if (k != "marker_list")}
        return max_marker_correlations(
            genes, marker_cor, **params, progress=progress, level=level, verbosity=verbosity, description=description
        )
    elif metric == "gene_corr":
        full_cor_mat = shared_results
        return gene_set_correlation_matrix(
            genes,
            full_cor_mat,
            ordered=True,
            progress=progress,
            level=level,
            verbosity=verbosity,
            description=description,
        )

    if progress and started:
        progress.stop()


def metric_summary(results: pd.DataFrame = None, metric: str = None, parameters: Dict = {}) -> Dict[str, Any]:
    """Simple final computations (per probe set) that aggregate evaluations to summary metrics.

    Args:
        results:
            The results of the previously calculated metric.
        metric:
            The name of the previously calculated metric.
        parameters:
            Parameters for the calculation of the metric.
    """

    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    summary: Dict[str, Any] = {}

    if metric == "cluster_similarity":
        nmis = results
        for s, val in summary_nmi_AUCs(nmis, parameters["AUC_borders"]).items():
            summary["cluster_similarity " + s] = val
    elif metric == "knn_overlap":
        means_df = results
        summary["knn_overlap mean_overlap_AUC"] = summary_knn_AUC(means_df)
    elif metric == "forest_clfs":
        conf_mat = results
        summary["forest_clfs accuracy"] = summary_metric_diagonal_confusion_mean(conf_mat)
        if "threshold" in parameters:
            th = parameters["threshold"]
            summary[f"forest_clfs perct acc > {th}"] = summary_metric_diagonal_confusion_percentage(
                conf_mat, threshold=th
            )
    elif metric == "marker_corr":
        cor_df = results
        for s, val in summary_marker_corr(cor_df).items():
            summary["marker_corr " + s] = val
    elif metric == "gene_corr":
        cor_mat = results
        summary["gene_corr 1 - mean"] = summary_metric_correlation_mean(cor_mat)
        if "threshold" in parameters:
            th = parameters["threshold"]
            summary[f"gene_corr perct max < {th}"] = summary_metric_correlation_percentage(cor_mat, threshold=th)

    return summary


#######################################
# cluster_similarity metric functions #
#######################################

# SHARED AND and PER PROBESET computations


def compute_clustering_and_update(
    adata: sc.AnnData, annotations: pd.DataFrame, resolution: float, tried_res_n: List[List], found_ns: List[int]
) -> Tuple[pd.DataFrame, List[List], List[int]]:
    """Compute a new leiden clustering and save the used resolution and resulting clusters.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        annotations:
            Contains clusterings for different numbers of clusters. Each row is a list of cluster assignments::

                n , obs1, obs2, obs3, ...
                2 ,   0 ,   0 ,   1 , ...
                4 ,   1 ,   0 ,   3 , ...
                8 ,   7 ,   7 ,   2 , ...
                3 ,   2 ,   1 ,   0 , ...

        resolution:
            Resolution to start computing clusterings.
        tried_res_n:
            Resolutions that were previously used for computing clusters.
        found_ns:
            The numbers of clusters resulting from previous clusters with the resolutions saved in :attr:`tried_res_n`.
    """
    sc.tl.leiden(adata, resolution=resolution, key_added="tmp")
    n = len(set(adata.obs["tmp"]))
    if n not in found_ns:
        annotations.loc[n, adata.obs_names] = list(adata.obs["tmp"])
        annotations.loc[n, "resolution"] = resolution
        found_ns.append(n)
    tried_res_n.append([resolution, n])
    tried_res_n.sort(key=lambda x: x[0])
    return annotations, tried_res_n, found_ns


def leiden_clusterings(
    adata: sc.AnnData,
    ns: Union[range, List[int]],
    start_res: float = 1.0,
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description: str = "Leiden clusterings...",
) -> pd.DataFrame:
    """Compute leiden clusters for different numbers of clusters.

    Leiden clusters are calculated with different resolutions.
    A search (similar to binary search) is applied to find the right resolutions for all defined n's.

    Args:
        adata:
            Adata object with data to compute clusters on.
        ns:
            The minimum (:attr:`ns[0]`) and maximum (:attr:`ns[1]`) number of clusters.
        start_res:
            Resolution to start computing clusterings.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.

    Returns:
        pd.DataFrame
            1st column refers to the number of clusters,
            2nd col refers to the resolution used to calculate the clustering
            each following column refer to individual cell's cluster assignments e.g.::

                n , res  , <adata.obs.index[0]>, <adata.obs.index[1]>, ...., <adata.obs.index[n_cells-1]>
                6 , 1.   ,         0           ,          4          , ....,              5
                2 , 0.67 ,         1           ,          0          , ....,              1
                13, 2.34 ,         9           ,          7          , ....,             12
                17, 2.78 ,         7           ,          7          , ....,              3
                .
                .
                .
    """

    progress, started = init_progress(progress, verbosity, level)
    if progress:
        task_leiden = progress.add_task(description, level=level, total=1)

    # Convert min and max n to list of ns
    if len(ns) != 2:
        raise ValueError("`ns` must be a list of two integers.")
    ns = range(ns[0], ns[1] + 1)

    # Clean adata and recalculate pca + neighbors graph
    a = adata.copy()
    clean_adata(a)
    sc.tl.pca(a)
    sc.pp.neighbors(a)

    # Initialize
    annotations = pd.DataFrame(index=ns, columns=["resolution"] + a.obs_names.tolist())
    annotations.index.name = "n"
    tried_res_n: List[List] = []
    found_ns: List[int] = []

    # First clustering step
    n_min = np.min(ns)
    n_max = np.max(ns)
    res = start_res
    annotations, tried_res_n, found_ns = compute_clustering_and_update(a, annotations, res, tried_res_n, found_ns)

    # Search for lower resolution border
    while np.min(np.unique([res_n[1] for res_n in tried_res_n])) > n_min:
        res *= 0.5
        annotations, tried_res_n, found_ns = compute_clustering_and_update(a, annotations, res, tried_res_n, found_ns)

    # Search for higher resolution border
    res = np.max([res_n[0] for res_n in tried_res_n])
    while np.max([res_n[1] for res_n in tried_res_n]) < n_max:
        res *= 2
        annotations, tried_res_n, found_ns = compute_clustering_and_update(a, annotations, res, tried_res_n, found_ns)

    # Search missing n's between neighboring found n's
    found_space = True
    while (not set(ns) <= set([res_n[1] for res_n in tried_res_n])) and (found_space):
        tmp_res_n = tried_res_n
        found_space = False
        for i in range(len(tmp_res_n) - 1):
            # check if two neighboring resolutions have different n's
            cond1 = tmp_res_n[i + 1][1] - tmp_res_n[i][1] > 1
            # check if we search an n between the two n's of the given resolutions
            cond2 = len([n for n in ns if n > tmp_res_n[i][1] and n < tmp_res_n[i + 1][1]]) > 0
            # we run in some error to recompute values between false pairs
            # I think this occurs since it can happen that sometimes a slightly higher
            # resolution leads to a lower number of clusters
            # TODO: check this properly
            # cond3 is added to account for that issue (probably not the best way)
            cond3 = abs(tmp_res_n[i + 1][0] - tmp_res_n[i][0]) > 0.00005  # 0.0003
            if cond1 and cond2 and cond3:
                res = (tmp_res_n[i][0] + tmp_res_n[i + 1][0]) * 0.5
                annotations, tried_res_n, found_ns = compute_clustering_and_update(
                    a, annotations, res, tried_res_n, found_ns
                )
                found_space = True

    if progress:
        progress.advance(task_leiden)
        if started:
            progress.stop()

    return annotations


def clustering_nmis(
    annotations: pd.DataFrame,
    ref_annotations: pd.DataFrame,
    ns: Union[range, List[int]],
    method: str = "arithmetic",
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description: str = "Cluster NMIs...",
) -> pd.DataFrame:
    """Compute NMI between clusterings and a reference set of clusterings.

    For different numbers of clusters (`ns`) the normalized mutual information
    NMI based on 2 different cluster annotations are computed.

    Args:
        annotations:
            Contains clusterings for different numbers of clusters. Each row is a list of cluster assignments::

                n , obs1, obs2, obs3, ...
                2 ,   0 ,   0 ,   1 , ...
                4 ,   1 ,   0 ,   3 , ...
                8 ,   7 ,   7 ,   2 , ...
                3 ,   2 ,   1 ,   0 , ...

        ref_annotations:
            Same as annotations for reference clusterings.
        ns:
            Minimum (:attr:`ns[0]`) and maximum (:attr:`ns[1]`) number of clusters.
        method:
            NMI implementation

                - 'max': scikit method with `average_method='max'`
                - 'min': scikit method with `average_method='min'`
                - 'geometric': scikit method with `average_method='geometric'`
                - 'arithmetic': scikit method with `average_method='arithmetic'`

            TODO: implement the following (see comment below and scib)

                - 'Lancichinetti': implementation by A. Lancichinetti 2009 et al.
                - 'ONMI': implementation by Aaron F. McDaid et al. (https://github.com/aaronmcdaid/Overlapping-NMI) Hurley 2011

        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.

    Returns:
        pd.DataFrame of NMI results::

            n (index), nmi
            2        , 1.0
            3        , 0.9989
            ...
    """

    from sklearn.metrics import normalized_mutual_info_score

    progress, started = init_progress(progress, verbosity, level)

    # Convert min and max n to list of ns
    if len(ns) != 2:
        raise ValueError("`ns` must be a list of two integers.")
    ns = range(ns[0], ns[1] + 1)

    nmis = pd.DataFrame(np.nan, index=ns, columns=["nmi"])

    # Drop column "resolution"
    ann = annotations[[c for c in annotations.columns if (c != "resolution")]].copy()
    ref_ann = ref_annotations[[c for c in ref_annotations.columns if (c != "resolution")]].copy()

    # Prepare shared ns
    found_ns_df = ~ann.isnull().any(axis=1)
    ref_found_ns_df = ~ref_ann.isnull().any(axis=1)
    found_ns = found_ns_df.loc[found_ns_df].index.tolist()
    ref_ns = ref_found_ns_df.loc[ref_found_ns_df].index.tolist()
    valid_ns = [n for n in found_ns if (n in ref_ns) and (n in ns)]

    if progress:
        task_nmi = progress.add_task(description, total=len(valid_ns), level=level)

    # Calculate nmis
    for n in valid_ns:
        labels = ann.loc[n].values
        ref_labels = ref_ann.loc[n].values
        nmis.loc[n, "nmi"] = normalized_mutual_info_score(labels, ref_labels, average_method=method)

        if progress:
            progress.advance(task_nmi)

    if progress and started:
        progress.stop()

    return nmis


# SUMMARY metrics
def AUC(series: pd.Series, n_min: int = 1, n_max: int = 60) -> float:
    """Calculate the Area Unter the Curve.

    Args:
        series:
            Series of values.
        n_min:
            Lower border of curve.
        n_max:
            Upper border of curve.
    """
    tmp = series.loc[(series.index >= n_min) & (series.index <= n_max)]
    n = len(tmp)
    return tmp.sum() / n


def summary_nmi_AUCs(nmis: pd.DataFrame, AUC_borders: List[List]) -> Dict[str, float]:
    """Calculate AUC over range of nmi values.

    Args:
        nmis:
            Table of NMI results::

                n (index), nmi
                2        , 1.0
                3        , 0.9643
                4        , NaN
                5        , 0.98
                ...

        AUC_borders:
            Calculates nmi AUCs over given borders. E.g. :attr:`AUC_borders = [[2,4],[5,20]]` calculates nmi over n
            ranges 2 to 4 and 5 to 20. Defined border shouldn't exceed values in :attr:`nmis`.
    """
    AUCs = {}
    for ths in AUC_borders:
        AUCs[f"nmi_{ths[0]}_{ths[1]}"] = AUC(nmis["nmi"].interpolate(), n_min=ths[0], n_max=ths[1])
    return AUCs


################################
# knn_overlap metric functions #
################################

# SHARED AND and PER PROBESET computations


def knns(
    adata: sc.AnnData,
    genes: Union[List, str] = "all",
    ks: List[int] = [10, 20],
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description: str = "KNNS...",
) -> pd.DataFrame:
    """Compute nearest neighbors of observations for different ks.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        genes:
            A list of selected genes or "all".
        ks:
            Calculate knn graphs for each k in :attr:`ks`.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.

    Returns:
        pd.DataFrame:
            Includes nearest neighbors for all ks::

                gene (index), k10_1, k10_2, ..., k20_1, k20_2, ...
                ISG15       , 3789 , 512  ,    , 9720 , 15   , ...
                TNFRSF4     , 678  , 713  ,    , 7735 , 6225 , ...
                ...
    """

    progress, started = init_progress(progress, verbosity, level)

    # Subset adata to gene set
    if isinstance(genes, str) and (genes == "all"):
        genes = adata.var_names
    a = adata[:, genes].copy()

    # Set n_pcs to 50 or number of genes if < 50
    n_pcs = np.min([50, len(genes) - 1])

    # Delete existing PCAs, neighbor graphs, etc. and calculate PCA for n_pcs
    uns = [key for key in a.uns]
    for u in uns:
        del a.uns[u]
    obsm = [key for key in a.obsm]
    for o in obsm:
        del a.obsm[o]
    varm = [key for key in a.varm]
    for v in varm:
        del a.varm[v]
    obsp = [key for key in a.obsp]
    for o in obsp:
        del a.obsp[o]
    sc.tl.pca(a, n_comps=n_pcs)  # use_highly_variable=False

    # Get nearest neighbors for each k
    df = pd.DataFrame(index=a.obs_names)
    if progress:
        task_knn = progress.add_task(description, level=level, total=len(ks))
    for k in ks:
        if "neighbors" in a.uns:
            del a.uns["neighbors"]
        if "connectivities" in a.obsp:
            del a.obsp["connectivities"]
        if "distances" in a.obsp:
            del a.obsp["distances"]
        sc.pp.neighbors(a, n_neighbors=k)
        rows, cols = a.obsp["distances"].nonzero()
        nns = []
        for r in range(a.n_obs):
            nns.append(cols[rows == r].tolist())
        nn_df = pd.DataFrame(nns, index=a.obs_names)
        nn_df.columns = [f"k{k}_{i}" for i in range(len(nn_df.columns))]
        df = pd.concat([df, nn_df], axis=1)

        if progress:
            progress.advance(task_knn)

    if progress and started:
        progress.stop()

    return df


def mean_overlaps(
    knn_df: pd.DataFrame,
    ref_knn_df: pd.DataFrame,
    ks: List[int],
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description: str = "Mean overlaps...",
) -> pd.DataFrame:
    """Calculate mean overlaps of knn graphs of different ks.

    Args:
        knn_df:
            The results of the knn calculations, that are independent of the shared calculations.
        ref_knn_df:
            The results of the metric calculations, that are not probe set specific.
        ks:
            Calculate knn graphs for each k in `ks`.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.


    Returns:
        pd.DataFrame:
            k (index), mean

    """

    progress, started = init_progress(progress, verbosity, level)

    if progress:
        task_meano_verlap = progress.add_task(description, level=level, total=len(ks))
    df = pd.DataFrame(index=ks, data={"mean": 0.0})
    for k in ks:
        overlaps = []
        cols_of_k = [col for col in knn_df.columns if (int(col.split("_")[0][1:]) == k)]
        ref_cols_of_k = [col for col in ref_knn_df.columns if (int(col.split("_")[0][1:]) == k)]
        nns1 = knn_df[cols_of_k].values.astype(int)
        nns2 = ref_knn_df[ref_cols_of_k].values.astype(int)
        for i in range(nns1.shape[0]):
            set1 = set(nns1[i])
            set2 = set(nns2[i])
            max_intersection = np.min([len(set1), len(set2)])
            overlaps.append(len(set1.intersection(set2)) / max_intersection)
        df.loc[k, "mean"] = np.mean(overlaps)

        if progress:
            progress.advance(task_meano_verlap)

    if progress and started:
        progress.stop()

    return df


# SUMMARY metrics


def summary_knn_AUC(means_df: pd.DataFrame) -> float:
    """Calculate AUC of mean overlaps over ks.

    Args:
        means_df: The results of the previously calculated metric (knn_overlap mean_overlap_AUC).
    """
    x = [int(x) for x in means_df.index.values]
    y = means_df["mean"].values
    tmp = pd.Series(index=range(np.min(x), np.max(x)), dtype="float64")
    for i in range(len(y)):
        tmp.loc[x[i]] = y[i]
    return AUC(tmp.interpolate(), n_min=np.min(x), n_max=np.max(x))


################################
# forest_clfs metric functions #
################################

# SHARED computations
# None

# PER PROBESET computations
def xgboost_forest_classification(
    adata: sc.AnnData,
    selection: Union[List, pd.DataFrame],
    celltypes: Union[List, str] = "all",
    ct_key: str = "Celltypes",
    n_cells_min: int = 40,
    max_depth: int = 3,
    lr: float = 0.2,
    colsample_bytree: float = 1,
    cv_splits: int = 5,
    min_child_weight: Union[float, None] = None,
    gamma: float = None,
    seed: int = 0,
    n_seeds: int = 5,
    verbosity: int = 0,
    return_train_perform: bool = False,
    return_clfs: bool = False,
    return_predictions: bool = False,
    n_jobs: int = 1,
    progress: Progress = None,
    level: int = 3,
    description: str = "XGB forest classification",
) -> List:
    """Measure celltype classification performance with gradient boosted forests.

    We train extreme gradient boosted forest classifiers on multi class classification of cell types. Cross validation
    is performed to get an average confusion matrix (normalised by ground truth counts and sample weights to weight cell
    types in a balanced way). To make the performance measure robust only cell types with at least :attr:`n_cells_min`
    are taken into account. To make the cross validation more robust we run it with :attr:`n_seeds`. I.e.
    :attr:`cv_splits x :attr:n_seeds` classifiers are trained and evaluated.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        selection:
            Forests are trained on genes of the list or genes defined in the bool column selection['selection'].
        celltypes:
            Forests are trained on the given celltypes.
        ct_key:
            Column name of adata.obs with cell type info.
        n_cells_min:
            Minimal number of cells to filter out cell types from the training set. Performance results are not robust
            for low :attr:`n_cells_min`.
        max_depth:
            The max_depth argument of XGBClassifier.
        lr:
            Learning rate of XGBClassifier.
        colsample_bytree:
            Fraction of features (randomly selected) that will be used to train each tree of XGBClassifier.
        cv_splits:
            Number of cross validation splits.
        min_child_weight:
            Minimum sum of instance weight(hessian) needed in a child.
        gamma:
            Regularisation parameter of XGBClassifier. Instruct trees to add nodes only if the associated loss gain is
            larger or equal to gamma.
        seed:
            Random number seed.
        n_seeds:
            Number of training repetitions with different seeds. We use multiple seeds to make results more robust.
            Also we don't want to increase the number of CV splits to keep a minimal test size.
        verbosity:
            Set to 2 for progress bar. Set to 3 to print test performance of each tree during training.
        return_train_perform:
            Wether to also return confusion matrix of training set.
        return_clfs:
            Wether to return the classifier objects.
        return_predictions:
            Whether to return a list of prediction dataframes
        n_jobs:
            Multiprocessing number of processes.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        description:
            Description of progress bar.


    Returns:
        pd.DataFrame:
            confusion matrix averaged over cross validations and seeds.
        pd.DataFrame:
            confusion matrix standard deviation over cross validations and seeds.
        if return_train_perform:
            pd.DataFrames as above for train set.
        if return_clfs:
            list of XGBClassifier objects of each cross validation step and seed.
        if return_predictions:
            list of dataframes with prediction results.
    """

    # if verbosity > 1:
    #     try:
    #         from tqdm.notebook import tqdm
    #     except ImportError:
    #         from tqdm import tqdm_notebook as tqdm
    #     desc = "XGBClassifier Cross Val."
    # else:
    #     tqdm = None

    progress, started = init_progress(progress, verbosity, level)

    # Define cell type list
    if celltypes == "all":
        celltypes = adata.obs[ct_key].unique().tolist()
    # Filter out cell types with less cells than n_cells_min
    cell_counts = adata.obs[ct_key].value_counts().loc[celltypes]
    if (cell_counts < n_cells_min).any() and (verbosity > 0):
        print(
            f"[bold yellow]The following cell types are not included in forest classifications since they have fewer "
            f"than {n_cells_min} cells: {cell_counts.loc[cell_counts < n_cells_min].index.tolist()}"
        )
        celltypes = [ct for ct in celltypes if (cell_counts.loc[ct] >= n_cells_min)]

    # Get data
    obs = adata.obs[ct_key].isin(celltypes)
    s = selection if isinstance(selection, list) else selection.loc[selection["selection"]].index.tolist()
    X = adata[obs, s].X
    ct_encoding = {ct: i for i, ct in enumerate(celltypes)}
    y = adata.obs.loc[obs, ct_key].astype(str).map(ct_encoding).values

    # Seeds
    # To have more robust results we use multiple seeds (especially important for cell types with low cell count)
    if n_seeds > 1:
        rng = np.random.default_rng(seed)
        seeds = list(rng.integers(low=0, high=100000, size=n_seeds))
    else:
        seeds = [seed]

    # Initialize variables for training
    confusion_matrices = []
    if return_train_perform:
        confusion_matrices_train = []
    if return_clfs:
        clfs = []
    if return_predictions:
        pred_dfs = []
        obs_names = np.array(adata.obs_names[obs])
        raw_df = pd.DataFrame(
            index=obs_names,
            data={
                "celltype": adata.obs.loc[obs, ct_key],
                "label": y,
                "train": False,
                "test": False,
                "pred": 0,
                "correct": False,
            },
        )
    n_classes = len(celltypes)

    # Cross validated random forest training
    # for seed in tqdm(seeds, desc="seeds", total=len(seeds)) if tqdm else seeds:
    if progress:
        task_forest = progress.add_task(description=description, total=len(seeds) * cv_splits, level=level)
    for seed in seeds:
        k_fold = StratifiedKFold(n_splits=cv_splits, random_state=seed, shuffle=True)
        # for train_ix, test_ix in tqdm(k_fold.split(X, y), desc=desc, total=cv_splits) if tqdm else k_fold.split(X, y):
        for train_ix, test_ix in k_fold.split(X, y):
            # Get train and test sets
            train_x, train_y, test_x, test_y = X[train_ix], y[train_ix], X[test_ix], y[test_ix]
            sample_weight_train = compute_sample_weight("balanced", train_y)
            sample_weight_test = compute_sample_weight("balanced", test_y)
            # Fit the classifier
            n_classes = len(np.unique(train_y))
            clf = XGBClassifier(
                max_depth=max_depth,
                num_class=n_classes,
                n_estimators=250,
                objective="multi:softmax" if n_classes > 2 else "binary:logistic",
                eval_metric="mlogloss",  # set this to get rid of warning
                learning_rate=lr,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                booster="gbtree",  # TODO: compare with 'dart',rate_drop= 0.1
                random_state=seed,
                use_label_encoder=False,  # To get rid of deprecation warning we convert labels into ints
                n_jobs=n_jobs,
            )
            clf.fit(
                train_x,
                train_y,
                sample_weight=sample_weight_train,
                early_stopping_rounds=5,
                eval_metric="mlogloss",
                eval_set=[(test_x, test_y)],
                sample_weight_eval_set=[sample_weight_test],
                verbose=verbosity > 2,
            )
            if return_clfs:
                clfs.append(clf)
            # Predict the labels of the test set samples
            y_pred = clf.predict(test_x)  # in case you try booster='dart' add, ntree_limit=1 (some value>0) check again
            # Append cv step results
            confusion_matrices.append(
                confusion_matrix(test_y, y_pred, normalize="true", sample_weight=sample_weight_test)
            )
            if return_predictions or return_train_perform:
                y_pred_train = clf.predict(train_x)
            if return_train_perform:
                confusion_matrices_train.append(
                    confusion_matrix(train_y, y_pred_train, normalize="true", sample_weight=sample_weight_train)
                )
            if return_predictions:
                df = raw_df.copy()
                df.loc[obs_names[train_ix], "train"] = True
                df.loc[obs_names[train_ix], "pred"] = y_pred_train
                df.loc[obs_names[test_ix], "test"] = True
                df.loc[obs_names[test_ix], "pred"] = y_pred
                df["correct"] = df["label"] == df["pred"]
                pred_dfs.append(df)

            if progress:
                progress.advance(task_forest)

    if started and progress:
        progress.stop()

    # Pool confusion matrices
    confusions_merged = np.concatenate([np.expand_dims(mat, axis=-1) for mat in confusion_matrices], axis=-1)
    confusion_mean = pd.DataFrame(index=celltypes, columns=celltypes, data=np.mean(confusions_merged, axis=-1))
    confusion_std = pd.DataFrame(index=celltypes, columns=celltypes, data=np.std(confusions_merged, axis=-1))
    if return_train_perform:
        confusions_merged = np.concatenate([np.expand_dims(mat, axis=-1) for mat in confusion_matrices_train], axis=-1)
        confusion_mean_train = pd.DataFrame(
            index=celltypes, columns=celltypes, data=np.mean(confusions_merged, axis=-1)
        )
        confusion_std_train = pd.DataFrame(index=celltypes, columns=celltypes, data=np.std(confusions_merged, axis=-1))

    # Return
    out = [confusion_mean, confusion_std]
    if return_train_perform:
        out += [confusion_mean_train, confusion_std_train]
    if return_clfs:
        out += [clfs]
    if return_predictions:
        out += [pred_dfs]
    return out


# SUMMARY metrics
def summary_metric_diagonal_confusion_mean(conf_mat: pd.DataFrame) -> np.ndarray:
    """Compute mean of diagonal elements of confusion matrix.

    Args:
        conf_mat:
            The results of the previously calculated metric (forest_clfs accuracy).
    """
    return np.diag(conf_mat).mean()


def linear_step(x: np.ndarray, low: float, high: float, descending: bool = True) -> np.ndarray:
    """Step function with linear transition between low and high.

    Args:
        x:
            Data.
        low:
            Lower border.
        high:
            Upper border.
        descending:
            Wether to go from 1 to 0 or the other way around.
    """

    b = 1.0
    m = 1 / (high - low)

    if descending:
        return np.where(x < low, b, np.where(x > high, 0, (x - low) * (-m) + b))
    else:
        return np.where(x < low, 0, np.where(x > high, b, (x - low) * m + 0))


def summary_metric_diagonal_confusion_percentage(
    conf_mat: pd.DataFrame, threshold: float = 0.9, tolerance: float = 0.05
) -> np.ndarray:
    """Compute percentage of diagonal elements of confusion matrix above threshold.

    Note:
        To make the metric more stable we smoothen the threshold with a linear transition from
        :attr:`threshold - tolerance` to :attr:`threshold + tolerance`.

    Args:
        conf_mat:
            The results of the previously calculated metric (forest_clfs accuracy).
        threshold:
            TODO add description
        tolerance:
            TODO add description
    """
    if tolerance:
        return np.mean(linear_step(np.diag(conf_mat), threshold - tolerance, threshold + tolerance, descending=False))
    else:
        return np.mean(np.diag(conf_mat) > threshold)


################################
# marker_corr metric functions #
################################

# SHARED computations
def marker_correlation_matrix(
    adata: sc.AnnData,
    marker_list: Union[str, Dict],
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description="Marker correlation...",
) -> pd.DataFrame:
    """Compute the correlation of each marker with all genes.

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        marker_list:
            Either path to marker list or dict, e.g. {celltype1:[marker1,marker2], celltype2:[marker3,..], ..}.
            If a path is provided the marker list needs to be a csv formatted as::

                celltype1, celltype2, ...
                marker11,  marker21,
                marker12,  marker22,
                        ,  marker23,
                        ,  marker24,
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.

    Returns:
        pd.DataFrame with genes of marker list as index

        columns:
            - "celltype" - cell type annotations of marker genes
            - "mean" - mean expression of markers
            - genes - for all genes of adata.var_names correlations with markers
    """

    full_cor_mat = correlation_matrix(
        adata, progress=progress, level=level, description=description, verbosity=verbosity
    )

    # Load marker_list as dict
    if isinstance(marker_list, str):
        marker_list_df = pd.read_csv(marker_list)
        marker_list_dict = dict_to_table(marker_list_df, genes_as_index=False, reverse=True)
    else:
        marker_list_dict = marker_list

    # Get markers and their corresponding celltype
    markers = [g for _, genes in marker_list_dict.items() for g in genes]
    markers = np.unique(markers).tolist()  # TODO: maybe throw a warning if genes occur twice
    # wrt celltype we take the first occuring celltype in marker_list for a given marker
    markers = [g for g in markers if g in adata.var_names]  # TODO: Throw warning if marker genes are not in adata
    ct_annot = []
    for g in markers:
        for ct, genes in marker_list_dict.items():
            if g in genes:
                ct_annot.append(ct)
                break  # to only add first celltype in list

    # Dataframes for calculations and potential min mean expression filters
    df = full_cor_mat.loc[markers]
    df_mean_tmp = gene_means(adata, genes=markers, key="mean", inplace=False)
    assert isinstance(df_mean_tmp, pd.DataFrame)
    df_mean = df_mean_tmp.loc[markers]
    # TODO: we might need to calculate the means at a previous point in case adata.X was scaled
    #      (or we just say this metric only makes sense on unscaled data if a min_mean is given)

    # Add mean expression and cell type annotation to dataframe
    df.insert(0, "mean", df_mean["mean"])
    ct_series = pd.Series(
        index=df.index, data=[ct for gene in df.index for ct, genes in marker_list_dict.items() if (gene in genes)]
    )
    df.insert(0, "celltype", ct_series)

    return df


# PER PROBESET computations
def max_marker_correlations(
    genes: List[str],
    marker_cor: pd.DataFrame,
    per_celltype: bool = True,
    per_marker: float = True,
    per_marker_min_mean: float = None,
    per_celltype_min_mean: float = None,
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description: str = "Max marker correlations...",
) -> pd.DataFrame:
    """Get maximal correlations with marker genes.

    Args:
        genes:
            The selected genes.
        marker_cor:
            Marker correlation matrix plus cell type annotations and mean expression (see output of
            :attr:`marker_correlation_matrix`).
        per_celltype:
            Wether to return columns with per cell type max correlations.
        per_marker:
            Wether to return columns with per marker max correlations.
        per_marker_min_mean:
            Add a column for correlation per marker that only takes into accounts markers with mean expression >
            min_mean_per_marker
        per_celltype_min_mean:
            Add a column for correlation per cell type that only takes into accounts markers with mean expression >
            min_mean_per_marker.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.

    Returns:
            pd.DataFrame with marker_genes as index

            columns:
                - "celltype":
                    cell type annotations
                - "mean":
                    mean expression of marker genes
                - max correlations:
                    - "per marker" maximal correlation of probeset and marker
                    - "per celltype" only highest correlation per cell type is not nan
                    - f"... mean > {min_mean}" filtered out markers with mean expression <= min_mean
    """

    progress, started = init_progress(progress, verbosity, level)

    if progress:
        task_max_corr = progress.add_task(description, level=level, total=1)

    cor_df = marker_cor[["celltype", "mean"]].copy()
    cor_df["per marker"] = marker_cor[genes].max(axis=1)
    if per_celltype:
        cor_df["per celltype"] = cor_df["per marker"]
        idxs = cor_df.groupby(["celltype"])["per celltype"].transform(max) == cor_df["per celltype"]
        cor_df.loc[~idxs, "per celltype"] = np.nan

    if (per_marker_min_mean is not None) and per_marker:
        min_mean = per_marker_min_mean
        cor_df[f"per marker mean > {min_mean}"] = cor_df["per marker"]
        cor_df.loc[cor_df["mean"] <= min_mean, f"per marker mean > {min_mean}"] = np.nan

    if (per_celltype_min_mean is not None) and per_celltype:
        min_mean = per_celltype_min_mean
        col = f"per celltype mean > {min_mean}"
        cor_df[col] = cor_df["per marker"]
        cor_df.loc[cor_df["mean"] <= min_mean, col] = np.nan
        idxs = cor_df.groupby(["celltype"])[col].transform(max) == cor_df[col]
        cor_df.loc[~idxs, col] = np.nan

    if not per_marker:
        del cor_df["per marker"]

    if progress:
        progress.advance(task_max_corr)
        if started:
            progress.stop()

    return cor_df


# SUMMARY metrics
def summary_marker_corr(cor_df: pd.DataFrame) -> Dict:
    """Means of maximal correlations with marker genes:

    Args:
        cor_df:
            Table with maximal correlations with marker genes. :attr:`cor_df` typically has multiple columns where
            different correlations are filtered out. See :attr:`max_marker_correlations` output for expected
            :attr:`cor_df`.
    """
    summaries = cor_df[[col for col in cor_df.columns if (col != "mean")]].mean(axis=0)
    return {summary: val for summary, val in summaries.items()}


##############################
# gene_corr metric functions #
##############################

# SHARED computations
def correlation_matrix(
    adata: sc.AnnData,
    var_names: List = None,
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description="Correlation matrix...",
) -> pd.DataFrame:
    """Compute correlation matrix of adata.X

    Args:
        adata:
            An already preprocessed annotated data matrix. Typically we use log normalised data.
        var_names:
            Calculate correlation on subset of variables.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.

    Returns:
        pd.DataFrame where index and columns are adata.var_names or var_names subset if defined. Values are correlations
        between genes.
    """
    progress, started = init_progress(progress, verbosity=verbosity, level=level)

    if progress:
        task_corr = progress.add_task(description, level=level, total=1)

    if var_names:
        a = adata[:, var_names]
    else:
        a = adata

    if issparse(a.X):
        cor_mat = pd.DataFrame(index=a.var.index, columns=a.var.index, data=np.corrcoef(a.X.toarray(), rowvar=False))
    else:
        cor_mat = pd.DataFrame(index=a.var.index, columns=a.var.index, data=np.corrcoef(a.X, rowvar=False))

    if progress:
        progress.advance(task_corr)
        if started:
            progress.stop()

    return cor_mat


# PER PROBESET computations
def gene_set_correlation_matrix(
    genes: List,
    full_cor_mat: pd.DataFrame,
    ordered: bool = True,
    progress: Progress = None,
    level: int = 2,
    verbosity: int = 2,
    description: str = "Gene set correlation matrix...",
) -> pd.DataFrame:
    """Return (ordered) correlation matrix of genes.

    Args:
        genes:
            The selected genes.
        full_cor_mat:
            Correlation matrix of genes that include at least `genes`.
        ordered:
            Wether to order the correlation matrix by a linkage clustering.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.
        description:
            Description for progress bar.
    """
    progress, started = init_progress(progress, verbosity, level)

    if progress:
        task_corr = progress.add_task(description, total=1, level=level)

    cor_mat = full_cor_mat.loc[genes, genes].copy()
    if ordered:
        cor_mat = cluster_corr(cor_mat)

    if progress:
        progress.advance(task_corr)
        if started:
            progress.stop()

    return cor_mat


# SUMMARY metrics
def summary_metric_correlation_mean(cor_matrix: pd.DataFrame) -> float:
    """Calculate 1 - mean correlation.

    Args:
        cor_matrix:
            Gene set correlation matrix.

    Returns:
        float:
            1 - mean correlation
    """
    cor_mat = cor_matrix.copy()
    cor_mat = np.abs(cor_mat)
    np.fill_diagonal(cor_mat.values, 0)
    cor_mat.values[~(np.arange(cor_mat.shape[0])[:, None] > np.arange(cor_mat.shape[1]))] = np.nan
    return 1 - np.nanmean(cor_mat)


def summary_metric_correlation_percentage(
    cor_matrix: pd.DataFrame, threshold: float = 0.8, tolerance: float = 0.05
) -> np.ndarray:
    """Calculate percentage of genes with max(abs(correlations)) < threshold.

    To make the metric more stable we smoothen the threshold with a linear transition from
    threshold - tolerance to threshold + tolerance.

    Args:
        cor_matrix:
            Gene set correlation matrix.
        threshold:
            TODO add description.
        tolerance:
            TODO add description

    Returns:
        float:
            percentage of genes with max(abs(correlations)) < threshold
    """
    cor_mat = cor_matrix.copy()
    cor_mat = np.abs(cor_mat)
    np.fill_diagonal(cor_mat.values, 0)
    if tolerance:
        return np.mean(
            linear_step(np.max(cor_mat, axis=0).values, threshold - tolerance, threshold + tolerance, descending=True)
        )
    else:
        return np.mean(np.max(cor_mat, axis=0).values < threshold)
