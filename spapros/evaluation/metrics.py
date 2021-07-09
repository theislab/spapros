import numpy as np
import pandas as pd
import scanpy as sc
from rich import print
from scipy.sparse import issparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from spapros.util.util import clean_adata
from spapros.util.util import cluster_corr
from spapros.util.util import dict_to_table
from spapros.util.util import gene_means
from xgboost import XGBClassifier


METRICS_PARAMETERS = {
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
def get_metric_names():
    return [metric for metric in METRICS_PARAMETERS]


def get_metric_parameter_names():
    names = {metric: [param for param in METRICS_PARAMETERS[metric]] for metric in METRICS_PARAMETERS}
    return names


def get_metric_default_parameters():
    return METRICS_PARAMETERS


############################
# General metric functions #
############################
# We define theses to systematically seperate all metrics in three different parts
# 1. shared computations: only need to compute one time for a given dataset
# 2. specific computations: Need to be computed for each probe set
# 3. summary: Simple final computations (per probe set) that aggregate evaluations to summary metrics


def metric_shared_computations(adata=None, metric=None, parameters={}):
    """ """
    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "cluster_similarity":
        return leiden_clusterings(adata, parameters["ns"])

    elif metric == "knn_overlap":
        return knns(adata, genes="all", ks=parameters["ks"])
    elif metric == "forest_clfs":
        pass
    elif metric == "marker_corr":
        return marker_correlation_matrix(adata, marker_list=parameters["marker_list"])
    elif metric == "gene_corr":
        return correlation_matrix(adata)


def metric_pre_computations(
    genes,
    adata=None,
    metric=None,
    parameters={},
):
    """
    Note: If there are no shared results needed at all to calculate a metric the computations are put in
    `metric_computations`, this is the case for e.g. forest_clfs.
    """
    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "cluster_similarity":
        return leiden_clusterings(adata[:, genes], parameters["ns"])
    elif metric == "knn_overlap":
        return knns(adata, genes=genes, ks=parameters["ks"])
    elif metric == "forest_clfs":
        return None
    elif metric == "marker_corr":
        return None
    elif metric == "gene_corr":
        return None


def metric_computations(
    genes, adata=None, metric=None, shared_results=None, pre_results=None, parameters={}, n_jobs=-1
):
    """ """

    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "cluster_similarity":
        ann = pre_results
        ref_ann = shared_results
        nmis = clustering_nmis(ann, ref_ann, parameters["ns"])
        return nmis
    elif metric == "knn_overlap":
        knn_df = pre_results
        ref_knn_df = shared_results
        return mean_overlaps(knn_df, ref_knn_df, parameters["ks"])
    elif metric == "forest_clfs":
        results = xgboost_forest_classification(
            adata, genes, celltypes="all", ct_key=parameters["ct_key"], n_jobs=n_jobs
        )
        conf_mat = results[0]
        return conf_mat
    elif metric == "marker_corr":
        marker_cor = shared_results
        params = {k: v for k, v in parameters.items() if (k != "marker_list")}
        return max_marker_correlations(genes, marker_cor, **params)
    elif metric == "gene_corr":
        full_cor_mat = shared_results
        return gene_set_correlation_matrix(genes, full_cor_mat, ordered=True)


def metric_summary(adata=None, results=None, metric=None, parameters={}):
    """ """

    if metric not in get_metric_names():
        raise ValueError(f"Unsupported metric: {metric}")

    summary = {}

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


def compute_clustering_and_update(adata, annotations, resolution, tried_res_n, found_ns):
    """ """
    sc.tl.leiden(adata, resolution=resolution, key_added="tmp")
    n = len(set(adata.obs["tmp"]))
    if n not in found_ns:
        annotations.loc[n, adata.obs_names] = list(adata.obs["tmp"])
        annotations.loc[n, "resolution"] = resolution
        found_ns.append(n)
    tried_res_n.append([resolution, n])
    tried_res_n.sort(key=lambda x: x[0])
    return annotations, tried_res_n, found_ns


def leiden_clusterings(adata, ns, start_res=1.0):
    """Compute leiden clusters for different numbers of clusters

    Leiden clusters are calculated with different resolutions.
    A search (similar to binary search) is applied to find the right resolutions for all defined n's.

    Arguments
    ---------
    adata: anndata object
        adata object with data to compute clusters on. Need to include a
        neighbors graph (and PCA?)  TODO: make this clear.
    ns: list of two ints
        minimum (`ns[0]`) and maximum (`ns[1]`) number of clusters.
    start_res: float
        resolution to start computing clusterings.
    verbose: bool
        if True a progress bar is shown

    Return
    ------
    pd.DataFrame

    csv file (path including the name of the file is given by save_to)
        1st column refers to the number of clusters,
        2nd col refers to the resolution used to calculate the clustering
        each following column refer to individual cell's cluster assignments
        e.g.
        n , res  , <adata.obs.index[0]>, <adata.obs.index[1]>, ...., <adata.obs.index[n_cells-1]>
        6 , 1.   ,         0           ,          4          , ....,              5
        2 , 0.67 ,         1           ,          0          , ....,              1
        13, 2.34 ,         9           ,          7          , ....,             12
        17, 2.78 ,         7           ,          7          , ....,              3
        .
        .
        .
    """

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
    tried_res_n = []
    found_ns = []

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

    # Search missing n's between neighbouring found n's
    found_space = True
    while (not set(ns) <= set([res_n[1] for res_n in tried_res_n])) and (found_space):
        tmp_res_n = tried_res_n
        found_space = False
        for i in range(len(tmp_res_n) - 1):
            # check if two neighbouring resolutions have different n's
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
    return annotations


def clustering_nmis(
    annotations,
    ref_annotations,
    ns,
    method="arithmetic",
):
    """Compute NMI between clusterings and a reference set of clusterings.

    For different numbers of clusters (`ns`) the normalized mutual information
    NMI based on 2 different cluster annotations are computed.

    Parameters
    ----------
    annotations: pd.DataFrame
        Contains clusterings for different numbers of clusters.
        n , obs1, obs2, obs3, ...
        2 ,   0 ,   0 ,   1 , ...
        4 ,   1 ,   0 ,   3 , ...
        8 ,   7 ,   7 ,   2 , ...
        3 ,   2 ,   1 ,   0 , ...
        (each row is a list of cluster assignments)
    ref_annotations: pd.DataFrame
        Same as annotations for reference clusterings.
    ns: list of two ints
        minimum (`ns[0]`) and maximum (`ns[1]`) number of clusters.
    method:
        NMI implementation
            'max': scikit method with `average_method='max'`
            'min': scikit method with `average_method='min'`
            'geometric': scikit method with `average_method='geometric'`
            'arithmetic': scikit method with `average_method='arithmetic'`
            TODO: implement the following (see comment below and scib)
            'Lancichinetti': implementation by A. Lancichinetti 2009 et al.
            'ONMI': implementation by Aaron F. McDaid et al. (https://github.com/aaronmcdaid/Overlapping-NMI) Hurley 2011

    Returns
    -------
    pd.DataFrame
        Table of NMI results:
            n (index), nmi
            2        , 1.0
            3        , 0.9989
            ...
    """

    from sklearn.metrics import normalized_mutual_info_score

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

    # Calculate nmis
    for n in valid_ns:
        labels = ann.loc[n].values
        ref_labels = ref_ann.loc[n].values
        nmis.loc[n, "nmi"] = normalized_mutual_info_score(labels, ref_labels, average_method=method)

    return nmis


# SUMMARY metrics
def AUC(series, n_min=1, n_max=60):
    """ """
    tmp = series.loc[(series.index >= n_min) & (series.index <= n_max)]
    n = len(tmp)
    return tmp.sum() / n


def summary_nmi_AUCs(nmis, AUC_borders):
    """Calculate AUC over range of nmi values

    nmis: pd.DataFrame

            n (index), nmi
            2        , 1.0
            3        , 0.9643
            4        , NaN
            5        , 0.98
            ...
    AUC_borders: list of lists of two ints
        Calculates nmi AUCs over given borders. E.g. `AUC_borders = [[2,4],[5,20]]` calculates nmi over n ranges
        2 to 4 and 5 to 20. Defined border shouldn't exceed values in `nmis`.

    """
    AUCs = {}
    for ths in AUC_borders:
        AUCs[f"nmi_{ths[0]}_{ths[1]}"] = AUC(nmis["nmi"].interpolate(), n_min=ths[0], n_max=ths[1])
    return AUCs


################################
# knn_overlap metric functions #
################################

# SHARED AND and PER PROBESET computations


def knns(adata, genes="all", ks=[10, 20]):
    """Compute nearest neighbors of observations for different ks

    adata: AnnData
    genes: "all" or list of strs
    ks: list of ints
        Calculate knn graphs for each k in `ks`.

    Returns
    -------
    pd.DataFrame
        Includes nearest neighbors for all ks
        gene (index), k10_1, k10_2, ..., k20_1, k20_2, ...
        ISG15       , 3789 , 512  ,    , 9720 , 15   , ...
        TNFRSF4     , 678  , 713  ,    , 7735 , 6225 , ...
        ...
    """

    # Set n_pcs to 50 or number of genes if < 50
    n_pcs = np.min([50, len(genes) - 1])

    if genes == "all":
        a = adata.copy()
    else:
        a = adata[:, genes].copy()

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
    sc.tl.pca(a, n_comps=n_pcs)

    # Get nearest neighbors for each k
    df = pd.DataFrame(index=a.obs_names)
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

    return df


def mean_overlaps(knn_df, ref_knn_df, ks):
    """Calculate mean overlaps of knn graphs of different ks

    knn_df: pd.DataFrame
    ref_knn_df: pd.DataFrame


    Returns
    -------
    pd.DataFrame
        k (index), mean

    """
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
    return df


# SUMMARY metrics


def summary_knn_AUC(means_df):
    """Calculate AUC of mean overlaps over ks"""
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
    adata,
    selection,
    celltypes="all",
    ct_key="Celltypes",
    n_cells_min=40,
    max_depth=3,
    lr=0.2,
    colsample_bytree=1,
    cv_splits=5,
    min_child_weight=None,
    gamma=None,
    seed=0,
    n_seeds=5,
    verbosity=0,
    return_train_perform=False,
    return_clfs=False,
    n_jobs=1,
):
    """Measure celltype classification performance with gradient boosted forests.

    We train extreme gradient boosted forest classifiers on multi class classification of cell types. Cross validation
    is performed to get an average confusion matrix (normalised by ground truth counts and sample weights to weight cell
    types in a balanced way). To make the performance measure robust only cell types with at least `n_cells_min` are
    taken into account. To make the cross validation more robust we run it with `n_seeds`. I.e. `cv_splits` x `n_seeds`
    classifiers are trained and evaluated.

    Parameters
    ----------
    adata: AnnData
        We expect log normalised data in adata.X.
    selection: list or pd.DataFrame
        Forests are trained on genes of the list or genes defined in the bool column selection['selection'].
    celltypes: 'all' or list
        Forests are trained on the given celltypes.
    ct_key: str
        Column name of adata.obs with cell type info.
    n_cells_min: int
        Minimal number of cells to filter out cell types from the training set. Performance results are not robust
        for low `n_cells_min`.
    max_depth: str
        max_depth argument of XGBClassifier.
    cv_splits: int
        Number of cross validation splits.
    lr: float
        Learning rate of XGBClassifier.
    colsample_bytree: float
        Fraction of features (randomly selected) that will be used to train each tree of XGBClassifier.
    gamma: float
        Regularisation parameter of XGBClassifier. Instruct trees to add nodes only if the associated loss gain is
        larger or equal to gamma.
    seed: int
    n_seeds: int
        Number of training repetitions with different seeds. We use multiple seeds to make results more robust.
        Also we don't want to increase the number of CV splits to keep a minimal test size.
    verbosity: int
        Set to 2 for progress bar. Set to 3 to print test performance of each tree during training.
    return_train_perform: bool
        Wether to also return confusion matrix of training set.
    return_clfs: str
        Wether to return the classifier objects.
    n_jobs: int
        Multiprocessing number of processes.

    Returns
    -------
    pd.DataFrame:
        confusion matrix averaged over cross validations and seeds.
    pd.DataFrame:
        confusion matrix standard deviation over cross validations and seeds.
    if return_train_perform:
        pd.DataFrames as above for train set.
    if return_clfs:
        list of XGBClassifier objects of each cross validation step and seed.

    """

    if verbosity > 1:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm_notebook as tqdm
        desc = "XGBClassifier Cross Val."
    else:
        tqdm = None

    # Define cell type list
    if celltypes == "all":
        celltypes = adata.obs[ct_key].unique().tolist()
    # Filter out cell types with less cells than n_cells_min
    cell_counts = adata.obs[ct_key].value_counts().loc[celltypes]
    if (cell_counts < n_cells_min).any():
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
        seeds = rng.integers(low=0, high=100000, size=n_seeds)
    else:
        seeds = [seed]

    # Initialize variables for training
    confusion_matrices = []
    if return_train_perform:
        confusion_matrices_train = []
    if return_clfs:
        clfs = []
    n_classes = len(celltypes)

    # Cross validated random forest training
    for seed in tqdm(seeds, desc="seeds", total=len(seeds)) if tqdm else seeds:
        k_fold = StratifiedKFold(n_splits=cv_splits, random_state=seed, shuffle=True)
        for train_ix, test_ix in tqdm(k_fold.split(X, y), desc=desc, total=cv_splits) if tqdm else k_fold.split(X, y):
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
            if return_train_perform:
                y_pred = clf.predict(train_x)
                confusion_matrices_train.append(
                    confusion_matrix(train_y, y_pred, normalize="true", sample_weight=sample_weight_train)
                )

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
    return out


# SUMMARY metrics
def summary_metric_diagonal_confusion_mean(conf_mat):
    """Compute mean of diagonal elements of confusion matrix"""
    return np.diag(conf_mat).mean()


def summary_metric_diagonal_confusion_percentage(conf_mat, threshold=0.9):
    """Compute percentage of diagonal elements of confusion matrix above threshold"""
    return np.mean(np.diag(conf_mat) > threshold)


################################
# marker_corr metric functions #
################################

# SHARED computations
def marker_correlation_matrix(adata, marker_list):
    """
    adata: AnnData
    marker_list: str or dict
        Either path to marker list or dict, e.g. {celltype1:[marker1,marker2], celltype2:[marker3,..], ..}.
        If a path is provided the marker list needs to be a csv formatted as:
            celltype1, celltype2, ...
            marker11,  marker21,
            marker12,  marker22,
                    ,  marker23,
                    ,  marker24,

    Returns
    -------
    pd.DataFrame
        index: genes of marker list
        columns:
            "celltype" - cell type annotations of marker genes
            "mean" - mean expression of markers
            genes - for all genes of adata.var_names correlations with markers

    """

    full_cor_mat = correlation_matrix(adata)

    # Load marker_list as dict
    if isinstance(marker_list, str):
        marker_list = pd.read_csv(marker_list)
        marker_list = dict_to_table(marker_list, genes_as_index=False, reverse=True)

    # Get markers and their corresponding celltype
    markers = [g for _, genes in marker_list.items() for g in genes]
    markers = np.unique(markers).tolist()  # TODO: maybe throw a warning if genes occur twice
    # wrt celltype we take the first occuring celltype in marker_list for a given marker
    markers = [g for g in markers if g in adata.var_names]  # TODO: Throw warning if marker genes are not in adata
    ct_annot = []
    for g in markers:
        for ct, genes in marker_list.items():
            if g in genes:
                ct_annot.append(ct)
                break  # to only add first celltype in list

    # Dataframes for calculations and potential min mean expression filters
    df = full_cor_mat.loc[markers]
    df_mean = gene_means(adata, genes=markers, key="mean", inplace=False).loc[markers]
    # TODO: we might need to calculate the means at a previous point in case adata.X was scaled
    #      (or we just say this metric only makes sense on unscaled data if a min_mean is given)

    # Add mean expression and cell type annotation to dataframe
    df.insert(0, "mean", df_mean["mean"])
    ct_series = pd.Series(
        index=df.index, data=[ct for gene in df.index for ct, genes in marker_list.items() if (gene in genes)]
    )
    df.insert(0, "celltype", ct_series)

    return df


# PER PROBESET computations
def max_marker_correlations(
    genes, marker_cor, per_celltype=True, per_marker=True, per_marker_min_mean=None, per_celltype_min_mean=None
):
    """Get maximal correlations with marker genes

    genes: list
        Gene set list.
    marker_cor: pd.DataFrame
        Marker correlation matrix plus cell type annotations and mean expression (see output of
        `marker_correlation_matrix`)
    per_celltype: bool
        Wether to return columns with per cell type max correlations.
    per_marker: bool
        Wether to return columns with per marker max correlations.
    per_marker_min_mean: float
        Add a column for correlation per marker that only takes into accounts markers with mean expression >
        min_mean_per_marker
    per_celltype_min_mean: float
        Add a column for correlation per cell type that only takes into accounts markers with mean expression >
        min_mean_per_marker

    Returns
    -------
    pd.DataFrame
        index: marker_genes
        columns:
            - "celltype" - cell type annotations
            - "mean" - mean expression of marker genes
            - max correlations:
                - "per marker" maximal correlation of probeset and marker
                - "per celltype" only highest correlation per cell type is not nan
                - f"... mean > {min_mean}" filtered out markers with mean expression <= min_mean

    """
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

    return cor_df


# SUMMARY metrics
def summary_marker_corr(cor_df):
    """Means of maximal correlations with marker genes

    cor_df: pd.DataFrame
        Table with maximal correlations with marker genes. cor_df typically has multiple columns where different
        correlations are filtered out. See `max_marker_correlations` output for expected `cor_df`.

    Returns
    -------
    dict of floats:
    """
    summaries = cor_df[[col for col in cor_df.columns if (col != "mean")]].mean(axis=0)
    return {summary: val for summary, val in summaries.items()}


##############################
# gene_corr metric functions #
##############################

# SHARED computations
def correlation_matrix(adata, var_names=None):
    """Compute correlation matrix of adata.X

    adata: AnnData
    var_names: list
        Calculate correlation on subset of variables

    Return
    ------
    pd.DataFrame:
        Index and columns are adata.var_names or var_names subset if defined. Values are correlations between genes
    """
    if var_names:
        a = adata[:, var_names]
    else:
        a = adata

    if issparse(a.X):
        cor_mat = pd.DataFrame(index=a.var.index, columns=a.var.index, data=np.corrcoef(a.X.toarray(), rowvar=False))
    else:
        cor_mat = pd.DataFrame(index=a.var.index, columns=a.var.index, data=np.corrcoef(a.X, rowvar=False))
    return cor_mat


# PER PROBESET computations
def gene_set_correlation_matrix(genes, full_cor_mat, ordered=True):
    """Return (ordered) correlation matrix of genes

    genes: list
        Gene set list
    full_cor_mat: pd.DataFrame
        Correlation matrix of genes that include at least `genes`.
    ordered: bool
        Wether to order the correlation matrix by a linkage clustering.

    Return
    ------

    """
    cor_mat = full_cor_mat.loc[genes, genes].copy()
    if ordered:
        cor_mat = cluster_corr(cor_mat)
    return cor_mat


# SUMMARY metrics
def summary_metric_correlation_mean(cor_matrix):
    """Calculate 1 - mean correlation

    cor_mat: pd.DataFrame and np.array
        Gene set correlation matrix.

    Return
    ------
    float:
        1 - mean correlation
    """
    cor_mat = cor_matrix.copy()
    cor_mat = np.abs(cor_mat)
    np.fill_diagonal(cor_mat.values, 0)
    cor_mat.values[~(np.arange(cor_mat.shape[0])[:, None] > np.arange(cor_mat.shape[1]))] = np.nan
    return 1 - np.nanmean(cor_mat)


def summary_metric_correlation_percentage(cor_matrix, threshold=0.8):
    """Calculate percentage of genes with max(abs(correlations)) < threshold

    cor_mat: pd.DataFrame and np.array
        Gene set correlation matrix.
    threshold: float

    Return
    ------
    float:
        percentage of genes with max(abs(correlations)) < threshold
    """
    cor_mat = cor_matrix.copy()
    cor_mat = np.abs(cor_mat)
    np.fill_diagonal(cor_mat.values, 0)
    return np.sum(np.max(cor_mat, axis=0) < threshold) / len(cor_mat)
