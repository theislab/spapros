from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from sklearn.decomposition import SparsePCA

from spapros.evaluation.evaluation import tree_classifications
from spapros.util.util import clean_adata
from spapros.util.util import preprocess_adata


def apply_correlation_penalty(scores, adata, corr_penalty, preselected_genes=None):
    """Compute correlations and iteratively penalize genes according max corr with selected genes

    This function is thoroughly tested.

    TODO: write docstring

    scores: pd.DataFrame
        no gene in preselected_genes should occur in scores.index
    """
    if preselected_genes is None:
        preselected_genes = []
    penalized_scores = scores.copy()

    # Compute correlations
    if scipy.sparse.issparse(adata.X):
        cor_mat = np.abs(np.corrcoef(adata.X.toarray(), rowvar=False))
    else:
        cor_mat = np.abs(np.corrcoef(adata.X, rowvar=False))
    cor_df = pd.DataFrame(index=adata.var.index, columns=adata.var.index, data=cor_mat)
    cor_df.fillna(0, inplace=True)

    # Eventually penalize according max correlation with preselected genes
    max_cor = cor_df.loc[penalized_scores.index, preselected_genes].max(axis=1).to_frame("max")
    if len(preselected_genes) > 0:
        tmp_scores = penalized_scores.mul(corr_penalty(max_cor), axis=0)
    else:
        tmp_scores = penalized_scores

    # Iteratively penalize according max correlations with theoretically selected genes
    for _ in range(len(scores.index)):
        gene = tmp_scores["scores"].idxmax()
        penalized_scores.loc[gene] = tmp_scores.loc[gene]
        max_cor.drop(index=gene, inplace=True)
        max_cor = max_cor.join(cor_df.loc[max_cor.index, gene])
        max_cor = max_cor.max(axis=1).to_frame(name="max")
        tmp_scores = scores.loc[max_cor.index].mul(corr_penalty(max_cor), axis=0)

    return penalized_scores


def apply_penalties(scores, adata, penalty_keys=[]):
    """
    adata: AnnData
        contains penalty_keys
    scores: pd.DataFrame
        index: genes, any columns. Values in a row are penalized in the same way
    penalty_keys: list of strs
        columns in adata.var containing penalty factors

    Returns
    -------
    pd.DataFrame
        scores multiplied with each gene's penalty factors
    """
    s = scores
    penalty = pd.DataFrame(
        index=scores.index,
        data={k: adata.var.loc[scores.index, k] for k in penalty_keys},
    )
    s = s.mul(penalty.product(axis=1), axis=0)
    return s


##########################################################################


def select_pca_genes(
    adata,
    n,
    variance_scaled=False,
    absolute=True,
    n_pcs=20,
    process_adata=None,
    penalty_keys=[],
    corr_penalty=None,
    inplace=True,
):
    """Select n features based on pca loadings

    Arguments
    ---------
    adata: AnnData
        adata
    n: int
        number of selected features
    variance_scaled: bool
        If True loadings are defined as eigenvector_component * sqrt(eigenvalue).
        If False loadings are defined as eigenvector_component.
    absolute: bool
        Take absolute value of loadings.
    n_pcs: int
        number of PCs used to calculate loadings sums.
    process_adata: list of strs
        Options to process adata before selection. Supported options are:
        - 'norm': normalise adata.X according adata.obs['size_factors']
        - 'log1p': log(adata.X + 1)
        - 'scale': scale genes of adata.X to zero mean and std 1 (adata.X no longer sparse)
        TODO: this selection function should only accept norm-log1p or norm-log1p-scaled data
              (...actually we might not want to include such constraint,
               maybe someone wants to try selections on other data than scRNA-seq)
    penalty_keys: list of strs
        List of keys for columns in adata.var that are multiplied with the scores
    corr_penalty: function
        Function that maps values from [0,1] to [0,1]. It describes an iterative penalty function
        that is applied on pca selected genes. The highest correlation with already selected genes
        to the next selected genes are penalized according the given function. (max correlation is
        recomputed after each selected gene)
    inplace: bool
        Save results in adata.var or return dataframe

    Returns
    -------
    if not inplace:
        pd.DataFrame (like adata.var) with columns
        - 'selection': bool indicator of selected genes
        - 'selection_score': pca loadings based score of each gene
        - 'selection_ranking': ranking according selection scores
    if inplace:
        Save results in adata.var[['selection','selection_score','selection_ranking']]
    """

    a = adata.copy()

    if n_pcs > a.n_vars:
        n_pcs = a.n_vars

    clean_adata(a, obs_keys=["size_factors"])
    if process_adata:
        preprocess_adata(a, options=process_adata)

    sc.pp.pca(
        a,
        n_comps=n_pcs,
        zero_center=True,
        svd_solver="arpack",
        random_state=0,
        return_info=True,
        copy=False,
    )

    loadings = a.varm["PCs"].copy()[:, :n_pcs]
    if variance_scaled:
        loadings *= np.sqrt(a.uns["pca"]["variance"][:n_pcs])
    if absolute:
        loadings = abs(loadings)

    scores = pd.DataFrame(index=adata.var.index, data={"scores": np.sum(loadings, axis=1)})
    scores = apply_penalties(scores, adata, penalty_keys=penalty_keys)
    if not (corr_penalty is None):
        scores = apply_correlation_penalty(scores, adata, corr_penalty)

    selected_genes = scores.nlargest(n, "scores").index.values
    selection = pd.DataFrame(
        index=scores.index,
        data={
            "selection": False,
            "selection_score": scores["scores"],
            "selection_ranking": scores["scores"].rank(method="dense", ascending=False),
        },
    )
    selection.loc[selected_genes, "selection"] = True

    if inplace:
        adata.var[["selection", "selection_score", "selection_ranking"]] = selection[
            ["selection", "selection_score", "selection_ranking"]
        ]
    else:
        return selection


def marker_scores(adata, obs_key="cell_types", groups="all", reference="rest", rankby_abs=False):
    """Compute marker scores for genes in adata

    adata: AnnData
        log normalised data
    obs_key: str
        column name of adata.obs for which marker scores are calculated
    groups, reference, rankby_abs: see sc.tl.rank_genes_groups()

    Returns
    -------
    pd.DataFrame
        index are genes as in adata.var.index, columns are names of groups in adata.obs[obs_key]
    """
    df = pd.DataFrame(index=adata.var.index)
    a = sc.tl.rank_genes_groups(
        adata,
        obs_key,
        use_raw=False,
        groups=groups,
        reference=reference,
        n_genes=adata.n_vars,
        rankby_abs=rankby_abs,
        copy=True,
        method="wilcoxon",
        corr_method="benjamini-hochberg",
    )
    names = a.uns["rank_genes_groups"]["scores"].dtype.names
    marker_scores = {
        name: pd.DataFrame(
            index=a.uns["rank_genes_groups"]["names"][name],
            data={name: a.uns["rank_genes_groups"]["scores"][name]},
        )
        for name in names
    }
    for name in names:
        df = df.join(marker_scores[name])
    return df


def select_DE_genes(
    adata,
    n,
    per_group=False,
    obs_key="cell_types",
    process_adata=None,
    penalty_keys=[],
    groups="all",
    reference="rest",
    rankby_abs=False,
    inplace=True,
):
    """Select genes based on wilxocon rank genes test

    adata: AnnData
        log normalised data
    n: int
        nr of genes to selected (in total if not per_group else per group)
    per_group: bool
        Select `n` genes per group of adata.obs[obs_key] (default: False).
        Note that the same gene can be selected for multiple groups.
    obs_key: str
        column name of adata.obs for which marker scores are calculated
    penalty_keys: list of strs
        penalty factor for gene selection.
    groups, reference, rankby_abs: see sc.tl.rank_genes_groups()

    Returns
    -------
    pd.DataFrame
        index are genes as in adata.var.index, bool column: 'selection'
    """
    if process_adata:
        a = preprocess_adata(adata, options=process_adata, inplace=False)
    else:
        a = adata

    group_counts = a.obs[obs_key].value_counts() > 2
    if not group_counts.values.all():
        groups = group_counts[group_counts].index.to_list()

    selection = pd.DataFrame(index=a.var.index, data={"selection": False})
    scores = marker_scores(a, obs_key=obs_key, groups=groups, reference=reference, rankby_abs=rankby_abs)
    scores = apply_penalties(scores, a, penalty_keys=penalty_keys)
    if per_group:
        genes = [gene for group in scores.columns for gene in scores.nlargest(n, group).index]
    else:
        ordered_gene_lists = [
            list(scores.sort_values(by=[group], ascending=False).index.values) for group in scores.columns
        ]
        n_groups = len(ordered_gene_lists)
        genes = []
        count = 0
        while len(genes) < n:
            i = count // n_groups
            j = count % n_groups
            gene = ordered_gene_lists[j][i]
            if gene not in genes:
                genes.append(gene)
            count += 1
    selection.loc[genes, "selection"] = True
    if inplace:
        adata.var["selection"] = selection["selection"]
    else:
        return selection


def add_tree_markers(
    adata,
    selection,
    f1,
    dec_genes,
    f1_ref,
    dec_genes_ref,
    ct_key="Celltypes",
    n_max_per_it=5,
    n_max=None,
    performance_th=0.1,
    importance_th=0.1,
    verbose=False,
):
    """Add markers till reaching classification performance of reference

    Classification trees are given for the pre selected genes and a reference. Classification performance and
    the feature importance of the reference trees are used to add markers.
    TODO: Write detailed description
    - maximal 1 marker per celltype per iteration
    - rank according the sum of importances across celltypes of interest
    - ...


    Arugments
    ---------
    adata: AnnData
    selection: list or pd.DataFrame
        Already selected genes. index: genes, boolean column 'selection'
    f1: pd.DataFrame
        f1 results table of ev.tree_classifications. This one should belong to a tree_classification on genes
        in `selection` (TODO: The function could also be written with the first tree_classification in here...)
    dec_genes: dict
        key: genes, values: importances for tree classifications. Result of ev.tree_classifications
    f1_ref: As f1 but for reference gene set
    dec_genes_ref: As dec_genes but for reference gene set
    ct_key: str
    n_mar_per_it: int
        Add n_mar_per_it genes per iteration. In each iteration ev.tree_classifications will be calculated.
        Note that per celltype only one gene per iteration is added.
    n_max: int
        Limit the upper number of added genes
    performance_th: float
        Further markers are only added for celltypes that have a performance difference above performance_th compared to
        the reference performance
    importance_th: float
        Only reference genes with at least importance_th as feature importance in the reference tree are added as markers.
        TODO: We're working with a relative importance measure here. An absolute value could be better.
              (If classification is bad for a given celltype then useless genes have a high importance)
    #save_load: Might be interesting to add some log of results...
    verbose: bool

    Returns
    -------
    pd.Dataframe as selection with added markers
    TODO: we could even have a ranking according our adding procedure. Might be interesting to return such ranking
    """
    if isinstance(selection, list):
        tmp = selection
        selection = pd.DataFrame(index=adata.var.index, data={"selection": False})
        selection.loc[tmp, "selection"] = True
    f1_diffs = f1_ref - f1
    f1_diffs = f1_diffs.loc[f1_diffs > performance_th]
    importances = pd.DataFrame(dec_genes_ref)
    importances = importances[importances > importance_th]
    importances = importances.dropna(axis=0, thresh=1).dropna(axis=1, thresh=1)
    celltypes = [ct for ct in f1_diffs.index if ct in importances.columns]
    selected = list(selection.loc[selection["selection"]].index)
    unselected = [g for g in importances.index if not (g in selected)]

    n_max_unreached = True
    selected_all = False
    n_added = 0
    while celltypes and n_max_unreached and (not selected_all):
        importances = importances.loc[unselected, celltypes]
        importances = importances.dropna(axis=0, thresh=1).dropna(axis=1, thresh=1)
        if n_max:
            if (n_max - n_added) < n_max_per_it:
                n_max_per_it = n_max - n_added
        if len(importances.idxmax().unique()) > n_max_per_it:
            new_markers = list(importances.sum(axis=1).nlargest(n_max_per_it).index)
        else:
            new_markers = list(importances.idxmax().unique())
        selected += new_markers
        n_added += len(new_markers)
        if verbose:
            print(f"Add new markers:\n{new_markers}")
        unselected = [g for g in importances.index if not (g in new_markers)]
        if verbose:
            print(f"Train decision trees on celltypes:\n{celltypes}")
        f1, _ = tree_classifications(
            adata,
            selected,
            celltypes=celltypes,
            ct_key=ct_key,
            plot=False,
            save_load=False,
        )
        if verbose:
            print("\t Training finished.")
        f1_diffs = f1_ref.loc[f1.index] - f1
        f1_diffs = f1_diffs.loc[f1_diffs > performance_th]
        celltypes = [ct for ct in f1_diffs.index if ct in importances.columns]

        if n_max and (n_added == n_max):
            n_max_unreached = False
        if len(unselected) == 0:
            selected_all = True

    df = selection.copy()
    df.loc[selected, "selection"] = True
    return df


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
# We want all of the following methods implemented as the methods above. This means
# adding multiplicative constraints (e.g. expression penalty). I am also thinking about
# adding the option of iterative penalties (i.e. a penalty based on already selected
# genes, e.g. a correlation penalty)
# Note: We can implement these constraints only for score based methods. This means sparse pca
#       and selfE do not work
#       - what would also be handy and work in general is a binary constraint (mask).
#         So you wouldn't need to subset the adata (which is important for the workflow, evaluation pipe, ...)


###### This is an old pca selection function with several different options how the loadings are aggregated to scores
###### My first runs showed that `method="sum"` performed best (as used in the function `select_pca_genes()`).
###### We should also include the old options to rerun the comparisons at the end.
def select_features_pca_loadings(adata, n, method="sum", absolute=True, n_pcs=30, inplace=True, verbose=True):
    """
    Arguments
    ---------
    n: int
        number of features to select
    method: str
        different loadings based method for feature selection
        "sum" - features with highest sum of loadings
        "max" - features with highest single loadings
        "max_PC" - features of `n` highest loadings of different PC components
        "max_PC_order" - features with highest loadings of first `n` PC components.
                         If a feature was already picked the 2nd highest is chosen
                         and so on.
                         --> make sure that n_pcs >= n if you use this method
    absolute: bool
        if True the absolute values of loadings are taken into account such
        that also negative loadings with high abs() can lead to chose features.
    n_pcs: int
        number of calculated PC components. If None n_pcs is set to the number of
        genes in adata. Only interesting for method = "sum" or if there are no
        existing PCA results
    inplace: bool
        If True add results to adata.var, else results are returned
    verbose: bool
        optionally print infos

    Returns
    -------
    if inplace: add
        - adata.var['selection'] - boolean pd.Series of n selected features
        - adata.var['selection_score'] - float pd.Series with selection scores (if scores exist for all features)
    else: return
        pd.Dataframe with columns 'selection', 'selection_score' (latter only for some methods)
    """

    a = adata.copy()

    if n_pcs > a.n_vars:
        n_pcs = a.n_vars
    if method == "max_PC_order":
        n_pcs = n

    clean_adata(a)

    if verbose:
        print("Compute PCA.")
    sc.pp.pca(
        a,
        n_comps=n_pcs,
        zero_center=True,
        svd_solver="arpack",
        random_state=0,
        return_info=True,
        copy=False,
    )

    loadings = a.varm["PCs"].copy()[:, :n_pcs]
    if absolute:
        loadings = abs(loadings)

    if method == "sum":
        scores = np.sum(loadings, axis=1)
        unique_scores = np.unique(scores)
        unique_scores[::-1].sort()
        feature_idxs = []
        count = 0
        for u in unique_scores:
            for index in np.where(scores == u)[0]:
                feature_idxs.append(index)
                count += 1
                if count == n:
                    break
            if count == n:
                break
        # scores = scores[feature_idxs]
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True
        a.var["selection_score"] = scores
    elif method == "max":
        scores = []
        unique_loadings = np.unique(loadings)
        unique_loadings[::-1].sort()
        feature_idxs = []
        pc_idxs = []
        count = 0
        for u in unique_loadings:
            i, j = np.where(loadings == u)
            for k in range(len(i)):
                feature_idx = i[k]
                pc_idx = j[k]
                if feature_idx not in feature_idxs:
                    feature_idxs.append(feature_idx)
                    scores.append(u)
                    pc_idxs.append(pc_idx)
                    count += 1
                if count == n:
                    break
            if count == n:
                break
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True
    elif method == "max_PC":
        scores = []
        unique_loadings = np.unique(loadings)
        unique_loadings[::-1].sort()
        feature_idxs = []
        pc_idxs = []
        count = 0
        for u in unique_loadings:
            i, j = np.where(loadings == u)
            for k in range(len(i)):
                feature_idx = i[k]
                pc_idx = j[k]
                if (feature_idx not in feature_idxs) and (pc_idx not in pc_idxs):
                    feature_idxs.append(feature_idx)
                    scores.append(u)
                    pc_idxs.append(pc_idx)
                    count += 1
                if count == n:
                    break
            if count == n:
                break
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True
    elif method == "max_PC_order":
        scores = []
        feature_idxs = []
        pc_idxs = [i for i in range(n)]
        for pc_idx in pc_idxs:
            found_feature = False
            pc_loadings = loadings[:, pc_idx]
            while not found_feature:
                feature_idx = np.nanargmax(pc_loadings)
                if feature_idx not in feature_idxs:
                    feature_idxs.append(feature_idx)
                    scores.append(np.nanmax(pc_loadings))
                    found_feature = True
                else:
                    pc_loadings[feature_idx] = np.nan
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True

    if inplace:
        adata.var["selection"] = a.var["selection"].copy()
        if "selection_score" in a.var.columns:
            adata.var["selection_score"] = a.var["selection_score"].copy()
    else:
        if "selection_score" in a.var.columns:
            return a.var[["selection", "selection_score"]].copy()
        else:
            return a.var[["selection"]].copy()


def select_highly_variable_features(adata, n, flavor="cell_ranger", inplace=True):
    a = adata.copy()
    clean_adata(a)
    sc.pp.highly_variable_genes(
        a,
        n_top_genes=n,
        n_bins=20,
        flavor=flavor,
        subset=False,
        inplace=True,
        batch_key=None,
    )
    if inplace:
        adata.var["selection"] = a.var["highly_variable"]
    else:
        a.var["selection"] = a.var["highly_variable"]
        return a.var[["selection"]].copy()


def random_selection(adata, n, seed=0, inplace=True):
    np.random.seed(seed=seed)
    f_idxs = np.random.choice(adata.n_vars, n, replace=False)
    df = pd.DataFrame(index=adata.var.index, columns=["selection"])
    df["selection"] = False
    df["selection"].iloc[f_idxs] = True
    if inplace:
        adata.var["selection"] = df["selection"].copy()
    else:
        return df


############################ Highest expressed genes ################################


def get_mean(X, axis=0):
    if scipy.sparse.issparse(X):
        mean = X.mean(axis=axis, dtype=np.float64)
        mean = np.array(mean)[0]
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
    return mean


def highest_expressed_genes(adata, n, inplace=True, process_adata=None, use_existing_means=False):
    """Select n highest expressed genes in adata"""
    if process_adata:
        a = preprocess_adata(adata, options=process_adata, inplace=False)
    else:
        a = adata

    df = pd.DataFrame(index=a.var.index, columns=["means"])
    if use_existing_means:
        if "means" in a.var:
            df["means"] = a.var["means"]
        else:
            raise ValueError('Column "means" in adata.var not found. Either add it or set use_existing_means=False.')
    else:
        df["means"] = get_mean(a.X)
    df["selection"] = False
    df.loc[df.nlargest(n, "means").index.values, "selection"] = True
    if inplace:
        adata.var["selection"] = df["selection"].copy()
    else:
        return df[["selection"]].copy()


##################################################################################
################################# Sparse PCA #####################################
##################################################################################


def sparse_pca(X, n_pcs, alpha, seed):
    transformer = SparsePCA(n_components=n_pcs, alpha=alpha, random_state=seed)
    transformer.fit(X)
    transformer.transform(X)
    loadings = transformer.components_
    features = np.any(np.abs(loadings) > 0, axis=0)
    return features, loadings


def sort_alphas(alphas, n_features):
    zipped_lists = zip(alphas, n_features)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    alphas, n_features = [list(t) for t in tuples]
    return alphas, n_features


def next_alpha(n, alphas, n_features):
    for i, a in enumerate(alphas):
        if (i == 0) and (n_features[i] < n):
            return a / 2
        elif (i < (len(alphas) - 1)) and ((n_features[i] > n) and (n_features[i + 1] < n)):
            return (alphas[i] + alphas[i + 1]) / 2
        elif (i == (len(alphas) - 1)) and (n_features[i] > n):
            return a * 2


def spca_feature_selection(
    adata,
    n,
    n_pcs=30,
    a_init=10,
    n_alphas_max=20,
    n_alphas_min=3,
    tolerance=0.05,
    seed=0,
    verbosity=1,
    inplace=True,
):
    """Select features based on sparse pca

    Sparse PCA is based on the regression formulation of pca with an additional lasso constraint.
    The lasso constraint leads to sparse loadings vectors for the found principal components.
    The higher the lasso parameter alpha the sparser the loadings vectors. We apply a search through
    different alphas to find sparse loadings vectors that are based on approximately n features in total.
    Since searching for the correct alpha takes a long time we define a tolerance: An accepted alpha is chosen
    if it comes with n_features selected such that
        n <= n_features <= (1+tolerance)*n


    One flaw: if we do not find exactly n features we use the sums of abs loadings as scores and neglect the last features
    with lowest scores to get n features. The problem compared with the non sparse pca selection is that we do not
    scale our loadings with the eigenvalue of the PCs. In simple pca we did that (as far as i know). It could be that
    a feature has very low loadings on the first PCs which are important PCs, the feature however is droped. IMO it's
    fine, the effect shouldn't be too high, since we only drop a few features at maximum and the features still have
    low loadings.. A question is also if we have a clear hierarchy which PC is more important in SPCA -> measure variance?
    (small update on this: our pca loadings based selection also does not take scaling with sqrt(eigenvalue) into account)

    Note that SPCA takes much longer for small alphas (less sparsity). Keep that in mind when choosing a_init etc.

    Parameters
    ----------
    adata:
    n: int
        number of features to select.
    n_pcs: int
        number of sparse PC components
    a_init: float
        first lasso alpha that is tried
    n_alphas_max: int (or None)
        The maximal number of alphas that is tried to find n_features in the defined tolerance to n
    n_alphas_min: int (or None)
        Minimal number of alphas to try. This is interesting to set if we are already in the tolerance
        but the search should still go on to ultimately find exactly n features.
    tolerance: float
        accept solutions for n_features such that n <= n_features <= (1+tolerance)*n
    seed: int
        random seed for sparse pca optimization
    verbosity: int
    inplace: bool
        if True save results in adata.var else return results

    Return
    ------
    if inplace:
        add
        - adata.var['selection'] (boolean column of n selected features)
        - adata.var['selection_scores'] (column with sum of abs loadings for each feature)
    else:
        return
        list: name of genes
        list: sum of abs loadings for each feature
    """

    # conditions for while loop
    n_features_in_tolerance = False
    n_features_equals_n = False
    max_alphas = False
    if (type(n_alphas_min) == int) and (n_alphas_min > 1):
        min_alphas = False
    else:
        min_alphas = True

    alphas = [a_init]
    if verbosity >= 1:
        t0 = datetime.now()
        print(f"Start alpha trial {len(alphas)} ({datetime.now() - t0})")
    features, loadings = sparse_pca(adata.X, n_pcs, alphas[0], seed)
    n_features = [np.sum(features)]
    if verbosity == 1:
        print(f"\t alpha = {alphas[0]}, n_features = {n_features[0]}")
    # adjust conditions
    if n_features[-1] == n:
        n_features_equals_n = True
    if (n <= n_features[-1]) and (n_features[-1] <= (1 + tolerance) * n):
        n_features_in_tolerance = True
    if (type(n_alphas_min) == int) and (len(alphas) >= n_alphas_min):
        min_alphas = True
    if (type(n_alphas_max) == int) and (len(alphas) == n_alphas_max):
        max_alphas = True

    while not (n_features_equals_n or (n_features_in_tolerance and min_alphas) or max_alphas):
        # get next alpha
        alphas, n_features = sort_alphas(alphas, n_features)
        alpha = next_alpha(n, alphas, n_features)
        alphas.append(alpha)
        if verbosity >= 1:
            print(f"Start alpha trial {len(alphas)} ({datetime.now() - t0})")
        # sparse pca
        features, loadings = sparse_pca(adata.X, n_pcs, alpha, seed)
        n_f = np.sum(features)
        n_features.append(n_f)
        if verbosity == 1:
            print(f"\t alpha = {alpha}, n_features = {n_f}")

        # adjust conditions
        if n_features[-1] == n:
            n_features_equals_n = True
        if (n <= n_features[-1]) and (n_features[-1] <= (1 + tolerance) * n):
            n_features_in_tolerance = True
        if (type(n_alphas_min) == int) and (len(alphas) >= n_alphas_min):
            min_alphas = True
        if (type(n_alphas_max) == int) and (len(alphas) == n_alphas_max):
            max_alphas = True

        if verbosity >= 2:
            print(f"########## {len(alphas)} ##########")
            print("alphas     : ", alphas)
            print("n_features : ", n_features)
            print(f"equal_n   : {n_features_equals_n}")
            print(f"in tol    : {n_features_in_tolerance}")
            print(f"min_alphas: {min_alphas}")
            print(f"max_alphas: {max_alphas}")

    if verbosity >= 1:
        if n_features_equals_n:
            print("Found solution with sparse PC components based on exactly n features")
        elif n_features_in_tolerance:
            print(
                "Found solution with number of features in defined tolerance. Excluded features with lowest loadings sums"
            )
        elif max_alphas:
            print("Maximal number of trials (n_alphas_max) was reached without finding a solution")
    if (not n_features_equals_n) and (not n_features_in_tolerance) and max_alphas:
        return 0

    scores = np.abs(np.sum(loadings, axis=0))
    selection_idxs = (-scores).argsort()[:n]
    selection = np.zeros_like(features, dtype=bool)
    selection[selection_idxs] = True

    if inplace:
        adata.var["selection_score"] = scores
        adata.var["selection"] = selection
    else:
        df = pd.DataFrame(index=adata.var.index, columns=["selection", "selection_scores"])
        df["selection"] = selection
        df["selection_score"] = scores
        return df


##################################################################################
################################### selfE ########################################
##################################################################################
# This was copied from my bachelor student and does not really work well. Takes for
# ever to compute. Either we reimplement the method or drop it completly. The original
# implementation was written in R - it's quite slow in general, especially for high `n`s


def select_selfE_features(adata, n, inplace=True, verbosity=0):
    """
    this method selects a subset of genes which expresses all the other genes best
    (SOMP-Algorithm)
    ... inspired by Rai et al ...

    Arguments
    ---------------
    adata: AnnData
        adata.X needs to be sc.sparse.csr_matrix, if a np.array is provided it's converted to a sprase matrix.
    n: int
        number of features to be selected
    inplace: bool
        save results in adata, otherwise return a dataframe
    verbosity: int

    Returns
    ---------------
    if inplace: add
        - adata.var['selection'] - boolean pd.Series of n selected features
    else: return
        pd.Dataframe with column 'selection'
    """

    a = adata.copy()

    if not scipy.sparse.issparse(a.X):
        a.X = scipy.sparse.csr.csr_matrix(a.X)

    if verbosity > 0:
        print(f"select {n} selfE genes")

    feature_idxs = []
    scores = []
    Y = a.X.copy()
    Phi = a.X.copy()
    R = Y.copy()

    if verbosity > 0:
        t0 = datetime.now()
    for i in range(0, n):
        if verbosity > 1:
            print(f"Select feature {i + 1}/{n} ({datetime.now() - t0})")
        K = abs(np.dot(R.transpose(), R))  # matrix product
        c = []

        c = np.sqrt(K.multiply(K).sum(1))
        pos = np.array(c).flatten().argsort()[::-1]

        feature_idxs.append(pos[0])
        scores.append(c[pos[0]])
        PhiS = Phi[:, feature_idxs].copy()
        # multiply extracted feature with its psuedo inverse and original data to minimize error.
        Yiter = PhiS.dot(scipy.sparse.csr_matrix(np.linalg.pinv(PhiS.toarray())).dot(Y))

        R = Y - Yiter

    if verbosity > 0:
        print(f"Selected {n} features ({datetime.now() - t0})")
    genes = a.var.index[feature_idxs].values
    df = pd.DataFrame(index=adata.var.index, columns=["selection"])
    df["selection"] = False
    df.loc[genes, "selection"] = True
    if inplace:
        adata.var["selection"] = df["selection"].copy()
    else:
        return df[["selection"]].copy()
