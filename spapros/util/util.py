import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs

##############
# Data Utils #
##############


def clean_adata(
    adata,
    obs_keys=None,
    var_keys=None,
    uns_keys=None,
    obsm_keys=None,
    varm_keys=None,
    obsp_keys=None,
    inplace=True,
):
    """Removes unwanted attributes of an adata object."""
    if not obs_keys:
        obs_keys = []
    if not var_keys:
        var_keys = []
    if not uns_keys:
        uns_keys = []
    if not obsm_keys:
        obsm_keys = []
    if not varm_keys:
        varm_keys = []

    if inplace:
        a = adata
    else:
        a = adata.copy()
    for obs in [o for o in a.obs_keys() if o not in obs_keys]:
        del a.obs[obs]
    for var in [v for v in a.var_keys() if v not in var_keys]:
        del a.var[var]
    for uns in [u for u in a.uns_keys() if u not in uns_keys]:
        del a.uns[uns]
    for obsm in [om for om in a.obsm_keys() if om not in obsm_keys]:
        del a.obsm[obsm]
    for varm in [vm for vm in a.varm_keys() if vm not in varm_keys]:
        del a.varm[varm]
    # for obsp in [op for op in a.obsp_keys() if not op in obsp_keys]:
    # 	del a.obsp[obsp]
    if not inplace:
        return a


def preprocess_adata(
    adata,
    options=["norm", "log1p", "scale"],  # noqa: B006
    size_factor_key="size_factors",
    inplace=True,
):
    """Apply standard preprocessings to adata.X .

    Arguments
    ---------
    adata: AnnData
        adata.X should be in the correct form for the provided options.
        If `'norm' in options` size factors are expected in `adata.obs[size_factor_key]`
    options: list of strs
        Preprocessing options
        - 'norm': normalise adata.X according `adata.obs[size_factor_key]`
        - 'log1p': log(adata.X + 1)
        - 'scale': scale genes of adata.X to zero mean and std 1 (adata.X no longer sparse)
    size_factor_key: str
        Key for normalisation size factors in adata.obs
    inplace: bool
        Process adata.X or return a copy of adata.

    Returns
    -------
    (if not inplace) AnnData with preprocessed AnnData.X.
    """
    a = adata if inplace else adata.copy()

    all_options = ["norm", "log1p", "scale"]

    for o in options:
        if o not in all_options:
            print(f"Preprocessing option {o} is not supported")
            return None

    if "norm" in options:
        if issparse(a.X):
            sparsefuncs.inplace_row_scale(a.X, 1 / a.obs["size_factors"].values)
        else:
            a.X /= a.obs["size_factors"].values[:, None]  # np.divide(X, counts[:, None], out=X)
    if "log1p" in options:
        sc.pp.log1p(a)
    if "scale" in options:
        sc.pp.scale(a)

    if not inplace:
        return a


def get_expression_quantile(adata, q=0.9, normalise=True, log1p=True, zeros_to_nan=False):
    """Compute each genes q'th quantile on normalised (and log1p) data.

    TODO: Add celltype weighting. (sc data does not represent correct celltype proportions)
          We should add
          - `group_key` = 'Celltypes'
          - `group_proportions` = {'AT1': 0.2, 'AT2': 0.4, 'Myeloid': 0.3, ....} (numbers proportional to counts)

    dataset: str or AnnData
        AnnData needs to contain raw counts (adata.X) and some size factor (adata.obs['size_factors'])
    normalise: bool
        Normalise data with a.obs['size_factors']
    log1p: bool
        log1p the data to get quantile values of log data.
    zeros_to_nan: bool
        Don't include zeros into quantile calculation (we might drop this option, it doesn't make sense)

    Returns
    -------
    Adds column adata.var[f'quantile_{q}']
    """
    a = adata.copy()
    if normalise:
        a.X /= a.obs["size_factors"].values[:, None]
    if log1p:
        sc.pp.log1p(a)
    df = pd.DataFrame(a.X, index=a.obs.index, columns=a.var.index)
    if zeros_to_nan:
        df[df == 0] = np.nan
    adata.var[f"quantile_{q}"] = df.quantile(q)


def gene_means(adata, genes="all", key="mean", inplace=False):
    """Compute each gene's mean expression

    Arguments
    ---------
    adata: AnnData
    genes: str or list of strs
        Genes for which mean is computed
    key: str
        Column name in which means are saved
    inplace: bool
        Wether to save results in adata.var or return a dataframe

    Returns
    -------
    pd.DataFrame (if not inplace)

    """
    a = adata if (genes == "all") else adata[:, genes]
    means = a.X.mean(axis=0) if issparse(adata.X) else np.mean(a.X, axis=0)
    if inplace:
        adata.var[key] = np.nan
        adata.var.loc[a.var_names, key] = means
    else:
        df = pd.DataFrame(index=adata.var_names, data={key: np.nan}, dtype="float64")
        df.loc[a.var_names, key] = means
        return df


def gene_stds(adata, genes="all", key="std", inplace=False):
    """Compute each gene's expression standard deviation

    Arguments
    ---------
    adata: AnnData
    genes: str or list of strs
        Genes for which std is computed
    key: str
        Column name in which stds are saved
    inplace: bool
        Wether to save results in adata.var or return a dataframe

    Returns
    -------
    pd.DataFrame (if not inplace)

    """
    a = adata if (genes == "all") else adata[:, genes]
    stds = a.X.std(axis=0) if issparse(adata.X) else np.std(a.X, axis=0)
    if inplace:
        adata.var[key] = np.nan
        adata.var.loc[a.var_names, key] = stds
    else:
        df = pd.DataFrame(index=adata.var_names, data={key: np.nan}, dtype="float64")
        df.loc[a.var_names, key] = stds
        return df


##############
# Plot Utils #
##############


def cluster_corr(corr_array, inplace=False):
    """Rearranges the correlation matrix, corr_array, so that groups of highly correlated variables are next to each other.

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix
    inplace : bool
        whether to rearrange in place or not

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def coexpression_plot(adata, figsize=(5, 5), colorbar=False, return_mean_abs=False):
    """Creates a coexpression plot.

    Args:
        adata: annData
        figsize: size of the created plot
        colorbar: colorbar to use for the plot
        return_mean_abs: whether to return mean absolute values
    """
    if scipy.sparse.issparse(adata.X):
        cor_mat = np.corrcoef(adata.X.toarray(), rowvar=False)
    else:
        cor_mat = np.corrcoef(adata.X, rowvar=False)
    ordered_cor_mat = cluster_corr(cor_mat, inplace=False)
    plt.figure(figsize=figsize)
    plt.imshow(ordered_cor_mat, cmap="seismic", vmin=-1, vmax=1)
    if colorbar:
        plt.colorbar()
    plt.show()
    if return_mean_abs:
        return np.mean(np.abs(cor_mat))


####################
# Constraint utils #
####################


def transfered_expression_thresholds(
    adata, lower=2, upper=6, tolerance=0.05, target_sum=10000, output_path: str = "./results/", plot=True
):
    """Transfer expression thresholds between different normalisations.

    If expression thresholds are known for normalisation with a given `target_sum` these limits
    are transfered to the normalisation given in adata.obs['size_factors'].

    Arguments
    ---------
    adata: AnnData
        adata.X must include raw counts. adata.obs['size_factors'] include the size factors for the normalisation
        of interest
    lower: float
        Lower expression threshold for target_sum normalisation
    upper: float
        Upper expression threshold for target_sum normalisation
    tolerance: float
        To estimate the thresholds in the target normalisation we sample expression values around the thresholds.
        Eventually increase this parameter for small datasets.
        (TODO: could be better to define a sample size instead of a tolerance)
    target_sum: float
        `target_sum` parameter of the reference normalisation (`sc.pp.normalize_total()`)
    plot: bool
        Plot histograms of mapped expressions around reference and target thresholds

    Returns
    -------
    2 floats:
        Lower and upper expression thresholds in the target normalisation

    """
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    if scipy.sparse.issparse(a.X):
        a.X = a.X.toarray()
    mask_lo = (a.X > (lower - tolerance)) & (a.X < (lower + tolerance))
    mask_hi = (a.X > (upper - tolerance)) & (a.X < (upper + tolerance))
    if plot:
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 2, 1)
        sns.distplot(a.X[mask_lo])
        plt.axvline(x=np.mean(a.X[mask_lo]), lw=0.5, ls="--", color="black")
        sns.distplot(a.X[mask_hi])
        plt.axvline(x=np.mean(a.X[mask_hi]), lw=0.5, ls="--", color="black")
        plt.title(f"Expressions around limits (target_sum = {target_sum})")

    a = adata.copy()
    if scipy.sparse.issparse(a.X):
        a.X = a.X.toarray()
    a.X /= a.obs["size_factors"].values[:, None]
    sc.pp.log1p(a)
    lo_expr = a.X[mask_lo]
    lo_mean = np.mean(lo_expr)
    hi_expr = a.X[mask_hi]
    hi_mean = np.mean(hi_expr)
    if plot:
        ax = plt.subplot(1, 2, 2)
        sns.distplot(lo_expr)
        plt.axvline(x=lo_mean, lw=0.5, ls="--", color="black")
        sns.distplot(hi_expr)
        plt.axvline(x=hi_mean, lw=0.5, ls="--", color="black")
        _, right = plt.xlim()
        plt.text(
            (lo_mean / right) * 1.02,
            0.9,
            f"{lo_mean:.3f}",
            horizontalalignment="left",
            transform=ax.transAxes,
        )
        plt.text(
            (hi_mean / right) * 0.98,
            0.9,
            f"{hi_mean:.3f}",
            horizontalalignment="right",
            transform=ax.transAxes,
        )
        plt.title("Expressions around limits (target normalisation)")
        plt.savefig(f"{output_path}expression_around_limits.png")
    return lo_mean, hi_mean


def plateau_penalty_kernel(var, x_min=None, x_max=None):
    """Return penalty function.

    The kernel can be one or two sided (one-sided if either x_min or x_max is None).
    The kernel is 1 between x_min and x_max. If one-sided it's 1 from x_min or till x_max.
    Outisde the defined range the kernel decays with a gaussian kernel with variance=var.
    """

    if type(var) == list:
        var_l = var[0]
        var_r = var[1]
    else:
        var_l = var
        var_r = var

    if not (x_min is None):

        def left_kernel(x):
            return np.exp(-np.power(x - x_min, 2.0) / (2 * var_l))

    if not (x_max is None):

        def right_kernel(x):
            return np.exp(-np.power(x - x_max, 2.0) / (2 * var_r))

    if not (x_min is None) and not (x_max is None):

        def function(x):
            return np.where(
                (x > x_min) & (x < x_max),
                1,
                np.where(x < x_min, left_kernel(x), right_kernel(x)),
            )

    elif not (x_min is None):

        def function(x):
            return np.where(x > x_min, 1, left_kernel(x))

    elif not (x_max is None):

        def function(x):
            return np.where(x < x_max, 1, right_kernel(x))

    else:

        def function(x):
            return np.ones_like(x)

    return function


#####################
# Marker List Utils #
#####################


def dict_to_table(marker_dict, genes_as_index=False, reverse=False):
    """Convert marker dictonary to pandas dataframe

    # TODO: Preference? Split this in two functions instead of the `reverse` argument?

    Two possible outputs:
    - each celltype's marker in a column (genes_as_index=False)
    - index are markers, one column with "celltype" annotation (genes_as_index=True)

    Arguments
    ---------
    marker_dict: dict or pd.DataFrame
        dict of form {'celltype':list of markers of celltype}. A DataFrame can be provided to reverse the
        transformation (reverse=True)
    genes_as_index: bool
        Wether to have genes in the dataframe index and one column for celltype annotations or genes listed in
        each celltype's column.
    reverse: bool
        Wether to transform a dataframe to a dict or a dict to a dataframe.

    Returns
    -------
    pd.DataFrame or dict (if reverse)

    """
    if isinstance(marker_dict, dict) and (not reverse):
        if genes_as_index:
            swap_dict = {g: ct for ct, genes in marker_dict.items() for g in genes}
            return pd.DataFrame.from_dict(swap_dict, orient="index", columns=["celltype"])
        else:
            max_length = max([len(genes) for _, genes in marker_dict.items()])
            marker_dict_padded = {ct: genes + [np.nan] * (max_length - len(genes)) for ct, genes in marker_dict.items()}
            return pd.DataFrame(data=marker_dict_padded)
    elif isinstance(marker_dict, pd.DataFrame) and reverse:
        output = {}
        if genes_as_index:
            for ct in marker_dict.celltype.unique():
                output[ct] = marker_dict.loc[marker_dict["celltype"] == ct].index.tolist()
        else:
            for ct in marker_dict.columns:
                output[ct] = marker_dict.loc[~marker_dict[ct].isnull(), ct].tolist()
        return output


def filter_marker_dict_by_penalty(marker_dict, adata, penalty_keys, threshold=1, verbose=True, return_filtered=False):
    """Filter out genes in marker_dict if a gene's penalty < threshold

    Parameters
    ----------
    marker_dict: dict
    adata: AnnData
    penalty_keys: str or list of strs
        keys of adata.var with penalty values
    threshold: float
        min value of penalty to keep gene

    Returns
    -------
    - filtered marker_dict
    - and if return_filtered a dict of the filtered genes
    """
    if isinstance(penalty_keys, str):
        penalty_keys = [penalty_keys]
    genes_in_adata = [g for ct, gs in marker_dict.items() for g in gs if g in adata.var.index]
    genes_not_in_adata = [g for ct, gs in marker_dict.items() for g in gs if not (g in adata.var.index)]
    df = adata.var.loc[genes_in_adata, penalty_keys].copy()
    penalized = (df < threshold).any(axis=1)
    genes = df.loc[~penalized].index.tolist() + genes_not_in_adata

    filtered_marker_dict = {}
    if verbose:
        filtered_out = {}
    for ct, gs in marker_dict.items():
        filtered_marker_dict[ct] = [g for g in gs if g in genes]
        filtered_genes_of_ct = [g for g in gs if g not in genes]
        if filtered_genes_of_ct and verbose:
            filtered_out[ct] = filtered_genes_of_ct
    if filtered_out and verbose:
        print(f"The following genes are filtered out due to the given penalty_keys (threshold: >= {threshold}):")
        max_str_length = max([len(ct) for ct in filtered_out])
        for ct, genes in filtered_out.items():
            print(f"\t {ct:<{max_str_length}} : {genes}")
    if genes_not_in_adata and verbose:
        print("The following genes couldn't be tested on penalties since they don't occur in adata.var.index:")
        print(f"\t {genes_not_in_adata}")

    if return_filtered:
        return filtered_marker_dict, filtered_out
    else:
        return filtered_marker_dict


def filter_marker_dict_by_shared_genes(marker_dict, verbose=True):
    """Filter out genes in marker_dict that occur multiple times"""
    genes = np.unique([g for _, gs in marker_dict.items() for g in gs])
    gene_cts_dict = {g: [] for g in genes}
    for ct, gs in marker_dict.items():
        for g in gs:
            gene_cts_dict[g].append(ct)
    multi_occurence_genes = [g for g, cts in gene_cts_dict.items() if (len(cts) > 1)]
    if multi_occurence_genes and verbose:
        max_str_length = max([len(g) for g in multi_occurence_genes])
        print("The following genes are filtered out since they occur multiple times in marker_dict:")
        for g in multi_occurence_genes:
            print(f"\t {g:<{max_str_length}} : {gene_cts_dict[g]}")
        print("\t If you want to include shared markers e.g. add a shared celltype key to the marker_dict")
    return {ct: [g for g in gs if g not in multi_occurence_genes] for ct, gs in marker_dict.items()}
