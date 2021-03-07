import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs


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
