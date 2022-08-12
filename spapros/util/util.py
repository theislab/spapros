from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs


##############
# Data Utils #
##############


def get_processed_pbmc_data(n_hvg: int = 1000):
    """Get log normalised pbmc AnnData with selection and evaluation relevant quantities

    Args:
        n_hvg:
            Number of highly variable genes

    Returns:
        processed AnnData.

    """
    adata = sc.datasets.pbmc3k()
    adata_tmp = sc.datasets.pbmc3k_processed()

    # Get infos from the processed dataset
    adata = adata[adata_tmp.obs_names, adata_tmp.var_names].copy()
    adata.obs["celltype"] = adata_tmp.obs["louvain"]
    adata.obsm["X_umap"] = adata_tmp.obsm["X_umap"]  # TODO: umap True/False
    del adata_tmp

    # Preprocess counts and get highly variable genes
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=1000)

    # TODO: with "quantiles" or "with_expr_penalty"  bool / Add note that these expression constraints might not fit
    #      real experiments

    return adata


def clean_adata(
    adata: sc.AnnData,
    obs_keys: Optional[List[str]] = None,
    var_keys: Optional[List[str]] = None,
    uns_keys: Optional[List[str]] = None,
    obsm_keys: Optional[List[str]] = None,
    varm_keys: Optional[List[str]] = None,
    obsp_keys: Optional[List[str]] = None,
    inplace: bool = True,
) -> Optional[sc.AnnData]:
    """Removes unwanted attributes of an adata object.

    Args:
        adata:
            Anndata object.
        obs_keys:
            Columns in `adata.obs` to keep.
        var_keys:
            Columns in `adata.var` to keep.
        uns_keys:
            Keys of `adata.uns` to keep.
        obsm_keys:
            Columns of `adata.obsm` to keep.
        varm_keys:
            Columns of `adata.varm` to keep.
        obsp_keys:
            Keys of `adata.obsp` to keep.
        inplace:

    Returns:
        If not inpace, the cleaned Anndata without any annotation.
    """
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
    else:
        return None


def preprocess_adata(
    adata: sc.AnnData,
    options: List[Literal["norm", "log1p", "scale"]] = ["norm", "log1p", "scale"],  # noqa: B006
    size_factor_key: str = "size_factors",
    inplace: bool = True,
) -> Optional[sc.AnnData]:
    """Apply standard preprocessings to `adata.X`.

    Args:
        adata:
            adata.X should be in the correct form for the provided options.
            If `'norm' in options` size factors are expected in `adata.obs[size_factor_key]`.
        options:
            Preprocessing options:

                - 'norm': normalise adata.X according `adata.obs[size_factor_key]`
                - 'log1p': log(adata.X + 1)
                - 'scale': scale genes of adata.X to zero mean and std 1 (adata.X no longer sparse)

        size_factor_key:
            Key for normalisation size factors in adata.obs.
        inplace:
            Process adata.X or return a copy of adata.

    Returns:
        sc.Anndata (if not inplace):
            AnnData with preprocessed AnnData.X.
    """
    a = adata if inplace else adata.copy()

    all_options = ["norm", "log1p", "scale"]

    for o in options:
        if o not in all_options:
            print(f"Preprocessing option {o} is not supported")
            return None

    if "norm" in options:
        if issparse(a.X):
            sparsefuncs.inplace_row_scale(a.X, 1 / a.obs[size_factor_key].values)
        else:
            a.X /= a.obs[size_factor_key].values[:, None]  # np.divide(X, counts[:, None], out=X)
    if "log1p" in options:
        sc.pp.log1p(a)
    if "scale" in options:
        sc.pp.scale(a)

    if not inplace:
        return a
    else:
        return None


def get_expression_quantile(
    adata: sc.AnnData, q: float = 0.9, normalise: bool = False, log1p: bool = False, zeros_to_nan: bool = False
) -> None:
    """Compute each genes q'th quantile on normalised (and log1p) data.

    Args:
        adata:
            AnnData object. If ``normalise is True`` we expect raw counts in ``adata.X`` and size factors in
            ``adata.obs['size_factors']``.
        q:
            Value between 0 = :attr:`q` = 1, the quantile to compute.
        normalise:
            Normalise data with a.obs['size_factors'].
        log1p:
            log1p the data to get quantile values of log data. Not necessary if log1p was already applied on
            ``adata.X``.
        zeros_to_nan:
            Don't include zeros into quantile calculation.

    Returns.
        Adds column ``adata.var[f'quantile_{q}']`` or ``adata.var[f'quantile_{} expr > 0']``:
    """
    # TODO: Add celltype weighting. (sc data does not represent correct celltype proportions)
    #       We should add
    #       - `group_key` = 'Celltypes'
    #       - `group_proportions` = {'AT1': 0.2, 'AT2': 0.4, 'Myeloid': 0.3, ....} (numbers proportional to counts)
    a = adata.copy()
    if normalise:
        a.X /= a.obs["size_factors"].values[:, None]
    if log1p:
        sc.pp.log1p(a)
    if issparse(a.X):
        a.X = a.X.toarray()
    df = pd.DataFrame(a.X, index=a.obs.index, columns=a.var.index)
    new_key = f"quantile_{q}"
    if zeros_to_nan:
        df[df == 0] = np.nan
        new_key = f"quantile_{q} expr > 0"
    adata.var[new_key] = df.quantile(q)


def gene_means(
    adata: sc.AnnData, genes: Union[Literal["all"], List[str]] = "all", key: str = "mean", inplace: bool = False
) -> Optional[pd.DataFrame]:
    """Compute each gene's mean expression.

    Args:
        adata:
            Anndata object with data in `adata.X`.
        genes:
            Genes for which mean is computed or "all".
        key:
            Column name in which means are saved.
        inplace:
            Wether to save results in adata.var or return a dataframe.

    Returns:
        Dataframe (if not inplace) with :attr:`adata.var_names` as index and a column :attr:`key` containing the
        expression means.
    """
    a = adata if (genes == "all") else adata[:, genes]
    means = a.X.mean(axis=0) if issparse(adata.X) else np.mean(a.X, axis=0)
    if inplace:
        adata.var[key] = np.nan
        adata.var.loc[a.var_names, key] = means
        return None
    else:
        df = pd.DataFrame(index=adata.var_names, data={key: np.nan}, dtype="float64")
        df.loc[a.var_names, key] = means
        return df


def gene_stds(
    adata: sc.AnnData, genes: Union[Literal["all"], List[str]] = "all", key: str = "std", inplace: bool = False
) -> Optional[pd.DataFrame]:
    """Compute each gene's expression standard deviation.

    Args:
        adata:
            Anndata object with data in `adata.X`.
        genes:
            Genes for which std is computed.
        key:
            Column name in which stds are saved.
        inplace:
            Wether to save results in adata.var or return a dataframe.

    Returns:
        Dataframe (if not inplace) with :attr:`adata.var_names` as index and a column :attr:`key` containing the
        expression standart deviation.
    """
    a = adata if (genes == "all") else adata[:, genes]
    stds = a.X.std(axis=0) if issparse(adata.X) else np.std(a.X, axis=0)
    if inplace:
        adata.var[key] = np.nan
        adata.var.loc[a.var_names, key] = stds
        return None
    else:
        df = pd.DataFrame(index=adata.var_names, data={key: np.nan}, dtype="float64")
        df.loc[a.var_names, key] = stds
        return df


##############
# Plot Utils #
##############


def cluster_corr(corr_array: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """Rearranges the correlation matrix, corr_array, so that groups of highly correlated variables are next to each
    other.

    Args:
        corr_array:
            A NxN correlation matrix.
    inplace :
        Whether to rearrange in place or not.

    Returns:
        pandas.DataFrame or numpy.ndarray:
            A NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
    idx = np.argsort(idx_to_cluster_array)

    corr_array = corr_array.copy()
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


def coexpression_plot(
    adata: sc.AnnData,
    figsize: Tuple[float, float] = (5, 5),
    colorbar: Optional[matplotlib.pyplot.colorbar] = None,
    return_mean_abs: bool = False,
) -> Optional[np.ndarray]:
    """Creates a coexpression plot.

    Args:
        adata:
            Anndata object with data in `adata.X`.
        figsize:
            Size of the created plot.
        colorbar:
            Colorbar to use for the plot.
        return_mean_abs:
            Whether to return mean absolute values.

    Returns:
        Mean absolute values if :attr:`return_mean_abs`.
    """
    if scipy.sparse.issparse(adata.X):
        cor_mat = np.corrcoef(adata.X.toarray(), rowvar=False)
    else:
        cor_mat = np.corrcoef(adata.X, rowvar=False)
    ordered_cor_mat = cluster_corr(cor_mat)
    plt.figure(figsize=figsize)
    plt.imshow(ordered_cor_mat, cmap="seismic", vmin=-1, vmax=1)
    if colorbar:
        plt.colorbar()
    plt.show()
    if return_mean_abs:
        return np.mean(np.abs(cor_mat))
    else:
        return None


####################
# Constraint utils #
####################


def transfered_expression_thresholds(
    adata: sc.AnnData,
    lower: float = 2,
    upper: float = 6,
    tolerance: float = 0.05,
    target_sum: float = 10000,
    output_path: str = "./results/",
    plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transfer expression thresholds between different normalisations.

    Note:
        If expression thresholds are known for normalisation with a given `target_sum` these limits
        are transfered to the normalisation given in adata.obs['size_factors'].

    Args:
        adata: AnnData
            Anndata with raw counts in :attr:`adata.X` and size factors for the normalisation in
            :attr:`adata.obs['size_factors']`.
        lower:
            Lower expression threshold for target_sum normalisation.
        upper:
            Upper expression threshold for target_sum normalisation.
        tolerance:
            To estimate the thresholds in the target normalisation we sample expression values around the thresholds.
            Eventually increase this parameter for small datasets.
        target_sum: float
            `target_sum` parameter of the reference normalisation (`sc.pp.normalize_total()`).
        output_path:
            Path where to save the figure.
        plot: bool
            Plot histograms of mapped expressions around reference and target thresholds.

    Returns:
        Lower and upper expression thresholds in the target normalisation.

    """
    # TODO: tolerance: could be better to define a sample size instead of a tolerance
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=target_sum)
    sc.pp.log1p(a)
    if scipy.sparse.issparse(a.X):
        a.X = a.X.toarray()
    mask_lo = (a.X > (lower - tolerance)) & (a.X < (lower + tolerance))
    mask_hi = (a.X > (upper - tolerance)) & (a.X < (upper + tolerance))
    if plot:
        df = pd.concat(
            [
                pd.DataFrame(data={"expression": a.X[mask_lo], "limit": "low"}),
                pd.DataFrame(data={"expression": a.X[mask_hi], "limit": "high"}),
            ]
        )
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 2, 1)
        sns.kdeplot(data=df, x="expression", hue="limit", fill=True, common_norm=False, legend=False)
        plt.axvline(x=np.mean(a.X[mask_lo]), lw=0.5, ls="--", color="black")
        plt.axvline(x=np.mean(a.X[mask_hi]), lw=0.5, ls="--", color="black")
        plt.ylabel("")
        plt.yticks([])
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
        df = pd.concat(
            [
                pd.DataFrame(data={"expression": lo_expr, "limit": "low"}),
                pd.DataFrame(data={"expression": hi_expr, "limit": "high"}),
            ]
        )
        ax = plt.subplot(1, 2, 2)
        sns.kdeplot(data=df, x="expression", hue="limit", fill=True, common_norm=False, legend=False)
        plt.axvline(x=lo_mean, lw=0.5, ls="--", color="black")
        plt.axvline(x=hi_mean, lw=0.5, ls="--", color="black")
        plt.ylabel("")
        plt.yticks([])
        left, right = plt.xlim()
        ax_len = right - left
        plt.text(
            ((lo_mean - left) / ax_len) * 1.02,
            0.9,
            f"{lo_mean:.3f}",
            horizontalalignment="left",
            transform=ax.transAxes,
        )
        plt.text(
            ((hi_mean - left) / ax_len) * 0.98,
            0.9,
            f"{hi_mean:.3f}",
            horizontalalignment="right",
            transform=ax.transAxes,
        )
        plt.title("Expressions around limits (target normalisation)")
        if output_path is not None:
            plt.savefig(f"{output_path}expression_around_limits.png")
    return lo_mean, hi_mean


def plateau_penalty_kernel(
    var: Union[float, List[float]], x_min: np.ndarray = None, x_max: np.ndarray = None
) -> Callable:
    """Return penalty function.

    The kernel can be one or two sided (one-sided if either :attr:`x_min` or :attr:`x_max` is `None`).
    The kernel is 1 between :attr:`x_min` and :attr:`x_max`. If one-sided it's 1 from :attr:`x_min` or till
    :attr:`x_max`. Outside the defined range the kernal decays with a gaussian function with variance = ``var``.

    Args:
        var:
            Outside the defined range, the kernel decays with a gaussian function with variance = ``var``.
        x_min:
            Lower border above which the kernel is 1.
        x_max:
            Upper border below which the kernel is 1.

    Returns:
        Penalty function.


    Example:

        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import spapros as sp

            penalty_fcts = {
                "left"  : sp.ut.plateau_penalty_kernel(var=0.5, x_min=2, x_max=None),
                "right" : sp.ut.plateau_penalty_kernel(var=2, x_min=None, x_max=5),
                "dual"  : sp.ut.plateau_penalty_kernel(var=[0.5,2], x_min=2, x_max=5),
            }

            x = np.linspace(0,10,100)
            _, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,2))
            for i, (title, penalty_fct) in enumerate(penalty_fcts.items()):
                axs[i].plot(x,penalty_fct(x))
                axs[i].set_title(title)
            plt.show()

        .. image:: ../../docs/plot_examples/Utils_plateau_penalty_kernel.png



    """

    if type(var) == list:
        var_l = var[0]
        var_r = var[1]
    else:
        assert isinstance(var, float) or isinstance(var, int)
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


def dict_to_table(
    marker_dict: Union[dict, pd.DataFrame], genes_as_index: bool = False, reverse: bool = False
) -> Union[pd.DataFrame, dict]:
    """Convert marker dictonary to pandas dataframe or reverse.

    Notes:
        Two possible outputs:
        - each celltype's marker in a column (genes_as_index=False)
        - index are markers, one column with "celltype" annotation (genes_as_index=True)

    Args:
        marker_dict:
            Dictionary of the form `{'celltype':list of markers of celltype}`. A DataFrame can be provided to reverse the
            transformation (:attr:`reverse`=True).
        genes_as_index:
            Wether to have genes in the dataframe index and one column for celltype annotations or genes listed in
            each celltype's column.
        reverse:
            Wether to transform a dataframe to a dict or a dict to a dataframe.

    Returns:
        Dataframe or dict (if reverse).
    """
    # TODO: Preference? Split this in two functions instead of the `reverse` argument?
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
    return {}


def filter_marker_dict_by_penalty(
    marker_dict: Dict[str, List[str]],
    adata: sc.AnnData,
    penalty_keys: Union[str, List[str]],
    threshold: float = 1,
    verbose: bool = True,
    return_filtered: bool = False,
) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], Dict[str, List[str]]]]:
    """Filter out genes in marker_dict if a gene's penalty < :attr:`threshold`.

    Args:
        marker_dict:
            Dictionary of the form `{'celltype':list of markers of celltype}`.
        adata:
            Anndata with data in :attr:`adata.X` and penalty values in :attr:`adata.var`.
        penalty_keys:
            Keys of :attr:`adata.var` with penalty values.
        threshold:
            Min value of penalty to keep gene.
        return_filtered:
            Return also the genes that were filtered out.
        verbose:
            Optionally print infos.

    Returns:
        Filtered marker_dict and if return_filtered a dict of the filtered genes.

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


def filter_marker_dict_by_shared_genes(marker_dict: Dict[str, List[str]], verbose: bool = True) -> Dict[str, List[str]]:
    """Filter out genes in marker_dict that occur multiple times.

    Args:
        marker_dict:
            Dictionary of the form `{'celltype':list of markers of celltype}`.
        verbose:
            Optionally print infos.

    Returns:
        Filtered :attr:`marker_dict`.
    """
    genes = np.unique([g for _, gs in marker_dict.items() for g in gs])
    gene_cts_dict: Dict[str, list] = {g: [] for g in genes}
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


# This is at the moment still needed for the ProbesetSelector. We also have such function in metrics. Should be combined
# at the end.


def correlation_matrix(
    adata: sc.AnnData,
    genes: Union[Literal["all"], List[str]] = "all",
    absolute: bool = True,
    diag_zero: bool = True,
    unknown_genes_to_zero: bool = False,
) -> pd.DataFrame:
    """Calculate gene correlation matrix.

    Args:
        adata:
            Anndata with data in :attr:`adata.X`.
        genes:
            Gene subset for correlation calculation.
        absolute:
            Wether to take the absolute values of correlations.
        diag_zero:
            Wether to set diagonal elements to zero.
        unknown_genes_to_zero:
            Wether to add genes that aren't in adata.var.index with zeros. (Otherwise an error is raised)

    Returns:
        Correlation matrix.
    """
    # TODO:
    #   - genes: add options for an unsymmetric cor_matrix
    #   - diag_zero: Add option to set a triangle to zero
    if genes == "all":
        genes = adata.var_names
    elif unknown_genes_to_zero:
        unknown = [g for g in genes if not (g in adata.var_names)]
        genes = [g for g in genes if not (g in unknown)]

    if issparse(adata.X):
        cor_mat = pd.DataFrame(index=genes, columns=genes, data=np.corrcoef(adata[:, genes].X.toarray(), rowvar=False))
    else:
        cor_mat = pd.DataFrame(index=genes, columns=genes, data=np.corrcoef(adata[:, genes].X, rowvar=False))

    if absolute:
        cor_mat = np.abs(cor_mat)
    if diag_zero:
        np.fill_diagonal(cor_mat.values, 0)
    if unknown_genes_to_zero and unknown:
        for g in unknown:
            cor_mat.loc[g] = 0.0
            cor_mat[g] = 0.0

    return cor_mat


def marker_mean_difference(
    adata: sc.AnnData, celltype: str, ct_key: str, genes: Union[Literal["all"], List[str]] = "all"
) -> np.ndarray:
    """Calculate the difference of the mean expression between the genes of one celltype.

    Args:
        adata:
            Anndata object with data in :attr:`andata.X`.
        celltype:
            Celltype with with the genes to calculate the mean are annotated in :attr:`adata.obs[celltype_key]`.
        ct_key:
            Column in :attr:`adata.obs` where celltype annotations are stored.
        genes:
            List of subset of the genes or "all".

    Returns:
        Difference of the mean expression of the genes of the celltype :attr:`celltype` and the mean expression of the
        remaining genes.
    """
    if genes == "all":
        genes = list(adata.var.index)
    mean_ct = adata[adata.obs[ct_key] == celltype, genes].X.mean(axis=0)
    mean_other = adata[~(adata.obs[ct_key] == celltype), genes].X.mean(axis=0)
    return mean_ct - mean_other


class NestedProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:

            # extract those self devined fields
            level = task.fields.get("level") if task.fields.get("level") else 1
            only_text = task.fields.get("only_text") if task.fields.get("only_text") else False
            header = task.fields.get("header") if task.fields.get("header") else False
            footer = task.fields.get("footer") if task.fields.get("footer") else False

            # layout
            indentation = (level - 1) * 2 * " "
            font_styles = {1: "bold blue", 2: "bold dim cyan", 3: "bold green", 4: "green"}

            # define columns for percentag and step progress
            steps_column = TextColumn("[progress.percentage]{task.completed: >2}/{task.total: <2}", justify="right")
            percentage_column = TextColumn("[progress.percentage]{task.percentage:>3.0f}% ", justify="right")
            fill = 92 if only_text else 58
            fill = fill - len(indentation)
            text_column = f"{indentation}[{font_styles[level]}][progress.description]{task.description:.<{fill}}"
            header_column = f"[bold black][progress.description]{task.description: <96}"
            footer_column = f"[bold black][progress.description]{task.description}"

            if not only_text:
                self.columns = (
                    text_column,
                    BarColumn(),
                    steps_column if task.total != 1 else percentage_column,
                    TimeElapsedColumn(),
                )
            else:
                self.columns = (text_column, "")
            if header:
                self.columns = (header_column, TimeElapsedColumn())
            if footer:
                self.columns = (footer_column, "")
            yield self.make_tasks_table([task])


def init_progress(progress, verbosity, level):
    started = False
    if verbosity < level:
        return None, started
    if not progress:
        progress = NestedProgress()
        progress.start()
        started = True
    return progress, started
