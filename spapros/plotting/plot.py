"""Plotting Module."""
import itertools
from typing import Dict
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.colors
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.cluster.hierarchy as sch
import seaborn as sns
from upsetplot import from_indicators
from upsetplot import UpSet
from venndata import venn
from spapros.plotting._masked_dotplot import MaskedDotPlot


#############################
## evaluation related plots ##
#############################


def ordered_confusion_matrices(conf_mats: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Rearranges confusion matrices by a linkage clustering.

    Notes:
        The matrices in conf_mats must have the same indices (and columns). We calculate the clustering on
        a concatenated list of confusion matrices and then reorder each matrix by the resulting order.

    Args:
        conf_mats:
            The confusion matrices to be reordered.

    Returns:
        list of pd.DataFrame:
            Reordered confusion matrices.
    """

    pooled = pd.concat(conf_mats, axis=1)

    pairwise_distances = sch.distance.pdist(pooled)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
    idx = np.argsort(idx_to_cluster_array)

    ordered_mats = [conf_mat.copy().iloc[idx, :].T.iloc[idx, :].T for conf_mat in conf_mats]
    return ordered_mats


def confusion_matrix(
    set_ids: List[str],
    conf_matrices: Dict[str, pd.DataFrame],
    ordered: Union[bool, list] = True,
    show: bool = True,
    save: Union[bool, str] = False,
    size_factor: float = 6,
    fontsize: int = 18,
    n_cols: int = 3,
    rotate_x_labels: bool = True,
) -> None:
    """Plot heatmap of cell type classification confusion matrices.

    Args:
        set_ids:
            List of probe set ids.
        conf_matrices:
            Confusion matrix of each probe set given in `set_ids`.
        ordered: bool or list
            If set to True a linkage clustering is computed to order cell types together that are hard to distinguish.
            If multiple set_ids are provided the same order is applied to all of them.
            Alternatively provide a list with a custom order.
        show:
            Show the figure.
        save:
            If `True` or a `str`, save the figure.
        size_factor:
             Factor for scaling the figure size.
        fontsize:
            Matplotlib fontsize.
        n_cols:
            Number of subplot columns.
        rotate_x_labels:
            Rotate the xticklabels by 45 degrees.
    """

    n_plots = len(set_ids)
    n_rows = (n_plots // n_cols) + int((n_plots % n_cols) > 0)

    if ordered:
        if isinstance(ordered, bool):
            tmp_matrices = [conf_matrices[set_id].copy() for set_id in set_ids]
            cms = ordered_confusion_matrices(tmp_matrices)
        elif isinstance(ordered, list):
            cms = [conf_matrices[set_id].copy().loc[ordered, ordered] for set_id in set_ids]
    else:
        cms = [conf_matrices[set_id].copy() for set_id in set_ids]

    fig = plt.figure(figsize=(size_factor * n_cols, 0.75 * size_factor * n_rows))
    for i, set_id in enumerate(set_ids):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        sns.heatmap(
            cms[i], cmap="OrRd", cbar=False, ax=ax, vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": fontsize}
        )
        # sns.heatmap(cms[i],cmap="OrRd",cbar=(i == (len(set_ids)-1)),ax=ax,vmin=0,vmax=1,annot=True,fmt=".2f")
        if i == 0:
            plt.tick_params(axis="both", which="major", bottom=False, labelbottom=False, top=True, labeltop=True)
            if rotate_x_labels:
                plt.setp(plt.gca().get_xticklabels(), ha="left", rotation=45)
        elif (i % n_cols) == 0:
            plt.tick_params(
                axis="both",
                which="major",
                bottom=False,
                labelbottom=False,
                top=False,
                labeltop=False,
                left=True,
                labelleft=True,
            )
        elif (i // n_cols) < 1:
            plt.tick_params(
                axis="both",
                which="major",
                bottom=False,
                labelbottom=False,
                top=True,
                labeltop=True,
                left=False,
                labelleft=False,
            )
            if rotate_x_labels:
                plt.setp(plt.gca().get_xticklabels(), ha="left", rotation=45)
        else:
            plt.tick_params(
                axis="both",
                which="major",
                bottom=False,
                labelbottom=False,
                top=False,
                labeltop=False,
                left=False,
                labelleft=False,
            )

        plt.title(set_id)
        ax.title.set_fontsize(fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

    # plt.subplots_adjust(top=1.54, bottom=0.08, left=0.05, right=0.95, hspace=0.20, wspace=0.25)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def correlation_matrix(
    set_ids: List[str],
    cor_matrices: Dict[str, pd.DataFrame],
    show: bool = True,
    save: Union[bool, str] = False,
    size_factor: float = 5,
    fontsize: int = 28,
    n_cols: int = 3,
    colorbar: bool = True,
) -> None:
    """Plot heatmap of gene correlation matrix.

    Args:
        set_ids:
            List of probe set ids.
        cor_matrices: dict of pd.DataFrames
            Correlation matrix of each probe set given in `set_ids`.
        show:
            Show the figure.
        save:
            Save plot to path.
        size_factor:
             Factor for scaling the figure size.
        fontsize:
            Matplotlib fontsize.
        colorbar:
            Whether to draw a colorbar.
        n_cols:
            Number of subplot columns.
    """

    n_plots = len(set_ids)
    n_rows = (n_plots // n_cols) + int((n_plots % n_cols) > 0)

    HSPACE_INCHES = 1 * n_rows
    WSPACE_INCHES = 0.5 * n_cols
    TOP_INCHES = 1
    BOTTOM_INCHES = 0.5
    RIGHT_INCHES = 2
    LEFT_INCHES = 0.5
    CBAR_WIDTH_INCHES = 0.4
    CBAR_RIGHT_INCHES = 1
    CBAR_LEFT_INCHES = RIGHT_INCHES - CBAR_WIDTH_INCHES - CBAR_RIGHT_INCHES

    FIGURE_WIDTH = (size_factor * n_cols) + (((n_cols - 1) / n_cols) * WSPACE_INCHES) + RIGHT_INCHES + LEFT_INCHES
    FIGURE_HEIGHT = (size_factor * n_rows) + (((n_rows - 1) / n_rows) * HSPACE_INCHES) + TOP_INCHES + BOTTOM_INCHES

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for i, set_id in enumerate(set_ids):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(cor_matrices[set_id].values, cmap="seismic", vmin=-1, vmax=1)
        plt.title(set_id, fontsize=fontsize)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    HSPACE = HSPACE_INCHES / FIGURE_HEIGHT
    WSPACE = WSPACE_INCHES / FIGURE_WIDTH
    TOP = 1 - (TOP_INCHES / FIGURE_HEIGHT)
    BOTTOM = BOTTOM_INCHES / FIGURE_HEIGHT
    RIGHT = 1 - (RIGHT_INCHES / FIGURE_WIDTH)
    LEFT = LEFT_INCHES / FIGURE_WIDTH
    SUBPLOT_HEIGHT = size_factor / FIGURE_HEIGHT
    CBAR_WIDTH = CBAR_WIDTH_INCHES / FIGURE_WIDTH
    CBAR_RIGHT = 1 - (CBAR_RIGHT_INCHES / FIGURE_WIDTH)
    CBAR_LEFT = CBAR_RIGHT - (CBAR_LEFT_INCHES / FIGURE_WIDTH)

    plt.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, hspace=HSPACE, wspace=WSPACE)  # top=1.54,

    if colorbar:
        cbar_ax = fig.add_axes([
            CBAR_LEFT,
            TOP - SUBPLOT_HEIGHT,
            CBAR_WIDTH,
            SUBPLOT_HEIGHT])
        cbar = plt.colorbar(cax=cbar_ax)
        cbar.ax.tick_params(labelsize=fontsize)

    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def cluster_similarity(
    selections_info: pd.DataFrame,
    nmi_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    groupby: Optional[str] = None,
    interpolate: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    fontsize: int = 18,
    title: Optional[str] = None,
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot normalized mutual information of clusterings over number of clusters.

    Args:
        selections_info:
            Information on each selection for plotting. The dataframe includes:

                - selection ids or alternative names as index
                - mandatory (only if ``nmi_dfs=None``) column: `path`: path to results csv of each selection (contains
                number of clusters and nmi values (as index) and nmi values in column `nmi`.)
                - optional columns:

                    - `color`: matplotlib color
                    - `linewidth`: matplotlib linewidth
                    - `linestyle`: matplotlib linestyle
                    - `<groupby>`: some annotation that can be used to group the legend.
                    Note that the legend order will follow the row order in :attr:`selections_info.

        nmi_dfs:
            NMI results for each selection.
        groupby:
            Column in ``selections_info`` to group the legend.
        interpolate
            Whether to interpolate the NMI values.
        figsize:
            Matplotlib figsize.
        fontsize:
            Matplotlib fontsize.
        title:
            Plot title.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    Returns:
        Figure can be shown (default `True`) and stored to path (default `None`).
        Change this with `show` and `save`.
    """

    # TODO: nmi_dfs: Check if it's useful to add this or if we always should load from files
    #  (or maybe even the other way around?)
    # TODO: instead of a list we could also use one dataframe with index: ns, column_names: probeset_ids,
    #       Check what's more practical, (think wrt Evaluator.plot_cluster_similarity())
    #  ==> my suggestion: dict of dfs just like evaluator.results["cluster_similarity"]

    df = selections_info.copy()

    # load NMI values from files if necessary
    if nmi_dfs is None:
        if "path" not in df:
            raise ValueError("The mandatory column 'path' is missing in 'selections_info'.")
        else:  # load NMI values from files
            nmi_dfs = {}
            for selection_nr, path in enumerate(df["path"]):
                nmi_dfs[str(selection_nr)] = pd.read_csv(path, index_col=0)
            # del df["path"]

    # check matplotlib style options
    for col in ["color", "linewidth", "linestyle"]:
        if col not in df.columns:
            df[col] = None
            # TODO: if groupby is provided assert that "color", "linewidth", "linestyle" are all the same for
            #  selections that belong to one group
            #   ==> done TODO test
            if groupby:
                group_check = df[["color", "linewidth", "linestyle", "groupby"]].groupby(by=groupby).nunique() > 1
                if any(group_check):
                    bad_group = group_check[group_check].dropna(how="all").dropna(how="all", axis=1)
                    raise ValueError(
                        "Grouping by "
                        + groupby
                        + " failed because the values of "
                        + str(bad_group.columns.values)
                        + " are not unique for group(s) "
                        + str(bad_group.index.values)
                        + "."
                    )

    fig = plt.figure(figsize=figsize)

    labels = []

    for selection_id, plot_df in nmi_dfs.items():
        label = selection_id if not groupby else df.loc[selection_id][groupby]

        plt.plot(
            plot_df["nmi"].interpolate() if interpolate else plot_df["nmi"],
            c=df.loc[selection_id]["color"],
            lw=df.loc[selection_id]["linewidth"],
            linestyle=df.loc[selection_id]["linestyle"],
            label=None if label in labels else label,
        )

        labels.append(label)

    if title:
        plt.title(title, fontsize=fontsize)

    plt.xlabel("number of clusters", fontsize=fontsize)
    plt.ylabel("NMI", fontsize=fontsize)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=fontsize)

    plt.tick_params(axis="both", labelsize=fontsize)

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def knn_overlap(
    selections_info: pd.DataFrame,
    knn_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    groupby: Optional[str] = None,
    interpolate: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    fontsize: int = 18,
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot plot mean overlap of knn clusterings over number of clusters.

    Args:
        selections_info:
            Information on each selection for plotting. The dataframe includes:

                - selection ids or alternative names as index
                - mandatory (only if ``nmi_dfs=None``) column: `path`: path to results csv of each selection (contains
                number of clusters and nmi values (as index) and nmi values in column `nmi`.)
                - optional columns:

                    - `color`: matplotlib color
                    - `linewidth`: matplotlib linewidth
                    - `linestyle`: matplotlib linestyle
                    - `<groupby>`: some annotation that can be used to group the legend.
                    Note that the legend order will follow the row order in :attr:`selections_info.

        knn_dfs:
            NMI results for each selection.
        groupby:
            Column in ``selections_info`` to group the legend.
        interpolate
            Whether to interpolate the NMI values.
        title:
            Plot title.
        figsize:
            Matplotlib figsize.
        fontsize:
            Matplotlib fontsize.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    Returns:
        Figure can be shown (default `True`) and stored to path (default `None`).
        Change this with `show` and `save`.
    """

    # TODO: nmi_dfs: Check if it's useful to add this or if we always should load from files
    #  (or maybe even the other way around?)
    # TODO: instead of a list we could also use one dataframe with index: ns, column_names: probeset_ids,
    #       Check what's more practical, (think wrt Evaluator.plot_cluster_similarity())
    #  ==> my suggestion: dict of dfs just like evaluator.results["cluster_similarity"]

    df = selections_info.copy()

    # load NMI values from files if necessary
    if knn_dfs is None:
        if "path" not in df:
            raise ValueError("The mandatory column 'path' is missing in 'selections_info'.")
        else:  # load NMI values from files
            knn_dfs = {}
            for selection_nr, path in enumerate(df["path"]):
                knn_dfs[str(selection_nr)] = pd.read_csv(path, index_col=0)

    # check matplotlib style options
    for col in ["color", "linewidth", "linestyle"]:
        if col not in df.columns:
            df[col] = None
            # TODO: if groupby is provided assert that "color", "linewidth", "linestyle" are all the same for
            #  selections that belong to one group
            #   ==> done TODO test
            if groupby:
                group_check = df[["color", "linewidth", "linestyle", "groupby"]].groupby(by=groupby).nunique() > 1
                if any(group_check):
                    bad_group = group_check[group_check].dropna(how="all").dropna(how="all", axis=1)
                    raise ValueError(
                        "Grouping by "
                        + groupby
                        + " failed because the values of "
                        + str(bad_group.columns.values)
                        + " are not unique for group(s) "
                        + str(bad_group.index.values)
                        + "."
                    )

    fig = plt.figure(figsize=figsize)

    labels = []

    for selection_id, plot_df in knn_dfs.items():
        label = selection_id if not groupby else df.loc[selection_id][groupby]

        plt.plot(
            plot_df["mean"].interpolate() if interpolate else plot_df["mean"],
            c=df.loc[selection_id]["color"],
            lw=df.loc[selection_id]["linewidth"],
            linestyle=df.loc[selection_id]["linestyle"],
            label=None if label in labels else label,
        )

        labels.append(label)

    if title:
        plt.title()

    plt.xlabel("number of neighbors", fontsize=fontsize)
    plt.ylabel("mean knn overlap", fontsize=fontsize)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=fontsize)

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def summary_table(
    table: pd.DataFrame,
    summaries: Union[Literal["all"], List[str]] = "all",
    color_maps: Dict[str, matplotlib.colors.Colormap] = {},
    rename_cols: Dict[str, str] = {},
    rename_rows: Dict[str, str] = {},
    time_format: List[str] = [],
    log_scale: List[str] = [],
    color_limits: Dict[str, List[float]] = {},
    nan_color: str = "lightgrey",
    threshold_ann: Dict = {},
    show: bool = True,
    save: Union[bool, str] = False,
) -> None:
    """Plot table of summary statistics.

    Args:
        table: pd.DataFrame

        summaries:
            List of summary metrics that are plotted.
        color_maps:
            Color maps assigned to summary metrics. Use the initial name and not the potential new name
            given via `rename_cols`.
        rename_cols:
            Rename summary metrics for plot.
        rename_rows:
            Rename set ids.
        time_format
            Summary names that are formatted to days, hours, mins and secs (seconds are expected as input).
        log_scale:
            Summary names for which a log scaled colormap is applied.
        color_limits: dict of lists of two floats
            For each summary metric optionally provide vmin and vmax for the colormap.
        nan_color:
            Color for nan values.
        threshold_ann: dict
            Special annotation for values above defined threshold. E.g. {"time":{"th":1000,"above":True,"ann":"> 1k"}}
        show:
            Show the figure.
        save:
            If `True` or a `str`, save the figure.
    """

    fsize = 15

    # Default order and colors
    default_order = [
        "cluster_similarity",
        "knn_overlap",
        "Greens",
        "forest_clfs",
        "marker_corr",
        "gene_corr",
        "penalty",
    ]
    default_cmaps = {
        "cluster_similarity": "Greens",
        "knn_overlap": "Greens",
        "forest_clfs": "Purples",  # "Reds",
        "marker_corr": "Purples",  # "Reds",
        "gene_corr": "Blues",
        "penalty": truncate_colormap(plt.get_cmap("Greys"), minval=0.05, maxval=0.7, n=100),  # "Greys",
        "other": "Greys",
    }
    if summaries == "all":
        summaries = table.columns.tolist()
        assert isinstance(summaries, list)
        for s in summaries:
            if s not in default_order:
                default_order.append(s.split()[0])
        # Order by default order of metrics and length of summary
        summaries.sort(key=lambda s: default_order.index(s.split()[0]) * 100 + len(s))

    cmaps = {}
    for summary in summaries:
        if summary in color_maps:
            cmaps[summary] = color_maps[summary]
        elif summary.split()[0] in default_cmaps:
            cmaps[summary] = default_cmaps[summary.split()[0]]
        else:
            cmaps[summary] = default_cmaps["other"]

    # Init final table for plotting
    df = table[summaries].copy()

    # Register potential new names of columns that are time formatted or log transformed
    for col in df.columns:
        if (col in time_format) and (col in rename_cols):
            time_format.append(rename_cols[col])
        if (col in log_scale) and (col in rename_cols):
            log_scale.append(rename_cols[col])
        if (col in color_limits) and (col in rename_cols):
            color_limits[rename_cols[col]] = color_limits[col]
        if (col in threshold_ann) and (col in rename_cols):
            threshold_ann[rename_cols[col]] = threshold_ann[col]

    # Rename columns
    df = df.rename(columns=rename_cols, index=rename_rows)

    # Replace old column names with new names in colormaps
    for summary, new_key in rename_cols.items():
        cmaps[new_key] = cmaps.pop(summary)

    n_cols = len(df.columns)
    n_sets = len(df.index)

    fig = plt.figure(figsize=(n_cols * 1.1, n_sets))
    gs1 = gridspec.GridSpec(1, n_cols)
    gs1.update(wspace=0.0, hspace=0.0)

    multi_col = {}
    cols = df.columns.tolist()
    for col in df.columns.unique():
        count = cols.count(col)
        if count > 1:
            multi_col[col] = 0

    for i, col in enumerate(df.columns):

        ax = plt.subplot(gs1[i])

        yticklabels = bool(i == 0)

        if col in multi_col:  # TODO: time formating multi col support? Better get col pos at the beginning + iloc
            col_pos = [i for i, c in enumerate(df.columns) if c == col][multi_col[col]]
            color_vals = np.log(df.iloc[:, [col_pos]]) if (col in log_scale) else df.iloc[:, [col_pos]]
            multi_col[col] += 1
        else:
            color_vals = np.log(df[[col]]) if (col in log_scale) else df[[col]]
            col_pos = [i for i, c in enumerate(df.columns) if c == col][0]

        if col in time_format:
            # annot = df[col].apply(format_time).values[:, np.newaxis]
            annot = df.iloc[:, col_pos].apply(format_time).values[:, np.newaxis]
            fmt = ""
        else:
            annot = True
            fmt = ".2f"
        if col in threshold_ann:
            formatter = lambda s: f"{s:.2f}"
            annot = df.iloc[:, col_pos].apply(formatter).values[:, np.newaxis] if isinstance(annot, bool) else annot
            tmp = threshold_ann[col]
            th_mask = (df.iloc[:, col_pos] > tmp["th"]) if tmp["above"] else (df.iloc[:, col_pos] < tmp["th"])
            annot[th_mask, :] = tmp["ann"]
            fmt = ""

        g = sns.heatmap(
            color_vals,
            cmap=cmaps[col],
            annot=annot,
            mask=color_vals.isnull(),
            cbar=False,
            square=True,
            yticklabels=yticklabels,
            fmt=fmt,
            annot_kws={"fontsize": fsize - 2},
            vmin=color_limits[col][0] if (col in color_limits) else None,
            vmax=color_limits[col][1] if (col in color_limits) else None,
        )
        g.set_facecolor(nan_color)
        plt.tick_params(
            axis="x", which="major", labelsize=fsize, labelbottom=False, bottom=False, top=True, labeltop=True
        )
        plt.tick_params(axis="y", which="major", labelsize=fsize)
        ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment="left", rotation=45)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


#############################
## selection related plots ##
#############################

def explore_constraint(
    a: List[sc.AnnData],
    selections_tmp: List[pd.DataFrame],
    penalty_kernels: List[Callable],
    key: str = "quantile_99",
    factors: List[int] = None,
    upper: float = 1,
    lower: float = 0,
    size_factor: int = 6,
    n_rows: int = 1,
    legend_size: int = 9,
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot histogram of quantiles for selected genes for different penalty kernels.
    Args:
        a:
        selections_tmp:
        factors:
        q:
        upper:
        lower:
        penalty_kernels:
        key:
        size_factor:
        n_rows:
        legend_size:
        show:
            Whether to display the plot.
        save:
            Save the plot to path.
    Returns:

    """
    # TODO:
    #  1) Fix explore_constraint plot. The following circular import is causing problems atm:
    #     DONE: moved parts of this method to selector.plot_expore_constraint --> this solves the problem
    #  2) How to generalize the plotting function, support:
    #     - any selection method with defined hyperparameters
    #     - any penalty kernel
    #     - any key to be plotted (not only quantiles)

    cols = len(factors)

    fig = plt.figure(figsize=(size_factor * cols, 0.7 * size_factor * n_rows))
    for i, factor in enumerate(factors):
        ax1 = plt.subplot(n_rows, cols, i + 1)
        hist_kws = {"range": (0, np.max(a[i].var[key]))}
        bins = 100
        sns.distplot(
            a[i].var[key],
            kde=False,
            label="highly_var",
            bins=bins,
            hist_kws=hist_kws,
        )
        sns.distplot(
            a[i][:, selections_tmp[i]["selection"]].var[key],
            kde=False,
            label="selection",
            bins=bins,
            hist_kws=hist_kws,
        )
        plt.axvline(x=lower, lw=0.5, ls="--", color="black")
        plt.axvline(x=upper, lw=0.5, ls="--", color="black")
        ax1.set_yscale("log")
        plt.legend(prop={"size": legend_size}, loc=[0.73, 0.74], frameon=False)
        plt.title(f"factor = {factor}")

        ax2 = ax1.twinx()
        x_values = np.linspace(0, np.max(a[i].var[key]), 240)
        plt.plot(x_values, 1 * penalty_kernels[i](x_values), label="penal.", color="green")
        plt.legend(prop={"size": legend_size}, loc=[0.73, 0.86], frameon=False)
        plt.ylim([0, 2])
        for label in ax2.get_yticklabels():
            label.set_color("green")

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def gene_overlap(
    selection_df: pd.DataFrame,
    style: Literal["upset", "venn"] = "upset",
    min_degree: int = 1,
    fontsize: int = 18,
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot the intersection of different selected gene sets.

    Args:
        selection_df:
            Table with gene sets. Gene names are given in the index, gene sets are given as boolean columns.
        style:
            Plot type. Options are

                - "upset": upset plot
                - "venn": venn diagram

        min_degree:
            Only for `style="upset"`: minimum degree of a subset to be shown in the plot.
        fontsize:
                Matplotlib fontsize.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    Returns:
        Figure can be shown (default `True`) and stored to path (default `None`).
        Change this with `show` and `save`.

    """

    if style == "venn":
        # remove emtpy sets:
        selection_df = selection_df.loc[:, selection_df.sum() > 0]

        # calculate plotting parameter
        labels, radii, actualOverlaps, disjointOverlaps = venn.df2areas(selection_df, fineTune=False)

        # draw venn diagram
        fig, ax = venn.venn(
            radii, actualOverlaps, disjointOverlaps, labels=labels, labelsize=fontsize, cmap="Blues", fineTune=False
        )

    elif style == "upset":
        # transform to compatible format
        upset_data = from_indicators(selection_df)

        # set up figure
        upset_plot = UpSet(upset_data, subset_size="count", min_degree=min_degree, show_counts=True)

        # draw figure
        fig = plt.figure()
        upset_plot.plot(fig=fig)

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def gene_overlap_grouped(
    selection_df: Dict[str, pd.DataFrame],
    groupby: str = "method",
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot the intersection of different selected gene sets grouped by the selection method.

    Args:
        selection_df:
            Boolean dataframe with gene identifiers as index and one column for each gene set.
        groupby:
            Name of a column that categorizes the gene sets, eg. the method they were selected with.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    Returns:
        Figure can be shown (default `True`) and stored to path (default `None`).
        Change this with `show` and `save`.

    """

    pass
    # TODO


def masked_dotplot(
    adata: sc.AnnData,
    selector,
    ct_key: str = "celltype",
    imp_threshold: float = 0.05,
    celltypes: Optional[List[str]] = None,
    n_genes: Optional[int] = None,
    comb_markers_only: bool = False,
    markers_only: bool = False,
    cmap: str = "Reds",
    comb_marker_color: str = "darkblue",
    marker_color: str = "green",
    non_adata_celltypes_color: str = "grey",
    save: Union[bool, str] = False,
):
    """Create dotplot with additional annotation masks.

    Args:
        adata:
            AnnData with `adata.obs[ct_key]` cell type annotations
        selector:
            ProbesetSelector object with selected selector.probeset
        ct_key:
            Column of `adata.obs` with cell type annotation.
        imp_threshold:
            Show genes as combinatorial marker only for those genes with importance > `imp_threshold`
        celltypes:
            Optional subset of celltypes (rows of dotplot)
        n_genes:
            Optionally plot top `n_genes` genes.
        comb_markers_only:
            Whether to plot only genes that are combinatorial markers for the plotted cell types. (can be combined with
            markers_only, in that case markers that are not comb markers are also shown)
        markers_only:
            Whether to plot only genes that are markers for the plotted cell types. (can be combined with comb_markers_only,
            in that case comb markers that are not markers are also shown)
        cmap:
            Colormap of mean expressions.
        comb_marker_color:
            Color for combinatorial markers.
        marker_color:
            Color for marker genes.
        non_adata_celltypes_color:
            Color for celltypes that don't occur in the data set
        save:
            If `True` or a `str`, save the figure.
    """
    from spapros.selection import ProbesetSelector

    if isinstance(selector, str):
        selector = ProbesetSelector(adata, ct_key, save_dir=selector)
        # TODO: think the last steps of the ProbesetSelector are still not saved..., needs to be fixed.

    # celltypes, possible origins:
    # - adata.obs[ct_key] (could include cts not used for selection)
    # - celltypes for selection (including markers, could include cts which are not in adata.obs[ct_key])
    # --> pool all together... order?

    if celltypes is not None:
        cts = celltypes
        a = adata[adata.obs[ct_key].isin(celltypes)].copy()
        # a.obs[ct_key] = a.obs[ct_key].astype(str).astype("category")
    else:
        # Cell types from adata
        cts = adata.obs[ct_key].unique().tolist()
        # Cell types from marker list only
        if "celltypes_marker" in selector.probeset:
            tmp = []
            for markers_celltypes in selector.probeset["celltypes_marker"].str.split(","):
                tmp += markers_celltypes
            tmp = np.unique(tmp).tolist()
            if "" in tmp:
                tmp.remove("")
            cts += [ct for ct in tmp if ct not in cts]
        a = adata

    # Get selected genes that are also in adata
    selected_genes = [
        g for g in selector.probeset[selector.probeset["selection"]].index.tolist() if g in adata.var_names
    ]

    # Get tree genes
    tree_genes = {}
    assert isinstance(selector.forest_results["forest"], list)
    assert len(selector.forest_results["forest"]) == 3
    assert isinstance(selector.forest_results["forest"][2], dict)
    for ct, importance_tab in selector.forest_results["forest"][2].items():
        if ct in cts:
            tree_genes[ct] = importance_tab["0"].loc[importance_tab["0"] > imp_threshold].index.tolist()
            tree_genes[ct] = [g for g in tree_genes[ct] if g in selected_genes]

    # Get markers
    marker_genes: Dict[str, list] = {ct: [] for ct in (cts)}
    for ct in cts:
        for gene in selector.probeset[selector.probeset["selection"]].index:
            if ct in selector.probeset.loc[gene, "celltypes_marker"].split(",") and (gene in adata.var_names):
                marker_genes[ct].append(gene)
        marker_genes[ct] = [g for g in marker_genes[ct] if g in selected_genes]

    # Optionally subset genes:
    # Subset to combinatorial markers of shown celltypes only
    if comb_markers_only or markers_only:
        allowed_genes = []
        if comb_markers_only:
            allowed_genes += list(itertools.chain(*[tree_genes[ct] for ct in tree_genes.keys()]))
        if markers_only:
            allowed_genes += list(itertools.chain(*[marker_genes[ct] for ct in marker_genes.keys()]))
        selected_genes = [g for g in selected_genes if g in allowed_genes]
    # Subset to show top n_genes only
    if n_genes:
        selected_genes = selected_genes[: min(n_genes, len(selected_genes))]
    # Filter (combinatorial) markers by genes that are not in the selected genes
    for ct in cts:
        marker_genes[ct] = [g for g in marker_genes[ct] if g in selected_genes]
    for ct in tree_genes.keys():
        tree_genes[ct] = [g for g in tree_genes[ct] if g in selected_genes]

    dp = MaskedDotPlot(
        a,
        var_names=selected_genes,
        groupby=ct_key,
        tree_genes=tree_genes,
        marker_genes=marker_genes,
        further_celltypes=[ct for ct in cts if ct not in adata.obs[ct_key].unique()],
        cmap=cmap,
        tree_genes_color=comb_marker_color,
        marker_genes_color=marker_color,
        non_adata_celltypes_color=non_adata_celltypes_color,
    )
    dp.make_figure()
    if save:
        plt.gcf().savefig(save, bbox_inches="tight", transparent=True)


######################
## helper functions ##
######################


def format_time(time: float) -> str:
    """Reformat a time stamp.

    Args:
        time:
            Time in seconds.

    Returns:
        str:
            The formatted time.
    """
    days = int(time // (3600 * 24))
    hours = int(time // 3600)
    mins = int(time // 60)
    secs = int(time // 1)
    unit = ["d", "h", "min", "sec"]
    for t, u in zip([days, hours, mins, secs], unit):
        if t > 0:
            return f"{t} {u}"
    return "0 sec"


def truncate_colormap(
    cmap: matplotlib.colors.Colormap, minval: float = 0.0, maxval: float = 1.0, n: int = 100
) -> matplotlib.colors.Colormap:
    """Truncate a colormap to a given number of colors and an interval.

    Args:
        cmap:
            Colormap name.
        minval:
            Smallest color value.
        maxval:
            Highest color value.
        n:
            Number of colors.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
