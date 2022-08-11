"""Plotting Module."""
import itertools
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from spapros.plotting._masked_dotplot import MaskedDotPlot
from upsetplot import from_indicators
from upsetplot import UpSet
from venndata import venn


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

    # TODO maybe move to utils ?

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
            Confusion matrix of each probe set given in ``set_ids``.
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
        cor_matrices:
            Ordered correlation matrix of each probe set given in ``set_ids``.
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

    plt.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, hspace=HSPACE, wspace=WSPACE)

    if colorbar:
        cbar_ax = fig.add_axes([CBAR_LEFT, TOP - SUBPLOT_HEIGHT, CBAR_WIDTH, SUBPLOT_HEIGHT])
        cbar = plt.colorbar(cax=cbar_ax)
        cbar.ax.tick_params(labelsize=fontsize)

    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def marker_correlation(
    marker_corr: Dict[str, pd.DataFrame],
    corr_metric: str = "per marker mean > 0.025",
    mask_below_max: bool = False,
    set_ids: Optional[List[str]] = None,
    rename_set_ids: Dict[str, str] = {},
    size_factor: float = 1.0,
    save: Optional[str] = None,
) -> None:
    """Plot maximal correlations with marker genes

    Args:
        marker_corr:
            For each set id a Dataframe containing maximal correlations for each marker gene. The index contains the marker
            genes symbols and the Dataframe needs to have one column named ``corr_metric`` with float values.
        corr_metric:
            Column in dataframes of ``marker_corr`` that contains the correlation values.
        mask_below_max:
            Whether to mask values that are lower than the maximum correlation with a marker gene for each cell type.
            The masking applies per set id.
        set_ids:
            Plot subset of set ids (must occur as keys in ``marker_corr``).
        rename_set_ids:
            Mapping to rename set_ids for tick labels.
        size_factor:
            Scale figure size.
        save:
            Optionally save figure to path.

    """

    #######################
    # Prepare data matrix #
    #######################

    if set_ids is None:
        set_ids = list(marker_corr.keys())
    set_id_names = [rename_set_ids[s] if (s in rename_set_ids) else s for s in set_ids]

    # Sort genes alphabetically by cell type group and within group alphabetically by gene name
    df_set0 = marker_corr[set_ids[0]]
    df_set0["genes"] = df_set0.index
    df_set0 = df_set0.sort_values(["celltype", "genes"])
    df_set0 = df_set0.loc[~df_set0["per marker mean > 0.025"].isnull()]
    genes = df_set0["genes"].values

    # Create dataframe for heatmap
    df = pd.DataFrame(
        index=set_id_names,
        data={gene: [marker_corr[set_id].loc[gene, corr_metric] for set_id in set_ids] for gene in genes},
    )

    # Create dataframe for masked heatmap
    df_masked = df.copy()
    for ct in df_set0["celltype"].unique():
        genes_tmp = df_set0.loc[df_set0["celltype"] == ct].index.tolist()
        for method, max_val in df[genes_tmp].max(axis=1).items():
            genes_reduced = [g for g in genes_tmp if (df.loc[method, g] != max_val)]
            df_masked.loc[method, genes_reduced] = np.nan

    ########
    # Plot #
    ########

    # plot parameters
    width_per_gene = 0.25
    height_per_set = 0.3
    pos_factor = 0.02

    # get positions for group brackets
    group_positions = []
    group_labels = []
    for ct in df_set0["celltype"].unique():
        tmp = pd.Series((df_set0["celltype"] == ct).values)
        group_positions.append([tmp.idxmax(), tmp.where(tmp).last_valid_index()])
        group_labels.append(ct)

    # Set nrows, ncols and figsize
    ncols_genes = len(df.columns)
    ncols_cbar = 2  # max(int(0.03 * ncols), 1)
    ncols = ncols_genes + ncols_cbar
    nrows = len(df) + 1  # 1 for group brackets
    fig = plt.figure(figsize=((width_per_gene * ncols) * size_factor, (height_per_set * nrows) * size_factor))

    # Define axes
    ax1 = plt.subplot2grid((nrows, ncols), (1, 0), colspan=ncols_genes, rowspan=nrows - 1)
    ax2 = plt.subplot2grid((nrows, ncols), (1, ncols_genes), colspan=1, rowspan=nrows - 1)

    # Plot heatmap and colorbar
    df_ = df_masked if mask_below_max else df
    hm = sns.heatmap(
        df_,
        square=False,
        cbar_ax=ax2,
        ax=ax1,
        cmap="Reds",
        xticklabels=df.columns,
        yticklabels=set_id_names,
        vmin=0,
        vmax=1,
        annot=False,
    )
    hm.set_facecolor("lightgrey")

    # Add group brackets
    for g_idx, group in enumerate(group_labels):
        n_cols_group = group_positions[g_idx][1] - group_positions[g_idx][0] + 1
        ax = plt.subplot2grid((nrows, ncols), (0, group_positions[g_idx][0]), colspan=n_cols_group, rowspan=1)

        ax.axis("off")
        ax.set_xlim([0, n_cols_group])
        ax.set_ylim([0, 1])
        x1, x2 = [0.2, n_cols_group - 0.2]
        y, h = [0, 0.6]  # [0,0.12]#[0,0.3]
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.0 * size_factor, c="black")
        plt.text(
            (x1 + x2) * 0.5, (y + h) * 1.5, group, ha="center", va="bottom", color="black", rotation=90
        )  # , fontsize=fsize)

    # Set axes labels
    # ax1.set_xlabel("marker gene")
    ax2.set_ylabel("max. correlation\nwith marker gene")

    # Show and save
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.show()


def _lineplot(
    selections_info: pd.DataFrame,
    data: Optional[Dict[str, pd.DataFrame]] = None,
    groupby: Optional[str] = None,
    interpolate: bool = True,
    title: Optional[str] = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 18,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """Plot interpolated lineplot

    Args:
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
        data:
            Dictionary with dataframes containing the data to plot for each selection. The keys need to be the same as
            the index of ``selections_info``.
        groupby:
            Column in ``selections_info`` to group the legend.
        interpolate
            Whether to interpolate the values.
        title:
            Plot title.
        xlabel
            X-axis label.
        ylabel:
            Y-axis label.
        figsize:
            Matplotlib figsize.
        fontsize:
            Matplotlib fontsize.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    """

    df = selections_info.copy()

    # check if data is there
    if data is None and "path" not in df:
        raise ValueError("No data provided.")

    if data is None:
        data = {}
    else:
        data = {selection_id: data[selection_id] for selection_id in df.index}

    # load data from files if necessary
    if "path" in df:
        for selection_id, path in zip(df.index, df["path"]):
            data[selection_id] = pd.read_csv(path, index_col=0)

    if groupby and groupby not in df:
        raise ValueError(f"To group by {groupby}, a correspondant column is necessary in selections_info.")

    # check style options
    for col in ["color", "linewidth", "linestyle"]:
        # if not given, use default matplotlib values except color if grouping
        if col not in df.columns:
            df[col] = None
        if col == "color" and groupby:
            # if necessary, set grouped line colors (note that just like matplotlib, we cycle over a limited number of
            # (default 10) colors)
            cycler = matplotlib.rcParams["axes.prop_cycle"]
            prop_cycler = itertools.cycle(cycler)
            for group in df[groupby].drop_duplicates():
                df["color"][df[groupby] == group] = next(prop_cycler)["color"]

    # assert that styles are equal for selections that belong to one group
    if groupby:
        group_check = df[["color", "linewidth", "linestyle", groupby]].groupby(by=groupby).nunique() > 1
        if group_check.any().any():
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

    for selection_id in df.index:

        if selection_id not in data:
            raise ValueError(f"Can't find {selection_id} in selections_info.")

        xlabel = data[selection_id].index.name if not xlabel else xlabel
        yname = data[selection_id].columns[0]
        ylabel = yname if not ylabel else ylabel

        plot_df = data[selection_id]
        label = selection_id if not groupby else df.loc[selection_id][groupby]

        plt.plot(
            plot_df[yname].interpolate() if interpolate else plot_df[yname],
            c=df.loc[selection_id]["color"],
            lw=df.loc[selection_id]["linewidth"],
            linestyle=df.loc[selection_id]["linestyle"],
            label=label,
        )

    if title:
        plt.title()

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=fontsize)
    plt.tick_params(axis="both", labelsize=fontsize)

    # plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def knn_overlap(
    selections_info: pd.DataFrame,
    data: Optional[Dict[str, pd.DataFrame]] = None,
    groupby: Optional[str] = None,
    interpolate: bool = True,
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 18,
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot mean knn overlap over k

    Args:
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
        data:
            Dictionary with dataframes containing the data to plot for each selection. The keys need to be the same as
            the index of ``selections_info``.
        groupby:
            Column in ``selections_info`` to group the legend.
        interpolate
            Whether to interpolate the values.
        figsize:
            Matplotlib figsize.
        fontsize:
            Matplotlib fontsize.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    """

    _lineplot(
        selections_info=selections_info,
        data=data,
        groupby=groupby,
        interpolate=interpolate,
        title=None,
        xlabel="number of neighbors",
        ylabel="mean knn overlap",
        figsize=figsize,
        fontsize=fontsize,
        show=show,
        save=save,
    )


def cluster_similarity(
    selections_info: pd.DataFrame,
    data: Optional[Dict[str, pd.DataFrame]] = None,
    groupby: Optional[str] = None,
    interpolate: bool = True,
    figsize: Tuple[float, float] = (8, 5),
    fontsize: int = 18,
    show: bool = True,
    save: Optional[str] = None,
):
    """Plot cluster similarity as NMI over number of clusters

    Args:
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
        data:
            Dictionary with dataframes containing the data to plot for each selection. The keys need to be the same as
            the index of ``selections_info``.
        groupby:
            Column in ``selections_info`` to group the legend.
        interpolate
            Whether to interpolate the values.
        figsize:
            Matplotlib figsize.
        fontsize:
            Matplotlib fontsize.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    """

    _lineplot(
        selections_info=selections_info,
        data=data,
        groupby=groupby,
        interpolate=interpolate,
        title=None,
        xlabel="number of clusters",
        ylabel="NMI",
        figsize=figsize,
        fontsize=fontsize,
        show=show,
        save=save,
    )


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
    """Plot table of summary statistics

    Args:
        table:
            Dataframe with set ids in the index and a metric in each column.
        summaries:
            List of summary metrics that are plotted.
        color_maps:
            Color maps assigned to summary metrics. Use the initial name and not the potential new name
            given via `rename_cols`.
        rename_cols:
            Rename summary metrics for plot.
        rename_rows:
            Rename set ids.
        time_format:
            Summary names that are formatted to days, hours, mins and secs (seconds are expected as input).
        log_scale:
            Summary names for which a log scaled colormap is applied.
        color_limits:
            For each summary metric optionally provide vmin and vmax for the colormap.
        nan_color:
            Color for nan values.
        threshold_ann:
            Special annotations for values above/below defined thresholds. E.g.
            ``{"time":{"th":1000,"above":True,"ann":"> 1k"}}``
        show:
            Show the figure.
        save:
            Save the plot to path.
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
    gs1 = GridSpec(1, n_cols)
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
    x_axis_key: str = "quantile_99",
    factors: List[float] = None,
    upper: float = 6,
    lower: float = 2,
    size_factor: int = 6,
    n_rows: int = 1,
    legend_size: int = 9,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """Plot histogram of quantiles for selected genes for different penalty kernels.

    Args:
        a:
            List of ``sc.AnnData`` objects containing the data for plotting.
        selections_tmp:
            Dataframe containing the selection.
        factors:
            List of titles for the subplots, i.e. the factors of each penalty kernel.
        upper:
            Lower border above which the kernel is 1.
        lower:
            Upper boder below which the kernel is 1.
        penalty_kernels:
            List of penalty kernels, which were used for selection.
        x_axis_key:
            Key of column in ``adata.var`` that is used for the x axis of the plotted histograms.
        size_factor:
             Factor for scaling the figure size.
        n_rows:
            Number of subplot rows.
        legend_size:
            Matplotlib legend size.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    Returns:

    """

    # TODO:
    #  remove --> now selection_histogram
    #  1) Fix explore_constraint plot. The following circular import is causing problems atm:
    #     DONE: moved parts of this method to selector.plot_expore_constraint --> this solves the problem
    #  2) How to generalize the plotting function, support:
    #     - any selection method with defined hyperparameters
    #     - any penalty kernel
    #     - any key to be plotted (not only quantiles)

    if factors is None:
        factors = [1]
    assert isinstance(factors, list)

    cols = len(factors)

    fig = plt.figure(figsize=(size_factor * cols, 0.7 * size_factor * n_rows))
    for i, factor in enumerate(factors):
        ax1 = plt.subplot(n_rows, cols, i + 1)
        hist_kws = {"range": (0, np.max(a[i].var[x_axis_key]))}
        bins = 100
        sns.distplot(
            a[i].var[x_axis_key],
            kde=False,
            label="highly_var",
            bins=bins,
            hist_kws=hist_kws,
        )
        idx = selections_tmp[i]["selection"].index[selections_tmp[i]["selection"]]
        sns.distplot(
            a[i][:, idx].var[x_axis_key],
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
        x_values = np.linspace(0, np.max(a[i].var[x_axis_key]), 240)
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


def selection_histogram(
    adata: sc.AnnData,
    selections_dict: Dict[str, pd.DataFrame],
    x_axis_keys: Dict[str, str],
    background_key: str = "highly_variable",
    penalty_kernels: Dict[str, Dict[str, Callable]] = None,
    penalty_keys: Dict[str, List[str]] = None,
    penalty_labels: Union[str, Dict[str, Dict[str, str]]] = "penalty",
    upper_borders: Union[bool, Dict[str, Union[float, bool]]] = None,
    lower_borders: Union[bool, Dict[str, Union[float, bool]]] = None,
    size_factor: float = 4.5,
    legend_size: float = 11,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """Plot histogram of quantiles for selected genes for different penalty kernels.

    Args:
        adata:
            ``sc.AnnData`` object containing the data for plotting the histogram in ``adata.var[x_axis_key]``.
        selections_dict:
            Dictionary of dataframes with gene identifiers as index and a boolean column indicating the selection. The
            dictionary keys are used as label in the plot and need to be equivalent to the keys in ``penalty_kernels``
            and ``penalty_keys``.
        x_axis_keys:
            Dictionary with the column in ``adata.var`` that is used for the x axis of the plotted histograms. There are
            two reasonable options:

                - If a penalty is plottet: the column containing the values of which the penalty scores were derived by
                  applying the penalty kernel. Use the ``penalty_key`` as dictionary key in this case.
                - If no penalty is plottet: a column containing some statistic, eg. 99% quantile. Use the same keys as
                  ``selection_dict`` in this case.

        background_key:
            Key of column in ``adata.var`` which is plottet as background histogram. If `None`, no background histogram
            is plotted. If `'all'`, all genes are used as background.
        penalty_kernels:
            Dictionary of penalty kernels, which were used for each selection. The outer key is the selection name. The
            inner keys are used as subplot titles.
            Additionally, ``penalty_keys`` can be provided. If both are None, only the histograms are plottet.
        penalty_keys:
            Dictionary of a list of column keys of ``adata.var`` containing penalty scores for each selection.
            Additionally, ``penalty_kernels`` can be provided. If both are None, only the histograms are plottet.
        penalty_labels:
            A legeng label for each selection and each penalty. The keys of the outer dictionary need to be the
            selection names. As keys for the inner dictionary, use the penalty keys.
        upper_borders:
            Dictionary with the lower borders above which the kernels are 1, which is indicated as a vertical line in
            the plot. Use the same dictionay keys as ``penalty_keys`` or ``penalty_kernels``. If None, it is
            automatically infered. If False or not in the dictionary, no vertical line at the upper border is drawn.
        lower_borders:
            Dictionary with the upper borders below which the kernels are 1, which is indicated as a vertical line in
            the plot. Use the same dictionay keys as ``penalty_keys`` or ``penalty_kernels``. If None, it is
            automatically infered. If False or not in the dictionary, no vertical line at the lower border is drawn.
        size_factor:
             Factor for scaling the figure size.
        legend_size:
            Matplotlib legend size.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    Returns:

    """

    x_values_dict = {}
    y_values_dict = {}

    if penalty_labels is None:
        penalty_labels = {}

    # apply penalty kernel if given
    if penalty_kernels is not None:
        for selection_label in selections_dict:

            x_values_dict[selection_label] = {}
            y_values_dict[selection_label] = {}

            if selection_label in penalty_kernels:
                for i, penalty_key in enumerate(penalty_kernels[selection_label]):

                    penalty_kernel = penalty_kernels[selection_label][penalty_key]

                    if penalty_key not in x_axis_keys:
                        raise ValueError(f"No adata.var key given for {penalty_key}.")
                    x_values = np.linspace(0, np.max(adata.var[x_axis_keys[penalty_key]]), 240)

                    x_values_dict[selection_label][penalty_key] = x_values
                    y_values_dict[selection_label][penalty_key] = penalty_kernel(x_values)

    # interpolate penalty scores if values in adata.var[[penalty_keys]] are given
    if penalty_keys is not None:
        for selection_label in selections_dict:

            x_values_dict[selection_label] = {}
            y_values_dict[selection_label] = {}

            if selection_label in penalty_keys:
                for i, penalty_key in enumerate(penalty_keys[selection_label]):

                    if penalty_key not in adata.var:
                        raise ValueError(f"Can't plot {penalty_key} because it was not found in adata.var. ")

                    if penalty_key not in x_axis_keys:
                        raise ValueError(f"No adata.var key given for {penalty_key}.")

                    penalty_interp = interp1d(
                        adata.var[x_axis_keys[penalty_key]], adata.var[penalty_key], kind="linear"
                    )
                    x_values = np.linspace(penalty_interp.x[0], penalty_interp.x[-1], 240)

                    x_values_dict[selection_label][penalty_key] = x_values
                    y_values_dict[selection_label][penalty_key] = penalty_interp(x_values)

    if (penalty_kernels is None) and (penalty_keys is None):
        x_values_dict = {set_id: [] for set_id in selections_dict}

    n_rows = len(selections_dict)
    cols = [len(x_values_dict[x]) for x in x_values_dict]
    n_cols = max(max(cols), 1)  # if cols else 1

    fig = plt.figure(figsize=(size_factor * n_cols, 0.7 * size_factor * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    i = -1
    for selection_label, selection_df in selections_dict.items():

        for j in range(n_cols):
            i = i + 1
            # if no penalty is given, plot only one histogram without penalty
            if len(x_values_dict[selection_label]) == 0 and j > 0:
                continue
            # if less penalties given than n_cols, let space empty
            elif 0 < len(x_values_dict[selection_label]) <= j:
                continue
            ax1 = fig.add_subplot(gs[i])

            # get adata.var key for histogram
            if len(x_values_dict[selection_label]) == 0:
                # no penalty to plot
                var_key = x_axis_keys[selection_label]
            else:
                # penalty kernel should be plottet
                penalty_key = list(x_values_dict[selection_label].keys())[j]
                var_key = x_axis_keys[penalty_key]
            assert isinstance(var_key, str)

            # get histogram data
            selected_genes = selection_df.loc[selection_df["selection"]].index  # selection_df.index[selection_df]
            mask = adata.var.index.isin(selected_genes)
            hist_data = adata[:, mask].var[var_key]
            hist_kws = {"range": (0, np.max(hist_data))}

            # get background data
            bg_data = None
            if background_key == "all":
                bg_data = adata.var[var_key]
            elif background_key:
                bg_data = adata[:, adata.var[background_key]].var[var_key]

            # plot background histogram
            if bg_data is not None:
                hist_kws = {"range": (0, np.max(bg_data))}
                bins = 100
                sns.distplot(
                    bg_data,
                    kde=False,
                    label=background_key,
                    bins=bins,
                    hist_kws=hist_kws,
                )

            # plot selection histogram
            bins = 100
            sns.distplot(
                hist_data,
                kde=False,
                label=selection_label,
                bins=bins,
                hist_kws=hist_kws,
            )
            ax1.set_yscale("log")
            if j == 0:
                ax1.set_ylabel("number of genes")

            # check if penalty is there to plot
            if len(x_values_dict[selection_label]) <= j:
                plt.legend(prop={"size": legend_size}, bbox_to_anchor=[0.99, 0.99], loc="upper right", frameon=False)
                continue
            assert isinstance(penalty_key, str)

            # draw vertical lines at lower and upper border
            ax2 = ax1.twinx()
            if lower_borders is None:
                lower = None
            else:
                if lower_borders is False:
                    lower = False
                else:
                    assert isinstance(lower_borders, dict)
                    if penalty_key not in lower_borders:
                        lower = False

            if lower is None:
                # infer lower border from penanalty values
                idx_of_first_one = np.where(y_values_dict[selection_label][penalty_key] >= 0.999)[0][0]
                lower = x_values_dict[selection_label][penalty_key][idx_of_first_one] if idx_of_first_one > 0 else False
            if lower is not False:
                assert isinstance(lower, float)
                plt.axvline(x=lower, lw=0.5, ls="--", color="black")

            if upper_borders is None:
                upper = None
            else:
                if upper_borders is False:
                    upper = False
                else:
                    assert isinstance(upper_borders, dict)
                    if penalty_key not in upper_borders:
                        upper = False

            if upper is None:
                # infer lower border from penanalty values
                idx_of_last_one = np.where(y_values_dict[selection_label][penalty_key] > 0.999)[0][-1]
                upper = (
                    x_values_dict[selection_label][penalty_key][idx_of_last_one]
                    if idx_of_last_one < len(x_values_dict[selection_label][penalty_key]) - 1
                    else False
                )
            if upper is not False:
                assert isinstance(upper, float)
                plt.axvline(x=upper, lw=0.5, ls="--", color="black")

            # plot penalty kernel (interpolated if from penalty_key)
            plt.title(penalty_key)
            penalty_label = "penalty"
            if selection_label in penalty_labels:
                if penalty_key in penalty_labels[selection_label]:
                    penalty_label = penalty_labels[selection_label][penalty_key]
            plt.plot(
                x_values_dict[selection_label][penalty_key],
                y_values_dict[selection_label][penalty_key],
                label=penalty_label,
                color="green",
            )

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            plt.legend(
                h1 + h2,
                l1 + l2,
                prop={"size": legend_size},
                bbox_to_anchor=[0.99, 0.99],
                loc="upper right",
                frameon=False,
            )

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
    min_degree: int = 0,
    fontsize: int = 18,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """Plot overlap of gene sets

    Args:
        selection_df:
            Table with gene sets. Gene names are given in the index, gene sets are given as boolean columns.
        style:
            Plot type. Options are

                - "upset": upset plot
                - "venn": venn diagram

        min_degree:
            Only for `style="upset"`: minimum degree (number of categories intersected) of a subset to be shown in the
            plot.
        fontsize:
            Matplotlib fontsize.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

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
        upset_plot = UpSet(
            upset_data, subset_size="count", min_degree=min_degree, show_counts=True, sort_by="cardinality"
        )

        # draw figure
        fig = plt.figure()
        upset_plot.plot(fig=fig)

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


# def gene_overlap_grouped(
#     selection_df: Dict[str, pd.DataFrame],
#     groupby: str = "method",
#     show: bool = True,
#     save: Optional[str] = None,
# ):
#     """Plot the intersection of different selected gene sets grouped by the selection method.
#
#     Args:
#         selection_df:
#             Boolean dataframe with gene identifiers as index and one column for each gene set.
#         groupby:
#             Name of a column that categorizes the gene sets, eg. the method they were selected with.
#         show:
#             Whether to display the plot.
#         save:
#             Save the plot to path.
#
#     Returns:
#         Figure can be shown (default `True`) and stored to path (default `None`).
#         Change this with ``show`` and ``save``.
#
#     """
#
#     pass
#     # TODO


def clf_genes_umaps(
    adata: sc.AnnData,  # TODO: adjust type hint
    df: pd.DataFrame,
    basis: str = "X_umap",
    ct_key: str = "celltype",
    n_cols: int = 4,
    size_factor: float = 1,
    fontsize: int = 18,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    """Plot umaps of genes needed for cell type classification of each cell type.

    Args:
        adata:
            ``AnnData`` with ``ct_key`` in ``adata.obs`` and ``basis`` in ``adata.obsm``.
        df:
            Dataframe with genes as index and the following columns:

                - `'decision_celltypes'`: List of celltypes for which the gene is needed for classification.
                - `'rank'`: Forest classification rank of the gene.
                - `'importance_score'`: Importance score of the gene.

            Optional columns: TODO: do we need to list them here?
                - `'marker_celltypes'`: List of celltypes for which the gene is a marker according to the known marker
                    list. TODO: check if also DE "markers" alowed?
                - `'decision_title'`: Subplot title.
                - `'marker_title'`: Subplot title used if gene is marker. TODO: why decision_title AND marker_title?
                - `'decision_cmap'`: Matplotlib colormap.
                - `'marker_cmap'`: Matplotlib colormap used if gene is marker. TODO: why decision_cmap and marker_cmap (and why cmap at all...)

        basis:
            Name of the ``obsm`` embedding to use.
        ct_key:
            Column name in ``adata.obs`` where celltypes are stored.
        n_cols:
            Number of subplot columns.
        fontsize:
            Matplotlib fontsize.
        size_factor:
            Scale factor for figure width and height.
        show:
            Whether to display the plot.
        save:
            Save the plot to path.

    Returns:
        Figure can be shown (default `True`) and stored to path (default `None`).
        Change this with ``show`` and ``save``.
    """

    # prepare data
    a = adata.copy()
    celltypes = list(set([y for x in df["decision_celltypes"] for y in x]))
    subplots_decision = {ct: list(df.index[df["decision_celltypes"].apply(lambda x: ct in x)]) for ct in celltypes}
    subplots_marker = {ct: [] for ct in celltypes}
    if "marker_celltypes" in df:
        subplots_marker = {ct: list(df.index[df["marker_celltypes"] == ct]) for ct in celltypes}

    # set titles and palettes:
    if "decision_title" not in df:
        df["decision_title"] = [
            f"{gene}: \n" f"rank={int(df.loc[gene]['rank'])}, " f"imp.={round(df.loc[gene]['importance_score'], 2)}"
            for gene in df.index
        ]
    if "marker_title" not in df:
        df["marker_title"] = [
            f"marker: {gene}: \n"
            f"rank={int(df.loc[gene]['rank'])}, "
            f"imp.={round(df.loc[gene]['importance_score'], 2)}"
            for gene in df.index
        ]
    if "decision_palette" not in df:
        # cmap = colors.LinearSegmentedColormap.from_list("mycmap", ["grey", "lime"])
        df["decision_cmap"] = "viridis"  # cmap
    if "marker_palette" not in df:
        # cmap = colors.LinearSegmentedColormap.from_list("mycmap", ["grey", "orangered"])
        df["marker_cmap"] = "viridis"  # cmap

    # numbers of subplots in rows and columns
    n_decision_genes = [len(subplots_decision[ct] + subplots_marker[ct]) for ct in subplots_decision]
    n_subplots = [n + 1 for n in n_decision_genes]
    if n_cols is None:
        n_cols = max(n_subplots)
    rows_per_ct = [np.ceil(s / n_cols) for s in n_subplots]
    n_rows = int(sum(rows_per_ct))
    row_ceils = [int(np.ceil(s / r)) for s, r in zip(n_subplots, rows_per_ct)]
    n_cols = max(row_ceils)

    CT_FONTSIZE = fontsize + 4
    PPI = 72
    CT_PADDING = CT_FONTSIZE / PPI  # space above celltype (additional to HSPACE)

    HSPACE_INCHES = fontsize / PPI * n_rows * 3.5
    WSPACE_INCHES = fontsize / PPI * n_cols * 4
    TOP_INCHES = -CT_PADDING
    BOTTOM_INCHES = 3
    LEFT_INCHES = 3
    RIGHT_INCHES = 3
    CT_HEIGHT_INCHES = CT_PADDING + (CT_FONTSIZE / PPI)
    SUBPLOT_HEIGHT_INCHES = 3
    SUBPLOT_WIDTH_INCHES = 3

    FIGURE_WIDTH = (
        (SUBPLOT_WIDTH_INCHES * n_cols) + (((n_cols - 1) / n_cols) * WSPACE_INCHES) + RIGHT_INCHES + LEFT_INCHES
    ) * size_factor
    FIGURE_HEIGHT = (
        ((SUBPLOT_HEIGHT_INCHES + CT_HEIGHT_INCHES) * n_rows)
        + (((n_rows - 1) / n_rows) * HSPACE_INCHES)
        + TOP_INCHES
        + BOTTOM_INCHES
    ) * size_factor

    HSPACE = HSPACE_INCHES / FIGURE_HEIGHT
    WSPACE = WSPACE_INCHES / FIGURE_WIDTH
    TOP = 1 - (TOP_INCHES / FIGURE_HEIGHT)
    BOTTOM = BOTTOM_INCHES / FIGURE_HEIGHT
    RIGHT = 1 - (RIGHT_INCHES / FIGURE_WIDTH)
    LEFT = LEFT_INCHES / FIGURE_WIDTH
    SUBPLOT_HEIGHT = SUBPLOT_HEIGHT_INCHES / FIGURE_HEIGHT
    CT_HEIGHT = CT_HEIGHT_INCHES / FIGURE_HEIGHT

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # note that every second row is just for legends
    gs = GridSpec(n_rows * 2, n_cols, figure=fig, height_ratios=[CT_HEIGHT, SUBPLOT_HEIGHT] * n_rows)
    i = -1
    for ct in celltypes:
        i += 1
        j = 0

        # prepare data
        a.obs[ct] = adata.obs[ct_key].astype(str)
        a.obs.loc[a.obs[ct] != ct, ct] = "other"
        a.obs[ct] = a.obs[ct].astype("category")

        # first subplot is the umap colored by celltype
        ax = fig.add_subplot(gs[2 * i + 1, j])
        ax = sc.pl.embedding(
            adata=a,
            basis=basis,
            color=ct,
            show=False,
            ax=ax,
            title="",
            palette=["blue", "grey"],
            legend_fontweight="heavy",
        )
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.title.set_fontsize(fontsize)
        ax.get_legend().remove()

        # add celltype header
        celltype_ax = fig.add_subplot(gs[2 * i, j])
        celltype_ax.text(
            x=0,
            y=0,
            s=ct,
            weight="bold",
            verticalalignment="baseline",
            horizontalalignment="left",
            fontsize=CT_FONTSIZE,
        )
        celltype_ax.set_axis_off()

        # subplots for decision genes:
        for gene in subplots_decision[ct]:

            j += 1
            if j >= n_cols:
                i += 1
                j = 0

            # set styling
            ax = fig.add_subplot(gs[2 * i + 1, j])
            ax = sc.pl.embedding(
                adata=a,
                basis=basis,
                color=gene,
                show=False,
                ax=ax,
                title=df["decision_title"][gene],
                cmap=df["decision_cmap"][gene],
            )
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)
            ax.title.set_fontsize(fontsize)
            cbar = ax.collections[-1].colorbar
            cbar.ax.tick_params(labelsize=fontsize)

        # subplots for marker genes
        for gene in subplots_marker[ct]:

            j += 1
            if j >= n_cols:
                i += 1
                j = 0

            ax = fig.add_subplot(gs[2 * i + 1, j])
            ax = sc.pl.embedding(
                adata=a,
                basis=basis,
                color=gene,
                show=False,
                ax=ax,
                title=df.loc[gene]["marker_title"],
                cmap=df["marker_cmap"][gene],
            )
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)
            ax.title.set_fontsize(fontsize)
            cbar = ax.collections[-1].colorbar
            cbar.ax.tick_params(labelsize=fontsize)

    plt.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, hspace=HSPACE, wspace=WSPACE)
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight", transparent=True)
    plt.close()


def masked_dotplot(
    adata: sc.AnnData,  # TODO: adjust type hint
    selector,
    ct_key: str = "celltype",
    imp_threshold: float = 0.05,
    celltypes: Optional[List[str]] = None,
    n_genes: Optional[int] = None,
    comb_markers_only: bool = False,
    markers_only: bool = False,
    cmap: str = "Reds",
    comb_marker_color: str = "darkblue",
    marker_color: str = "blue",
    non_adata_celltypes_color: str = "grey",
    save: Union[bool, str] = False,
):
    """Create dotplot with additional annotation masks.

    Args:
        adata:
            AnnData with cell type annotations in `adata.obs[ct_key]`.
        selector:
            `ProbesetSelector` object with selected `selector.probeset`.
        ct_key:
            Column of `adata.obs` with cell type annotation.
        imp_threshold:
            Annotate genes as "Spapros marker" only for those genes with importance > ``imp_threshold``.
        celltypes:
            Optional subset of celltypes (rows of dotplot).
        n_genes:
            Optionally plot top ``n_genes`` genes.
        comb_markers_only:
            Whether to plot only genes that are "Spapros markers" for the plotted cell types. (can be combined with
            markers_only, in that case markers that are not Spapros markers are also shown)
        markers_only:
            Whether to plot only genes that are markers for the plotted cell types. (can be combined with
            ``comb_markers_only``, in that case Spapros markers that are not markers are also shown)
        cmap:
            Colormap of mean expressions.
        comb_marker_color:
            Color for "Spapros markers".
        marker_color:
            Color for marker genes.
        non_adata_celltypes_color:
            Color for celltypes that don't occur in the data set.
        save:
            If `True` or a `str`, save the figure.


    Example:

        (Takes a few minutes to calculate)

        .. code-block:: python

            import spapros as sp
            adata = sp.ut.get_processed_pbmc_data()
            selector = sp.se.ProbesetSelector(adata, "celltype", n=30, verbosity=0)
            selector.select_probeset()

            sp.pl.masked_dotplot(adata,selector)

        .. image:: ../../docs/plot_examples/masked_dotplot.png


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
