import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from spapros.plotting._masked_dotplot import MaskedDotPlot
import itertools

# from spapros.util.util import plateau_penalty_kernel

# TODO: Fix explore_constraint plot. The following circular import is causing problems atm:
#
# from spapros.selection.selection_methods import select_pca_genes
#
# def explore_constraint(adata, factors=None, q=0.99, lower=0, upper=1):
#    """Plot histogram of quantiles for selected genes for different penalty kernels
#
#    How to generalize the plotting function, support:
#    - any selection method with defined hyperparameters
#    - any penalty kernel
#    - any key to be plotted (not only quantiles)
#    """
#    if factors is None:
#        factors = [10, 1, 0.1]
#    legend_size = 9
#    factors = [10, 1, 0.1]
#    rows = 1
#    cols = len(factors)
#    sizefactor = 6
#
#    gaussians = []
#    a = []
#    selections_tmp = []
#    for i, factor in enumerate(factors):
#        x_min = lower
#        x_max = upper
#        var = [factor * 0.1, factor * 0.5]
#        gaussians.append(plateau_penalty_kernel(var=var, x_min=x_min, x_max=x_max))
#
#        a.append(adata.copy())
#
#        a[i].var["penalty_expression"] = gaussians[i](a[i].var[f"quantile_{q}"])
#        selections_tmp.append(
#            select_pca_genes(
#                a[i],
#                100,
#                variance_scaled=False,
#                absolute=True,
#                n_pcs=20,
#                process_adata=["norm", "log1p", "scale"],
#                penalty_keys=["penalty_expression"],
#                corr_penalty=None,
#                inplace=False,
#                verbose=True,
#            )
#        )
#        print(f"N genes selected: {np.sum(selections_tmp[i]['selection'])}")
#
#    plt.figure(figsize=(sizefactor * cols, 0.7 * sizefactor * rows))
#    for i, factor in enumerate(factors):
#        ax1 = plt.subplot(rows, cols, i + 1)
#        hist_kws = {"range": (0, np.max(a[i].var[f"quantile_{q}"]))}
#        bins = 100
#        sns.distplot(
#            a[i].var[f"quantile_{q}"],
#            kde=False,
#            label="highly_var",
#            bins=bins,
#            hist_kws=hist_kws,
#        )
#        sns.distplot(
#            a[i][:, selections_tmp[i]["selection"]].var[f"quantile_{q}"],
#            kde=False,
#            label="selection",
#            bins=bins,
#            hist_kws=hist_kws,
#        )
#        plt.axvline(x=x_min, lw=0.5, ls="--", color="black")
#        plt.axvline(x=x_max, lw=0.5, ls="--", color="black")
#        ax1.set_yscale("log")
#        plt.legend(prop={"size": legend_size}, loc=[0.73, 0.74], frameon=False)
#        plt.title(f"factor = {factor}")
#
#        ax2 = ax1.twinx()
#        x_values = np.linspace(0, np.max(a[i].var[f"quantile_{q}"]), 240)
#        plt.plot(x_values, 1 * gaussians[i](x_values), label="penal.", color="green")
#        plt.legend(prop={"size": legend_size}, loc=[0.73, 0.86], frameon=False)
#        plt.ylim([0, 2])
#        for label in ax2.get_yticklabels():
#            label.set_color("green")
#    plt.show()


def ordered_confusion_matrices(conf_mats):
    """Rearranges confusion matrices by a linkage clustering

    The matrices in conf_mats must have the same indices (and columns). We calculate the clustering on
    a concatenated list of confusion matrices and then reorder each matrix by the resulting order.

    Parameters
    ----------
    conf_mats : list of pd.DataFrame

    Returns
    -------
    list of pd.DataFrame
        Reordered confusion matrices
    """

    pooled = pd.concat(conf_mats, axis=1)

    pairwise_distances = sch.distance.pdist(pooled)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
    idx = np.argsort(idx_to_cluster_array)

    ordered_mats = [conf_mat.copy().iloc[idx, :].T.iloc[idx, :].T for conf_mat in conf_mats]
    return ordered_mats


def confusion_heatmap(
    set_ids, 
    conf_matrices, 
    ordered=True, 
    show=True, 
    save=False, 
    size_factor=6, 
    n_cols=2, 
    rotate_x_labels=True
):
    """Plot heatmap of cell type classification confusion matrices

    set_ids: list
        List of probe set ids.
    conf_matrices: dict of pd.DataFrames
        Confusion matrix of each probe set given in `set_ids`.
    ordered: bool or list
        If set to True a linkage clustering is computed to order cell types together that are hard to distinguish.
        If multiple set_ids are provided the same order is applied to all of them.
        Alternatively provide a list with a custom order.

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
        sns.heatmap(cms[i], cmap="OrRd", cbar=False, ax=ax, vmin=0, vmax=1, annot=True, fmt=".2f")
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
    # plt.subplots_adjust(top=1.54, bottom=0.08, left=0.05, right=0.95, hspace=0.20, wspace=0.25)
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight")
    plt.close()


def correlation_matrix(set_ids, cor_matrices, show=True, save=False, size_factor=6, n_cols=5):
    """
    set_ids: list
        List of probe set ids.
    cor_matrices: dict of pd.DataFrames
        Correlation matrix of each probe set given in `set_ids`.
    """

    n_plots = len(set_ids)
    n_rows = (n_plots // n_cols) + int((n_plots % n_cols) > 0)

    fig = plt.figure(figsize=(size_factor * n_cols, 0.75 * size_factor * n_rows))
    for i, set_id in enumerate(set_ids):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(cor_matrices[set_id].values, cmap="seismic", vmin=-1, vmax=1)
        plt.title(set_id)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(top=1.54, bottom=0.08, left=0.05, right=0.95, hspace=0.20, wspace=0.25)
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches="tight")
    plt.close()


def format_time(time):
    """
    time: float
        in seconds.

    Return
    ------
    str
        formatted time
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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def summary_table(
    table,
    summaries="all",
    color_maps={},
    rename_cols={},
    rename_rows={},
    time_format=[],
    log_scale=[],
    color_limits={},
    nan_color='lightgrey',
    threshold_ann={},
    show=True,
    save=False,
):
    """Plot table of summary statistics

    table: pd.DataFrame

    summaries: "all" or list of strs
        List of summary metrics that are plotted.
    color_maps: dict
        Color maps assigned to summary metrics. Use the initial name and not the potential new name
        given via `rename_cols`.
    rename_cols: dict
        Rename summary metrics for plot.
    rename_rows: dict
        Rename set ids.
    time_format: list of strs
        Summary names that are formatted to days, hours, mins and secs (seconds are expected as input).
    log_scale: list of strs
        Summary names for which a log scaled colormap is applied.
    color_limits: dict of lists of two floats
        For each summary metric optionally provide vmin and vmax for the colormap.
    nan_color: str
        Color for nan values.
    threshold_ann: dict
        Special annotation for values above defined threshold. E.g. {"time":{"th":1000,"above":True,"ann":"> 1k"}}

    """

    fsize = 15

    # Default order and colors
    default_order = ["cluster_similarity", "knn_overlap", "Greens", "forest_clfs", "marker_corr", "gene_corr",
                     "penalty"]
    default_cmaps = {
        "cluster_similarity": "Greens",
        "knn_overlap": "Greens",
        "forest_clfs": "Purples",#"Reds",
        "marker_corr": "Purples",#"Reds",
        "gene_corr": "Blues",
        "penalty": truncate_colormap(plt.get_cmap('Greys'), minval=0.05, maxval=0.7, n=100), #"Greys",
        "other": "Greys",
    }

    if summaries == "all":
        summaries = table.columns.tolist()
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

        if col in multi_col: # TODO: time formating multi col support? Better get col pos at the beginning + iloc
            col_pos = [i for i, c in enumerate(df.columns) if c == col][multi_col[col]]
            color_vals = np.log(df.iloc[:,[col_pos]]) if (col in log_scale) else df.iloc[:,[col_pos]]
            multi_col[col] += 1
        else:
            color_vals = np.log(df[[col]]) if (col in log_scale) else df[[col]]
            col_pos = [i for i, c in enumerate(df.columns) if c == col][0]
            
        if col in time_format:
            #annot = df[col].apply(format_time).values[:, np.newaxis]
            annot = df.iloc[:,col_pos].apply(format_time).values[:, np.newaxis]
            fmt = ""
        else:
            annot = True
            fmt = ".2f"
        if col in threshold_ann:
            formatter = lambda s: f"{s:.2f}"
            annot = df.iloc[:,col_pos].apply(formatter).values[:, np.newaxis] if isinstance(annot,bool) else annot
            tmp = threshold_ann[col]            
            th_mask = (df.iloc[:,col_pos] > tmp["th"]) if tmp["above"] else (df.iloc[:,col_pos] < tmp["th"])
            annot[th_mask,:] = tmp["ann"]
            fmt=""

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
        fig.savefig(save, bbox_inches="tight")
    plt.close()

    
    
def masked_dotplot(
    adata, 
    selector, 
    ct_key="celltype", 
    imp_threshold = 0.05,
    celltypes=None, 
    n_genes=None, 
    comb_markers_only=False,
    markers_only=False,
    cmap="Reds",
    comb_marker_color="darkblue",
    marker_color="green",
    non_adata_celltypes_color="grey",
    save=None,
):
    """
    Arguments
    ---------
    adata
        AnnData with adata.obs[ct_key] cell type annotations
    selector
        ProbesetSelector object with selected selector.probeset
    ct_key
    imp_threshold
        Show genes as combinatorial marker only for those genes with importance > `imp_threshold`
    celltypes
        Optional subset of celltypes (rows of dotplot)
    n_genes
        Optionally plot top `n_genes` genes.
    comb_markers_only
        Whether to plot only genes that are combinatorial markers for the plotted cell types. (can be combined with 
        markers_only, in that case markers that are not comb markers are also shown)
    markers_only
        Whether to plot only genes that are markers for the plotted cell types. (can be combined with comb_markers_only,
        in that case comb markers that are not markers are also shown)
    cmap
        Colormap of mean expressions
    comb_marker_color
        Color for combinatorial markers
    marker_color
        Color for marker genes            
    non_adata_celltypes_color
        Color for celltypes that don't occur in the data set
    save
        Save figure to path
    """
    
    if isinstance(selector,str):
        selector = select.ProbesetSelector(adata,ct_key,save_dir=selector)
        # TODO: think the last steps of the ProbesetSelector are still not saved..., needs to be fixed.
    
    # celltypes, possible origins:
    # - adata.obs[ct_key] (could include cts not used for selection)
    # - celltypes for selection (including markers, could include cts which are not in adata.obs[ct_key])
    # --> pool all together... order?
    
    if celltypes is not None:
        cts = celltypes
        a = adata[adata.obs[ct_key].isin(celltypes)].copy()
        #a.obs[ct_key] = a.obs[ct_key].astype(str).astype("category")
    else:
        # Cell types from adata
        cts = adata.obs[ct_key].unique().tolist()
        # Cell types from marker list only
        if 'celltypes_marker' in selector.probeset:
            tmp = []
            for markers_celltypes in selector.probeset['celltypes_marker'].str.split(','):
                tmp += markers_celltypes
            tmp = np.unique(tmp).tolist()
            if '' in tmp:
                tmp.remove('')
            cts += [ct for ct in tmp if ct not in cts]
        a = adata
    
    # Get selected genes that are also in adata
    selected_genes = [g for g in selector.probeset[selector.probeset["selection"]].index.tolist() if g in adata.var_names]
    
    # Get tree genes
    tree_genes = {}
    for ct,importance_tab in selector.forest_results['forest'][2].items():
        if ct in cts:
            tree_genes[ct] = importance_tab['0'].loc[importance_tab['0']>imp_threshold].index.tolist()
            tree_genes[ct] = [g for g in tree_genes[ct] if g in selected_genes]
        
    # Get markers
    marker_genes = {ct:[] for ct in (cts)}
    for ct in (cts):
        for gene in selector.probeset[selector.probeset["selection"]].index:
            if ct in selector.probeset.loc[gene,'celltypes_marker'].split(',') and (gene in adata.var_names):
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
        selected_genes = selected_genes[:min(n_genes,len(selected_genes))]
    # Filter (combinatorial) markers by genes that are not in the selected genes
    for ct in cts:
        marker_genes[ct] = [g for g in marker_genes[ct] if g in selected_genes]
    for ct in tree_genes.keys():
        tree_genes[ct] = [g for g in tree_genes[ct] if g in selected_genes]        
    
    dp = MaskedDotPlot(a,
                       var_names=selected_genes,
                       groupby=ct_key,
                       tree_genes=tree_genes,
                       marker_genes=marker_genes,
                       further_celltypes=[ct for ct in cts if ct not in adata.obs[ct_key].unique()],
                       cmap = cmap,
                       tree_genes_color = comb_marker_color,
                       marker_genes_color = marker_color,
                       non_adata_celltypes_color = non_adata_celltypes_color,
                       )
    dp.make_figure()
    if save:
        plt.gcf().savefig(save, bbox_inches="tight")