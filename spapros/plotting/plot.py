import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from spapros.util.util import plateau_penalty_kernel
import scipy.cluster.hierarchy as sch

# TODO: Fix explore_constraint plot. The following circular import is causing problems atm:
#
# from spapros.selection.selection_methods import select_pca_genes
#
#def explore_constraint(adata, factors=None, q=0.99, lower=0, upper=1):
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
    
    pooled = pd.concat(conf_mats,axis=1)
    
    pairwise_distances = sch.distance.pdist(pooled)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
    idx = np.argsort(idx_to_cluster_array)

    ordered_mats = [conf_mat.copy().iloc[idx, :].T.iloc[idx, :].T for conf_mat in conf_mats]
    return ordered_mats
    
def confusion_heatmap(set_ids,conf_matrices,ordered=True,show=True,save=False,size_factor=6,n_cols=2):
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
    n_rows = (n_plots//n_cols) + int((n_plots%n_cols) > 0)
    
    if ordered:
        if isinstance(ordered,bool):
            tmp_matrices = [conf_matrices[set_id].copy() for set_id in set_ids]
            cms = ordered_confusion_matrices(tmp_matrices)
        elif isinstance(ordered,list):
            cms = [conf_matrices[set_id].copy().loc[ordered,ordered] for set_id in set_ids]
    else:
        cms = [conf_matrices[set_id].copy() for set_id in set_ids]
    
    fig = plt.figure(figsize=(size_factor*n_cols,0.75*size_factor*n_rows))    
    for i,set_id in enumerate(set_ids):
        ax = plt.subplot(n_rows,n_cols,i+1)
        sns.heatmap(cms[i],cmap="OrRd",cbar=False,ax=ax,vmin=0,vmax=1,annot=True,fmt=".2f")
        #sns.heatmap(cms[i],cmap="OrRd",cbar=(i == (len(set_ids)-1)),ax=ax,vmin=0,vmax=1,annot=True,fmt=".2f")
        if i == 0:
            plt.tick_params(axis='both', which='major', bottom=False, labelbottom = False, top=True, labeltop=True)
        elif ((i % n_cols) == 0):
            plt.tick_params(axis='both', which='major', bottom=False, labelbottom = False, top=False, labeltop=False,
                            left=True,labelleft=True)
        elif ((i // n_cols) < 1):
            plt.tick_params(axis='both', which='major', bottom=False, labelbottom = False, top=True, labeltop=True, 
                            left=False,labelleft=False)
        else:
            plt.tick_params(axis='both', which='major', bottom=False, labelbottom = False, top=False, labeltop=False, 
                            left=False,labelleft=False)            
            
        plt.title(set_id)
    #plt.subplots_adjust(top=1.54, bottom=0.08, left=0.05, right=0.95, hspace=0.20, wspace=0.25)
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches = "tight")
    plt.close()     
    
    
def correlation_matrix(set_ids,cor_matrices,show=True,save=False,size_factor=6,n_cols=5):
    """
    set_ids: list
        List of probe set ids.
    cor_matrices: dict of pd.DataFrames
        Correlation matrix of each probe set given in `set_ids`.
    """
    
    n_plots = len(set_ids)
    n_rows = (n_plots//n_cols) + int((n_plots%n_cols) > 0)
    
    fig = plt.figure(figsize=(size_factor*n_cols,0.75*size_factor*n_rows))    
    for i,set_id in enumerate(set_ids):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(cor_matrices[set_id].values, cmap='seismic',vmin=-1,vmax=1)
        plt.title(set_id)
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)        
    plt.subplots_adjust(top=1.54, bottom=0.08, left=0.05, right=0.95, hspace=0.20, wspace=0.25)
    if show:
        plt.show()
    if save:
        fig.savefig(save, bbox_inches = "tight")
    plt.close()    
    
    
    

def summary_table(table,summaries="all",color_maps={},rename_cols={},rename_rows={},show=True,save=False):
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
    
    """

    fsize = 15    
    
    # Default order and colors
    default_order = ["cluster_similarity", "knn_overlap","Greens","forest_clfs","marker_corr","gene_corr"]
    default_cmaps = {
        "cluster_similarity":"Greens",        
        "knn_overlap":"Greens",
        "forest_clfs":"Reds",
        "marker_corr":"Reds",
        "gene_corr":"Blues",        
    }    
    
    if summaries == "all":
        summaries = table.columns.tolist()
        # Order by default order of metrics and length of summary
        summaries.sort(key = lambda s: default_order.index(s.split()[0])*100 + len(s))  
    
    cmaps = {}
    for summary in summaries:
        if (summary in color_maps):
            cmaps[summary] = color_maps[summary]
        else:
            cmaps[summary] = default_cmaps[summary.split()[0]]               
    
    df = table[summaries].copy()
    
    df = df.rename(columns=rename_cols, index=rename_rows)
    
    for summary, new_key in rename_cols.items():
        cmaps[new_key] = cmaps.pop(summary)
    
    n_cols = len(df.columns)
    n_sets = len(df.index)

    fig = plt.figure(figsize = (n_cols*1.1,n_sets))
    gs1 = gridspec.GridSpec(1, n_cols)
    gs1.update(wspace=0.0, hspace=0.0)

    for i,col in enumerate(df.columns):    
        ax = plt.subplot(gs1[i])      
        
        yticklabels=bool(i == 0)

        annot = True
        fmt = '.2f'
        ax1 = sns.heatmap(df[[col]],
                          cmap=cmaps[col], 
                          annot=annot, 
                          cbar=False,
                          square=True,
                          yticklabels=yticklabels, 
                          fmt=fmt, 
                          annot_kws={"fontsize":fsize-2}
                         )
        plt.tick_params(axis='x', 
                        which='major', 
                        labelsize=fsize, 
                        labelbottom = False, 
                        bottom=False, 
                        top = True, 
                        labeltop=True
                       )
        plt.tick_params(axis='y', which='major', labelsize=fsize)
        ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='left',rotation=45)  
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if show:
        plt.show()
    if save:
        fig.savefig(save,bbox_inches='tight')
    plt.close()