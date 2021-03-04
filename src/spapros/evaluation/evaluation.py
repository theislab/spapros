import csv
import os
import pickle
import time
import warnings
from pathlib import Path

import anndata as ann
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
from sklearn import tree
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm


def plot_gene_expressions(f_idxs, adata, fig_title=None, save_to=None):
    a = adata.copy()
    gene_obs = []
    for _, f_idx in enumerate(f_idxs):
        gene_name = a.var.index[f_idx]
        a.obs[gene_name] = a.X[:, f_idx]
        gene_obs.append(gene_name)

    fig = sc.pl.umap(a, color=gene_obs, ncols=4, return_fig=True)
    fig.suptitle(fig_title)
    if save_to is not None:
        fig.savefig(save_to)
    plt.show()


##########################################################
####### Helper functions for clustering_sets() ###########
##########################################################


def get_found_ns(csv_file):
    found_ns = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(f, None)  # skip header # TODO: test
        for row in reader:
            found_ns.append(int(row[0]))
    return found_ns


def write_assignments_to_csv(n, resolution, assignments, csv_file):
    new_row = [n, resolution] + assignments
    with open(csv_file, "a") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(new_row)


def write_tried_res_n_to_csv(n, resolution, csv_file):
    new_row = [resolution, n]
    with open(csv_file, "a") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(new_row)


def create_leiden_results_csv(csv_file, adata):
    header = ["n", "resolution"] + list(adata.obs.index)
    with open(csv_file, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(header)
    with open(csv_file[:-4] + "_tried.csv", "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["resolution", "n"])


def get_tried_res_n(csv_file):
    tried_res_n = []
    if os.path.isfile(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(f, None)  # skip header
            for row in reader:
                tried_res_n.append([float(row[0]), int(row[1])])
        tried_res_n.sort(key=lambda x: x[0])
    return tried_res_n


def compute_clustering(adata, ns, resolution, tried_res_n, csv_file, progress_bar=None):
    """Compute leiden clustering for the given resolution, save result if new n was found, and update the variables."""
    sc.tl.leiden(adata, resolution=resolution, key_added="tmp")
    n = len(set(adata.obs["tmp"]))
    tried_res_n.append([resolution, n])
    tried_res_n.sort(key=lambda x: x[0])  # sort list of list by resolutions
    found_ns = get_found_ns(csv_file)
    write_tried_res_n_to_csv(n, resolution, csv_file[:-4] + "_tried.csv")
    if n not in found_ns:
        write_assignments_to_csv(n, resolution, list(adata.obs["tmp"]), csv_file)
        if n in ns:
            progress_bar.update(n=1)


##########################################################
################# clustering_sets() ######################
##########################################################


def clustering_sets(adata, ns, save_to, start_res=1.0, verbose=False):
    """Compute leiden clusters for different numbers of clusters

    Leiden clusters are calculated with different resolutions.
     A search (similar to binary search) is applied to find the right resolutions for all defined n's.

    Arguments
    ---------
    adata: anndata object
        adata object with data to compute clusters on. Need to include a
        neighbors graph (and PCA?)  TODO: make this clear.
    ns: list of ints
        list of numbers of clusters
    save_to: str
        path to save results (e.g. /path/to/file.csv)
    start_res: float
        resolution to start computing clusterings.
    verbose: bool
        if True a progress bar is shown

    Return
    ------
    nothing

    Save
    ----
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
    csv file (at path, save_to[:-4]+"_tried.csv")
        this file includes all resolutions that were tried, some resolution lead to same
        numbers of clusters that have already been observed, such resolutions are not saved in
        the first csv file. Therefore the second is added, such that results can be easily
        extended if the function is called again. File structure:
        1nd column refers to the resolution used to calculate the clustering
        2st column refers to the number of clusters,
        e.g.
        res   , n
        1.    , 6
        0.5   , 4
        0.25  , 2
        0.375 , 2
        0.4375, 3
        .
        .
        .
        (note: this listing keeps the order of the conducted trials)
    """
    tried_res_n = get_tried_res_n(save_to[:-4] + "_tried.csv")  # list of [resolution,n] lists

    if verbose:
        bar = tqdm(total=len(ns))
        if len(tried_res_n) > 0:
            bar.n = len(set(ns).intersection(set([res_n[1] for res_n in tried_res_n])))
            bar.last_print_n = len(set(ns).intersection(set([res_n[1] for res_n in tried_res_n])))
            bar.refresh()
    else:
        bar = None

    if not os.path.isfile(save_to):
        save_dir = save_to.rsplit("/", 1)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        create_leiden_results_csv(save_to, adata)

    n_min = np.min(ns)
    n_max = np.max(ns)
    if not len(tried_res_n) > 0:
        res = start_res
        compute_clustering(adata, ns, res, tried_res_n, save_to, progress_bar=bar)

    # search for lower resolution border
    res = np.min([res_n[0] for res_n in tried_res_n])
    while np.min(np.unique([res_n[1] for res_n in tried_res_n])) > n_min:
        res *= 0.5
        compute_clustering(adata, ns, res, tried_res_n, save_to, progress_bar=bar)

    # search for higher resolution border
    res = np.max([res_n[0] for res_n in tried_res_n])
    while np.max([res_n[1] for res_n in tried_res_n]) < n_max:
        res *= 2
        compute_clustering(adata, ns, res, tried_res_n, save_to, progress_bar=bar)

    # search missing n's between neighbouring found n's
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
                compute_clustering(adata, ns, res, tried_res_n, save_to, progress_bar=bar)
                found_space = True


############## Helper function for nmi () ###################


def get_assignments(csv_file, n):
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(f, None)  # skip header
        for row in reader:
            if int(row[0]) == n:
                return [int(val) for val in row[2:]]


####################### NMI () ##############################


def nmi(
    files,
    reference_file,
    save_to,
    ns,
    method="arithmetic",
    names=None,
    verbose=True,
    save_every=10,
):
    """Compute NMI between sets  of clusterings and a reference set of clusterings.

    For different numbers of clusters (`ns`) the normalized mutual information
    NMI based on 2 different cluster assignments are computed. This is done for
    all cluster assignments of each element in files wrt the assignment
    in reference_file with the according `n`.
    Results are saved to a csv file. If a csv file already exists nmis are calculated
    for missing sets. Existing results are not deleted.

    Parameters
    ----------
    files: list of strs
        file pathes to csv files that contain clusterings for different
        numbers of clusters. Each file refers to one set of selected
        features. file structure:
        n , name1, name2, name3, ...
        2 ,   0  ,   0  ,   1  , ...
        4 ,   1  ,   0  ,   3  , ...
        8 ,   7  ,   7  ,   2  , ...
        3 ,   2  ,   1  ,   0  , ...
        (each row is a list of cluster assignments)
    reference_file: str
        file path to reference file (same file structure as in groups)
    save_to: str
        path to output file.
    ns: list of ints
        list of numbers of clusters for which NMIs are computed.
    method:
        NMI implementation
            'max': scikit method with `average_method='max'`
            'min': scikit method with `average_method='min'`
            'geometric': scikit method with `average_method='geometric'`
            'arithmetic': scikit method with `average_method='arithmetic'`
            TODO: implement the following (see comment below and scib)
            'Lancichinetti': implementation by A. Lancichinetti 2009 et al.
            'ONMI': implementation by Aaron F. McDaid et al. (https://github.com/aaronmcdaid/Overlapping-NMI) Hurley 2011
    names: list of strs
        Optionally provide a list with names refering to groups. The saved file
        has those names as column names. If names is None (default) the
        file names of files are used.
    save_every: int
        save results after every `save_every` minutes

    Return
    ------
    nothing

    Save
    ----
    csv file (pandas dataframe)
        dataframe of NMI results. Structure of dataframe:
             n (index),     name1  ,     name2  , ....
             1        ,     1.0    ,     1.0    , ....
             2        ,     0.9989 ,     0.9789 , ....
             ...
    """

    if save_every not in None:
        start = time.time()
        minute_count = 0

    if names is None:
        names = []
        for file in files:
            tmp = file.rsplit("/")[-1]
            tmp = tmp.rsplit(".", 1)[0]
            names.append(tmp)

        # Create nmi dataframe or extent existing with NaN values for nmis to calculate
    if (save_to is not None) and os.path.isfile(save_to):
        nmis = pd.read_csv(save_to, index_col=0)
        # add NaN cols
        for name in [s for s in names if s not in nmis.columns]:
            nmis[name] = np.nan
        # add NaN rows
        old_ns = list(nmis.index)
        all_ns = old_ns + [n for n in ns if n not in old_ns]
        nmis.reindex(all_ns)
    else:
        save_dir = save_to.rsplit("/", 1)[0]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        nmis = pd.DataFrame(np.nan, index=ns, columns=names)

    # Search which nmis can be calculated with given files
    result_exists = pd.DataFrame(False, index=nmis.index, columns=nmis.columns)
    ref_ns_full = get_found_ns(reference_file)
    ref_ns = [n for n in ref_ns_full if n in ns]
    for i, name in enumerate(names):
        found_ns = get_found_ns(files[i])
        valid_ns = [n for n in found_ns if n in ref_ns]
        result_exists.loc[result_exists.index.isin(valid_ns), name] = True

    for n in ref_ns:
        ref_labels = get_assignments(reference_file, n)
        for i, name in enumerate(names):
            if np.isnan(nmis.loc[n, name]) and result_exists.loc[n, name]:
                labels = get_assignments(files[i], n)
                if len(ref_labels) != len(labels):
                    raise ValueError(
                        f"different lengths in {file} - ({len(ref_labels)}) and reference - ({len(labels)})"
                    )
                if method in ["max", "min", "geometric", "arithmetic"]:
                    from sklearn.metrics import normalized_mutual_info_score

                    nmi_value = normalized_mutual_info_score(labels, ref_labels, average_method=method)
                else:
                    raise ValueError(f"Method {method} not valid")
                nmis.at[n, name] = nmi_value

                if save_every is not None:
                    atm = time.time()
                    minutes, seconds = divmod(atm - start, 60)
                    if minutes // save_every > minute_count:
                        minute_count += 1
                        nmis.to_csv(save_to)
    nmis.to_csv(save_to)


def plot_nmis(results_path, cols=None, colors=None, labels=None, legend=None):
    """Custom legend: e.g. legend = [custom_lines,line_names]

    custom_lines = [Line2D([0], [0], color='red',    lw=linewidth),
                    Line2D([0], [0], color='orange', lw=linewidth),
                    Line2D([0], [0], color='green',  lw=linewidth),
                    Line2D([0], [0], color='blue',   lw=linewidth),
                    Line2D([0], [0], color='cyan',   lw=linewidth),
                    Line2D([0], [0], color='black',  lw=linewidth),
                    ]
    line_names = ["dropout", "dropout 1 donor", "pca", "marker", "random"]
    """
    df = pd.read_csv(results_path, index_col=0)
    fig = plt.figure(figsize=(10, 6))
    if cols is not None:
        for col in df.columns:
            plt.plot(df.index, df[col], label=col)
    else:
        for i, col in enumerate(cols):
            if labels is None:
                label = col
            else:
                label = labels[i]
            if colors is None:
                plt.plot(df.index, df[col], label=label)
            else:
                plt.plot(df.index, df[col], label=label, c=colors[i])
    if legend is None:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(
            legend[0],
            legend[1],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
    plt.xlabel("n clusters")
    plt.ylabel("NMI")
    plt.show()
    return fig


def NMI_AUCs(results_path, sections=None, full_AUC=True):
    """Calculate the AUC of NMI vs n clusters plots

    Note that the function assumes that NMI values are given for n's with
    stepsize of one 1 between the n values. Otherwise a factor of the step
    size would make the results comparable. (if stepsizes change a n value
    specific factor needs to be taken into account...)

    Arguments
    ---------
    results_path: str
        path to csv file with NMI results (created with evaluate_nmi.py script).
    sections: list of floats
        The interval of n values is divided in sections defined by the thresholds
        given in sections. E.g.
            ns = [1,2,3,...,300]
            sections = [50, 100, 200]
            AUCs are separately calculated for
                1. ns = [1,2, ..., 50]
                2. ns = [51,52, ..., 100]
                3. ns = [101,102, ..., 200]
                4. ns = [201,202, ..., 300]
    full_AUC: bool
        if True the AUC over the full set of n is calculated. This is interesting
        if you want to calculate AUCs on sections but also on the full set.

    """
    df = pd.read_csv(results_path, index_col=0)

    ns = list(df.index)
    sects = []
    if full_AUC:
        sects.append([np.min(ns), np.max(ns)])
    if sections is not None and len(sections) > 0:
        sections.sort()
        for i, s in enumerate(sections):
            if i == 0:
                sects.append([np.min(ns), s])
            else:
                sects.append([sections[i - 1] + 1, s])
        sects.append([sections[-1] + 1, np.max(ns)])

    for i, interval in enumerate(sects):
        if i == 0:
            tmp = df.loc[(df.index >= interval[0]) & (df.index <= interval[1])].mean()
            # print(list(tmp))
            AUCs = pd.DataFrame([list(tmp)], columns=tmp.index, index=[f"{interval[0]}-{interval[1]}"])
        else:
            tmp = df.loc[(df.index >= interval[0]) & (df.index <= interval[1])].mean()
            tmp.name = f"{interval[0]}-{interval[1]}"
            AUCs = AUCs.append(tmp)  # list(tmp))

    return AUCs


##########################################################
################## knn_similarity() ######################
##########################################################


def neighbors_csv(adata, k, csv_file):
    """Save nearest neighbors of each cell in a csv file"""
    if "distances" in adata.obsp:
        neighbors = adata.obsp["distances"]
    elif "distances" in adata.uns["neighbors"]:
        neighbors = adata.uns["neighbors"]["distances"]
    rows, cols = neighbors.nonzero()
    k_nns = {}
    for r in range(adata.n_obs):
        k_nns[str(r)] = []
    for i in range(len(rows)):
        k_nns[str(rows[i])].append(cols[i])
    max_k = 0
    for r in k_nns:
        if len(k_nns[r]) > max_k:
            max_k = len(k_nns[r])

    header = ["cell_idx", "n_neighbors"] + [f"neighbor_{i}" for i in range(max_k)]
    with open(csv_file, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(header)
        for cell in k_nns:
            row = [cell, len(k_nns[cell])] + k_nns[cell]
            csv_writer.writerow(row)

        # def compute_similarity(csv_file_1,csv_file_2):


# 	"""Compute len(intersection) of two sets of k nearest neighbors for each cell
#
# 	"""
# 	with open(csv_file_1, 'r') as f1, open(csv_file_2, 'r') as f2:
# 		reader1 = csv.reader(f1, delimiter=",")
# 		reader2 = csv.reader(f2, delimiter=",")
# 		# skip headers
# 		next(f1, None)
# 		next(f2, None)
# 		common_neighbors = []
# 		for row1,row2 in zip(reader1,reader2):
# 			set1 = set([int(val) for val in row1[2:]])
# 			set2 = set([int(val) for val in row2[2:]])
# 			common_neighbors.append(len(set1.intersection(set2)))
# 	return common_neighbors


def compute_similarity(csv_file_1, csv_file_2):
    """Compute overlap of two sets of k nearest neighbors for each cell

    The overlap is the len(intersection) of both sets. To make the values comparable
    in a reasonable way they are divided by the number of neighbors in the smaller set.

    """
    with open(csv_file_1, "r") as f1, open(csv_file_2, "r") as f2:
        reader1 = csv.reader(f1, delimiter=",")
        reader2 = csv.reader(f2, delimiter=",")
        # skip headers
        next(f1, None)
        next(f2, None)
        overlaps = []
        for row1, row2 in zip(reader1, reader2):
            set1 = set([int(val) for val in row1[2:]])
            set2 = set([int(val) for val in row2[2:]])
            max_intersection = np.min([len(set1), len(set2)])
            overlaps.append(len(set1.intersection(set2)) / max_intersection)
    return overlaps


def knn_similarity(
    gene_set,
    reference,
    ks=[15],
    save_dir=None,
    save_name=None,
    reference_dir=None,
    reference_name=None,
    bbknn_ref_key=None,
):
    """Compute the nr of identical neighbors for each cell

    For each `k` in `ks` neighbor graphs for reference adata and adata[:,gene_set] are calculated.
    If results for a given `k` already exist they are not recalculated.
    For each cell the number of identical neighbors in both neighbor graphs are calculated.
    Neighbor graphs are calculated on 50 PCA components

    Parameters
    ----------
    gene_set: list
        list of genes or int indices for selected genes
    reference: anndata or str
        Either reference (full gene set) anndata object or path to .h5ad file
    ks: list
        list of ints. Number of neighbors for which neighbor graphs are calculated
    save_dir: str
        path where files and results for `gene_set` are saved
    save_name: str
        name that is added to saved neighbors of `gene_set`'s neighbor graph and similarity results.
    reference_dir: str
        directory where neighbor info files of `reference` are saved
    reference_name: str
        name for saving reference files
    bbknn_ref: None or str
        If str: calculate the neighbors graph for the reference with bbknn on provided batch key

    Save
    ----
    for each k in `ks`:
    1. csv file of `gene_set`'s neighbors at path = <save_dir> + f"nns_k{k}" + <save_name> + ".csv"
        Structure of file:
        cell_idx,  n_neighbors, neighbor_0, neighbor_1, ..., neighbor_<n_neighbors-1>
            0   ,      14     ,    307    ,    432    , ...,      30356
            1   ,      14     ,    256    ,    289    , ...,      43220
        ...
        (Caution! rows can be longer than the original k-1 because the neighbors search algorithm
        finds more than k-1, happens super rarely, idk why. Ideally k-1 == n_neighbors. If such case
        occurs in your dataset rows in the csv file may have different length)
    2. csv file of `reference`'s neighbors at path = <reference_dir> + f"nns_k{k}" + <reference_name> + ".csv"
        Same structure as 1.
    3. csv file of results at path = <save_dir> + "nn_similarity_" + <save_name> + ".csv"
        Structure of file:
        (header: cell_idx, ks[0], ks[1], ..., ks[-1])
        e.g.
        (cell_idx),  15  ,  30  , ...,  100
            0     ,   7  ,  22  , ...,   73
            1     ,  14  ,  28  , ...,   82
            2     ,   0  ,   3  , ...,   37
        ...

    """

    if (save_dir is None) or (save_name is None) or (reference_dir is None) or (reference_name is None):
        raise ValueError("Names and directories for saving results must be specified.")

    if type(reference) == str:
        adata = ann.read_h5ad(reference)
    else:
        adata = reference.copy()
    gene_set = [g for g in gene_set if g in adata.var.index]
    n_pcs = np.min([50, len(gene_set) - 1])
    adata_red = adata[:, gene_set]

    # Delete existing PCAs, neighbor graphs, etc. and calculate PCA for n_pcs
    for a in [adata, adata_red]:
        uns = [key for key in a.uns]
        for u in uns:
            del a.uns[u]
        obsm = [key for key in a.obsm]
        for o in obsm:
            del a.obsm[o]
        varm = [key for key in a.varm]
        for v in varm:
            del a.varm[v]
        sc.tl.pca(a, n_comps=n_pcs)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(reference_dir).mkdir(parents=True, exist_ok=True)
    results_path = save_dir + "nn_similarity_" + save_name + ".csv"
    if os.path.isfile(results_path):
        with open(results_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            existing_ks = [int(header[i]) for i in range(1, len(header))]
        df = pd.read_csv(results_path, index_col=0)
    # print(df)
    else:
        index = [i for i in range(len(adata.obs))]
        df = pd.DataFrame(index=index)
        existing_ks = []

    # create files where neighbors are saved
    for k in [_ for _ in ks if not (_ in existing_ks)]:
        path = save_dir + f"nns_k{k}" + save_name + ".csv"
        ref_path = reference_dir + f"nns_k{k}" + reference_name + ".csv"
        # if bbknn_ref_key == None:
        for p, a in [(path, adata_red), (ref_path, adata)]:
            if not os.path.isfile(p):
                if "neighbors" in a.uns:
                    del a.uns["neighbors"]
                if "connectivities" in a.obsp:
                    del a.obsp["connectivities"]
                if "distances" in a.obsp:
                    del a.obsp["distances"]
                sc.pp.neighbors(a, n_neighbors=k)
                neighbors_csv(a, k, p)
    # else:
    # 	for p,a in [(path,adata_red)]:
    # 		if not os.path.isfile(p):
    # 			if 'neighbors' in a.uns:
    # 				del a.uns['neighbors']
    # 			sc.pp.neighbors(a,n_neighbors=k)
    # 			neighbors_csv(a,k,p)
    # 	for p,a in [(ref_path,adata)]:
    # 		if not os.path.isfile(p):
    # 			if 'neighbors' in a.uns:
    # 				del a.uns['neighbors']
    # 			sce.pp.bbknn(a,n_neighbors=k,batch_key=bbknn_key)
    # 			#sc.pp.neighbors(a,n_neighbors=k)
    # 			neighbors_csv(a,k,p)

    # save similarities in csv
    for k in [_ for _ in ks if not (_ in existing_ks)]:
        path = save_dir + f"nns_k{k}" + save_name + ".csv"
        ref_path = reference_dir + f"nns_k{k}" + reference_name + ".csv"
        common_neighbors = compute_similarity(path, ref_path)
        # print(len(common_neighbors))
        df[f"{k}"] = common_neighbors
        df.to_csv(results_path)


full_marker_dict = {
    "Epithelial": {
        "AT1": ["RTKN2", "AGER", "NCKAP5", "AQP4"],
        "AT2": ["SFTPC", "SFTPA1", "SFTPA2", "SFTPD", "NAPSA", "SFTPB", "SFTPA2"],
        "basal": [
            "DLK2",
            "KRT15",
            "MIR205HG",
            "KRT5",
            "NPPC",
            "TP63",
            "THSD4",
            "S1PR5",
            "SLC5A7",
            "NKX2-1",
            "SFTA3",
            "DPP4",
        ],
        "Cycling": ["PBK", "MKI67", "TOP2A", "UBE2C", "BIRC5"],
        "Suprabasal": {
            "Suprabasal": ["KRT5", "SERPINB4"],
            "Suprabasal Nasal": [
                "KRT6A",
                "KRT13",
                "NMU",
                "PAX7",
                "SIX3",
                "PAX3",
                "PAX6",
            ],
            "Suprabasal TB": ["BBOX1", "SERPINB13"],
        },
        "Club": {
            "Club": ["CYP2F1", "CEACAM6", "VSIG2", "MSLN"],
            "Club Nasal": [
                "PI3",
                "LYPD2",
                "MUC1",
                "OTX2",
                "PAX7",
                "SIX3",
                "PAX3",
                "PAX6",
            ],
            "Club TB": [
                "SCGB1A1",
                "BPIFB1",
                "SCGB3A1",
                "SHH",
                "NKX2-1",
                "SFTA3",
                "DPP4",
            ],
        },
        "Goblet": {
            "Goblet": ["MUC5AC", "CEACAM6", "VSIG2", "MSLN"],
            "Goblet Nasal": [
                "PI3",
                "BPIFA2",
                "CEACAM5",
                "FUT6",
                "LYNX1",
                "RDH10",
                "OTX2",
                "PAX7",
                "SIX3",
                "PAX3",
                "PAX6",
            ],
            "Goblet TB": ["MUC5B", "FOXA3", "TSPAN8", "SHH", "NKX2-1", "SFTA3", "DPP4"],
        },
        "Multiciliated": {
            "Multiciliated": [
                "OMG",
                "TPPP3",
                "RSPH1",
                "SNTN",
                "CCDC78",
                "CFAP157",
                "DNAH12",
                "DTHD1",
                "DNAH12",
                "DCDC2B",
                "CFAP157",
                "ANKUB1",
                "TSPAN19",
            ],
            "MCC Nasal": [
                "SAA4",
                "PALLD",
                "CDKN2A",
                "SAA2",
                "OTX2",
                "PAX7",
                "SIX3",
                "PAX3",
                "PAX6",
            ],
            "MCC TB": ["DOC2A", "CORO2B", "RYR3", "NKX2-1", "SFTA3", "DPP4"],
        },
        "Deuterosomal": [
            "CDC20B",
            "E2F7",
            "ANLN",
            "CDC20",
            "CDC20Bshort",
            "PLK4",
            "FOXN4",
            "NEK2",
            "DEUP1",
            "HES6",
        ],
        "Goblet multiciliating": ["MUC5AC", "FOXJ1"],
        "Mucous": [
            "BPIFB2",
            "MUC5B",
            "GOLM1",
            "PART1",
            "TCN1",
            "AZGP1",
            "NKX3-1",
            "PRR4",
            "SPDEF",
            "XBP1",
            "FCGBP",
            "CBR3",
            "TSPAN8",
        ],
        "Serous": [
            "LTF",
            "AZGP1",
            "ZG16B",
            "PIP",
            "LYZ",
            "PRR4",
            "TCN1",
            "S100A1",
            "C6orf58",
            "PRB3",
            "SCGB3A2",
            "LPO",
            "ODAM",
            "PRH2",
        ],
        "Brush": ["RGS13", "BMX", "HEPACAM2", "PLCG2", "BIK", "VAMP2"],
        "Ionocyte": [
            "ATP6V0B",
            "SEC11C",
            "ASCL3",
            "AKR1B1",
            "HEPACAM2",
            "CLCNKB",
            "SCNN1B",
            "ATP6V1A",
            "TMEM61",
            "ATP6V1B1",
            "FOXI1",
            "BSND",
            "IGF1",
            "ATP6V1F",
            "STAP1",
            "CLCNKA",
            "ATP6V1G3",
            "CFTR",
            "ATP6V1C2",
            "ATP6V0A4",
            "TMPRSS11E",
            "FXYD2",
            "ATP6V0D2",
            "AMACR",
            "PLCG2",
        ],
        "Neuroendocrine": [
            "GRP",
            "PCSK1N",
            "PHGR1",
            "SCGN",
            "MIAT",
            "BEX1",
            "CHGB",
            "SCG2",
            "NEB",
            "APLP1",
        ],
    },
    "Endothelial": {
        "Endothelial": [
            "AQP1",
            "VWF",
            "IL3RA",
            "PLVAP",
            "ACKR1",
            "SELE",
            "EMCN",
            "GNG11",
            "CD34",
            "NOSTRIN",
            "LIFR",
            "EGFL7",
            "CLDN5",
            "JAM2",
            "ESAM",
            "FAM110D",
            "HYAL2",
            "CXCL12",
            "SEMA3D",
            "SNCG",
            "CCL21",
            "FLT4",
            "BMX",
            "EDNRB",
            "S100A3",
            "IL7R",
        ]
    },
    "Mesenchymal": {  # Stromal
        "Fibroblast": [
            "COL1A2",
            "DCN",
            "C1R",
            "COL1A1",
            "COL3A1",
            "LUM",
            "DPT",
            "LTBP1",
            "FBLN2",
            "PTGDS",
            "HTRA3",
            "SFRP2",
            "SFRP4",
            "TCF21",
            "COL6A3",
            "MFAP4",
            "C1S",
            "FBLN1",
            "PDGFRA",
            "PDGFRL",
            "VCAN",
            "SCARA5",
            "PI16",
            "MFAP5",
            "CD248",
            "FGFR4",
            "ITGA8",
            "SCN7A",
        ],
        "Pericyte": [
            "HIGD1B",
            "GJC1",
            "NOTCH3",
            "RGS5",
            "FAM162B",
            "COX4I2",
            "NDUFA4L2",
            "ACTA2",
            "TAGLN",
            "LGI4",
            "ITGA7",
            "CDH6",
        ],
        "Smooth Muscle": [
            "ACTA2",
            "TAGLN",
            "CALD1",
            "TAGLN",
            "TPM2",
            "LMOD1",
            "MYH11",
            "MYL9",
            "MYLK",
            "SPARCL1",
            "PLN",
            "ACTG2",
            "CAV1",
            "DES",
            "CNN1",
            "KCNMB1",
            "SORBS1",
            "SPEG",
            "EDNRB",
        ],
    },
    "Immune": {
        "B cell": [
            "AIM2",
            "BANK1",
            "CD19",
            "CD22",
            "CD79A",
            "CD79B",
            "LINC00926",
            "LTB",
            "MS4A1",
            "TLR10",
            "VPREB3",
        ],
        "plasma cell": ["MZB1", "JCHAIN", "FKBP11", "IGLL5", "DERL3", "TNFRSF17"],
        "T/NKT cell": [
            "CD2",
            "CD3D",
            "CAMK4",
            "CCL5",
            "IL32",
            "CD3E",
            "CD96",
            "XCL1",
            "CST7",
            "GZMH",
            "GZMB",
            "LTB",
            "CD7",
            "GZMA",
            "NKG7",
            "GNLY",
            "CCL4",
            "CTSW",
            "CD160",
            "CD247",
            "IL2RB",
            "KLRD1",
            "PRF1",
            "SLA2",
            "XCL2",
            "CD8A",
            "CD8B",
            "CCR6",
            "CD3G",
            "CD6",
            "TRAT1",
            "KLRB1",
            "MYBL1",
            "CXCR3",
            "TRAF1",
        ],
        "mast cell": [
            "TPSAB1",
            "CPA3",
            "HPGDS",
            "SLC18A2",
            "RGS13",
            "KIT",
            "GATA2",
            "RGS1",
            "HDC",
            "VWA5A",
            "LTC4S",
        ],
        "macrophage": [
            "APOC1",
            "OLR1",
            "APOE",
            "FABP4",
            "INHBA",
            "MARCO",
            "RETN",
            "MCEMP1",
            "TREM1",
            "CCL18",
            "GLDN",
            "GPD1",
            "MS4A7",
            "MSR1",
            "VSIG4",
            "HLA-DMB",
            "HLA-DPA1",
            "HLA-DPB1",
            "HLA-DQA1",
            "HLA-DQB1",
            "HLA-DRB1",
            "AIF1",
            "MNDA",
        ],
        "dendritic cell": [
            "CD1E",
            "RGS1",
            "RGS10",
            "RNASE6",
            "CD1C",
            "CLEC10A",
            "FCER1A",
            "FCGR2B",
            "LGALS2",
            "MS4A6A",
            "NLRP3",
            "PLD4",
        ],
        "monocytes": [
            "VCAN",
            "CD300E",
            "FCN1",
            "S100A12",
            "EREG",
            "APOBEC3A",
            "PLA2G7",
        ],
    },
}


def get_markers(level=3, return_dict=True):
    """Get marker gene names

    Arguments
    ---------
    level: int
        There are three annotation levels for celltypes. Genes are specified at least till
        level 2. Some are also specified till level 3. Possible values: 1 (rough), 2, 3 (detailed)
    return_dict: bool
        Return a dict, otherwise return a list without celltype annotations
    """
    if return_dict:
        marker_dict = {}
        if level == 1:
            marker_dict = {key: [] for key in full_marker_dict}
        elif level == 2:
            for key1 in full_marker_dict:
                for key2 in full_marker_dict[key1]:
                    marker_dict.update({key2: []})
        elif level == 3:
            for key1 in full_marker_dict:
                for key2 in full_marker_dict[key1]:
                    if type(full_marker_dict[key1][key2]) == dict:
                        for key3 in full_marker_dict[key1][key2]:
                            marker_dict.update({key3: []})
                    else:
                        marker_dict.update({key2: []})

        for key1 in full_marker_dict:
            for key2 in full_marker_dict[key1]:
                if type(full_marker_dict[key1][key2]) == dict:
                    for key3 in full_marker_dict[key1][key2]:
                        if level == 1:
                            marker_dict[key1] += full_marker_dict[key1][key2][key3]
                        elif level == 2:
                            marker_dict[key2] += full_marker_dict[key1][key2][key3]
                        elif level == 3:
                            marker_dict[key3] += full_marker_dict[key1][key2][key3]
                else:
                    if level == 1:
                        marker_dict[key1] += full_marker_dict[key1][key2]
                    else:
                        marker_dict[key2] += full_marker_dict[key1][key2]
        return marker_dict
    else:
        marker_list = []
        for key1 in full_marker_dict:
            for key2 in full_marker_dict[key1]:
                if type(full_marker_dict[key1][key2]) == dict:
                    for key3 in full_marker_dict[key1][key2]:
                        marker_list += full_marker_dict[key1][key2][key3]
                else:
                    marker_list += full_marker_dict[key1][key2]
        return marker_list


##########################################################
############### tree_classifications() ###################
##########################################################
# Note that the single tree classifications are too noisy
# we go with random forest classifications now, picking the best tree


def split_train_test_sets(adata, split=4, seed=2020, verbose=True, obs_key=None):
    """Split data to train and test set

    obs_key: str
        Provide a column name of adata.obs. If an obs_key is provided each group is split with the defined ratio.
    """
    if not obs_key:
        n_train = (adata.n_obs // (split + 1)) * split
        np.random.seed(seed=seed)
        train_obs = np.random.choice(adata.n_obs, n_train, replace=False)
        test_obs = np.array([True for i in range(adata.n_obs)])
        test_obs[train_obs] = False
        train_obs = np.invert(test_obs)
        if verbose:
            print(f"Split data to ratios {split}:1 (train:test)")
            print(f"datapoints: {adata.n_obs}")
            print(f"train data: {np.sum(train_obs)}")
            print(f"test data: {np.sum(test_obs)}")
        adata.obs["train_set"] = train_obs
        adata.obs["test_set"] = test_obs
    else:
        adata.obs["train_set"] = False
        adata.obs["test_set"] = False
        for group in adata.obs[obs_key].unique():
            df = adata.obs.loc[adata.obs[obs_key] == group]
            n_obs = len(df)
            n_train = (n_obs // (split + 1)) * split
            np.random.seed(seed=seed)
            train_obs = np.random.choice(n_obs, n_train, replace=False)
            test_obs = np.array([True for i in range(n_obs)])
            test_obs[train_obs] = False
            train_obs = np.invert(test_obs)
            if verbose:
                print(f"Split data for group {group}")
                print(f"to ratios {split}:1 (train:test)")
                print(f"datapoints: {n_obs}")
                print(f"train data: {np.sum(train_obs)}")
                print(f"test data: {np.sum(test_obs)}")
            adata.obs.loc[df.index, "train_set"] = train_obs
            adata.obs.loc[df.index, "test_set"] = test_obs


def tree_classifications(
    adata,
    selection,
    celltypes="all",
    ct_key="Celltypes",
    plot=False,
    save_load=False,
    seed=0,
):
    """Compute or load decision tree classification results

    Parameters
    ----------
    adata: AnnData
    selection: list or pd.DataFrame
        Trees are trained on genes of the list or genes defined in the bool column selection['selection'].
    celltypes: 'all' or list
        Trees are trained on the given celltypes
    ct_key: str
        Column name of adata.obs with celltype infos
    plot: bool
        Plot decision tree draft, confusion matrix and summary statistics table for each decision tree.
    save_load: str or False
        If not False load results if the given file exists, otherwise save results after computation.

    Returns
    -------
    f1_table: pd.Series
        Average f1 score for each decision tree. (rows: celltypes, name: 'macro avg f1')
    decision_genes: dict
        Genes used in the decision tree of the given celltype (dict key).

    """

    if save_load:
        if os.path.exists(save_load):
            if plot:
                warnings.warn("Can't plot decision trees when results are loaded from file.")
            return pickle.load(open(save_load, "rb"))
        if "/" in save_load:
            Path(save_load.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)

    decision_genes = {}
    if isinstance(selection, list):
        genes = selection
    elif isinstance(selection, pd.DataFrame):
        genes = list(selection.loc[selection["selection"]].index)
    a = adata[:, genes].copy()
    split_train_test_sets(a, split=4, seed=2020, verbose=False, obs_key=ct_key)
    if celltypes == "all":
        celltypes = np.unique(a.obs[ct_key].values)
    celltypes_tmp = [
        ct
        for ct in celltypes
        if (ct in a.obs.loc[a.obs["train_set"], ct_key].values) and (ct in a.obs.loc[a.obs["test_set"], ct_key].values)
    ]
    for c in [ct for ct in celltypes if ct not in celltypes_tmp]:
        warnings.warn(f"Zero cells of celltype {c} in train or test set. No tree is calculated for celltype {c}.")
    celltypes = celltypes_tmp
    f1_table = pd.Series(index=celltypes, name="macro avg f1", dtype="float64")

    if scipy.sparse.issparse(a.X):
        X_train = a[a.obs["train_set"], :].X.toarray()
        X_test = a[a.obs["test_set"], :].X.toarray()
    else:
        X_train = a[a.obs["train_set"], :].X.copy()
        X_test = a[a.obs["test_set"], :].X.copy()
    y_train = {}
    y_test = {}
    for ct in celltypes:
        y_train[ct] = np.where(a[a.obs["train_set"], :].obs[ct_key] == ct, ct, "other")
        y_test[ct] = np.where(a[a.obs["test_set"], :].obs[ct_key] == ct, ct, "other")

    test_ct_idxs = {}
    for ct in np.unique(a[a.obs["test_set"], :].obs[ct_key]):
        test_ct_idxs[ct] = np.where(a[a.obs["test_set"], :].obs[ct_key] == ct)[0]

    ct_trees = {}
    for ct in celltypes:
        ct_trees[ct] = tree.DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=3,
            random_state=seed,  # 0,
            class_weight="balanced",
        )
        ct_trees[ct] = ct_trees[ct].fit(X_train, y_train[ct])

    if plot:
        sizefactor = 10
        cols = 2
        rows = len(celltypes)
        plt.figure(figsize=(sizefactor * cols, sizefactor * rows))
    for i, ct in enumerate(celltypes):
        if plot:
            plt.subplot(rows, cols, i * cols + 1)
            tree.plot_tree(
                ct_trees[ct],
                impurity=False,
                label="none",
                feature_names=genes,
                max_depth=2,
                fontsize=10,
            )  # ,proportion=True)
            ax = plt.subplot(rows * 2, cols, i * 2 * cols + 2)
            sklearn.metrics.plot_confusion_matrix(
                ct_trees[ct],
                X_test,
                y_test[ct],
                cmap=plt.cm.Blues,
                normalize="true",
                ax=ax,
            )
            plt.title(ct)
            plt.yticks(rotation=90, va="center")
            plt.subplot(rows * 2, cols, i * 2 * cols + 4)
            plt.text(
                0.1,
                0.3,
                classification_report(y_test[ct], ct_trees[ct].predict(X_test)),
                **{"fontname": "Helvetica", "fontfamily": "monospace"},
            )
            plt.axis("off")
        report = classification_report(y_test[ct], ct_trees[ct].predict(X_test), output_dict=True)
        f1_table.loc[ct] = report["macro avg"]["f1-score"]
        decision_genes[ct] = {
            a.var.index.values[i]: ct_trees[ct].feature_importances_[i]
            for i in range(len(a.var))
            if (ct_trees[ct].feature_importances_[i] != 0)
        }

    if plot:
        plt.show()
    if save_load:
        with open(save_load, "wb") as f:
            pickle.dump([f1_table, decision_genes], f)
    return f1_table, decision_genes


############## FOREST CLASSIFICATIONS ##############
def sample_train_set_by_ct(adata, ct_key, subsample=500, seed=2020, celltypes="all"):
    """Subsample `subsample` cells per celltype"""
    a = adata[adata.obs["train_set"], :]
    if celltypes == "all":
        celltypes = list(a.obs["ct_key"].unique())
        # Get subsample for each celltype
    obs = []
    for ct in celltypes:
        df = a.obs.loc[a.obs[ct_key] == ct]
        n_obs = len(df)
        np.random.seed(seed=seed)
        if n_obs > subsample:
            train_obs = np.random.choice(n_obs, subsample, replace=False)
            obs += list(df.iloc[train_obs].index.values)
        else:
            train_obs = np.random.choice(n_obs, subsample, replace=True)
            obs += list(df.iloc[train_obs].index.values)

    X_train = a[obs, :].X.toarray()
    y_train = {}
    for ct in celltypes:
        y_train[ct] = np.where(a[obs, :].obs[ct_key] == ct, ct, "other")

    return X_train, y_train


def forest_classifications(
    adata,
    selection,
    celltypes="all",
    ct_key="Celltypes",
    save_load=False,
    seed=0,
    n_trees=50,
    subsample=1000,
    full_info=False,
    verbose=False,
):
    """Compute or load decision tree classification results

    Parameters
    ----------
    adata: AnnData
    selection: list or pd.DataFrame
        Trees are trained on genes of the list or genes defined in the bool column selection['selection'].
    celltypes: 'all' or list
        Trees are trained on the given celltypes
    ct_key: str
        Column name of adata.obs with celltype infos
    save_load: str or False
        If not False load results if the given file exists, otherwise save results after computation.
    full_info: bool
        Additionally return performance table and importances of all trees
    subsample: int
        For each trained tree we use samples of maximal size=`subsample` for each celltype. If fewer cells
        are present for a given celltype all cells are used.

    Returns
    -------
    f1_table: pd.Series
        Average f1 score for each decision tree. (rows: celltypes, name: 'macro avg f1')
    decision_genes: dict
        Genes used in the decision tree of the given celltype (dict key).

    """

    if save_load:
        if os.path.exists(save_load):
            if plot:  # noqa: F821  TODO Louis
                warnings.warn("Can't plot decision trees when results are loaded from file.")
            return pickle.load(open(save_load, "rb"))
        if "/" in save_load:
            Path(save_load.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)

    if isinstance(selection, list):
        genes = selection
    elif isinstance(selection, pd.DataFrame):
        genes = list(selection.loc[selection["selection"]].index)
    a = adata[:, genes].copy()

    # apply a train test set split one time just to get test set numbers for each celltype to eventually filter out celltypes
    split_train_test_sets(a, split=4, seed=2020, verbose=False, obs_key=ct_key)
    if celltypes == "all":
        celltypes = np.unique(a.obs[ct_key].values)
    celltypes_tmp = [
        ct
        for ct in celltypes
        if (ct in a.obs.loc[a.obs["train_set"], ct_key].values) and (ct in a.obs.loc[a.obs["test_set"], ct_key].values)
    ]
    for c in [c for c in celltypes if c not in celltypes_tmp]:
        warnings.warn(f"Zero cells of celltype {c} in train or test set. No tree is calculated for celltype {c}.")
    celltypes = celltypes_tmp

    if scipy.sparse.issparse(a.X):
        X_test = a[a.obs["test_set"], :].X.toarray()
    else:
        X_test = a[a.obs["test_set"], :].X.copy()
    y_test = {}
    for ct in celltypes:
        y_test[ct] = np.where(a[a.obs["test_set"], :].obs[ct_key] == ct, ct, "other")

        # f1_table = pd.Series(index=celltypes, name='macro avg f1', dtype='float64')
    f1_table = pd.DataFrame(index=celltypes, columns=[str(i) for i in range(n_trees)], dtype="float64")
    importances = {
        ct: pd.DataFrame(index=a.var.index, columns=[str(i) for i in range(n_trees)], dtype="float64")
        for ct in celltypes
    }
    ct_trees = {ct: [] for ct in celltypes}
    decision_genes = {ct: [] for ct in celltypes}
    np.random.seed(seed=seed)
    seeds = np.random.choice(100000, n_trees, replace=False)
    for i in range(n_trees):
        if verbose:
            print(f"~~~ Trees number {i} ~~~")
        X_train, y_train = sample_train_set_by_ct(a, ct_key, subsample=subsample, seed=seeds[i], celltypes=celltypes)
        for ct in celltypes:
            ct_trees[ct].append(
                tree.DecisionTreeClassifier(
                    criterion="gini",
                    splitter="best",
                    max_depth=3,
                    random_state=seeds[i],  # 0,
                    class_weight="balanced",
                )
            )
            ct_trees[ct][-1] = ct_trees[ct][-1].fit(X_train, y_train[ct])

            report = classification_report(y_test[ct], ct_trees[ct][-1].predict(X_test), output_dict=True)
            f1_table.loc[ct, str(i)] = report["macro avg"]["f1-score"]
            importances[ct][str(i)] = ct_trees[ct][-1].feature_importances_

    best_f1_table = pd.Series(index=f1_table.index, name="macro avg f1", dtype="float64")
    best_tree_genes = {}
    for ct, best_tree in f1_table.idxmax(axis=1).items():
        best_f1_table.loc[ct] = f1_table.loc[ct, best_tree]
        best_tree_genes[ct] = {c: im for c, im in importances[ct][best_tree].items() if (im > 0)}

    if save_load:
        with open(save_load, "wb") as f:
            pickle.dump([f1_table, decision_genes], f)
    if full_info:
        return best_f1_table, best_tree_genes, f1_table, importances
    else:
        return best_f1_table, best_tree_genes
