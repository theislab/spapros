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
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from spapros.util.mp_util import _get_n_cores
from spapros.util.mp_util import parallelize
from spapros.util.mp_util import Signal
from tqdm.notebook import tqdm
from xgboost import XGBClassifier


def plot_gene_expressions(adata, f_idxs, fig_title=None, save_to=None):
    a = adata.copy()  # TODO Think we can get rid of this copy and just work with the views
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
        if n in ns and progress_bar:
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

    if save_every is not None:
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
    else:
        index = [i for i in range(len(adata.obs))]
        df = pd.DataFrame(index=index)
        existing_ks = []

    # create files where neighbors are saved
    for k in [_ for _ in ks if not (_ in existing_ks)]:
        path = save_dir + f"nns_k{k}" + save_name + ".csv"
        ref_path = reference_dir + f"nns_k{k}" + reference_name + ".csv"
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

    # save similarities in csv
    for k in [_ for _ in ks if not (_ in existing_ks)]:
        path = save_dir + f"nns_k{k}" + save_name + ".csv"
        ref_path = reference_dir + f"nns_k{k}" + reference_name + ".csv"
        common_neighbors = compute_similarity(path, ref_path)
        # print(len(common_neighbors))
        df[f"{k}"] = common_neighbors
        df.to_csv(results_path)


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


############## FOREST CLASSIFICATIONS ##############


def get_celltypes_with_too_small_test_sets(adata, ct_key, min_test_n=20, split_kwargs={"seed": 0, "split": 4}):
    """Get celltypes whith test set sizes below `min_test_n`

    We split the observations in adata into train and test sets for forest training. Check if the resulting
    test sets have at least `min_test_n` samples.

    Arguments
    ---------
    adata: AnnData
    ct_key: str
        adata.obs key for cell types
    min_test_n: int
        Minimal number of samples in each celltype's test set
    split_kwargs: dict
        Keyword arguments for ev.split_train_test_sets()

    Returns
    -------
    cts_below_min: list
        celltypes with too small test sets
    counts_below_min: list
        test set sample count for each celltype in celltype_list

    """
    a = adata.copy()
    split_train_test_sets(a, verbose=False, obs_key=ct_key, **split_kwargs)
    a = a[a.obs["test_set"], :]

    counts = a.obs.groupby(ct_key)["test_set"].value_counts()
    below_min = counts < min_test_n
    cts_below_min = [idx[0] for idx in below_min[below_min].index]
    counts_below_min = counts[below_min].tolist()
    return cts_below_min, counts_below_min


def uniform_samples(adata, ct_key, set_key="train_set", subsample=500, seed=2020, celltypes="all"):
    """Subsample `subsample` cells per celltype

    If the number of cells of a celltype is lower we're oversampling that celltype.
    """
    a = adata[adata.obs[set_key], :]
    if celltypes == "all":
        celltypes = list(a.obs[ct_key].unique())
    # Get subsample for each celltype
    all_obs = []
    for ct in celltypes:
        df = a.obs.loc[a.obs[ct_key] == ct]
        n_obs = len(df)
        np.random.seed(seed=seed)
        if n_obs > subsample:
            obs = np.random.choice(n_obs, subsample, replace=False)
            all_obs += list(df.iloc[obs].index.values)
        else:
            obs = np.random.choice(n_obs, subsample, replace=True)
            all_obs += list(df.iloc[obs].index.values)

    if scipy.sparse.issparse(a.X):
        X = a[all_obs, :].X.toarray()
    else:
        X = a[all_obs, :].X.copy()
    y = {}
    for ct in celltypes:
        y[ct] = np.where(a[all_obs, :].obs[ct_key] == ct, ct, "other")

    cts = a[all_obs].obs[ct_key].values

    return X, y, cts


def save_forest(results, path):
    """Save forest results to file

    results: list
        Output from forest_classifications()
    path: str
        Path to save file
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_forest(path):
    """Load forest results from path

    path: str
        Path to file
    """
    return pickle.load(open(path, "rb"))


def get_reference_masks(cts, ct_to_ref):
    """Get celltype specific boolean masks over celltype annotation vector

    cts: list
        celltype annotations.
    ct_to_ref: dict
        Each celltype's list of reference celltypes e.g.:
        {'AT1':['AT1','AT2','Club'],'Pericytes':['Pericytes','Smooth muscle']}
    """
    masks = {}
    for ct, ref in ct_to_ref.items():
        masks[ct] = np.in1d(cts, ref)
    return masks


def train_ct_tree_helper(celltypes, X_train, y_train, seed, max_depth=3, masks=None, queue=None):
    """Train decision trees parallelized over celltypes

    TODO: Write docstring
    """
    ct_trees = {}
    for ct in celltypes:
        ct_trees[ct] = tree.DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=max_depth, random_state=seed, class_weight="balanced"  # 3,
        )
        if masks is None:
            ct_trees[ct] = ct_trees[ct].fit(X_train, y_train[ct])
        elif np.sum(masks[ct]) > 0:
            ct_trees[ct] = ct_trees[ct].fit(X_train[masks[ct], :], y_train[ct][masks[ct]])

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return ct_trees


def pool_train_ct_tree_helper(ct_trees_dicts):
    """Combine list of dictonaries to one dict

    TODO: Write docstring
    """
    tmp = [ct for trees_dict in ct_trees_dicts for ct in trees_dict]
    if len(set(tmp)) < len(tmp):
        raise ValueError("Multiple trees for the same celltype are in the results of the parallelized execution")
    ct_trees = {ct: tree for ct_trees_dict in ct_trees_dicts for ct, tree in ct_trees_dict.items()}
    return ct_trees


def eval_ct_tree_helper(ixs, celltypes, ref_celltypes, ct_trees, X_test, y_test, cts_test, masks=None, queue=None):
    """
    TODO: Write docstring

    Returns
    -------
    f1_scores:
    """
    tree_names = [str(i) for i in ixs]
    summary_metric = pd.DataFrame(index=celltypes, columns=tree_names, dtype="float64")
    specificities = {ct: pd.DataFrame(index=ref_celltypes, columns=tree_names, dtype="float64") for ct in celltypes}
    for i in ixs:
        for ct in celltypes:
            if (masks is None) or (np.sum(masks[ct]) > 0):
                X = X_test if (masks is None) else X_test[masks[ct], :]
                y = y_test[ct] if (masks is None) else y_test[ct][masks[ct]]
                prediction = ct_trees[ct][i].predict(X)
                report = classification_report(y, prediction, output_dict=True)
                summary_metric.loc[ct, str(i)] = report["macro avg"]["f1-score"]
                for ct2 in [c for c in ref_celltypes if not (c == ct)]:
                    idxs = (cts_test == ct2) if (masks is None) else (cts_test[masks[ct]] == ct2)
                    if np.sum(idxs) > 0:
                        specificity = np.sum(prediction[idxs] == y[idxs]) / np.sum(idxs)
                        specificities[ct].loc[ct2, str(i)] = specificity

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return summary_metric, specificities


def pool_eval_ct_tree_helper(f1_and_specificities):
    """
    TODO: Write docstring

    We parallelize over n_trees
    """
    summary_metric_dfs = [val[0] for val in f1_and_specificities]
    specificity_df_dicts = [val[1] for val in f1_and_specificities]
    summary_metric = pd.concat(summary_metric_dfs, axis=1)
    cts = list(specificity_df_dicts[0])
    specificities = {
        ct: pd.concat([specificity_df_dicts[i][ct] for i in range(len(specificity_df_dicts))], axis=1) for ct in cts
    }
    return summary_metric, specificities


def single_forest_classifications(
    adata,
    selection,
    celltypes="all",
    ref_celltypes="all",
    ct_key="Celltypes",
    ct_spec_ref=None,
    save=False,
    seed=0,
    n_trees=50,
    max_depth=3,
    subsample=1000,
    test_subsample=3000,
    sort_by_tree_performance=True,
    verbose=False,
    return_clfs=False,
    n_jobs=1,
    backend="loky",
):
    """Compute or load decision tree classification results

    TODO: This doc string is partially from an older version. Update it! (Descripiton and Return is already up to date)
    TODO: Add progress bars to trees, and maybe change verbose to verbosity levels

    As metrics we use:
    macro f1 score as summary statistic - it's a uniformly weighted statistic wrt celltype groups in 'others' since
    we sample uniformly.
    For the reference celltype specific metric we use specificity = TN/(FP+TN) (also because FN and TP are not feasible
    in the given setting)

    Parameters
    ----------
    adata: AnnData
    selection: list or pd.DataFrame
        Trees are trained on genes of the list or genes defined in the bool column selection['selection'].
    celltypes: 'all' or list
        Trees are trained on the given celltypes
    ct_key: str
        Column name of adata.obs with celltype infos
    ct_spec_ref: dict of lists
        Celltype specific references (e.g.: {'AT1':['AT1','AT2','Club'],'Pericytes':['Pericytes','Smooth muscle']}).
        This argument was introduced to train secondary trees.
    save_load: str or False
        If not False load results if the given file exists, otherwise save results after computation.
    max_depth: str
        max_depth argument of DecisionTreeClassifier.
    subsample: int
        For each trained tree we use samples of maximal size=`subsample` for each celltype. If fewer cells
        are present for a given celltype all cells are used.
    sort_by_tree_performance: str
        Wether to sort results and trees by tree performance (best first) per celltype
    return_clfs: str
        Wether to return the sklearn tree classifier objects. (if `return_clfs` and `save_load` we still on
        save the results tables, if you want to save the classifiers this needs to be done separately).
    n_jobs: int
        Multiprocessing number of processes.
    backend: str
        Which backend to use for multiprocessing. See class `joblib.Parallel` for valid options.

    Returns
    -------
    Note in all output files trees are ordered according macro f1 performance.

    summary_metric: pd.DataFrame
        macro f1 scores for each celltype's trees (Ordered according best performing trees)
    ct_specific_metric: dict of pd.DataFrame
        For each celltype's tree: specificity (= TN / (FP+TN)) wrt each other celltype's test sample
    importances: dict of pd.DataFrame
        Gene's feature importances for each tree.

    if return_clfs:
        return [summary_metric,ct_specific_metric,importances], forests

    """

    if verbose:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm_notebook as tqdm
    else:
        tqdm = None

    n_jobs = _get_n_cores(n_jobs)

    if isinstance(selection, (list, np.ndarray, np.generic)):
        genes = selection
    elif isinstance(selection, pd.Index):
        genes = selection.values
    elif isinstance(selection, pd.DataFrame):
        genes = list(selection.loc[selection["selection"]].index)
    a = adata[:, genes].copy()

    # apply a train test set split one time just to get test set numbers for each celltype to eventually filter out celltypes
    split_train_test_sets(a, split=4, seed=2020, verbose=False, obs_key=ct_key)
    if celltypes == "all":
        celltypes = np.unique(a.obs[ct_key].values)
    if ref_celltypes == "all":
        ref_celltypes = np.unique(a.obs[ct_key].values)
    celltypes_tmp = [
        ct
        for ct in celltypes
        if (ct in a.obs.loc[a.obs["train_set"], ct_key].values) and (ct in a.obs.loc[a.obs["test_set"], ct_key].values)
    ]
    ref_celltypes_tmp = [
        ct
        for ct in ref_celltypes
        if (ct in a.obs.loc[a.obs["train_set"], ct_key].values) and (ct in a.obs.loc[a.obs["test_set"], ct_key].values)
    ]
    for c in [c for c in celltypes if c not in celltypes_tmp]:
        warnings.warn(f"Zero cells of celltype {c} in train or test set. No tree is calculated for celltype {c}.")
    for c in [c for c in ref_celltypes if c not in ref_celltypes_tmp]:
        warnings.warn(
            f"Zero cells of celltype {c} in train or test set. Celltype {c} is not included as reference celltype."
        )
    celltypes = celltypes_tmp
    ref_celltypes = ref_celltypes_tmp
    cts_not_in_ref = [ct for ct in celltypes if not (ct in ref_celltypes)]
    if cts_not_in_ref:
        warnings.warn(
            f"For celltypes {cts_not_in_ref} trees are computed, they are not listed in reference celltypes though. Added them..."
        )
        ref_celltypes += cts_not_in_ref

    # Reset subsample and test_subsample if dataset is actually too small.
    max_counts_train = a.obs.loc[a.obs["train_set"], ct_key].value_counts().loc[ref_celltypes].max()
    max_counts_test = a.obs.loc[a.obs["test_set"], ct_key].value_counts().loc[ref_celltypes].max()
    if subsample > max_counts_train:
        subsample = max_counts_train
    if test_subsample > max_counts_test:
        test_subsample = max_counts_test

    X_test, y_test, cts_test = uniform_samples(
        a, ct_key, set_key="test_set", subsample=test_subsample, seed=seed, celltypes=ref_celltypes
    )
    if ct_spec_ref is not None:
        masks_test = get_reference_masks(cts_test, ct_spec_ref)
    else:
        masks_test = None

    ct_trees = {ct: [] for ct in celltypes}
    np.random.seed(seed=seed)
    seeds = np.random.choice(100000, n_trees, replace=False)
    # Compute trees (for each tree index we parallelize over celltypes)
    for i in tqdm(range(n_trees), desc="Train trees") if tqdm else range(n_trees):
        # if verbose: print(f"\t\t ~~~ Trees number {i} ~~~")
        X_train, y_train, cts_train = uniform_samples(
            a, ct_key, set_key="train_set", subsample=subsample, seed=seeds[i], celltypes=ref_celltypes
        )
        if ct_spec_ref is not None:
            masks = get_reference_masks(cts_train, ct_spec_ref)
        else:
            masks = None
        ct_trees_i = parallelize(
            callback=train_ct_tree_helper,
            collection=celltypes,
            n_jobs=n_jobs,
            backend=backend,
            extractor=pool_train_ct_tree_helper,
            show_progress_bar=False,  # verbose,
        )(X_train=X_train, y_train=y_train, seed=seeds[i], max_depth=max_depth, masks=masks)
        for ct in celltypes:
            ct_trees[ct].append(ct_trees_i[ct])
    # Get feature importances
    importances = {
        ct: pd.DataFrame(index=a.var.index, columns=[str(i) for i in range(n_trees)], dtype="float64")
        for ct in celltypes
    }
    for i in range(n_trees):
        for ct in celltypes:
            if (masks_test is None) or (np.sum(masks_test[ct]) > 0):
                importances[ct][str(i)] = ct_trees[ct][i].feature_importances_
    # Evaluate trees (we parallelize over tree indices)
    summary_metric, ct_specific_metric = parallelize(
        callback=eval_ct_tree_helper,
        collection=[i for i in range(n_trees)],
        n_jobs=n_jobs,
        backend=backend,
        extractor=pool_eval_ct_tree_helper,
        show_progress_bar=verbose,
        desc="Evaluate trees",
    )(
        celltypes=celltypes,
        ref_celltypes=ref_celltypes,
        ct_trees=ct_trees,
        X_test=X_test,
        y_test=y_test,
        cts_test=cts_test,
        masks=masks_test,
    )

    # Sort results
    if sort_by_tree_performance:
        for ct in summary_metric.index:
            if (masks_test is None) or (np.sum(masks_test[ct]) > 0):
                order = summary_metric.loc[ct].sort_values(ascending=False).index.values.copy()
                order_int = [summary_metric.loc[ct].index.get_loc(idx) for idx in order]
                summary_metric.loc[ct] = summary_metric.loc[ct, order].values
                ct_specific_metric[ct].columns = order.copy()
                importances[ct].columns = order.copy()
                ct_specific_metric[ct] = ct_specific_metric[ct].reindex(
                    sorted(ct_specific_metric[ct].columns, key=int), axis=1
                )
                importances[ct] = importances[ct].reindex(sorted(importances[ct].columns, key=int), axis=1)
                ct_trees[ct] = [ct_trees[ct][i] for i in order_int]

    # We change the f1 summary metric now, since we can't summarize this value anymore when including secondary trees.
    # When creating the results for secondary trees we take the specificities according reference celltypes of each tree.
    # Our new metric is just the mean of these specificities. Btw we still keep the ordering to be based on f1 scores.
    # Think that makes more sense since it's the best balanced result.
    # TODO TODO TODO: Change misleading variable names in other functions (where the old "f1_table" is used)
    summary_metric = summarize_specs(ct_specific_metric)

    if save:
        save_forest([summary_metric, ct_specific_metric, importances], save)

    if return_clfs:
        return [summary_metric, ct_specific_metric, importances], ct_trees
    else:
        return summary_metric, ct_specific_metric, importances


def summarize_specs(specs):
    """Summarize specificities to summary metrics per celltype"""
    cts = [ct for ct in specs]
    df = pd.DataFrame(index=cts, columns=specs[cts[0]].columns)
    for ct in specs:
        df.loc[ct] = specs[ct].mean()
    return df


def combine_tree_results(primary, secondary, with_clfs=False):
    """Combine results of primary and secondary trees

    There are three parts in the forest results:
    1. f1_table
    2. classification specificities
    3. feature_importances

    The output for 2. and 3. will be in the same form as the input.
    Specificities are taken from secondary where existend, otherwise from primary.
    Feature_importances are summed up (reasoning: distinguishing celltypes that are
    hard to distinguish is very important and therefore good to rank respective genes high).
    The f1 tables are just aggregated to a list

    primary: dict of pd.DataFrames
    secondary: dict of pd.DataFrames
    with_clfs: bool
        Whether primary, secondary and the output each contain a list of forest results and the forest classifiers
        or only the results.

    """
    expected_len = 2 if with_clfs else 3
    if (len(primary) != expected_len) or (len(secondary) != expected_len):
        raise ValueError(
            f"inputs primary and secondary are expected to be lists of length == {expected_len}, not {len(primary)},"
            f"{len(secondary)}"
        )

    if with_clfs:
        primary, primary_clfs = primary
        secondary, secondary_clfs = secondary

    combined = [0, {}, {}]
    ## f1 (exchanged by summary stats below)
    # for f1_table in [primary[0],secondary[0]]:
    #    if isinstance(f1_table,list):
    #        combined[0] += f1_table
    #    else:
    #        combined[0].append(f1_table)
    # specificities
    celltypes = [key for key in secondary[1]]
    combined[1] = {ct: df.copy() for ct, df in primary[1].items()}
    for ct in celltypes:
        filt = ~secondary[1][ct].isnull().all(axis=1)
        combined[1][ct].loc[filt] = secondary[1][ct].loc[filt]
    # summary stats
    combined[0] = summarize_specs(combined[1])
    # feature importance
    combined[2] = {ct: df.copy() for ct, df in primary[2].items()}
    for ct in celltypes:
        combined[2][ct] += secondary[2][ct].fillna(0)
        combined[2][ct] = combined[2][ct].div(combined[2][ct].sum(axis=0), axis=1)

    if with_clfs:
        combined_clfs = primary_clfs
        for ct in combined_clfs:
            if ct in secondary_clfs:
                combined_clfs[ct] += secondary_clfs[ct]
        return combined, combined_clfs
    else:
        return combined


def outlier_mask(df, n_stds=1, min_outlier_dif=0.02, min_score=0.9):
    """Get mask over df.index based on values in df columns"""
    crit1 = df < (df.mean(axis=0) - (n_stds * df.std(axis=0))).values[np.newaxis, :]
    crit2 = df < (df.mean(axis=0) - min_outlier_dif).values[np.newaxis, :]
    crit3 = df < min_score
    return (crit1 & crit2) | crit3


def get_outlier_reference_celltypes(specs, n_stds=1, min_outlier_dif=0.02, min_score=0.9):
    """For each celltype's best tree get reference celltypes with low performance

    specs: dict of pd.DataFrames
        Each celltype's specificities of reference celltypes.
    """
    outliers = {}
    for ct, df in specs.items():
        outliers[ct] = df.loc[outlier_mask(df[["0"]], n_stds, min_outlier_dif, min_score).values].index.tolist() + [ct]
        if len(outliers[ct]) == 1:
            outliers[ct] = []
    return outliers


def forest_classifications(adata, selection, max_n_forests=3, verbosity=1, outlier_kwargs={}, **forest_kwargs):
    """Train best trees including secondary trees

    max_n_forests: int
        Number of best trees considered as a tree group. Including the primary tree.

    """

    if verbosity > 0:
        try:
            from tqdm.notebook import tqdm
        except ImportError:
            from tqdm import tqdm_notebook as tqdm
    else:
        tqdm = None

    ct_spec_ref = None
    res = None
    with_clfs = "return_clfs" in forest_kwargs

    for _ in tqdm(range(max_n_forests), desc="Train hierarchical trees") if tqdm else range(max_n_forests):
        new_res = single_forest_classifications(
            adata, selection, ct_spec_ref=ct_spec_ref, verbose=verbosity > 1, **forest_kwargs
        )
        res = new_res if (res is None) else combine_tree_results(res, new_res, with_clfs=with_clfs)
        specs = res[0][1] if with_clfs else res[1]
        ct_spec_ref = get_outlier_reference_celltypes(specs, **outlier_kwargs)

    return res


def forest_rank_table(importances, celltypes="all", return_ct_specific_rankings=False):
    """Rank genes according importances of the celltypes' forests

    celltypes: str or list of strs
        If 'all' create ranking based on all celltypes in importances. Otherwise base the ranking only on
        the trees of the subset `celltypes`.
    importances: dict of pd.DataFrame
        Output from `forest_classifications()`. DataFrame for each celltype's forest.
        Each column refers to genes of one tree. The columns are sorted according performance (best first)

    Returns
    -------
    pd.DataFrame
        index: Genes found in `importances`
        columns:
            - 'rank': integer, tree rank in which a gene occured the first time (best over all celltypes)
            - 'importance_score': max feature importance of gene over celltypes where the gene occured at `rank`
            - ct in celltypes:
    """
    im = (
        importances.copy()
        if (celltypes == "all")
        else {ct: df.copy() for ct, df in importances.items() if ct in celltypes}
    )

    # ranking per celltype
    worst_rank = max([len(im[ct].columns) + 1 for ct in im])
    for ct in im:
        im[ct] = im[ct].reindex(columns=im[ct].columns.tolist() + ["tree", "rank", "importance_score"])
        rank = 1
        for tree_idx, col in enumerate(im[ct].columns):
            filt = im[ct]["rank"].isnull() & (im[ct][col] > 0)
            if filt.sum() > 0:
                im[ct].loc[filt, ["tree", "rank"]] = [tree_idx, rank]
                im[ct].loc[filt, "importance_score"] = im[ct].loc[filt, col]
                rank += 1
        filt = im[ct]["rank"].isnull()
        im[ct].loc[filt, ["rank", "importance_score"]] = [worst_rank, 0]

    # Save celltype specific rankings in current form if later returned
    if return_ct_specific_rankings:
        im_ = {ct: df.copy() for ct, df in im.items()}

    # collapse to general ranking.
    for ct in im:
        im[ct].columns = [f"{ct}_{col}" for col in im[ct]]
    tab = pd.concat([df for _, df in im.items()], axis=1)
    tab["rank"] = tab[[f"{ct}_rank" for ct in im]].min(axis=1)
    tab["importance_score"] = 0
    tab = tab.reindex(columns=tab.columns.tolist() + [ct for ct in im])
    tab[[ct for ct in im]] = False
    for gene in tab.index:
        tmp_cts = [ct for ct in im if (tab.loc[gene, f"{ct}_rank"] == tab.loc[gene, "rank"])]
        tmp_scores = [
            tab.loc[gene, f"{ct}_importance_score"]
            for ct in im
            if (tab.loc[gene, f"{ct}_rank"] == tab.loc[gene, "rank"])
        ]
        tab.loc[gene, "importance_score"] = max(tmp_scores) if tmp_scores else 0
        if tab.loc[gene, "rank"] != worst_rank:
            tab.loc[gene, tmp_cts] = True

    tab = tab[["rank", "importance_score"] + [ct for ct in im]]
    tab = tab.sort_values(["rank", "importance_score"], ascending=[True, False])
    tab["rank"] = tab["rank"].rank(axis=0, method="dense", na_option="keep", ascending=True)

    if return_ct_specific_rankings:
        im_ = {ct: df.sort_values(["rank", "importance_score"], ascending=[True, False]) for ct, df in im_.items()}
        return tab, im_
    else:
        return tab
           
def XGBoost_forest_classification(
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
    n_jobs=1
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
        warnings.warn(f"The following cell types are not included in forest classifications since they have fewer "\
                      f"than {n_cells_min} cells: {cell_counts.loc[cell_counts < n_cells_min].index.tolist()}")
        celltypes = [ct for ct in celltypes if (cell_counts.loc[ct] >= n_cells_min)]
        
    # Get data
    obs = adata.obs[ct_key].isin(celltypes)
    s = selection if isinstance(selection,list) else selection.loc[selection['selection']].index.tolist()
    X = adata[obs,s].X
    ct_encoding = {ct:i for i,ct in enumerate(celltypes)}
    y = adata.obs.loc[obs,ct_key].astype(str).map(ct_encoding).values
    
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
        k_fold = StratifiedKFold(n_splits=cv_splits,random_state=seed,shuffle=True)            
        for train_ix, test_ix in tqdm(k_fold.split(X,y), desc=desc, total=cv_splits) if tqdm else k_fold.split(X,y):
            # Get train and test sets
            train_x, train_y, test_x, test_y = X[train_ix], y[train_ix], X[test_ix], y[test_ix]
            sample_weight_train = compute_sample_weight('balanced', train_y)
            sample_weight_test = compute_sample_weight('balanced', test_y)
            # Fit the classifier
            n_classes = len(np.unique(train_y))
            clf = XGBClassifier(max_depth=max_depth, 
                                num_class=n_classes,
                                n_estimators=250,
                                objective="multi:softmax" if n_classes > 2 else "binary:logistic",
                                eval_metric='mlogloss', # set this to get rid of warning
                                learning_rate=lr,
                                colsample_bytree=colsample_bytree,
                                min_child_weight=min_child_weight,
                                gamma=gamma,
                                booster='gbtree',#TODO: compare with 'dart',rate_drop= 0.1
                                random_state=seed,
                                use_label_encoder=False, # To get rid of deprecation warning we convert labels into ints
                                n_jobs=n_jobs)
            clf.fit(train_x, 
                    train_y, 
                    sample_weight=sample_weight_train, 
                    early_stopping_rounds=5,
                    eval_metric="mlogloss", 
                    eval_set=[(test_x,test_y)], 
                    sample_weight_eval_set=[sample_weight_test], 
                    verbose=verbosity>2)
            if return_clfs:
                clfs.append(clf)
            # Predict the labels of the test set samples
            y_pred = clf.predict(test_x) # in case you try booster='dart' add, ntree_limit=1 (some value>0) check again
            # Append cv step results        
            confusion_matrices.append(confusion_matrix(test_y, 
                                                       y_pred, 
                                                       normalize="true", 
                                                       sample_weight=sample_weight_test)
                                     )
            if return_train_perform:
                y_pred = clf.predict(train_x)
                confusion_matrices_train.append(confusion_matrix(train_y, 
                                                                 y_pred, 
                                                                 normalize="true", 
                                                                 sample_weight=sample_weight_train)
                                               )
        
    # Pool confusion matrices
    confusions_merged = np.concatenate([np.expand_dims(mat, axis=-1) for mat in confusion_matrices],axis=-1)
    confusion_mean = pd.DataFrame(index=celltypes,columns=celltypes,data=np.mean(confusions_merged, axis=-1))
    confusion_std = pd.DataFrame(index=celltypes,columns=celltypes,data=np.std(confusions_merged, axis=-1))
    if return_train_perform:
        confusions_merged = np.concatenate([np.expand_dims(mat, axis=-1) for mat in confusion_matrices_train],axis=-1)
        confusion_mean_train = pd.DataFrame(index=celltypes,columns=celltypes,data=np.mean(confusions_merged, axis=-1))
        confusion_std_train = pd.DataFrame(index=celltypes,columns=celltypes,data=np.std(confusions_merged, axis=-1))
       
    # Return
    out = [confusion_mean, confusion_std]
    if return_train_perform:
        out += [confusion_mean_train, confusion_std_train]
    if return_clfs:
        out += [clfs]
    return out