import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

#################################
# Configure Evaluation Pipeline #
#################################
from spapros.evaluation.evaluation import knn_similarity, clustering_sets, nmi
from spapros.util.util import clean_adata, preprocess_adata

RESULTS_DIR = "../results/evaluation/"
NAME = "210222_test_eval"
PROBESETS_FILE = "../results/probesets/selections_genesets_1.csv"
PROBESET_IDS = ["genesets_1_0", "genesets_1_1", "genesets_1_13"]  # 'all'

dataset_params = {
    "data_path": ["../data/"],
    "dataset": ["small_data_raw_counts.h5ad"],
    "process_adata": [["norm", "log1p"], ["norm", "log1p", "scale"]],
}

metric_configs = {
    # Clustering similarity via normalized mutual information
    "nmi": {
        "ns": list(range(5, 21, 1)),
        "AUC_borders": [[7, 14], [15, 20]],
    },
    # Similarity of knn graphs
    "knn": {
        "ks": [5, 10, 15, 20, 25, 30],
    },
    # Marker list correlation
    "marker_corr": {
        # FOLLOWS SOON
    },
    # Forest classification
    "forests": {
        # FOLLOWS SOON
    },
    # Gene redundancy via coexpression
    "coexpr": {
        # FOLLOWS SOON
    },
}

######################################
# Produce some vars used in the pipe #
######################################

reference_dir = RESULTS_DIR + "references/"  # handy to reuse reference results for further evaluations
results_dir = RESULTS_DIR + NAME + "/"

Path(reference_dir).mkdir(parents=True, exist_ok=True)
Path(results_dir).mkdir(parents=True, exist_ok=True)

cartesian_product = list(itertools.product(*[param_list for _, param_list in dataset_params.items()]))
dataset_configs = [{key: val for key, val in zip(dataset_params, val_list)} for val_list in cartesian_product]

if PROBESET_IDS == "all":
    df = pd.read_csv(PROBESETS_FILE, index_col=0)
    probesets = df.columns.to_list()
    del df
else:
    probesets = PROBESET_IDS

####################################
# Save dataset configuration infos #
####################################

metric_cols = [f"{metric}_{k}" for metric, m_config in metric_configs.items() for k in m_config]
metric_vals = [v for metric, m_config in metric_configs.items() for _, v in m_config.items()]
dataset_cols = [k for k in dataset_params]
dataset_cols.remove("process_adata")

# not very elegant to save these infos in df_info maybe...
df_info = pd.DataFrame(
    columns=["results_directory", "reference_directory", "normalised", "log1p", "scaled"] + dataset_cols + metric_cols
)

for i, d_config in enumerate(dataset_configs):
    # dataset_name = d_config['dataset'].rsplit('.',1)[0]
    config_id = f"data_config_{i}"
    df_info.loc[config_id, ["results_directory", "reference_directory"]] = [results_dir, reference_dir]
    tmp = d_config["process_adata"] if ("process_adata" in d_config) else []
    pp_options = [("norm" in tmp), ("log1p" in tmp), ("scale" in tmp)]
    df_info.loc[config_id, ["normalised", "log1p", "scaled"]] = pp_options
    df_info.loc[config_id, dataset_cols] = [d_config[k] for k in dataset_cols]
    df_info.loc[config_id, metric_cols] = metric_vals

df_info.to_csv(RESULTS_DIR + NAME + ".csv")

############################################################
################ Actual Evaluation Pipeline ################
############################################################

###############################
# Clustering similarity (NMI) #
###############################
# This metric normally takes hours to calculate for a single probeset (on a dataset with 60k cells)
# For the prepared probesets it takes a few mins, further reduce the dataset size or the range of m_config['ns'] to make it faster
#
# In the Clustering similarity metric we compute leiden clusterings for different resolutions.
# - We are searching clusterings with all numbers of cluster as defined in 'ns'
#  (search through different resolutions till all ns are captured)
# - the clusterings are computed for all genes in adata (as reference/"ground truth") and only on the genes of the probeset
# - we only need to calculate the reference clusterings once, we compare all probesets against the reference
# - important note: the leiden algorithmus eventually does not find a resolution for a searched number of clusters.
# So there are some gaps that are interpolated
# it can be problematic if the gap is very large, ideally we have a warning for that case

for i, d_config in enumerate(dataset_configs):
    config_id = f"data_config_{i}"

    # Load dataset
    a = sc.read(d_config["data_path"] + d_config["dataset"])
    if d_config["process_adata"]:
        preprocess_adata(a, options=d_config["process_adata"])
    clean_adata(a)

    # Alias for current metric config
    m_config = metric_configs["nmi"]

    # Calculate the reference clusterings
    adata = a.copy()
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    clustering_sets(adata, m_config["ns"], reference_dir + f"{config_id}_assignments.csv", start_res=1.0, verbose=False)

    # Calculate the clusterings for each probeset
    for set_id in probesets:
        selection = pd.read_csv(PROBESETS_FILE, usecols=["index", set_id], index_col=0)
        genes = [g for g in selection.loc[selection[set_id]].index.to_list() if g in a.var.index]
        adata = a[:, genes].copy()
        clean_adata(adata)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        output_file = results_dir + f"{config_id}_{set_id}_assignments.csv"
        clustering_sets(adata, m_config["ns"], output_file, start_res=1.0, verbose=False)

    # Calculate nmis between reference and probeset clusterings and save them in `save_to`
    nmi_results_file = results_dir + f"nmi_{config_id}.csv"
    reference_file = reference_dir + f"{config_id}_assignments.csv"
    probeset_files = [results_dir + f"{config_id}_{set_id}_assignments.csv" for set_id in probesets]
    nmi(
        probeset_files,
        reference_file,
        nmi_results_file,
        m_config["ns"],
        method="arithmetic",
        names=None,
        verbose=False,
        save_every=10,
    )

    # Calculate AUCs of nmis
    # - Just realized it's a little problematic to have this in the pipeline since you normally want to first
    #   plot curves of the results from `nmi_results_file` before you set m_config['AUC_borders']. But I'd rly
    #   like to have the whole thing till the scalar metrics (AUCs) in one pipeline. A simple interactive plot
    #   as for expression constraints won't work, the nmi calculations take too long. Maybe the user can readjust the
    #   'AUC_borders' for a given data_config afterwards to recalculate the final results? idk..
    def AUC(series, n_min=1, n_max=60):
        tmp = series.loc[(series.index >= n_min) & (series.index <= n_max)]
        n = len(tmp)
        return tmp.sum() / n

    df = pd.read_csv(nmi_results_file, index_col=0)
    nmi_results_AUC_file = results_dir + f"nmi_AUCs_{config_id}.csv"
    df_nmi_AUCs = pd.DataFrame(index=probesets, columns=[f"nmi_{ths[0]}-{ths[1]}" for ths in m_config["AUC_borders"]])
    for probeset in probesets:
        for ths in m_config["AUC_borders"]:
            df_nmi_AUCs.loc[probeset, f"nmi_{ths[0]}-{ths[1]}"] = AUC(
                df[f"{config_id}_{probeset}_assignments"].interpolate(), n_min=ths[0], n_max=ths[1]
            )
    df_nmi_AUCs.to_csv(nmi_results_AUC_file)

########################
# Knn graph similarity #
########################
# Actually the ev.knn_similarity function that I wrote takes care of reference knn graph and selections knn graph calc
# note: if it's called several times simultaneously it might try to calculate the reference results several times.
# In practice I ran it once for the first probeset and then for all others afterwards in parallel
for i, d_config in enumerate(dataset_configs):
    config_id = f"data_config_{i}"

    # Load dataset
    adata = sc.read(d_config["data_path"] + d_config["dataset"])
    if d_config["process_adata"]:
        preprocess_adata(adata, options=d_config["process_adata"])
    clean_adata(adata)

    m_config = metric_configs["knn"]

    for set_id in probesets:
        selection = pd.read_csv(PROBESETS_FILE, usecols=["index", set_id], index_col=0)
        genes = [g for g in selection.loc[selection[set_id]].index.to_list() if g in adata.var.index]
        knn_similarity(
            genes,
            adata,
            ks=m_config["ks"],
            save_dir=results_dir,
            save_name=f"{config_id}_{set_id}",
            reference_dir=reference_dir,
            reference_name=config_id,
            bbknn_ref_key=None,
        )

    # Calculate AUC
    knn_results_AUC_file = results_dir + f"knn_AUCs_{config_id}.csv"
    df_knn_AUCs = pd.DataFrame(index=probesets, columns=["knn_AUC"])

    for set_id in probesets:
        df = pd.read_csv(results_dir + f"nn_similarity_{config_id}_{set_id}.csv", index_col=0)
        x = [int(x) for x in df.mean().index.values]
        y = df.mean().values
        tmp = pd.Series(index=range(np.min(x), np.max(x)))
        for i in range(len(y)):
            tmp.loc[x[i]] = y[i]
        df_knn_AUCs.loc[set_id, "knn_AUC"] = AUC(tmp.interpolate(), n_min=np.min(x), n_max=np.max(x))
    df_knn_AUCs.to_csv(knn_results_AUC_file)

#################################
# The other metrics will follow #
#################################
