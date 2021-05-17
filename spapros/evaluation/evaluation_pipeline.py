import itertools
from pathlib import Path
from spapros.evaluation.evaluation import clustering_sets
from spapros.evaluation.evaluation import knn_similarity
from spapros.evaluation.evaluation import nmi
from spapros.util.util import clean_adata
from spapros.util.util import cluster_corr
from spapros.util.util import dict_to_table
from spapros.util.util import gene_means
from spapros.util.util import preprocess_adata
from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
from rich.console import Console
from rich.progress import Progress
from scipy.sparse import issparse

console = Console()

# TODO below must become a parameter :)
spapros_dir = "/mnt/home/icb/louis.kuemmerle/projects/st_probesets/spapros/"  # "/home/zeth/PycharmProjects/spapros/"

dataset_params = {
    "data_path": [spapros_dir + "data/"],
    "dataset": ["small_data_raw_counts.h5ad"],
    "process_adata": [["norm", "log1p"], ["norm", "log1p", "scale"]],
}

metric_configs: Dict[str, Any] = {
    # Clustering similarity via normalized mutual information
    "nmi": {
        "ns": list(range(5, 21, 1)),
        "AUC_borders": [[7, 14], [15, 20]],
    },
    # Similarity of knn graphs
    "knn": {
        "ks": [5, 10, 15, 20, 25, 30],
        # TODO: Add uniform weighted metric. And here some argument for different weightings
        # "weighting": {"no key":["data"], f"{CT_KEY}":["uniform"]} ... need to think about this
    },
    # Marker list correlation
    "marker_corr": {
        "marker_list": spapros_dir + "data/small_data_marker_list.csv",
        "per_celltype": True,
        "per_marker": True,
        "per_celltype_min_mean": [None],
        "per_marker_min_mean": [0.025],
    },
    # Forest classification
    "forests": {
        # FOLLOWS SOON
    },
    # Gene redundancy via coexpression
    "coexpr": {
        "threshold": [0.8],
    },
}


def run_evaluation(probeset: str, result_dir: str) -> None:
    NAME = "210222_test_eval"
    PROBESET_IDS: Union[str, list] = ["genesets_1_0", "genesets_1_1", "genesets_1_13"]  # 'all'

    reference_dir = result_dir + "references/"  # handy to reuse reference results for further evaluations
    results_dir = result_dir + NAME + "/"

    Path(reference_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    cartesian_product = list(itertools.product(*[param_list for _, param_list in dataset_params.items()]))  # type: ignore
    dataset_configs = [{key: val for key, val in zip(dataset_params, val_list)} for val_list in cartesian_product]

    if PROBESET_IDS == "all":
        df = pd.read_csv(probeset, index_col=0)
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
        columns=["results_directory", "reference_directory", "normalised", "log1p", "scaled"]
        + dataset_cols
        + metric_cols
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

    df_info.to_csv(results_dir + NAME + ".csv")

    ##############################
    # Actual Evaluation Pipeline #
    ##############################

    with Progress() as progress:
        evaluation_task = progress.add_task(
            "[bold blue]Performing evaluation...", total=4
        )  # COOKIETEMPLE TODO: Currently hardcoded!
        # TODO: Preference? I added the bars right at the beginning to directly have an overview which metrics gets calculated when
        #       if that's bad practice: move each task back to the beginning of the metric computation
        nmi_task = progress.add_task("[bold blue]Clustering similarity (NMI)", total=len(dataset_configs))
        knn_task = progress.add_task("[bold blue]KNN Graph Similarity", total=len(dataset_configs))
        marker_task = progress.add_task("[bold blue]Marker List Correlation", total=len(dataset_configs))
        coexpr_task = progress.add_task("[bold blue]Gene Redundancy By Correlation", total=len(dataset_configs))

        ###############################
        # Clustering similarity (NMI) #
        ###############################
        # This metric normally takes hours to calculate for a single probeset (on a dataset with 60k cells)
        # In the Clustering similarity metric we compute leiden clusterings for different resolutions.
        # - We are searching clusterings with all numbers of cluster as defined in 'ns'
        #  (search through different resolutions till all ns are captured)
        # - the clusterings are computed for all genes in adata (as reference/"ground truth") and only on the genes of the probeset
        # - we only need to calculate the reference clusterings once, we compare all probesets against the reference
        # - important note: the leiden algorithmus eventually does not find a resolution for a searched number of clusters.
        # So there are some gaps that are interpolated
        # it can be problematic if the gap is very large, ideally we have a warning for that case
        nmi_res_dir = results_dir + "clustering_similarity/"
        Path(nmi_res_dir).mkdir(parents=True, exist_ok=True)

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
            clustering_sets(
                adata, m_config["ns"], reference_dir + f"{config_id}_assignments.csv", start_res=1.0, verbose=False
            )

            # Calculate the clusterings for each probeset
            for set_id in probesets:
                selection = pd.read_csv(probeset, usecols=["index", set_id], index_col=0)
                genes = [g for g in selection.loc[selection[set_id]].index.to_list() if g in a.var.index]
                adata = a[:, genes].copy()
                clean_adata(adata)
                sc.tl.pca(adata)
                sc.pp.neighbors(adata)
                output_file = nmi_res_dir + f"{config_id}_{set_id}_assignments.csv"
                clustering_sets(adata, m_config["ns"], output_file, start_res=1.0, verbose=False)

            # Calculate nmis between reference and probeset clusterings and save them in `save_to`
            nmi_results_file = nmi_res_dir + f"nmi_{config_id}.csv"
            reference_file = reference_dir + f"{config_id}_assignments.csv"
            probeset_files = [nmi_res_dir + f"{config_id}_{set_id}_assignments.csv" for set_id in probesets]
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
            nmi_results_AUC_file = results_dir + f"summary_nmi_AUCs_{config_id}.csv"
            df_nmi_AUCs = pd.DataFrame(
                index=probesets, columns=[f"nmi_{ths[0]}-{ths[1]}" for ths in m_config["AUC_borders"]]
            )
            for set_id in probesets:
                for ths in m_config["AUC_borders"]:
                    df_nmi_AUCs.loc[set_id, f"nmi_{ths[0]}-{ths[1]}"] = AUC(
                        df[f"{config_id}_{set_id}_assignments"].interpolate(), n_min=ths[0], n_max=ths[1]
                    )
            df_nmi_AUCs.to_csv(nmi_results_AUC_file)

            progress.advance(nmi_task)

        progress.advance(evaluation_task)

        ########################
        # Knn graph similarity #
        ########################
        # Actually the ev.knn_similarity function that I wrote takes care of reference knn graph and selections knn graph calc
        # note: if it's called several times simultaneously it might try to calculate the reference results several times.
        # In practice I ran it once for the first probeset and then for all others afterwards in parallel
        knn_res_dir = results_dir + "knn_similarity/"
        Path(knn_res_dir).mkdir(parents=True, exist_ok=True)

        for i, d_config in enumerate(dataset_configs):
            config_id = f"data_config_{i}"

            # Load dataset
            adata = sc.read(d_config["data_path"] + d_config["dataset"])
            if d_config["process_adata"]:
                preprocess_adata(adata, options=d_config["process_adata"])
            clean_adata(adata)

            m_config = metric_configs["knn"]

            for set_id in probesets:
                selection = pd.read_csv(probeset, usecols=["index", set_id], index_col=0)
                genes = [g for g in selection.loc[selection[set_id]].index.to_list() if g in adata.var.index]
                knn_similarity(
                    genes,
                    adata,
                    ks=m_config["ks"],
                    save_dir=knn_res_dir,
                    save_name=f"{config_id}_{set_id}",
                    reference_dir=reference_dir,
                    reference_name=config_id,
                    bbknn_ref_key=None,
                )

            # Calculate AUC
            knn_results_AUC_file = results_dir + f"summary_knn_AUCs_{config_id}.csv"
            df_knn_AUCs = pd.DataFrame(index=probesets, columns=["knn_AUC"])

            for set_id in probesets:
                df = pd.read_csv(knn_res_dir + f"nn_similarity_{config_id}_{set_id}.csv", index_col=0)
                x = [int(x) for x in df.mean().index.values]
                y = df.mean().values
                tmp = pd.Series(index=range(np.min(x), np.max(x)))
                for i in range(len(y)):
                    tmp.loc[x[i]] = y[i]
                df_knn_AUCs.loc[set_id, "knn_AUC"] = AUC(tmp.interpolate(), n_min=np.min(x), n_max=np.max(x))
            df_knn_AUCs.to_csv(knn_results_AUC_file)

            progress.advance(knn_task)

        progress.advance(evaluation_task)

        ###########################
        # Marker List Correlation #
        ###########################
        marker_res_dir = results_dir + "marker_list/"
        Path(marker_res_dir).mkdir(parents=True, exist_ok=True)

        for i, d_config in enumerate(dataset_configs):
            config_id = f"data_config_{i}"

            # Load dataset
            adata = sc.read(d_config["data_path"] + d_config["dataset"])
            if d_config["process_adata"]:
                preprocess_adata(adata, options=d_config["process_adata"])
            clean_adata(adata)

            m_config = metric_configs["marker_corr"]

            # Compute correlation matrix
            if issparse(adata.X):
                full_cor_mat = pd.DataFrame(
                    index=adata.var.index,
                    columns=adata.var.index,
                    data=np.abs(np.corrcoef(adata.X.toarray(), rowvar=False)),
                )
            else:
                full_cor_mat = pd.DataFrame(
                    index=adata.var.index, columns=adata.var.index, data=np.abs(np.corrcoef(adata.X, rowvar=False))
                )

            marker_list = m_config["marker_list"]

            # Load marker_list as dict
            if isinstance(marker_list, str):
                marker_list = pd.read_csv(marker_list, index_col=0)
                marker_list = dict_to_table(marker_list, genes_as_index=False, reverse=True)

            # Get markers and their corresponding celltype
            markers = [g for _, genes in marker_list.items() for g in genes]
            markers = np.unique(markers).tolist()  # TODO: maybe throw a warning if genes occur twice
            # wrt celltype we take the first occuring celltype in marker_list for a given marker
            markers = [
                g for g in markers if g in adata.var_names
            ]  # TODO: Throw warning if marker genes are not in adata
            ct_annot = []
            for g in markers:
                for ct, genes in marker_list.items():
                    if g in genes:
                        ct_annot.append(ct)
                        break  # to only add first celltype in list

            # Dataframes for calculations and potential min mean expression filters
            corr_df_marker = full_cor_mat.loc[markers]
            df_mean_filter = gene_means(adata, genes=markers, key="mean", inplace=False).loc[markers]
            # TODO: we might need to calculate the means at a previous point in case adata.X was scaled
            #      (or we just say this metric only makes sense on unscaled data if a min_mean is given)

            # Prepare names (`cols`) of resulting summary statistics
            cols = []
            for mode in ["per_celltype", "per_marker"]:
                if m_config[mode]:
                    for min_mean in m_config[f"{mode}_min_mean"]:
                        tmp_str = "" if (min_mean is None) else f"_mean>{min_mean}"
                        cols.append(f"{mode}{tmp_str}")
                        if min_mean and not (str(min_mean) in df_mean_filter.columns):
                            df_mean_filter[str(min_mean)] = df_mean_filter["mean"] > min_mean

            # Run metric for each probeset
            result_tables = {}
            for set_id in probesets:
                selection = pd.read_csv(probeset, usecols=["index", set_id], index_col=0)
                genes = [g for g in selection.loc[selection[set_id]].index.to_list() if g in adata.var.index]
                corrs = pd.DataFrame(index=corr_df_marker.index, columns=["max_cor", "celltype"] + cols)
                corrs["celltype"] = ct_annot
                corrs["max_cor"] = corr_df_marker[genes].max(axis=1)

                if m_config["per_celltype"]:
                    for min_mean in m_config["per_celltype_min_mean"]:
                        col = "per_celltype" if (min_mean is None) else f"per_celltype_mean>{min_mean}"
                        corrs[col] = corrs["max_cor"]
                        if not (min_mean is None):
                            corrs.loc[~df_mean_filter[str(min_mean)], col] = np.nan
                        idxs = corrs.groupby(["celltype"])["max_cor"].transform(max) == corrs["max_cor"]
                        corrs.loc[~idxs, col] = np.nan

                if m_config["per_marker"]:
                    for min_mean in m_config["per_marker_min_mean"]:
                        col = "per_marker" if (min_mean is None) else f"per_marker_mean>{min_mean}"
                        corrs[col] = corrs["max_cor"]
                        if not (min_mean is None):
                            corrs.loc[~df_mean_filter[str(min_mean)], col] = np.nan

                corrs.to_csv(marker_res_dir + f"correlations_{set_id}_{config_id}.csv")
                result_tables[set_id] = corrs  # TODO: same scheme as for the other metrics would be to only save
                # the intermediate results (corrs) and load them instead of using result_tables

            # Calculate and save summary statistics
            df = pd.DataFrame(index=probesets, columns=cols)
            for set_id in probesets:
                df.loc[set_id] = result_tables[set_id][cols].mean(axis=0)
            df.to_csv(results_dir + f"summary_marker_list_correlatios_{config_id}.csv")

            progress.advance(marker_task)

        progress.advance(evaluation_task)

        ####################
        # Gene correlation #
        ####################
        cor_res_dir = results_dir + "coexpression/"
        Path(cor_res_dir).mkdir(parents=True, exist_ok=True)

        for i, d_config in enumerate(dataset_configs):
            config_id = f"data_config_{i}"
            # Load dataset
            adata = sc.read(d_config["data_path"] + d_config["dataset"])
            if d_config["process_adata"]:
                preprocess_adata(adata, options=d_config["process_adata"])
            clean_adata(adata)

            m_config = metric_configs["coexpr"]

            if issparse(adata.X):
                full_cor_mat = pd.DataFrame(
                    index=adata.var.index, columns=adata.var.index, data=np.corrcoef(adata.X.toarray(), rowvar=False)
                )
            else:
                full_cor_mat = pd.DataFrame(
                    index=adata.var.index, columns=adata.var.index, data=np.corrcoef(adata.X, rowvar=False)
                )

            df = pd.DataFrame(
                index=probesets, columns=["1 - mean corr"] + [f"corr < {th}" for th in m_config["threshold"]]
            )
            for set_id in probesets:
                selection = pd.read_csv(probeset, usecols=["index", set_id], index_col=0)
                genes = [g for g in selection.loc[selection[set_id]].index.to_list() if g in adata.var.index]
                cor_mat = full_cor_mat.loc[genes, genes]
                if True:  # intermediate_results.
                    # TODO: decide if we want to provide the option to not calculate intermediate results (speed up).
                    #       for the summary metrics the following three lines are not necessary, so we could skip them.
                    #       But for more specific plots they are necessary.
                    cluster_corr(cor_mat, inplace=True)
                    cor_mat.to_csv(cor_res_dir + f"cor_matrix_{set_id}_{config_id}.csv")
                cor_mat = np.abs(cor_mat)
                np.fill_diagonal(cor_mat.values, 0)
                percent_below_th = []
                for th in m_config["threshold"]:
                    percent_below_th.append((np.sum(np.max(cor_mat, axis=0) < th) / len(cor_mat)))
                cor_mat.values[~(np.arange(cor_mat.shape[0])[:, None] > np.arange(cor_mat.shape[1]))] = np.nan
                one_minus_mean_corr = 1 - np.nanmean(cor_mat)

                df.loc[set_id, ["1 - mean corr"] + [f"corr < {th}" for th in m_config["threshold"]]] = [
                    one_minus_mean_corr
                ] + percent_below_th
            df.to_csv(results_dir + f"summary_coexpression_{config_id}.csv")

            progress.advance(coexpr_task)

        progress.advance(evaluation_task)

    #################################
    # The other metrics will follow #
    #################################

    console.print(f"[bold blue]Wrote results to {results_dir}")
