from pathlib import Path
from typing import Union

import pandas as pd
import scanpy as sc
from rich.console import Console
from rich.progress import Progress
from ruamel.yaml import YAML

from spapros.evaluation import ProbesetEvaluator
from spapros.util.util import preprocess_adata

# CHANGES
# - use ProbesetEvaluator
#     - intrinsic file hierarchy
#     - generalised metric usage
# - reduce pipeline complexity
#     - we're supporting only one setting of reference data (run pipeline multiple times if you want multiple settings)
#     - for metric parameters we also only allow one setting... i don't know if for future extensions this might be
#       annoying, but for now it suffices and makes things simpler


console = Console()


def run_evaluation(
    adata_path: str,
    probeset: str,
    marker_file: str,
    result_dir: str,
    probeset_ids: Union[str, list] = "all",
    parameters_file: str = None,
) -> None:
    if parameters_file is not None:
        yaml = YAML(typ="safe")
        parameters = yaml.load(Path(parameters_file))
    else:
        parameters = {  # TODO create some reasonable default
            "data": {
                "name": "small_data",
                "process_adata": ["norm", "log1p"],
                "celltype_key": "celltype",
            },
            "metrics": {
                # Clustering similarity via normalized mutual information
                "cluster_similarity": {
                    "ns": [5, 21],
                    "AUC_borders": [[7, 14], [15, 20]],
                },
                # Similarity of knn graphs
                "knn_overlap": {
                    "ks": [5, 10, 15, 20, 25, 30],
                },
                # Forest classification
                "forest_clfs": {
                    "threshold": 0.8,
                },
                # Marker list correlation
                "marker_corr": {
                    "per_celltype": True,
                    "per_marker": True,
                    "per_celltype_min_mean": None,
                    "per_marker_min_mean": 0.025,
                },
                # Gene redundancy via coexpression
                "gene_corr": {
                    "threshold": 0.8,
                },
            },
        }

    dataset_params = parameters["data"]
    metric_configs = parameters["metrics"]

    results_dir = result_dir + f"{dataset_params['name']}/"
    # reference_dir = result_dir + "references/"  # handy to reuse reference results for further evaluations
    # reference_dir is a little tricky atm, since I don't save the info of previous hyperparameters the saved
    # reference files might be produced with wrong hyperparameters, therefore atm we use the default
    # reference_dir (of ProbesetEvaluator) which is within `results_dir`.

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # cartesian_product = list(itertools.product(*[param_list for _, param_list in dataset_params.items()]))  # type: ignore
    # dataset_configs = [{key: val for key, val in zip(dataset_params, val_list)} for val_list in cartesian_product]

    if probeset_ids == "all":
        df = pd.read_csv(probeset, index_col=0)
        probesets = df.columns.to_list()
        del df
    else:
        probesets = probeset_ids

    ####################################
    # Save dataset configuration infos #
    ####################################

    # WE LEAVE THIS OUT FOR NOW, would be nice to save the info in general though...
    #
    # metric_cols = [f"{metric}_{k}" for metric, m_config in metric_configs.items() for k in m_config]
    # metric_vals = [v for metric, m_config in metric_configs.items() for _, v in m_config.items()]
    # dataset_cols = [k for k in dataset_params]
    # dataset_cols.remove("process_adata")
    #
    ## not very elegant to save these infos in df_info maybe...
    # df_info = pd.DataFrame(
    #    columns=["results_directory", "reference_directory", "normalised", "log1p", "scaled"]
    #    + dataset_cols
    #    + metric_cols
    # )
    #
    # for i, d_config in enumerate(dataset_configs):
    #    # dataset_name = d_config['dataset'].rsplit('.',1)[0]
    #    config_id = f"data_config_{i}"
    #    df_info.loc[config_id, ["results_directory", "reference_directory"]] = [results_dir, reference_dir]
    #    tmp = d_config["process_adata"] if ("process_adata" in d_config) else []
    #    pp_options = [("norm" in tmp), ("log1p" in tmp), ("scale" in tmp)]
    #    df_info.loc[config_id, ["normalised", "log1p", "scaled"]] = pp_options
    #    df_info.loc[config_id, dataset_cols] = [d_config[k] for k in dataset_cols]
    #    df_info.loc[config_id, metric_cols] = metric_vals
    #
    # df_info.to_csv(results_dir + NAME + ".csv")

    ##############################
    # Actual Evaluation Pipeline #
    ##############################

    with Progress() as progress:
        evaluation_task = progress.add_task("[bold blue]Performing evaluation...", total=8)

        shared_tasks = progress.add_task(
            "[bold blue]Computations shared for each probeset", total=len([key for key in metric_configs]) - 1
        )
        clust_sim_pre_task = progress.add_task("[bold blue]Clustering similarity step 1", total=len(probesets))
        knn_pre_task = progress.add_task("[bold blue]KNN Graph Overlap step 1", total=len(probesets))
        forest_task = progress.add_task("[bold blue]Forest Cell Type Classification", total=len(probesets))
        clust_sim_task = progress.add_task("[bold blue]Clustering similarity", total=len(probesets))
        knn_task = progress.add_task("[bold blue]KNN Graph Overlap", total=len(probesets))
        corr_task = progress.add_task("[bold blue]Gene and Marker Correlation", total=len(probesets))
        summary_task = progress.add_task("[bold blue]Summary Statistics", total=1)

        adata = sc.read(adata_path)  # type: ignore
        if dataset_params["process_adata"]:
            preprocess_adata(adata, options=dataset_params["process_adata"])

        def get_genes(set_id, probesets_file=probeset, var_names=adata.var_names):
            """ """
            selection = pd.read_csv(probesets_file, usecols=["index", set_id], index_col=0)
            genes = [g for g in selection.loc[selection[set_id]].index.to_list() if g in var_names]
            return genes

        evaluator_kwargs = dict(
            scheme="custom",
            results_dir=results_dir,
            celltype_key=dataset_params["celltype_key"],
            marker_list=marker_file,
            metrics_params=metric_configs,
            reference_name=dataset_params["name"],
        )

        ##############################
        # 1.1 Compute shared results #
        ##############################
        # Note: forest_clfs doesn't have any shared computations

        # Process 1.1.1 shared cluster_similarity
        # output file: results_dir+f"references/{dataset_params["name"]}_cluster_similarity.csv"

        evaluator = ProbesetEvaluator(adata, metrics=["cluster_similarity"], **evaluator_kwargs)
        evaluator.compute_or_load_shared_results()
        progress.advance(shared_tasks)

        # Process 1.1.2 shared knn_overlap
        # output file: results_dir+f"references/{dataset_params["name"]}_knn_overlap.csv"

        evaluator = ProbesetEvaluator(adata, metrics=["knn_overlap"], **evaluator_kwargs)
        evaluator.compute_or_load_shared_results()
        progress.advance(shared_tasks)

        # Process 1.1.3 shared other metrics (pool metrics since they don't take too long)
        # output files:
        #    - results_dir+f"references/{dataset_params["name"]}_gene_corr.csv"
        #    - results_dir+f"references/{dataset_params["name"]}_marker_corr.csv"

        evaluator = ProbesetEvaluator(adata, metrics=["gene_corr", "marker_corr"], **evaluator_kwargs)
        evaluator.compute_or_load_shared_results()
        progress.advance(shared_tasks)
        progress.advance(shared_tasks)

        progress.advance(evaluation_task)

        ##############################################
        # 1.2 Compute probe set specific pre results #
        ##############################################
        # Note: no pre results for forest_clfs, marker_corr, gene_corr

        # Process 1.2.1 probeset specific cluster_similarity pre results
        # Parallelised: one process for each set_id
        # output file: results_dir+f"cluster_similarity/{dataset_params["name"]}_{set_id}_pre.csv"

        for set_id in probesets:
            evaluator = ProbesetEvaluator(adata, metrics=["cluster_similarity"], **evaluator_kwargs)
            genes = get_genes(set_id)
            evaluator.evaluate_probeset(genes, set_id=set_id, pre_only=True)
            progress.advance(clust_sim_pre_task)

        progress.advance(evaluation_task)

        # Process 1.2.2 probeset specific knn_overlap pre results
        # Parallelised: one process for each set_id
        # output file: results_dir+f"knn_overlap/{dataset_params["name"]}_{set_id}_pre.csv"

        for set_id in probesets:
            evaluator = ProbesetEvaluator(adata, metrics=["knn_overlap"], **evaluator_kwargs)
            genes = get_genes(set_id)
            evaluator.evaluate_probeset(genes, set_id=set_id, pre_only=True)
            progress.advance(knn_pre_task)

        progress.advance(evaluation_task)

        ##########################################################
        # 1.3 Compute probe set specific results for forest_clfs #
        ##########################################################

        # Process 1.3 probeset specific forest_clfs results
        # Parallelised: one process for each set_id
        # output file: results_dir+f"forest_clfs/{dataset_params["name"]}_{set_id}.csv"

        for set_id in probesets:
            evaluator = ProbesetEvaluator(adata, metrics=["forest_clfs"], **evaluator_kwargs)
            genes = get_genes(set_id)
            evaluator.evaluate_probeset(genes, set_id=set_id, update_summary=False)
            progress.advance(forest_task)

        progress.advance(evaluation_task)

        ##########################################
        # 2.1 Compute probe set specific results #
        ##########################################

        # Process 2.1.1 probeset specific cluster_similarity results
        # Prerequisites: Process 1.1.1 and Processes 1.2.1
        # Sequential: over set_ids (no need to parallelize, doesn't take too long)
        # output files:
        #    - for all set_ids: results_dir+f"cluster_similarity/{dataset_params["name"]}_{set_id}.csv"
        # input files:
        #    - for all set_ids: results_dir+f"cluster_similarity/{dataset_params["name"]}_{set_id}_pre.csv"
        #    - results_dir+f"references/{dataset_params["name"]}_cluster_similarity.csv"

        evaluator = ProbesetEvaluator(adata, metrics=["cluster_similarity"], **evaluator_kwargs)
        for set_id in probesets:
            genes = get_genes(set_id)
            evaluator.evaluate_probeset(genes, set_id=set_id, update_summary=False)
            progress.advance(clust_sim_task)

        progress.advance(evaluation_task)

        # Process 2.1.2 probeset specific knn_overlap results
        # Prerequisites: Process 1.1.2 and Processes 1.2.2
        # Sequential: over set_ids (no need to parallelize, doesn't take too long)
        # output files:
        #    - for all set_ids: results_dir+f"knn_overlap/{dataset_params["name"]}_{set_id}.csv"
        # input files:
        #    - for all set_ids: results_dir+f"knn_overlap/{dataset_params["name"]}_{set_id}_pre.csv"
        #    - results_dir+f"references/{dataset_params["name"]}_knn_overlap.csv"

        evaluator = ProbesetEvaluator(adata, metrics=["knn_overlap"], **evaluator_kwargs)
        for set_id in probesets:
            genes = get_genes(set_id)
            evaluator.evaluate_probeset(genes, set_id=set_id, update_summary=False)
            progress.advance(knn_task)

        progress.advance(evaluation_task)

        # Process 2.1.3 probeset specific marker_corr and gene_corr results
        # Prerequisites: Process 1.1.3
        # Sequential: over set_ids (no need to parallelize, doesn't take too long)
        # output files:
        #    - for all set_ids: results_dir+f"gene_corr/{dataset_params["name"]}_{set_id}.csv"
        #    - for all set_ids: results_dir+f"marker_corr/{dataset_params["name"]}_{set_id}.csv"
        # input files:
        #    - results_dir+f"references/{dataset_params["name"]}_gene_corr.csv"
        #    - results_dir+f"references/{dataset_params["name"]}_marker_corr.csv"

        evaluator = ProbesetEvaluator(adata, metrics=["gene_corr", "marker_corr"], **evaluator_kwargs)
        for set_id in probesets:
            genes = get_genes(set_id)
            evaluator.evaluate_probeset(genes, set_id=set_id, update_summary=False)
            progress.advance(corr_task)

        progress.advance(evaluation_task)

        #############################
        # 3. Get summary statistics #
        #############################

        # Process 3 Get summary statistics
        # Prerequisites: All previous Processes
        # output_file: results_dir+f"{dataset_params["name"]}_summary.csv"
        # input_files:
        #   - for all metrics and all set_ids: results_dir+f"{metric}/{dataset_params["name"]}_{set_id}.csv"

        evaluator = ProbesetEvaluator(
            adata,
            metrics=["cluster_similarity", "knn_overlap", "forest_clfs", "gene_corr", "marker_corr"],
            **evaluator_kwargs,
        )
        evaluator.summary_statistics(probesets)
        progress.advance(summary_task)
        progress.advance(evaluation_task)

    console.print(f"[bold blue]Wrote results to {results_dir}")
