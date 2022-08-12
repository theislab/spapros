import itertools
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import scanpy as sc
from rich.console import Console
from rich.progress import Progress
from spapros.selection.selection_methods import highest_expressed_genes
from spapros.selection.selection_methods import random_selection
from spapros.selection.selection_methods import select_DE_genes
from spapros.selection.selection_methods import select_pca_genes
from spapros.util.util import plateau_penalty_kernel
from spapros.util.util import preprocess_adata
from spapros.util.util import transfered_expression_thresholds

console = Console()


def run_selection(adata_path: str, output_path: str) -> None:
    """Runs Spapros probeset selection.

    Args:
        adata_path: Path to the AnnData object.
        output_path: Output path of the selection results.

    Example:

    .. code-block:: python

        import spapros as sp
        selection_result = sp.run_selection
    """
    adata = sc.read(adata_path)

    a = preprocess_adata(adata, options=["norm", "log1p"], inplace=False)

    with console.status("Selecting genes with PCA..."):
        selected_df = select_pca_genes(
            a,
            100,
            variance_scaled=False,
            absolute=True,
            n_pcs=20,
            penalty_keys=[],
            corr_penalty=None,
            inplace=False,
        )
        console.print(f"[bold blue]Using {len(selected_df)} genes")

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Map expression thresholds
    # From experimental partners we know good thresholds for the basic normalisation to a target_sum of 10000.
    # However, we use scran normalisation for our datasets.
    lower_th, upper_th = transfered_expression_thresholds(
        adata, lower=2, upper=6, tolerance=0.05, target_sum=10000, output_path=output_path, plot=True
    )

    # From the plots we decide for factor = 0.1
    factor = 0.1
    var = [factor * 0.1, factor * 0.5]
    penalty = plateau_penalty_kernel(var=var, x_min=lower_th, x_max=upper_th)
    adata.var["expression_penalty"] = penalty(adata.var["quantile_0.99"])
    a = preprocess_adata(adata, options=["norm", "log1p"], inplace=False)
    select_pca_genes(
        a,
        100,
        variance_scaled=False,
        absolute=True,
        n_pcs=20,
        penalty_keys=["expression_penalty"],
        corr_penalty=None,
        inplace=True,
    )

    #######################################################################################
    # Systematic selections pipeline to create probeset files for the evaluation pipeline #
    #######################################################################################
    NAME = "genesets_1"  # Provide a name for the systematic selections in case we want to combine different experiments later

    general_params = {
        "n": [20, 100],
        "penalty_keys": [[]],
        "gene_subset": [None],
        # We could add a key from adata.var here: e.g. different numbers of highly variable genes
        # (the key needs to be prepared in `adata.var[key]` (dtype=bool) --> adata = adata[:,adata.var[key]] )
        "obs_fraction": [None],  # For measuring method stability
        "fraction_seed": [None],  # (obs_fraction and fraction_seed only work in combination)
    }
    pca_params = {
        "variance_scaled": [False, True],
        "absolute": [True],
        "n_pcs": [10, 20, 30],
        "process_adata": [["norm", "log1p"], ["norm", "log1p", "scale"]],
        "corr_penalty": [None],
    }
    DE_params = {
        "obs_key": ["celltype"],
        "rankby_abs": [False],
        "process_adata": [["norm", "log1p"]],
    }  # note we left out several parameters where we want the standard settings
    random_params = {
        "seed": list(np.random.choice(10000, 50, replace=False))
    }  # seed: important to measure stochasticity of stochastic methods
    highest_expr_params = {"process_adata": [["norm", "log1p"]]}

    method_params = {
        "pca": pca_params,
        "DE": DE_params,
        "random": random_params,
        "highest_expr": highest_expr_params,
    }

    methods = {
        "pca": select_pca_genes,
        "DE": select_DE_genes,
        "random": random_selection,
        "highest_expr": highest_expressed_genes,
    }

    methods_kwargs = {
        "pca": [
            "n",
            "variance_scaled",
            "absolute",
            "n_pcs",
            "process_adata",
            "penalty_keys",
            "corr_penalty",
        ],
        "DE": ["n", "obs_key", "process_adata", "penalty_keys", "rankby_abs"],
        "random": ["n", "seed"],
        "highest_expr": ["n", "process_adata"],
    }

    # Reshape configs for single selections #
    with console.status("Preparing selection..."):
        cartesian_product = list(
            itertools.product(*[param_list for _, param_list in general_params.items()])  # type: ignore
        )
        general_configs = [{key: val for key, val in zip(general_params, val_list)} for val_list in cartesian_product]

        method_configs = {}
        for method, params in method_params.items():
            cartesian_product = list(itertools.product(*[param_list for _, param_list in params.items()]))  # type: ignore
            method_configs[method] = [
                {key: val for key, val in zip(params, val_list)} for val_list in cartesian_product  # type: ignore
            ]

        Path(output_path).mkdir(parents=True, exist_ok=True)

        param_keys = list(
            np.unique(
                [k for k in general_params] + [k for _, m_params in method_params.items() for k in m_params]  # type: ignore
            )
        )
        param_keys.remove("process_adata")

        df_info = pd.DataFrame(
            columns=["directory", "method", "normalised", "log1p", "scaled", "time_seconds"] + param_keys
        )
        df_info.index.name = "set_id"
        df_sets = pd.DataFrame(index=adata.var.index)

    count = 0
    with Progress() as progress:
        configuration_task = progress.add_task("[bold blue]Performing selection...", total=len(general_configs))
        for g_config in general_configs:
            method_task = progress.add_task("[bold blue]Running method grid...", total=len(method_configs.keys()))
            for method, m_configs in method_configs.items():
                progress.console.print(f"[bold green]Running: {method}")
                for m_config in m_configs:
                    adata = sc.read(adata_path)
                    kwargs = {k: v for k, v in g_config.items() if k in methods_kwargs[method]}
                    kwargs.update(
                        {k: v for k, v in m_config.items() if (k in methods_kwargs[method]) & (k != "process_adata")}
                    )
                    if "process_adata" in m_config:
                        preprocess_adata(adata, options=m_config["process_adata"], inplace=True)
                    start = timer()
                    generated_selection = methods[method](adata, **kwargs, inplace=False)  # type: ignore
                    # TODO actually the dataset processing shouldn't be part of the measured computation time...
                    computation_time = timer() - start

                    df_info.loc[f"{NAME}_{count}", ["directory", "method", "time_seconds"]] = [
                        output_path,
                        method,
                        computation_time,
                    ]
                    g_config_cols = [k for k in g_config if k in df_info.columns]
                    g_config_values = [v if (not isinstance(v, list)) else "-".join(v) for k, v in g_config.items()]
                    df_info.loc[f"{NAME}_{count}", g_config_cols] = g_config_values
                    kwarg_cols = [k for k in kwargs if k in df_info.columns]
                    kwarg_values = [v if (not isinstance(v, list)) else "-".join(v) for k, v in kwargs.items()]
                    df_info.loc[f"{NAME}_{count}", kwarg_cols] = kwarg_values
                    tmp = m_config["process_adata"] if ("process_adata" in m_config) else []
                    pp_options = [("norm" in tmp), ("log1p" in tmp), ("scale" in tmp)]
                    df_info.loc[f"{NAME}_{count}", ["normalised", "log1p", "scaled"]] = pp_options

                    df_sets[f"{NAME}_{count}"] = generated_selection["selection"]

                    count += 1
                progress.advance(method_task)
            progress.advance(configuration_task)

            df_info.to_csv(output_path + f"selections_info_{NAME}.csv")
            df_sets.to_csv(output_path + f"selections_{NAME}.csv")

    console.print(f"[bold blue]Wrote results to {output_path}")
