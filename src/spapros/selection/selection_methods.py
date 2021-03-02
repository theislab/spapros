import itertools
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import selection_methods as select
import utils

# load test dataset
adata = sc.read("../data/small_data_raw_counts.h5ad")

####################################################################
# Demonstrate selections and how they should be saved in the adata #
####################################################################
# We use `select_pca_genes()` as an example here

# Return dataframe with info (`inplace=False`)
# - the dataframe contains a boolean column 'selection' with the 100 selected genes
# - method specific additional infos are also saved (e.g. here: 'selection_score' and 'selection_ranking')
select_df = select.select_pca_genes(
    adata,
    100,
    variance_scaled=False,
    absolute=True,
    n_pcs=20,
    process_adata=["norm", "log1p"],
    penalty_keys=[],
    corr_penalty=None,
    inplace=False,
    verbose=True,
)
print(select_df)


# Add selection info to adata (`inplace=True`)
select.select_pca_genes(
    adata,
    100,
    variance_scaled=False,
    absolute=True,
    n_pcs=20,
    process_adata=["norm", "log1p"],
    penalty_keys=[],
    corr_penalty=None,
    inplace=True,
    verbose=True,
)


##########################
# Demonstrate constraint #
##########################

### Map expression thresholds
# From experimental partners we know good thresholds for the basic normalisation to a target_sum of 10000.
# However, we use scran normalisation for our datasets.
lower_th, upper_th = utils.transfered_expression_thresholds(
    adata, lower=2, upper=6, tolerance=0.05, target_sum=10000, plot=True
)


def explore_constraint(factors=[10, 1, 0.1], q=0.99, lower=lower_th, upper=upper_th):
    """Plot histogram of quantiles for selected genes for different penalty kernels

    How to generalize the plotting function, support:
    - any selection method with defined hyperparameters
    - any penalty kernel
    - any key to be plotted (not only quantiles)
    """
    legend_size = 9
    factors = [10, 1, 0.1]
    rows = 1
    cols = len(factors)
    sizefactor = 6

    gaussians = []
    a = []
    selections_tmp = []
    for i, factor in enumerate(factors):
        x_min = lower_th
        x_max = upper_th
        var = [factor * 0.1, factor * 0.5]
        gaussians.append(
            utils.plateau_penalty_kernel(var=var, x_min=x_min, x_max=x_max)
        )

        a.append(adata.copy())

        a[i].var["penalty_expression"] = gaussians[i](a[i].var[f"quantile_{q}"])
        selections_tmp.append(
            select.select_pca_genes(
                a[i],
                100,
                variance_scaled=False,
                absolute=True,
                n_pcs=20,
                process_adata=["norm", "log1p", "scale"],
                penalty_keys=["penalty_expression"],
                corr_penalty=None,
                inplace=False,
                verbose=True,
            )
        )
        print(f"N genes selected: {np.sum(selections_tmp[i]['selection'])}")

    plt.figure(figsize=(sizefactor * cols, 0.7 * sizefactor * rows))
    for i, factor in enumerate(factors):
        ax1 = plt.subplot(rows, cols, i + 1)
        hist_kws = {"range": (0, np.max(a[i].var[f"quantile_{q}"]))}
        bins = 100
        sns.distplot(
            a[i].var[f"quantile_{q}"],
            kde=False,
            label="highly_var",
            bins=bins,
            hist_kws=hist_kws,
        )
        sns.distplot(
            a[i][:, selections_tmp[i]["selection"]].var[f"quantile_{q}"],
            kde=False,
            label="selection",
            bins=bins,
            hist_kws=hist_kws,
        )
        plt.axvline(x=x_min, lw=0.5, ls="--", color="black")
        plt.axvline(x=x_max, lw=0.5, ls="--", color="black")
        ax1.set_yscale("log")
        plt.legend(prop={"size": legend_size}, loc=[0.73, 0.74], frameon=False)
        plt.title(f"factor = {factor}")

        ax2 = ax1.twinx()
        x_values = np.linspace(0, np.max(a[i].var[f"quantile_{q}"]), 240)
        plt.plot(x_values, 1 * gaussians[i](x_values), label="penal.", color="green")
        plt.legend(prop={"size": legend_size}, loc=[0.73, 0.86], frameon=False)
        plt.ylim([0, 2])
        for label in ax2.get_yticklabels():
            label.set_color("green")
    plt.show()


# From the plots we decide for factor = 0.1
factor = 0.1
var = [factor * 0.1, factor * 0.5]
penalty = utils.plateau_penalty_kernel(var=var, x_min=lower_th, x_max=upper_th)
adata.var["expression_penalty"] = penalty(adata.var["quantile_0.99"])
# Our final selection would then be
select.select_pca_genes(
    adata,
    100,
    variance_scaled=False,
    absolute=True,
    n_pcs=20,
    process_adata=["norm", "log1p"],
    penalty_keys=["expression_penalty"],
    corr_penalty=None,
    inplace=True,
    verbose=True,
)
print(adata.var)

#######################################################################################
# Systematic selections pipeline to create probeset files for the evaluation pipeline #
#######################################################################################

###### Define experiment configurations ######
RESULTS_DIR = "../results/probesets/"
NAME = "genesets_1"  # Provide a name for the systematic selections in case we want to combine different experiments later

general_params = {
    "n": [20, 100],
    "penalty_keys": [[]],
    "dataset": ["small_data_raw_counts.h5ad"],
    "data_path": [
        "../data/"
    ],  # It actually doesn't make sense to provide more than one path here I guess
    # How to handle datasets that are in different directories? (overkill?)
    "gene_subset": [
        None
    ],  # We could add a key from adata.var here: e.g. different numbers of highly variable genes
    # (the key needs to be prepared in `adata.var[key]` (dtype=bool) --> adata = adata[:,adata.var[key]] )
    "obs_fraction": [None],  # For measuring method stability
    "fraction_seed": [
        None
    ],  # (obs_fraction and fraction_seed only work in combination)
    #
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
    "pca": select.select_pca_genes,
    "DE": select.select_DE_genes,
    "random": select.random_selection,
    "highest_expr": select.highest_expressed_genes,
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

###### Reshape configs for single selections ######
cartesian_product = list(
    itertools.product(*[param_list for _, param_list in general_params.items()])  # type: ignore
)
general_configs = [
    {key: val for key, val in zip(general_params, val_list)}
    for val_list in cartesian_product
]

method_configs = {}
for method, params in method_params.items():
    cartesian_product = list(
        itertools.product(*[param_list for _, param_list in params.items()])  # type: ignore
    )
    method_configs[method] = [
        {key: val for key, val in zip(params, val_list)}  # type: ignore
        for val_list in cartesian_product
    ]

###### Run selections and save results in RESULTS_DIR+f'selections_info_{NAME}.csv' and RESULTS_DIR+f'selections_{NAME}.csv' ######
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

param_keys = list(
    np.unique(
        [k for k in general_params]
        + [k for _, m_params in method_params.items() for k in m_params]  # type: ignore
    )
)
param_keys.remove("process_adata")

df_info = pd.DataFrame(
    columns=["directory", "method", "normalised", "log1p", "scaled", "time_seconds"]
    + param_keys
)
df_info.index.name = "set_id"
df_sets = pd.DataFrame(index=adata.var.index)

count = 0
for g_config in general_configs:
    for method, m_configs in method_configs.items():
        print(method)
        for m_config in m_configs:
            adata = sc.read(g_config["data_path"] + g_config["dataset"])
            kwargs = {k: v for k, v in g_config.items() if k in methods_kwargs[method]}
            kwargs.update(
                {k: v for k, v in m_config.items() if k in methods_kwargs[method]}
            )
            start = timer()
            s = methods[method](adata, **kwargs, inplace=False)  # , verbose=False)
            computation_time = (
                timer() - start
            )  # actually the dataset processing shouldn't be part of the measured computation time...

            df_info.loc[f"{NAME}_{count}", ["directory", "method", "time_seconds"]] = [
                RESULTS_DIR,
                method,
                computation_time,
            ]
            g_config_cols = [k for k in g_config if k in df_info.columns]
            df_info.loc[f"{NAME}_{count}", g_config_cols] = [
                v for k, v in g_config.items()
            ]
            kwarg_cols = [k for k in kwargs if k in df_info.columns]
            df_info.loc[f"{NAME}_{count}", kwarg_cols] = [
                v for k, v in kwargs.items() if k in kwarg_cols
            ]

            tmp = kwargs["process_adata"] if ("process_adata" in kwargs) else []
            pp_options = [("norm" in tmp), ("log1p" in tmp), ("scale" in tmp)]
            df_info.loc[
                f"{NAME}_{count}", ["normalised", "log1p", "scaled"]
            ] = pp_options

            df_sets[f"{NAME}_{count}"] = s["selection"]

            count += 1
            print(count)

df_info.to_csv(RESULTS_DIR + f"selections_info_{NAME}.csv")
df_sets.to_csv(RESULTS_DIR + f"selections_{NAME}.csv")
