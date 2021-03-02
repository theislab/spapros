import numpy as np
import pandas as pd
import scanpy as sc
import utils
import selection_methods as select
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from timeit import default_timer as timer

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


# Some selection methods in selection_methods.py are not on the latest versions (`select_pca_genes()` is always good for testing, other methods see below)
# - The options `process_adata`, `penalty_keys`, (`corr_penalty`) and eventually others needs to be added
# - happy to adjust these functions when we decided how to handle the methods generally


##########################
# Demonstrate constraint #
##########################
# Here we show how an expression penalty kernel is applied
# - the expression penalty is based on computed quantiles (adata.var['quantile_0.99'] precomputed via `utils.get_expression_quantile()`)
# - we want to penalize genes that are above and below certain expression thresholds (to prevent image saturation and undetectable genes)
# - we choose genes that exceed the lower tresholds for at least 1% of cells (user defined threshold) and don't exceed the upper threshold
#   for more than 1% of cells (therefore the 0.99 quantile)

### Map expression thresholds
# From experimental partners we know good thresholds for the basic normalisation to a target_sum of 10000. However, we use scran normalisation for our datasets.
lower_th, upper_th = utils.transfered_expression_thresholds(
    adata, lower=2, upper=6, tolerance=0.05, target_sum=10000, plot=True
)

### Now we have mapped limits, we want to apply a smoothed penalty kernel: genes close to the threshold are still chosen if the score is very high
# To decide on a proper kernel the user must be able to test selections with different kernel parameters. For that purpose I prepared an interactive plotting function.
# This is an interesting point for the package design:
# - we want to do multiple selections with our current selection method settings for different penalty kernels.
# - The following plot function specifically includes the pca selection method. We want it to be generalized for any method though.
#   --> Guess the best way is to define selection methods as class objects (?)
# since the plotting function is far from generalized I don't put it in some imported module atm. I'll rewrite it later.


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

        a[i].var[f"penalty_expression"] = gaussians[i](a[i].var[f"quantile_{q}"])
        selections_tmp.append(
            select_pca_genes(
                a[i],
                100,
                variance_scaled=False,
                absolute=True,
                n_pcs=20,
                process_adata=["norm", "log1p", "scale"],
                penalty_keys=[f"penalty_expression"],
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
adata.var["expression_penalty"] = penalty(adata.var[f"quantile_0.99"])
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
# So for this we will need some config file to define the systematic selections
# - we want to be able to easily define multiple methods
# - and easily define shared and specific parameters
#     - e.g. the number of selected genes n could be shared
#     - hyperparameter `n_pcs` for pca selection would be specific
#     - we should also be able to set general parameters as method specific:
#          - e.g. for some experiments we want `process_adata` to be the same for all methods, and in other experiments not
#          - the option `n_pcs` could also be shared between standard pca selection and sparse pca (general parameter, applicable on some methods)

# example options of systematic selections:
# - different methods
# - different hyperparameters
# - with and without constraints (not shown here)
# - different dataset parameters: n_hvg (not shown here), scaled/unscaled
#
# The systematic selection are saved in the two files
# - ../results/probesets/selections_genesets_1.csv (includes the selected probesets)
# - ../results/probesets/selections_info_genesets_1.csv (includes infos on each selected probeset)
#
# usage at the end (systematic selections):
# for the project so far I wrote a script that runs parallelized jobs for each selection to speed things up, however I think this job distribution can be done much nicer (so far hyperparameters are not saved nicely, specifications are too entangled).
# I don't know if this systematic selection procedure should be part of the pipeline? I guess yes, but then the pipeline would have two aspects: 1. systematic probeset selection, 2. evaluation of the probesets

# for the probesets csv file I'd like to indicate from which genes the methods could select from ("full_gene_set"). Two options:
# 1. we have different csv files for each different full_gene_set (I don't like that option, if we iterate over many HVGs we have a lot of files)
# 2. instead of boolean columns we could use three categories: 'selected', 'not selected', 'not given'
# --> i think 2. would be okay, we'd need an utility fct that maps to a bool vector
# --> Ah I think atm we might even have a bug here: if two different datasets are in the systematic selection configs this might
#     raise errors when creating the gene list in the csv output files..


###### Define experiment configurations ######
RESULTS_DIR = "../results/probesets/"  #
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
    itertools.product(*[param_list for _, param_list in general_params.items()])
)
general_configs = [
    {key: val for key, val in zip(general_params, val_list)}
    for val_list in cartesian_product
]

method_configs = {}
for method, params in method_params.items():
    cartesian_product = list(
        itertools.product(*[param_list for _, param_list in params.items()])
    )
    method_configs[method] = [
        {key: val for key, val in zip(params, val_list)}
        for val_list in cartesian_product
    ]

###### Run selections and save results in RESULTS_DIR+f'selections_info_{NAME}.csv' and RESULTS_DIR+f'selections_{NAME}.csv' ######
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

param_keys = list(
    np.unique(
        [k for k in general_params]
        + [k for _, m_params in method_params.items() for k in m_params]
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
