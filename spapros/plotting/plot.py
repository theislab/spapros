from spapros.selection.selection_methods import select_pca_genes
from spapros.util.util import plateau_penalty_kernel

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def explore_constraint(adata, factors=None, q=0.99, lower=0, upper=1):
    """Plot histogram of quantiles for selected genes for different penalty kernels

    How to generalize the plotting function, support:
    - any selection method with defined hyperparameters
    - any penalty kernel
    - any key to be plotted (not only quantiles)
    """
    if factors is None:
        factors = [10, 1, 0.1]
    legend_size = 9
    factors = [10, 1, 0.1]
    rows = 1
    cols = len(factors)
    sizefactor = 6

    gaussians = []
    a = []
    selections_tmp = []
    for i, factor in enumerate(factors):
        x_min = lower
        x_max = upper
        var = [factor * 0.1, factor * 0.5]
        gaussians.append(plateau_penalty_kernel(var=var, x_min=x_min, x_max=x_max))

        a.append(adata.copy())

        a[i].var["penalty_expression"] = gaussians[i](a[i].var[f"quantile_{q}"])
        selections_tmp.append(
            select_pca_genes(
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
