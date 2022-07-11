from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from rich.progress import Progress
from rich.progress import TaskID
from sklearn.decomposition import SparsePCA
from spapros.evaluation.evaluation import forest_classifications
from spapros.util.util import clean_adata


def apply_correlation_penalty(
    scores: pd.DataFrame, adata: sc.AnnData, corr_penalty: Callable, preselected_genes: list = []
) -> pd.DataFrame:
    """Compute correlations and iteratively penalize genes according max corr with selected genes.

    Args:
        scores:
            Sum of PCA loading variance for each gene. No gene in preselected_genes should occur in
            :attr:`scores.index`.
        adata:
            Data with log normalised counts in adata.X.
        corr_penalty:
            Function that maps values from [0,1] to [0,1]. It describes an iterative penalty function
            that is applied on pca selected genes. The highest correlation with already selected genes
            to the next selected genes are penalized according the given function. (max correlation is
            recomputed after each selected gene).
        preselected_genes:
            Pre selected genes (these will also have the highest ranking in the final list).

    """
    #     TODO: proofread dodstring
    #     Note that the param preselected_genes is never set!
    #     This function is thoroughly tested.

    penalized_scores = scores.copy()

    # Compute correlations
    if scipy.sparse.issparse(adata.X):
        cor_mat = np.abs(np.corrcoef(adata.X.toarray(), rowvar=False))
    else:
        cor_mat = np.abs(np.corrcoef(adata.X, rowvar=False))
    cor_df = pd.DataFrame(index=adata.var.index, columns=adata.var.index, data=cor_mat)
    cor_df.fillna(0, inplace=True)

    # Eventually penalize according max correlation with preselected genes
    max_cor = cor_df.loc[penalized_scores.index, preselected_genes].max(axis=1).to_frame("max")
    if len(preselected_genes) > 0:
        tmp_scores = penalized_scores.mul(corr_penalty(max_cor), axis=0)
    else:
        tmp_scores = penalized_scores

    # Iteratively penalize according max correlations with theoretically selected genes
    for _ in range(len(scores.index)):
        gene = tmp_scores["scores"].idxmax()
        penalized_scores.loc[gene] = tmp_scores.loc[gene]
        max_cor.drop(index=gene, inplace=True)
        max_cor = max_cor.join(cor_df.loc[max_cor.index, gene])
        max_cor = max_cor.max(axis=1).to_frame(name="max")
        tmp_scores = scores.loc[max_cor.index].mul(corr_penalty(max_cor), axis=0)

    return penalized_scores


def apply_penalties(scores: pd.DataFrame, adata: sc.AnnData, penalty_keys: list = []) -> pd.DataFrame:
    """Multipy penalty scores with the data.

    Args:
        scores:
            Dataframe with genes as index and columns with any name containing penalty scores. Values in a row are
            penalized in the same way.
        adata:
            Data with log normalised counts in adata.X and columns :attr:`penalty_keys`in adata.var.
        penalty_keys:
            List of the columns in adata.var containing penalty factors.

    Returns
        pd.DataFrame
            scores multiplied with each gene's penalty factors
    """
    s = scores
    penalty = pd.DataFrame(
        index=scores.index,
        data={k: adata.var.loc[scores.index, k] for k in penalty_keys},
    )
    s = s.mul(penalty.product(axis=1), axis=0)
    return s


def select_pca_genes(
    adata: sc.AnnData,
    n: int,
    variance_scaled: bool = False,
    absolute: bool = True,
    n_pcs: int = 20,
    penalty_keys: list = [],
    corr_penalty: Callable = None,
    inplace: bool = True,
    progress: Optional[Progress] = None,
    level: int = 1,
    verbosity: int = 2,
) -> pd.DataFrame:
    """Select n features based on pca loadings.

    Args:
        adata:
            Data with log normalised counts in adata.X.
        n:
            Number of selected features.
        variance_scaled:
            If `True`, loadings are defined as `eigenvector_component * sqrt(eigenvalue)`.
            If False loadings are defined as eigenvector_component.
        absolute:
            Take absolute value of loadings.
        n_pcs:
            Number of PCs used to calculate loadings sums.
        penalty_keys:
            List of keys for columns in adata.var that are multiplied with the scores.
        corr_penalty:
            Function that maps values from [0,1] to [0,1]. It describes an iterative penalty function
            that is applied on pca selected genes. The highest correlation with already selected genes
            to the next selected genes are penalized according the given function. (max correlation is
            recomputed after each selected gene).
        inplace:
            Save results in :attr:`adata.var` or return dataframe.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        level:
            Progress bar level.
        verbosity:
            Verbosity level.

    Returns:

        - if not inplace:
            pd.DataFrame (like adata.var) with columns:

                - 'selection': bool indicator of selected genes
                - 'selection_score': pca loadings based score of each gene
                - 'selection_ranking': ranking according selection scores

        - if inplace:
            Save results in `adata.var[['selection','selection_score','selection_ranking']]`.

    """

    a = adata.copy()

    if n_pcs > a.n_vars:
        n_pcs = a.n_vars

    clean_adata(a)

    if progress and 2 * verbosity >= level:
        pca_task = progress.add_task("Select pca genes...", total=1, level=1)

    sc.pp.pca(
        a,
        n_comps=n_pcs,
        zero_center=True,
        svd_solver="arpack",
        random_state=0,
        return_info=True,
        copy=False,
    )

    if progress and 2 * verbosity >= level:
        progress.advance(pca_task)

    loadings = a.varm["PCs"].copy()[:, :n_pcs]
    if variance_scaled:
        loadings *= np.sqrt(a.uns["pca"]["variance"][:n_pcs])
    if absolute:
        loadings = abs(loadings)

    scores = pd.DataFrame(index=adata.var.index, data={"scores": np.sum(loadings, axis=1)})
    scores = apply_penalties(scores, adata, penalty_keys=penalty_keys)
    if not (corr_penalty is None):
        scores = apply_correlation_penalty(scores, adata, corr_penalty)

    selected_genes = scores.nlargest(n, "scores").index.values
    selection = pd.DataFrame(
        index=scores.index,
        data={
            "selection": False,
            "selection_score": scores["scores"],
            "selection_ranking": scores["scores"].rank(method="dense", ascending=False),
        },
    )
    selection.loc[selected_genes, "selection"] = True

    if inplace:
        adata.var[["selection", "selection_score", "selection_ranking"]] = selection[
            ["selection", "selection_score", "selection_ranking"]
        ]
    else:
        return selection


def marker_scores(
    adata: sc.AnnData,
    obs_key: str = "celltype",
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = "rest",
    rankby_abs: bool = False,
) -> pd.DataFrame:
    """Compute marker scores for genes in adata.

    Args:
        adata:
            Data with log normalised counts in adata.X.
        obs_key:
            Column name of adata.obs for which marker scores are calculated.
        groups:
            See sc.tl.rank_genes_groups().
        reference:
            See sc.tl.rank_genes_groups().
        rankby_abs:
            See sc.tl.rank_genes_groups().

    Returns:
        pd.DataFrame:
            Index are genes as in adata.var.index, columns are names of groups in adata.obs[obs_key].
    """
    df = pd.DataFrame(index=adata.var.index)
    adata_ = adata if isinstance(reference, str) else adata[adata.obs[obs_key].isin(reference)]
    ref = reference if isinstance(reference, str) else "rest"
    a = sc.tl.rank_genes_groups(
        adata_,
        obs_key,
        use_raw=False,
        groups=groups,
        reference=ref,
        n_genes=adata.n_vars,
        rankby_abs=rankby_abs,
        copy=True,
        method="wilcoxon",
        corr_method="benjamini-hochberg",
    )
    names = a.uns["rank_genes_groups"]["scores"].dtype.names
    marker_scores = {
        name: pd.DataFrame(
            index=a.uns["rank_genes_groups"]["names"][name], data={name: a.uns["rank_genes_groups"]["scores"][name]}
        )
        for name in names
    }
    for name in names:
        df = df.join(marker_scores[name])
    return df


def select_DE_genes(
    adata: sc.AnnData,
    n: int,
    per_group: bool = False,
    obs_key: str = "celltype",
    penalty_keys: list = [],
    groups: Union[Literal["all"], Iterable[str]] = "all",
    reference: str = "rest",
    rankby_abs: bool = False,
    inplace: bool = True,
    progress: Optional[Progress] = None,
    verbosity: int = 2,
    level: int = 2,
    task: str = "Select DE genes...",
) -> pd.DataFrame:
    """Select genes based on wilxocon rank genes test.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        n: int
            Number of genes to selected (in total if not :attr:`per_group` else per group).
        per_group:
            Select :attr:`n` genes per group of adata.obs[obs_key] (default: False). Note that the same gene can be
            selected for multiple groups.
        obs_key:
            Column name of `adata.obs` for which marker scores are calculated.
        penalty_keys:
            Penalty factor for gene selection.
        groups:
            See sc.tl.rank_genes_groups().
        reference:
            See sc.tl.rank_genes_groups() but extended to lists of reference groups. Note that in case of providing
            such a list you need to include all elements of `groups` in `reference`.
        rankby_abs:
            See :meth:`sc.tl.rank_genes_groups()`.
        inplace:
            Save results in :attr:`adata.var` or return dataframe.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown.
        verbosity:
            Verbosity level.
        level:
            Progress bar level.
        task:
            Description of progress task.

    Returns:
        pd.DataFrame:
            Index are genes as in `adata.var.index`. It contains a bool column named 'selection'.
    """

    a = adata

    if groups == "all":
        group_counts = a.obs[obs_key].value_counts() > 2
    else:
        group_counts = a[a.obs[obs_key].isin(groups)].obs[obs_key].value_counts() > 2
    if not group_counts.values.all():
        groups = group_counts[group_counts].index.to_list()

    selection = pd.DataFrame(index=a.var.index, data={"selection": False})
    scores = marker_scores(a, obs_key=obs_key, groups=groups, reference=reference, rankby_abs=rankby_abs)
    scores = apply_penalties(scores, a, penalty_keys=penalty_keys)
    if per_group:
        if progress and 2 * verbosity >= level:
            de_task = progress.add_task(task, total=len(scores.columns), level=level)
        for group in scores.columns:
            scores.loc[scores.index.difference(scores.nlargest(n, group).index), group] = 0
            if progress and 2 * verbosity >= level:
                progress.advance(de_task)
        genes = scores.loc[(scores > 0).any(axis=1)].index.tolist()
    else:
        genes = []
        groups_of_genes = {}
        tmp_scores = scores.copy()
        if progress and 2 * verbosity >= level:
            de_task = progress.add_task("Select DE genes...", total=n, level=level)
        while len(genes) < n:
            for group in scores.columns:
                gene = tmp_scores[group].idxmax()
                if (len(genes) < n) and (gene not in genes):
                    genes.append(gene)
                    groups_of_genes[gene] = [group]
                elif gene in genes:
                    groups_of_genes[gene].append(group)
            tmp_scores = scores.loc[~scores.index.isin(genes)].copy()
            if progress and 2 * verbosity >= level:
                progress.advance(de_task)
        scores.loc[~scores.index.isin(list(groups_of_genes.keys()))] = 0
        for gene_g, groups in groups_of_genes.items():
            scores.loc[gene_g, [group for group in scores.columns if group not in groups]] = 0
    selection.loc[genes, "selection"] = True
    selection[scores.columns] = scores
    if inplace:
        adata.var["selection"] = selection["selection"]
    else:
        return selection


#################################################################################
# Procedure to add DE genes with specific reference groups to trees of interest #
#################################################################################
# Toroughly tested.


def outlier_mask(
    df: pd.DataFrame, n_stds: float = 1.0, min_outlier_dif: float = 0.02, min_score: float = 0.9
) -> pd.DataFrame:
    """Get mask over df.index based on values in df columns.

    Args:
        df:
            Dataframe where columns are the celltypes of interest and the index are potential outlier celltypes for a
            given celltype of interest.
        n_stds:
            Celltypes are considered as outliers if their specificity is `n_stds` standard deviations below the mean
            specificity.
        min_outlier_dif:
            Lower bound of `n_stds` x standard_deviation in the outlier criterion
        min_score:
            Minimal specificty score to not classify a reference celltype as outlier.

    Returns:
        pd.DataFrame:
            Same shape as input. The values are boolean indicating whether criteria 1 (mean specificity) and 2 (outlier)
            or criterion 3 (min specificity)3 is fullfilled.

    """
    # TODO proofread docstring

    crit1 = df < (df.mean(axis=0) - (n_stds * df.std(axis=0))).values[np.newaxis, :]
    crit2 = df < (df.mean(axis=0) - min_outlier_dif).values[np.newaxis, :]
    crit3 = df < min_score
    return (crit1 & crit2) | crit3


def add_DE_genes_to_trees(
    adata: sc.AnnData,
    tree_results: list,
    ct_key: str = "Celltypes",
    n_DE: int = 1,
    min_score: float = 0.9,
    n_stds: float = 1.0,
    penalty_keys: list = [],
    min_outlier_dif: float = 0.02,
    n_terminal_repeats: int = 3,
    max_step: int = 12,
    tree_clf_kwargs: dict = {},
    verbosity: int = 1,
    save: Union[str, bool] = False,
    return_clfs: bool = False,
    progress: Optional[Progress] = None,
    level: int = 2,
    baseline_task=None,
) -> Union[
    Tuple[List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]], pd.DataFrame, pd.DataFrame],  # return_clfs = True
    Tuple[List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]], pd.DataFrame],  # return_clfs = False
]:
    """Improve decision trees by adding DE genes wrt reduced sets of reference celltypes.

    Notes:
        Typically we train trees on genes preselected by simple 1-vs-all differential expression.
        We now test for the best tree of each celltype of interest if some other celltype is hard to distinguish.
        If that's the case we add DE genes that are calculated in a 1-vs-"hard to distinguish celltypes" fashion.
        The adding procedure is as follows:

            - For each tree for a celltype of interest (ct_oi) we take specificity scores of all other celltypes
            (ct_others)
            - Based on those scores we find celltypes that are low specificity outliers.
            - we consider celltypes as outliers if

               - specificity < :attr:`min_score` or
               - (specificity < mean_specificy - `n_stds` x standard deviation) && (specificity < mean_specificy -
               min_outlier_dif)

            - if we have outliers: add :attr:`n_DE` genes from a DE test ct_oi-vs-outliers
            - recalculate trees and repeat
            - stop criteria: either no more outliers, or the same outliers occured in the last
            :attr:`n_terminal_repeats` (each tree is stopped individually), or :attr:`max_step` is reached.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        tree_results:
            Results of :meth:`ev.forest_classifications()`.
        ct_key:
            Key of adata.obs with celltype annotations.
        n_DE:
            Number of DE genes added per tree and iteration.
        min_score:
            Minimal specificty score to not classify a reference celltype as outlier.
        n_stds:
            Celltypes are considered as outliers if their specificity is :attr:`n_stds` standard deviations below the
            mean specificity.
        penalty_keys:
            Keys of adata.obs for penalties applied to DE genes selection.
        min_outlier_dif:
            Lower bound of :attr:`n_stds` x standard_deviation in the outlier criterion.
        n_terminal_repeats:
            If the same outliers are identified :attr:`n_terminal_repeats` times for a given celltype then optimisation
            for that celltype is stopped.
        max_step:
            Maximal number of iterations as a manual step threshold also to stop the while loop in case of unexpected
            behavior.
            Unexpected behavior really shouldn't happen but we can't say that for sure.
        tree_clf_kwargs:
            Keyword arguments that are passed to :meth:`ev.forest_classifications()`. The same arguments that were used to
            compute :attr:`tree_results` should be used here.
        verbosity:
            Verbosity level.
        save:
            Path to save the final classification forest.
        return_clfs:
            Wether to return the sklearn tree classifiers of the final retrained trees.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown for each step of adding genes from reference
            trees.
        level:
            Progress bar level.
        baseline_task:
            ID of the supertask.

    Returns:
        forest_results: list
            Output of forest_classification() on the final set of genes (initial genes + added DE genes).
        forest_classifiers (if :attr:`return_clfs`):
            dict: keys are celltypes, values are list of sklearn tree classifiers.
        DE_info: pd.DataFrame
            Infos of added DE genes:

                - index: added genes, sorted by 'rank' (no sorting within the same rank)
                - columns:

                    - column = 'step': iteration step where DE gene was selected
                    - column.names = "celltypes": float DE scores for each selected gene (the score is only > 0 if a
                      gene was selected for a given celltype)
                    - column.names = "celltypes (ref)": if celltype was in the reference of a given DE test (these might
                      include pooled references of multiple tests which yielded the same selected DE gene)
    """

    # Throw out "return_clfs" from forest_kwargs since we set them to False except for the last forest
    if "return_clfs" in tree_clf_kwargs:
        del tree_clf_kwargs["return_clfs"]

    # Note: in specificities we have celltype keys and dataframes with reference celltypes as index
    # the ct keys might be fewer and in a different order then the reference cts in the index
    # - we want to create an "outlier" dataframe with symmetric celltype order in index (references) and columns
    #   (celltypes of interest)
    # - therefore we need to add columns that are actually not used to identify outliers (we just set the performance
    #   to 1 in those cols)
    # - the design could have been different (but i first only regarded the case where cts_oi and reference_cts were the
    #   same...)
    init_summary, specificities, im = tree_results
    all_celltypes = specificities[list(specificities)[0]].index.tolist()
    # all_celltypes = [ct for ct in specificities]

    # Get all input genes of the initial forest training (the `genes` variable will be updated with new selected DE
    # genes each iteration)
    genes = im[list(im)[0]].index.tolist()

    # Specificities table (True negative rates) of best trees
    TN_rates = pd.DataFrame(index=all_celltypes, columns=all_celltypes, data=1)
    new_TN_rates = pd.concat([d["0"] for ct, d in specificities.items()], axis=1)
    celltypes = [ct for ct in specificities]
    new_TN_rates.columns = celltypes
    TN_rates[celltypes] = new_TN_rates
    # TN_rates = pd.concat([d['0'] for ct,d in specificities.items()],axis=1)
    # TN_rates.columns = [ct for ct in all_celltypes]

    # Get outlier mask. Note the columns are the celltypes of interest, and the index are potential outlier celltypes
    # for a given celltype of interest
    outliers = outlier_mask(TN_rates, n_stds, min_outlier_dif, min_score)
    celltypes = outliers.columns[outliers.any(axis=0)].tolist()

    # Variable to measure how many steps in a row we have the same outliers for a celltype (stop criterion)
    repeats = {ct: 0 for ct in TN_rates.columns}

    # Initialize DE_info table with infos about iteratively added DE genes
    ref_columns = [f"{ct} (ref)" for ct in all_celltypes]
    DE_info = pd.DataFrame(columns=["step"] + all_celltypes + ref_columns)

    step = 1

    # if verbosity > 0:
    #     print("Add DE genes with specific reference groups to improve tree performance...")

    if progress and 2 * verbosity >= level:
        DE_tree_task = progress.add_task(
            "Iteratively add DE genes to DE_baseline forest...", total=max_step, level=level
        )
    while (outliers.values.sum() > 0) and celltypes and (step < max_step + 1):
        # if verbosity > 1:
        #     print(f"  Iteration step {step}:")
        # Get mapping of each celltype to its test reference group
        reference = outliers.copy()
        np.fill_diagonal(reference.values, True)
        ct_to_reference = {
            ct: reference[ct].loc[reference[ct]].index.tolist() for ct in reference.columns if (reference[ct].sum() > 1)
        }

        # DE selection for celltypes with outliers
        # if verbosity > 1:
        #     print("\t Select DE genes...")
        if progress and 2 * verbosity >= level + 1:
            select_DE_task = progress.add_task(
                "Select DE genes for celltypes with outliers...", total=len(ct_to_reference), level=level + 1
            )
        for ct in ct_to_reference:
            # if verbosity > 2:
            #     print(f"\t ...for celltype {ct} with reference group {ct_to_reference[ct]}")
            df_ct_DE = select_DE_genes(
                adata[:, [g for g in adata.var_names if not (g in genes)]],
                n_DE,
                per_group=True,
                obs_key=ct_key,
                penalty_keys=penalty_keys,
                groups=[ct],
                reference=ct_to_reference[ct],
                rankby_abs=False,
                inplace=False,
                progress=progress,
                level=level + 2,
                task=f"Select DE genes for {ct}...",
                verbosity=verbosity,
            )
            if verbosity > 2:
                print(f"\t\t Selected: {df_ct_DE[df_ct_DE['selection']].index.tolist()}")
            # Add rows and update values in DE_info
            df_ct_DE = df_ct_DE[df_ct_DE["selection"]]
            del df_ct_DE["selection"]
            df_ct_DE[ref_columns] = pd.DataFrame(
                np.tile(reference[ct].values.copy(), (len(df_ct_DE), 1)), index=df_ct_DE.index
            )
            already_added = [g for g in df_ct_DE.index if g in DE_info.index]
            df_ct_DE.loc[already_added, ref_columns] = (
                df_ct_DE.loc[already_added, ref_columns] | DE_info.loc[already_added, ref_columns]
            )
            DE_info = DE_info.reindex(DE_info.index.union(df_ct_DE.index))
            DE_info.loc[df_ct_DE.index, df_ct_DE.columns] = df_ct_DE
            DE_info.loc[df_ct_DE.index, "step"] = step

            if progress and verbosity > 1:
                progress.advance(select_DE_task)

        new_genes = DE_info.loc[~DE_info.index.isin(genes)].index.tolist()
        genes += new_genes

        # Retrain forests
        # if verbosity > 1:
        #     print(f"\t Train decision trees on celltypes:\n\t\t {celltypes}")
        *_, specificities, _ = forest_classifications(
            adata,
            genes,
            celltypes=celltypes,
            ref_celltypes="all",
            ct_key=ct_key,
            save=False,
            **tree_clf_kwargs,
            verbosity=verbosity,
            return_clfs=False,
            progress=progress,
            level=level + 1,
            task="Train forest with added DE genes...",
        )
        # if verbosity > 1:
        #     print("\t\t Training finished.")

        if verbosity > 2:
            print("\t Identify outliers with new trees...")
        # Calculate outliers of new trees, and kick out celltypes with outliers that repeated `n_terminal_repeats` times
        new_TN_rates = pd.concat([d["0"] for ct, d in specificities.items()], axis=1)
        new_TN_rates.columns = celltypes
        TN_rates[celltypes] = new_TN_rates
        TN_rates.columns = [ct for ct in all_celltypes]
        new_outliers = outlier_mask(TN_rates, n_stds, min_outlier_dif, min_score)
        for ct in celltypes:
            if (new_outliers[ct] == outliers[ct]).all():
                repeats[ct] += 1
                if repeats[ct] == n_terminal_repeats:
                    new_outliers[ct] = False
        outliers = new_outliers.copy()
        celltypes = outliers.columns[outliers.any(axis=0)].tolist()

        step += 1

        if progress and 2 * verbosity >= level:
            progress.advance(DE_tree_task)

    if progress and 2 * verbosity >= level:
        progress.update(DE_tree_task, completed=3)

    if progress and 2 * verbosity >= level - 1 and baseline_task:
        progress.advance(baseline_task)

    # Sort DE info
    DE_info = DE_info.sort_values("step")

    # Train final forest on all celltypes
    # if verbosity > 1:
    #     print("Train final trees on all celltypes, now with the added genes...")
    celltypes = init_summary.index.tolist()

    forest_results = forest_classifications(
        adata,
        genes,
        celltypes=all_celltypes,
        ref_celltypes="all",
        ct_key=ct_key,
        save=save,
        **tree_clf_kwargs,
        verbosity=verbosity,
        return_clfs=return_clfs,
        progress=progress,
        level=level,
        task="Train final baseline forest on all celltypes...",
    )
    if return_clfs:
        assert isinstance(forest_results, tuple)
        results = forest_results[0]
        tree_clfs = forest_results[1]
        summary, ct_spec_summary, im = results
    else:
        assert isinstance(forest_results, list)
        summary, ct_spec_summary, im = forest_results

        # if verbosity > 1:
        #     print("\t Finished...")
        if save:
            print(f"\t Results saved at {save}")

    if verbosity > 2:
        print("\t Final performance table of training with all new markers:")
        f1_diffs = summary["0"] - init_summary["0"]
        tmp = pd.concat([summary["0"], init_summary["0"], f1_diffs], axis=1)
        tmp.columns = ["final f1 score", "initial f1 score", "difference"]
        print(tmp)  # display(tmp)

    if return_clfs:
        return [summary, ct_spec_summary, im], tree_clfs, DE_info
    else:
        return [summary, ct_spec_summary, im], DE_info


##################################################################################
# Procedure to add important genes from second set of trees to trees of interest #
##################################################################################
# Thoroughly tested.


def add_tree_genes_from_reference_trees(
    adata: sc.AnnData,
    tree_results: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    ref_tree_results: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    ct_key: str = "Celltypes",
    ref_celltypes: Union[str, list] = "all",
    n_max_per_it: int = 5,
    n_max: int = None,
    performance_th: float = 0.02,
    importance_th: float = 0,
    verbosity: int = 1,
    max_step: int = 12,
    save: Union[str, bool] = False,
    tree_clf_kwargs: dict = {},
    return_clfs: bool = False,
    final_forest_task: Optional[TaskID] = None,
    progress: Optional[Progress] = None,
    level: int = 2,
    task: str = "Iteratively add genes from DE_baseline_forest...",
) -> Union[
    Tuple[List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]], list],  # len(f1_diffs) == 0 && return_clfs=True
    Tuple[
        pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]
    ],  # (len(f1_diffs) == 0 && return_clfs=False) | return_clfs=False
    Tuple[List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]], dict],  # return_clfs=True
]:
    """Add markers till reaching classification performance of reference.

    Notes:
        Classification trees are given for the pre selected genes and a reference. Classification performance and
        the feature importance of the reference trees are used to add markers.
        Markers are added according the following procedure:
            - Take best tree of each celltype from :attr:`tree_results` (trees_1) and :attr:`ref_tree_results` (trees_2)
            respectively.
            - Whenever we retrain trees (for checking the performance after adding genes) we train on all genes in
            :attr:`tree_results` and the already iteratively added new genes from trees_2.
            - At each step we add 1 gene per celltype from trees_2 for those celltypes with: performance
            difference(trees_2 - trees_1) > :attr:`performance_th`.
            - Added genes are chosen according the feature importance in trees_2.
            - If the same gene occurs in more than one tree of trees_2 the importance is summed over the multiple
            celltypes and if the importance of the multi occuring gene is the highest, then only this one gene is added
            for the multiple celltypes.
            - Additionally we only add :attr:`n_max_per_it` genes per step.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        tree_results:
            Info of trees that we want to improve. List of output variables of the function
            :meth:`ev.forest_classifications()`.
            Note that tree_results include the info on which genes the trees were trained. We still allow all genes for
            the updated table, not only those that are in the initial best trees.
        ref_tree_results:
            Like :attr:`tree_results` but for reference trees.
        ct_key:
            Key of adata.obs with celltype annotations.
        ref_celltypes:

        n_max_per_it:
            Add :attr:`n_max_per_it` genes per iteration. In each iteration tree classifications will be calculated.
            Note that per celltype only one gene per iteration is added.
        n_max:
            Limit the upper number of added genes.
        performance_th:
            Further markers are only added for celltypes that have a performance difference above :attr:`performance_th`
            compared to the reference performance.
        importance_th:
            Only reference genes with at least :attr:`importance_th` as feature importance in the reference tree are
            added as markers.
        verbosity:
            Verbosity level.
        max_step:
            Number of maximal iteration steps.
        save:
            Path to save final forest results.
        tree_clf_kwargs:
            Keyword arguments that are passed to :meth:`ev.forest_classifications()`. The same arguments that were used
            to compute :attr:`tree_results` should be used here.
        return_clfs:
            Wether to return the sklearn tree classifiers of the final retrained trees.
        final_forest_task:
            :attr:`Task_ID` of taks of progress for final forest training.
        progress:
            :attr:`rich.Progress` object if progress bars should be shown for each step of adding genes from reference
            trees.
        level:
            Progress bar level.
        task:
            Description of progress task.

    Returns:
        object:
            Results of the final trees trained on union of old genes and new markers
            (results are in the form of the output from :meth:`ev.forest_classifications()`)
        dict (if `return_clfs`):
            Forest classifiers. Keys are celltypes, values are list of sklearn tree classifiers.

    """

    #  TODO:
    #   - check if the parameter n_max_per_it works, and eventually fix the bug (i think it actually works, thought there is
    #     a bug since the number of potential markers dropped more than difference of n_max_per_it, but this can be the
    #     case when we reach the wanted classification performance for several celltypes)
    #   - write docstring for ref_celltypes
    #   - importance_th: We're working with a relative importance measure here. An absolute value could be better.
    #     (If classification is bad for a given celltype then useless genes have a high importance)
    #   - add Arguments for kwargs for forest_calssifications()
    #     (it's mainly about subsample=1000,test_subsample=3000 - but it's probably best to also include n_trees and
    #     ref_celltypes!!!) ... ok n_trees is also in there now - don't know about ref_celltypes

    # Throw out "return_clfs" from forest_kwargs since we set them to False except for the last forest
    if "return_clfs" in tree_clf_kwargs:
        del tree_clf_kwargs["return_clfs"]

    initial_summary, initial_ct_spec_summary, im = tree_results
    assert isinstance(initial_summary, pd.DataFrame)
    assert isinstance(initial_ct_spec_summary, dict)
    assert isinstance(im, dict)
    initial_summary_ref, _, im_ref = ref_tree_results
    assert isinstance(initial_summary_ref, pd.DataFrame)
    assert isinstance(im_ref, dict)
    # get summary metrics from best trees
    f1 = initial_summary["0"]
    f1_ref = initial_summary_ref["0"]

    # Start performance difference
    f1_diffs = f1_ref - f1
    f1_diffs = f1_diffs.loc[f1_diffs > performance_th]

    # Return initial results if all cell types' performances are good enough already
    if len(f1_diffs) == 0:

        if progress and 2 * verbosity >= level:
            progress.add_task("Initial results are good enough. No genes are added...", only_text=True, level=level)

        if progress and 2 * verbosity >= (level - 1) and final_forest_task:
            progress.advance(final_forest_task)  # retrain final tree is skipped

        if return_clfs:
            # raise ValueError("No classifiers were trained since no cell type needs a performance improvement. "\
            #                 "Set return_clfs=False or scip the function call.")
            # TODO: ideally we would have the following option (it's not too important though):
            #       return [initial_summary, initial_ct_spec_summary, im], initial_tree_clfs
            return [initial_summary, initial_ct_spec_summary, im], []
        else:
            return initial_summary, initial_ct_spec_summary, im

    # Get reference importance table of the celltypes' best trees
    importances = pd.concat([im_ref[ct]["0"] for ct in im_ref], axis=1)
    importances.columns = [ct for ct in im_ref]
    # Set reference importances to nan if below threshold
    importances = importances[importances > importance_th]
    # Drop rows with only nan
    importances = importances.dropna(axis=0, thresh=1).dropna(axis=1, thresh=1)
    # Reduce to celltypes for which we have genes in reference tree above threshold
    celltypes = [ct for ct in f1_diffs.index if ct in importances.columns]
    # Genes on which we trained the trees initially
    selected = im[celltypes[0]].index.tolist()
    # Potential new genes from reference
    unselected = [g for g in importances.index if not (g in selected)]
    # Filter importances table to unselected genes
    importances = importances.loc[unselected, celltypes]
    importances = importances.dropna(axis=0, thresh=1).dropna(axis=1, thresh=1)

    n_max_unreached = True
    selected_all = False
    n_added = 0
    # if verbosity > 0:
    #     print("Adding genes from reference tree...")
    if verbosity > 2:
        print("\t Performance table before training with new markers:")
        tmp = pd.concat([f1, f1_ref, f1_diffs], axis=1)
        tmp.columns = ["f1", "reference f1", "difference"]
        print(tmp)  # display(tmp)

    max_step_reached = False
    step = 0

    if progress and 2 * verbosity >= level:
        add_ref_task = progress.add_task(task, total=max_step if max_step else 12, level=level)

    while celltypes and n_max_unreached and (not selected_all) and (not importances.empty) and (not max_step_reached):

        if n_max:
            if (n_max - n_added) < n_max_per_it:
                n_max_per_it = n_max - n_added

        if len(importances.idxmax().unique()) > n_max_per_it:
            new_markers = list(importances.sum(axis=1).nlargest(n_max_per_it).index)
        else:
            new_markers = list(importances.idxmax().unique())
        selected += new_markers
        n_added += len(new_markers)
        unselected = [g for g in importances.index if not (g in new_markers)]
        # if verbosity > 1:
        #     print(f"\t Added new markers:\n\t\t {new_markers} \n\t\t ({len(unselected)} potential new markers left)")
        # if verbosity > 1:
        #     print(f"\t Train decision trees on celltypes:\n\t\t {celltypes}")
        summary, *_, _ = forest_classifications(
            adata,
            selected,
            celltypes=celltypes,
            ref_celltypes=ref_celltypes,
            ct_key=ct_key,
            save=False,
            verbosity=verbosity,
            return_clfs=False,
            progress=progress,
            level=level + 1,
            task="Train forest with added genes...",
            **tree_clf_kwargs,
        )
        assert isinstance(summary, pd.DataFrame)
        f1 = summary["0"].copy()
        # if verbosity > 1:
        #     print("\t\t Training finished.")
        f1_diffs = f1_ref.loc[f1.index] - f1
        f1_diffs = f1_diffs.loc[f1_diffs > performance_th]
        if (verbosity > 1) and not np.all([(ct in importances.columns) for ct in f1_diffs.index]):
            print("\t There are no more new markers in the reference trees for celltypes:")
            print(f"\t\t {[ct for ct in f1_diffs.index if not ct in importances.columns]}")
            print("\t\t even though the performance threshold was not reached")
            print("\t\t (statistical randomness, consider increasing n_trees or performance_th)")

        celltypes = [ct for ct in f1_diffs.index if ct in importances.columns]

        if verbosity > 2:
            print("\t Performance table after training with new markers:")
            tmp = pd.concat([f1.loc[celltypes], f1_ref.loc[celltypes], f1_diffs], axis=1)
            tmp.columns = ["f1", "reference f1", "difference"]
            print(tmp)  # display(tmp)

        if n_max and (n_added == n_max):
            n_max_unreached = False
        if len(unselected) == 0:
            selected_all = True

        importances = importances.loc[unselected, celltypes]
        importances = importances.dropna(axis=0, thresh=1).dropna(axis=1, thresh=1)

        step += 1
        if max_step and (step >= max_step):
            max_step_reached = True
            if progress and 2 * verbosity >= level + 1:
                progress.add_task(f"Maximal iteration step ({step}) reached.", total=0, level=level + 1, only_text=True)
            # if verbosity > 1:
            #     print(f"\t\t Maximal iteration step ({step}) reached.")

        if progress and 2 * verbosity >= level:
            progress.advance(add_ref_task)

    if progress and 2 * verbosity >= level:
        progress.update(add_ref_task, completed=max_step if max_step else 12)

    if progress and 2 * verbosity >= (level - 1) and final_forest_task:
        progress.advance(final_forest_task)

    # if verbosity > 0:
    #     print("Train final trees on all celltypes, now with the added genes...")

    celltypes = initial_summary.index.tolist()
    forest_results = forest_classifications(
        adata,
        selected,
        celltypes=celltypes,
        ref_celltypes=ref_celltypes,
        ct_key=ct_key,
        save=save,
        verbosity=verbosity,
        return_clfs=return_clfs,
        progress=progress,
        level=level,
        task="Train final forest on all celltypes...",
        **tree_clf_kwargs,
    )

    if return_clfs:
        assert isinstance(forest_results, tuple)
        results = forest_results[0]
        tree_clfs = forest_results[1]
        summary, ct_spec_summary, im = results
    else:
        assert isinstance(forest_results, list)
        summary, ct_spec_summary, im = forest_results

        # if verbosity > 0:
        #     print("\t Finished...")
        if save:
            print(f"\t Results saved at {save}")

    if verbosity > 2:
        print("\t Final performance table of training with all new markers:")
        f1_diffs = f1_ref - summary["0"]
        tmp = pd.concat([summary["0"], f1_ref, f1_diffs], axis=1)
        tmp.columns = ["f1", "reference f1", "difference"]
        print(tmp)  # display(tmp)

    if return_clfs:
        return [summary, ct_spec_summary, im], tree_clfs
    else:
        return summary, ct_spec_summary, im


################################################
# Selection from marker list related functions #
################################################


def get_markers_and_correlated_genes(
    cor_mat: pd.DataFrame, markers: list, selection: Union[pd.DataFrame, list], n_min: int = 2, th: float = 0.5
) -> Tuple[List[str], List[str]]:
    """Get markers and marker correlated genes for seletion from marker list.

    Notes:
        Typically use this function for markers of one celltype in a marker_list dictionary.

        We consider gene pairs when selecting based on correlations, e.g. n_min = 2 and th = 0.5:

            - take gene A from selection with the highest correlation to all markers (let marker M and gene A have the
            highest correlation)
            - if correlation(gene A,marker M) > th=0.5 add gene A to `correlated_genes` (we then have 1 of n_min=2 genes
            selected)
            - if not we add the first n_min - len(already_selected) markers
            - for the next correlation checks we exclude gene A and marker M (since we want to make sure to capture
            n_min markers with n_min genes) and repeat the previous steps till we reached n_min selected genes (or the
            possible maximum)

    Args:
        cor_mat:
            Dataframe with index = columns = genes. Genes in cor_mat must include all genes in :attr:`markers` and all
            genes with `selection['selection'] == True`.
        markers:
            List of marker genes for one celltype.
        selection:
            List or dataframe of the selected genes.
        n_min:
            Minimal number of markers per celltype that are captured with a min correlation of :attr:`th`.
        th:
            Minimal correlation to consider a gene as captures.

    Returns:
        add_markers: list
            The markers that are added to reach `n_min` (if there are enough markers provided).
        correlated_genes: list
            Genes in `selection` that have correlations > `th` with specific markers

    """
    # TODO
    #   - check attr markers
    #   - proofread docstrings

    # Initialize output lists
    add_markers: List[str] = []
    correlated_genes: List[str] = []

    # Prepare gene list of already selected genes
    if isinstance(selection, pd.DataFrame):
        selected_genes = selection.loc[selection["selection"]].index.tolist()
    else:
        selected_genes = selection

    # Transfer markers from `markers` into `already_selected` if markers are already in the selection
    already_selected = [g for g in markers if g in selected_genes]
    markers = [ge for ge in markers if ge not in already_selected]
    selected_genes = [g for g in selected_genes if g not in already_selected]
    if already_selected:
        add_markers += already_selected

    if (len(add_markers) >= n_min) or (markers == []):
        return add_markers[: min(len(add_markers), n_min)], correlated_genes

    genes = [g for g in (markers + selected_genes)]
    cm = cor_mat.loc[genes, genes]
    n_genes = len(add_markers)

    while (n_genes < n_min) and markers:
        max_marker = cm.loc[selected_genes, markers].max().idxmax()
        max_gene = cm.loc[markers, selected_genes].max().idxmax()
        if cor_mat.loc[max_marker, max_gene] > th:
            correlated_genes.append(max_gene)
            markers.remove(max_marker)
            selected_genes.remove(max_gene)
        else:
            add_markers += markers[: min([len(markers), n_min - n_genes])]
            markers = []
        n_genes = len(add_markers + correlated_genes)
    return add_markers, correlated_genes


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
# We want all of the following methods implemented as the methods above. This means
# adding multiplicative constraints (e.g. expression penalty). I am also thinking about
# adding the option of iterative penalties (i.e. a penalty based on already selected
# genes, e.g. a correlation penalty)
# Note: We can implement these constraints only for score based methods. This means sparse pca
#       and selfE do not work
#       - what would also be handy and work in general is a binary constraint (mask).
#         So you wouldn't need to subset the adata (which is important for the workflow, evaluation pipe, ...)


###### This is an old pca selection function with several different options how the loadings are aggregated to scores
###### My first runs showed that `method="sum"` performed best (as used in the function `select_pca_genes()`).
###### We should also include the old options to rerun the comparisons at the end.
def select_features_pca_loadings(
    adata: sc.AnnData,
    n: int,
    method: Literal["sum", "max", "max_PC", "max_PC_order"] = "max",
    absolute: bool = True,
    n_pcs: int = 30,
    inplace: bool = True,
    verbose: bool = True,
):
    """Select genes based on PCA loadings.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        n:
            Number of features to select.
        method:
            Different loadings based method for feature selection:

                - "sum" - features with highest sum of loadings
                - "max" - features with highest single loadings
                - "max_PC" - features of `n` highest loadings of different PC components
                - "max_PC_order" - features with highest loadings of first :attr:`n` PC components.
                If a feature was already picked the 2nd highest is chosen and so on.
                --> make sure that n_pcs >= n if you use this method

        absolute:
            If True the absolute values of loadings are taken into account such
            that also negative loadings with high abs() can lead to chose features.
        n_pcs:
            Number of calculated PC components. If None, n_pcs is set to the number of
            genes in adata. Only interesting for :attr:`method` = "sum" or if there are no
            existing PCA results.
        inplace:
            If True add results to `adata.var`, else results are returned.
        verbose:
            Optionally print infos.

    Returns:
        if inplace: add

            - adata.var['selection'] - boolean pd.Series of n selected features
            - adata.var['selection_score'] - float pd.Series with selection scores (if scores exist for all features)

        else: return
            pd.Dataframe with columns 'selection', 'selection_score' (latter only for some methods)
    """

    a = adata.copy()

    if n_pcs > a.n_vars:
        n_pcs = a.n_vars
    if method == "max_PC_order":
        n_pcs = n

    clean_adata(a)

    if verbose:
        print("Compute PCA.")
    sc.pp.pca(
        a,
        n_comps=n_pcs,
        zero_center=True,
        svd_solver="arpack",
        random_state=0,
        return_info=True,
        copy=False,
    )

    loadings = a.varm["PCs"].copy()[:, :n_pcs]
    if absolute:
        loadings = abs(loadings)

    if method == "sum":
        scores = np.sum(loadings, axis=1)
        unique_scores = np.unique(scores)
        unique_scores[::-1].sort()
        feature_idxs = []
        count = 0
        for u in unique_scores:
            for index in np.where(scores == u)[0]:
                feature_idxs.append(index)
                count += 1
                if count == n:
                    break
            if count == n:
                break
        # scores = scores[feature_idxs]
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True
        a.var["selection_score"] = scores
    elif method == "max":
        scores = []
        unique_loadings = np.unique(loadings)
        unique_loadings[::-1].sort()
        feature_idxs = []
        pc_idxs = []
        count = 0
        for u in unique_loadings:
            i, j = np.where(loadings == u)
            for k in range(len(i)):
                feature_idx = i[k]
                pc_idx = j[k]
                if feature_idx not in feature_idxs:
                    feature_idxs.append(feature_idx)
                    scores.append(u)
                    pc_idxs.append(pc_idx)
                    count += 1
                if count == n:
                    break
            if count == n:
                break
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True
    elif method == "max_PC":
        scores = []
        unique_loadings = np.unique(loadings)
        unique_loadings[::-1].sort()
        feature_idxs = []
        pc_idxs = []
        count = 0
        for u in unique_loadings:
            i, j = np.where(loadings == u)
            for k in range(len(i)):
                feature_idx = i[k]
                pc_idx = j[k]
                if (feature_idx not in feature_idxs) and (pc_idx not in pc_idxs):
                    feature_idxs.append(feature_idx)
                    scores.append(u)
                    pc_idxs.append(pc_idx)
                    count += 1
                if count == n:
                    break
            if count == n:
                break
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True
    elif method == "max_PC_order":
        scores = []
        feature_idxs = []
        pc_idxs = [i for i in range(n)]
        for pc_idx in pc_idxs:
            found_feature = False
            pc_loadings = loadings[:, pc_idx]
            while not found_feature:
                feature_idx = np.nanargmax(pc_loadings)
                if feature_idx not in feature_idxs:
                    feature_idxs.append(feature_idx)
                    scores.append(np.nanmax(pc_loadings))
                    found_feature = True
                else:
                    pc_loadings[feature_idx] = np.nan
        a.var["selection"] = False
        a.var["selection"].iloc[feature_idxs] = True

    if inplace:
        adata.var["selection"] = a.var["selection"].copy()
        if "selection_score" in a.var.columns:
            adata.var["selection_score"] = a.var["selection_score"].copy()
    else:
        if "selection_score" in a.var.columns:
            return a.var[["selection", "selection_score"]].copy()
        else:
            return a.var[["selection"]].copy()


def select_highly_variable_features(
    adata: sc.AnnData,
    n: int,
    flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = "cell_ranger",
    inplace: bool = True,
):
    """Select genes based on variability.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        n:
            Number of features to select.
        flavor:
            See sc.pp.highly_variable genes. Options are `'seurat', 'cell_ranger' and 'seurat_v3'.
        inplace:
            If True add results to `adata.var`, else results are returned.

    Returns:
        if inplace: add
            adata.var['selection'] - boolean pd.Series of n selected features

        else: return
            pd.Dataframe with column 'selection'
    """
    a = adata.copy()
    clean_adata(a)
    sc.pp.highly_variable_genes(
        a,
        n_top_genes=n,
        n_bins=20,
        flavor=flavor,
        subset=False,
        inplace=True,
        batch_key=None,
    )
    if inplace:
        adata.var["selection"] = a.var["highly_variable"]
    else:
        a.var["selection"] = a.var["highly_variable"]
        return a.var[["selection"]].copy()


def random_selection(adata: sc.AnnData, n: int, seed: int = 0, inplace: bool = True):
    """Select random genes.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        n:
            Number of features to select.
        seed:
            Random number seed.
        inplace:
            If True add results to `adata.var`, else results are returned.

    Returns:
        if inplace: add
            adata.var['selection'] - boolean pd.Series of n selected features

        else: return
            pd.Dataframe with column 'selection'
    """
    np.random.seed(seed=seed)
    f_idxs = np.random.choice(adata.n_vars, n, replace=False)
    df = pd.DataFrame(index=adata.var.index, columns=["selection"])
    df["selection"] = False
    df["selection"].iloc[f_idxs] = True
    if inplace:
        adata.var["selection"] = df["selection"].copy()
    else:
        return df


############################ Highest expressed genes ################################


def get_mean(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """Get mean of array.

    Args:
        X:
            Data matrix, eg. `adata.X`.
        axis:
            Axis along which the means are computed.

    Returns:
        np.ndarray:
            Array containing the means.

    """
    if scipy.sparse.issparse(X):
        mean = X.mean(axis=axis, dtype=np.float64)
        mean = np.array(mean)[0]
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
    return mean


def highest_expressed_genes(adata: sc.AnnData, n: int, inplace: bool = True, use_existing_means: bool = False):
    """Select n highest expressed genes in adata.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        n:
            Number of features to select.
        inplace:
            If True add results to `adata.var`, else results are returned.
        use_existing_means:
            Means can already be found in the columns 'means'. It `False`, calculate the means.

    Returns:
        if inplace: add
            adata.var['selection'] - boolean pd.Series of n selected features

        else: return
            pd.Dataframe with column 'selection'
    """
    a = adata

    df = pd.DataFrame(index=a.var.index, columns=["means"])
    if use_existing_means:
        if "means" in a.var:
            df["means"] = a.var["means"]
        else:
            raise ValueError('Column "means" in adata.var not found. Either add it or set use_existing_means=False.')
    else:
        df["means"] = get_mean(a.X)
    df["selection"] = False
    df.loc[df.nlargest(n, "means").index.values, "selection"] = True
    if inplace:
        adata.var["selection"] = df["selection"].copy()
    else:
        return df[["selection"]].copy()


##################################################################################
################################# Sparse PCA #####################################
##################################################################################


def sparse_pca(X: np.ndarray, n_pcs: int, alpha: float, seed: int, n_jobs: int) -> Tuple[np.ndarray, Any]:
    """Extract the sparse components that can optimally reconstruct the data.

    Args:
        X:
            Data matrix, eg. `adata.X`.
        n_pcs:
            Number of sparse PC components.
        alpha:
            Sparsity controlling parameter. Higher values lead to sparser components.
        seed:
            Random number seed. Used during dictionary learning.
        n_jobs:
            Number of parallel jobs to run. `None` means 1, `-1` means using all processors.

    Returns:
        np.ndarray:
            Features with any abs(loading) > 0.
        Any:
            Transformed data.
    """
    transformer = SparsePCA(n_components=n_pcs, alpha=alpha, random_state=seed, n_jobs=n_jobs)
    transformer.fit(X)
    transformer.transform(X)
    loadings = transformer.components_
    features = np.any(np.abs(loadings) > 0, axis=0)
    return features, loadings


def sort_alphas(alphas: list, n_features: list) -> Tuple[list, list]:
    """Sort lists of parameter values in ascending order of :attr:`alphas`.

    Args:
        alphas:
            List of already applied alpha values. Alpha is the parsity controlling parameter. Higher values lead to
            sparser components.
        n_features:
            Lists of already selected number of features.

    Returns:
        list:
            List of a alphas sorted in ascending order.
        list:
            List of n_pcs sorted according to :attr:`alphas`.
    """
    zipped_lists = zip(alphas, n_features)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    alphas, n_features = [list(t) for t in tuples]
    return alphas, n_features


def next_alpha(n: int, alphas: list, n_features: list):
    """

    Args:
        n:
            Number of features to select.
        alphas:
            List of already applied alpha values. Alpha is the parsity controlling parameter. Higher values lead to
            sparser components.
        n_features:
            Lists of already selected number of features.
    """
    for i, a in enumerate(alphas):
        if (i == 0) and (n_features[i] < n):
            return a / 2
        elif (i < (len(alphas) - 1)) and ((n_features[i] > n) and (n_features[i + 1] < n)):
            return (alphas[i] + alphas[i + 1]) / 2
        elif (i == (len(alphas) - 1)) and (n_features[i] > n):
            return a * 2


def spca_feature_selection(
    adata: sc.AnnData,
    n: int,
    n_pcs: int = 30,
    a_init: float = 150,
    n_alphas_max: Optional[int] = 25,
    n_alphas_min: Optional[int] = 5,
    tolerance: float = 0.1,
    seed: int = 0,
    verbosity: int = 1,
    inplace: bool = True,
    n_jobs: int = -1,
):
    """Select features based on sparse pca.

    Sparse PCA is based on the regression formulation of pca with an additional lasso constraint.
    The lasso constraint leads to sparse loadings vectors for the found principal components.
    The higher the lasso parameter alpha the sparser the loadings vectors. We apply a search through
    different alphas to find sparse loadings vectors that are based on approximately n features in total.
    Since searching for the correct alpha takes a long time we define a tolerance: An accepted alpha is chosen
    if it comes with n_features selected such that::

        n <= n_features <= (1+tolerance)*n


    One flaw: if we do not find exactly n features we use the sums of abs loadings as scores and neglect the last features
    with lowest scores to get n features. The problem compared with the non sparse pca selection is that we do not
    scale our loadings with the eigenvalue of the PCs. In simple pca we did that (as far as i know). It could be that
    a feature has very low loadings on the first PCs which are important PCs, the feature however is droped. IMO it's
    fine, the effect shouldn't be too high, since we only drop a few features at maximum and the features still have
    low loadings.. A question is also if we have a clear hierarchy which PC is more important in SPCA -> measure variance?
    (small update on this: our pca loadings based selection also does not take scaling with sqrt(eigenvalue) into account)

    Note that SPCA takes much longer for small alphas (less sparsity). Keep that in mind when choosing a_init etc.

    Args:
        adata:
            Data with log normalised counts in `adata.X`.
        n:
            Number of features to select.
        n_pcs:
            Number of sparse PC components.
        a_init:
            First lasso alpha that is tried.
        n_alphas_max:
            The maximal number of alphas that is tried to find n_features in the defined tolerance to :attr:`n`.
        n_alphas_min:
            Minimal number of alphas to try. This is interesting to set if we are already in the tolerance
            but the search should still go on to ultimately find exactly :attr:`n` features.
        tolerance:
            Accept solutions for n_features such that n <= n_features <= (1+tolerance)*n
        seed:
            Random seed for sparse pca optimization.
        verbosity:
            Verbosity level.
        inplace: bool
            if True save results in adata.var else return results
        n_jobs: int
            Number of parallel jobs to run.

    Returns:
        if inplace: add

            - `adata.var['selection']` - boolean column of n selected features
            - `adata.var['selection_scores']` - column with sum of abs loadings for each feature

        else:
            pd.Dataframe with columns 'selection', 'selection_scores'
    """
    print(type(adata.X), flush=True)
    if scipy.sparse.issparse(adata.X):
        if verbosity > 0:
            print("Convert adata.X from sparse representation to standard format.")
        if adata.is_view:
            adata = adata.copy()
        adata.X = adata.X.toarray()
    print(type(adata.X), flush=True)
    # conditions for while loop
    n_features_in_tolerance = False
    n_features_equals_n = False
    max_alphas = False
    if (type(n_alphas_min) == int) and (n_alphas_min > 1):
        min_alphas = False
    else:
        min_alphas = True

    alphas = [a_init]
    if verbosity >= 1:
        t0 = datetime.now()
        print(f"Start alpha trial {len(alphas)} ({datetime.now() - t0})", flush=True)
    features, loadings = sparse_pca(adata.X, n_pcs, alphas[0], seed, n_jobs)
    n_features = [np.sum(features)]
    if verbosity == 1:
        print(f"\t alpha = {alphas[0]}, n_features = {n_features[0]}", flush=True)
    # adjust conditions
    if n_features[-1] == n:
        n_features_equals_n = True
    if (n <= n_features[-1]) and (n_features[-1] <= (1 + tolerance) * n):
        n_features_in_tolerance = True
    if (type(n_alphas_min) == int) and (len(alphas) >= n_alphas_min):
        min_alphas = True
    if (type(n_alphas_max) == int) and (len(alphas) == n_alphas_max):
        max_alphas = True

    while not (n_features_equals_n or (n_features_in_tolerance and min_alphas) or max_alphas):
        # get next alpha
        alphas, n_features = sort_alphas(alphas, n_features)
        alpha = next_alpha(n, alphas, n_features)
        assert alpha is not None
        alphas.append(alpha)
        if verbosity >= 1:
            print(f"Start alpha trial {len(alphas)} ({datetime.now() - t0})", flush=True)
        # sparse pca
        features, loadings = sparse_pca(adata.X, n_pcs, alpha, seed, n_jobs)
        n_f = np.sum(features)
        n_features.append(n_f)
        if verbosity == 1:
            print(f"\t alpha = {alpha}, n_features = {n_f}", flush=True)

        # adjust conditions
        if n_features[-1] == n:
            n_features_equals_n = True
        if (n <= n_features[-1]) and (n_features[-1] <= (1 + tolerance) * n):
            n_features_in_tolerance = True
        if (type(n_alphas_min) == int) and (len(alphas) >= n_alphas_min):
            min_alphas = True
        if (type(n_alphas_max) == int) and (len(alphas) == n_alphas_max):
            max_alphas = True

        if verbosity >= 2:
            print(f"########## {len(alphas)} ##########", flush=True)
            print("alphas     : ", alphas, flush=True)
            print("n_features : ", n_features, flush=True)
            print(f"equal_n   : {n_features_equals_n}", flush=True)
            print(f"in tol    : {n_features_in_tolerance}", flush=True)
            print(f"min_alphas: {min_alphas}", flush=True)
            print(f"max_alphas: {max_alphas}", flush=True)

    if verbosity >= 1:
        if n_features_equals_n:
            print("Found solution with sparse PC components based on exactly n features", flush=True)
        elif n_features_in_tolerance:
            print(
                "Found solution with number of features in defined tolerance. Excluded features with lowest loadings sums",
                flush=True,
            )
        elif max_alphas:
            print("Maximal number of trials (n_alphas_max) was reached without finding a solution", flush=True)
    if (not n_features_equals_n) and (not n_features_in_tolerance) and max_alphas:
        return 0

    scores = np.abs(np.sum(loadings, axis=0))
    selection_idxs = (-scores).argsort()[:n]
    selection = np.zeros_like(features, dtype=bool)
    selection[selection_idxs] = True

    if inplace:
        adata.var["selection_score"] = scores
        adata.var["selection"] = selection
    else:
        df = pd.DataFrame(index=adata.var.index, columns=["selection", "selection_scores"])
        df["selection"] = selection
        df["selection_score"] = scores
        return df


##################################################################################
################################### selfE ########################################
##################################################################################
# This was copied from my bachelor student and does not really work well. Takes for
# ever to compute. Either we reimplement the method or drop it completly. The original
# implementation was written in R - it's quite slow in general, especially for high `n`s


def select_selfE_features(adata: sc.AnnData, n: int, inplace: bool = True, verbosity: int = 0):
    """Select a subset of genes which expresses all the other genes best (SOMP-Algorithm).
    ... inspired by Rai et al ...

    Args:
        adata:
            adata.X needs to be sc.sparse.csr_matrix, if a np.array is provided it's converted to a sprase matrix.
        n:
            Number of features to be selected.
        inplace:
            Save results in adata, otherwise return a dataframe.
        verbosity:
            Verbosity level.

    Returns:
        if inplace: add
            `adata.var['selection']` - boolean pd.Series of n selected features
        else: return
            pd.Dataframe with column 'selection'
    """

    a = adata.copy()

    if not scipy.sparse.issparse(a.X):
        a.X = scipy.sparse.csr.csr_matrix(a.X)

    if verbosity > 0:
        print(f"select {n} selfE genes")

    feature_idxs = []
    scores = []
    Y = a.X.copy()
    Phi = a.X.copy()
    R = Y.copy()

    if verbosity > 0:
        t0 = datetime.now()
    for i in range(0, n):
        if verbosity > 1:
            print(f"Select feature {i + 1}/{n} ({datetime.now() - t0})")
        K = abs(np.dot(R.transpose(), R))  # matrix product
        c = []

        c = np.sqrt(K.multiply(K).sum(1))
        pos = np.array(c).flatten().argsort()[::-1]

        feature_idxs.append(pos[0])
        scores.append(c[pos[0]])
        PhiS = Phi[:, feature_idxs].copy()
        # multiply extracted feature with its psuedo inverse and original data to minimize error.
        Yiter = PhiS.dot(scipy.sparse.csr_matrix(np.linalg.pinv(PhiS.toarray())).dot(Y))

        R = Y - Yiter

    if verbosity > 0:
        print(f"Selected {n} features ({datetime.now() - t0})")
    genes = a.var.index[feature_idxs].values
    df = pd.DataFrame(index=adata.var.index, columns=["selection"])
    df["selection"] = False
    df.loc[genes, "selection"] = True
    if inplace:
        adata.var["selection"] = df["selection"].copy()
    else:
        return df[["selection"]].copy()
