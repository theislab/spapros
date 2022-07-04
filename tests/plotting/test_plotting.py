import random
import pytest
from matplotlib.testing.compare import compare_images
from spapros import pl
import scanpy as sc
import numpy as np


# Note: The figures depend somehow on the environment!
# Tests might fail if compared figures derived from different envs eg development env and test env


#############
# selection #
#############


def test_masked_dotplot(tiny_adata, selector, tmp_path):
    ref_name = "tests/plotting/test_data/masked_dotplot.png"
    fig_name = f"{tmp_path}/tmp_masked_dotplot.png"
    random.seed(0)
    pl.masked_dotplot(tiny_adata, selector, save=fig_name)
    pl.masked_dotplot(tiny_adata, selector, save=ref_name)
    assert compare_images(ref_name, fig_name, 0.001) is None


@pytest.mark.parametrize(
    "fun, kwargs, id", [
        # plot_histogram -> selection_histogram:
        ("plot_histogram", {
            "x_axis_keys": None,
            "selections": None,
            "penalty_keys": None,
            "unapplied_penalty_keys": None,
            "background_key": None,
        }, "default"),
        ("plot_histogram", {
            "x_axis_keys": {"expression_penalty_upper": "quantile_0.99",
                            "expression_penalty_lower": "quantile_0.9 expr > 0",
                            "marker": "quantile_0.99"},
            "selections": ["marker"],
            "penalty_keys": {"marker": []},
            "unapplied_penalty_keys": {"marker": []},
            "background_key": True
        }, "marker_w_bg"),
        ("plot_histogram", {
            "x_axis_keys": {"expression_penalty_upper": "quantile_0.99",
                            "expression_penalty_lower": "quantile_0.9 expr > 0",
                            "marker": "quantile_0.99"},
            "selections": ["marker"],
            "penalty_keys": {"marker": None},
            "unapplied_penalty_keys": {"marker": None},
            "background_key": None
        }, "marker_w_penal"),
        ("plot_histogram", {
            "x_axis_keys": {"expression_penalty_upper": "quantile_0.99",
                            "expression_penalty_lower": "quantile_0.9 expr > 0",
                            "marker": "quantile_0.99"},
            "selections": ["marker"],
            "penalty_keys": {"marker": []},
            "unapplied_penalty_keys": {"marker": []},
            "background_key": "all"
        }, "marker_w_bg_all"),
        # plot_coexpression -> correlation_matrix
        ("plot_coexpression", {
            "selections": None,
            "n_cols": 3,
            "scale": True
        }, "n_cols_3_scaled"),
        ("plot_coexpression", {
            "selections": None,
            "n_cols": 1,
            "scale": False
        }, "n_cols_1_unscaled"),
        ("plot_coexpression", {
            "selections": ["marker"],
            "colorbar": False
        }, "marker_wo_cbar"),

        # plot_classification_rule_umaps -> classification_rule_umaps
        ("plot_classification_rule_umaps", {
            "till_rank": 5,
            "importance_th": 0.3
        }, "till_rank_5_imp_th_03"),

        # overlap:
        ("plot_gene_overlap", {"style": "venn"}, "venn"),
        ("plot_gene_overlap", {"style": "upset"}, "upset")
    ]
)
# TODO maybe add "plot_confusion_matrix_difference", "plot_marker_correlation"
# TODO add further kwargs for each fun
def test_selection_plots(selector_with_penalties, fun, tmp_path, kwargs, id):
    ref_name = f"tests/plotting/test_data/selection_{fun}_{id}.png"
    fig_name = f"{tmp_path}/selection_{fun}_{id}.png"
    getattr(selector_with_penalties, fun)(save=fig_name, show=False, **kwargs)
    # getattr(selector_with_penalties, fun)(save=ref_name, show=False, **kwargs)
    assert compare_images(ref_name, fig_name, 0.001) is None


##############
# evaluation #
##############


def test_plot_summary(evaluator, tmp_path):
    ref_name = "tests/plotting/test_data/plot_summary.png"
    fig_name = f"{tmp_path}/tmp_plot_summary.png"
    evaluator.plot_summary(show=False, save=fig_name)
    # evaluator.plot_summary(show=False, save=ref_name)
    assert compare_images(ref_name, fig_name, 0.001) is None


# old version
# @pytest.mark.parametrize("metric", ["gene_corr", "forest_clfs"])
# def test_plot_evaluations_gene_corr(evaluator, small_probeset, metric, tmp_path):
#     ref_name = f"tests/plotting/test_data/plot_evaluations_{metric}.png"
#     fig_name = f"{tmp_path}/plot_evaluations_{metric}.png"
#     evaluator.plot_evaluations(metrics=[metric], show=False, save=fig_name)
#     # evaluator.plot_evaluations(metrics=[metric], show=False, save=ref_name)
#     assert compare_images(ref_name, fig_name, 0.001) is None


@pytest.mark.parametrize(
    "fun, kwargs",
    [
        ("plot_confusion_matrix", {}),
        ("plot_correlation_matrix", {}),
        ("plot_cluster_similarity", {}),

        # ev.plot_knn_overlap --> pl.knn_overlap
        ("plot_knn_overlap", {
            "set_ids": ["ref_DE", "ref_PCA", "spapros_selection"],
            "selections_info": None
        }),
        ("plot_knn_overlap", {
            "set_ids": None,
            "selections_info": "selections_info_1"
        }),
        ("plot_knn_overlap", {
            "set_ids": None,
            "selections_info": "selections_info_2"
        }),
    ],
)
# TODO maybe add "plot_confusion matrix_difference", "plot_marker_correlation"
# TODO add further kwargs for each fun
def test_evaluation_plots(evaluator_4_sets, fun, tmp_path, kwargs, request):
    ref_name = f"tests/plotting/test_data/evaluation_{fun}_{kwargs}.png"
    fig_name = f"{tmp_path}/evaluations_{fun}_{kwargs}.png"
    if "selections_info" in kwargs:
        if kwargs["selections_info"] is not None:
            kwargs["selections_info"] = request.getfixturevalue(kwargs["selections_info"])
    getattr(evaluator_4_sets, fun)(save=fig_name, show=False, **kwargs)
    # getattr(evaluator_4_sets, fun)(save=ref_name, show=False, **kwargs)
    assert compare_images(ref_name, fig_name, 0.001) is None


