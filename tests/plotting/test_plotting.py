import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.compare import compare_images

from spapros import pl

# Use universal non-interactive matplotlib backend such that saved images are the same on all systems
matplotlib.use("Agg")

plt.rcParams["figure.dpi"] = 40
plt.rcParams["savefig.dpi"] = 40


# Note: The figures depend somehow on the environment!
# Tests might fail if compared figures derived from different envs eg development env and test env


def _transform_string(s):
    """Transforms a string by replacing ': ' with '-' and ', ' with '_',
    and then removing problematic characters for Windows filenames.

    In the tests we use the kwargs of the functions as strings to name output files. To get a valid name for windows
    we need to replace certain characters.
    """
    # Initial replacements
    transformed = s.replace(": ", "-").replace(", ", "_")

    # Additional removals
    for char in [":", ",", "{", "}", "'", "[", "]"]:
        transformed = transformed.replace(char, "")

    return transformed


#############
# selection #
#############


def test_masked_dotplot(small_adata, selector, out_dir):  # tmp_path):
    ref_name = Path("tests/plotting/test_data/masked_dotplot.png")
    # fig_name = Path(f"{tmp_path}/masked_dotplot.png")
    fig_name = Path(f"{out_dir}/masked_dotplot.png")
    random.seed(0)
    small_adata = small_adata[random.sample(range(small_adata.n_obs), 100), :]
    pl.masked_dotplot(small_adata, selector, save=fig_name)
    # pl.masked_dotplot(small_adata, selector, save=ref_name)
    assert compare_images(ref_name, fig_name, 0.001) is None


@pytest.mark.parametrize(
    "fun, kwargs",
    [
        # plot_histogram -> selection_histogram: #TODO: Add again at some point, this fct is a bit broken an complicated
        # (
        #    "plot_histogram",
        #    {
        #        "x_axis_keys": None,
        #        "selections": None,
        #        "penalty_keys": None,
        #        "unapplied_penalty_keys": None,
        #        "background_key": None,
        #    },
        # ),
        # (
        #    "plot_histogram",
        #    {
        #        "x_axis_keys": {
        #            "expression_penalty_upper": "quantile_0.99",
        #            "expression_penalty_lower": "quantile_0.9 expr > 0",
        #            "marker": "quantile_0.99",
        #        },
        #        "selections": ["marker"],
        #        "penalty_keys": {"marker": []},
        #        "unapplied_penalty_keys": {"marker": []},
        #        "background_key": True,
        #    },
        # ),
        # (
        #    "plot_histogram",
        #    {
        #        "x_axis_keys": {
        #            "expression_penalty_upper": "quantile_0.99",
        #            "expression_penalty_lower": "quantile_0.9 expr > 0",
        #            "marker": "quantile_0.99",
        #        },
        #        "selections": ["marker"],
        #        "penalty_keys": {"marker": []},
        #        "unapplied_penalty_keys": {"marker": []},
        #        "background_key": None,
        #    },
        # ),
        # (
        #    "plot_histogram",
        #    {
        #        "x_axis_keys": {
        #            "expression_penalty_upper": "quantile_0.99",
        #            "expression_penalty_lower": "quantile_0.9 expr > 0",
        #            "marker": "quantile_0.99",
        #        },
        #        "selections": ["marker"],
        #        "penalty_keys": {"marker": []},
        #        "unapplied_penalty_keys": {"marker": []},
        #        "background_key": "all",
        #    },
        # ),
        # plot_coexpression -> correlation_matrix
        (
            "plot_coexpression",
            {
                "selections": None,
                "n_cols": 3,
            },
        ),
        (
            "plot_coexpression",
            {
                "selections": None,
                "n_cols": 1,
            },
        ),
        ("plot_coexpression", {"selections": ["marker"], "colorbar": False}),
        ## plot_clf_genes -> classification_rule_umaps #TODO: Fix - somehow not reproducible, mabye some seed issue?
        # ("plot_clf_genes", {"till_rank": 2, "importance_th": 0.8}),
        # overlap:
        ("plot_gene_overlap", {"style": "venn"}),
        ("plot_gene_overlap", {"style": "upset"}),
    ],
)
# TODO maybe add "plot_confusion_matrix_difference", "plot_marker_correlation"
# TODO add further kwargs for each fun
def test_selection_plots(selector_with_marker, fun, out_dir, kwargs):  # tmp_path
    ref_name = Path(_transform_string(f"tests/plotting/test_data/selection_{fun}_{kwargs}.png"))
    # fig_name = Path(_transform_string(f"{tmp_path}/selection_{fun}_{kwargs}.png"))
    fig_name = Path(_transform_string(f"{out_dir}/selection_{fun}_{kwargs}.png"))
    getattr(selector_with_marker, fun)(save=fig_name, show=False, **kwargs)
    # getattr(selector_with_marker, fun)(save=ref_name, show=False, **kwargs)
    assert compare_images(ref_name, fig_name, 0.001) is None


##############
# evaluation #
##############


def test_plot_summary(evaluator, out_dir):
    ref_name = Path("tests/plotting/test_data/tmp_plot_summary.png")
    fig_name = Path(f"{out_dir}/tmp_plot_summary.png")
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
        ("plot_coexpression", {}),  # ("plot_correlation_matrix", {}),
        ("plot_cluster_similarity", {}),
        # ev.plot_knn_overlap --> pl.knn_overlap
        ("plot_knn_overlap", {"set_ids": ["ref_DE", "ref_PCA", "spapros_selection"], "selections_info": None}),
        ("plot_knn_overlap", {"set_ids": None, "selections_info": "selections_info_1"}),
        ("plot_knn_overlap", {"set_ids": None, "selections_info": "selections_info_2"}),
    ],
)
# TODO maybe add "plot_confusion matrix_difference", "plot_marker_correlation"
# TODO add further kwargs for each fun
def test_evaluation_plots(evaluator_4_sets, fun, out_dir, kwargs, request):
    ref_name = Path(_transform_string(f"tests/plotting/test_data/evaluation_{fun}_{kwargs}.png"))
    fig_name = Path(_transform_string(f"{out_dir}/evaluation_{fun}_{kwargs}.png"))
    if "selections_info" in kwargs:
        if kwargs["selections_info"] is not None:
            kwargs["selections_info"] = request.getfixturevalue(kwargs["selections_info"])
    getattr(evaluator_4_sets, fun)(save=fig_name, show=False, **kwargs)
    # getattr(evaluator_4_sets, fun)(save=ref_name, show=False, **kwargs)
    assert compare_images(ref_name, fig_name, 0.001) is None
