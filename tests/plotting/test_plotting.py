import random

import pytest
from matplotlib.testing.compare import compare_images
from spapros import pl


# Note: The figures depend somehow on the environment!
# Tests might fail if compared figures derived from different envs eg development env and test env


#############
# selection #
#############


def test_masked_dotplot(small_adata, selector, tmp_path):
    ref_name = "tests/plotting/test_data/masked_dotplot.png"
    fig_name = f"{tmp_path}/tmp_masked_dotplot.png"
    random.seed(0)
    small_adata = small_adata[random.sample(range(small_adata.n_obs), 100), :]
    pl.masked_dotplot(small_adata, selector, save=fig_name)
    # pl.masked_dotplot(small_adata, selector, save=ref_name)
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
        ("plot_knn_overlap", {}),
    ],
)
@pytest.mark.parametrize("set_ids", [None, range(100)])
# TODO maybe add "plot_confusion matrix_difference", "plot_marker_correlation"
# TODO add further kwargs for each fun
def test_evalution_plots(evaluator, fun, tmp_path, set_ids, kwargs):
    ref_name = f"tests/plotting/test_data/evaluation_{fun}_{set_ids}_{kwargs}.png"
    fig_name = f"{tmp_path}/evaluations_{fun}_{set_ids}_{kwargs}.png"
    getattr(evaluator, fun)(save=fig_name, **kwargs)
    # getattr(evaluator, fun)(save=ref_name, **kwargs)
    assert compare_images(ref_name, fig_name, 0.001) is None


@pytest.mark.parametrize(
    "fun, kwargs", [("plot_gene_overlap", {"style": "venn"}), ("plot_gene_overlap", {"style": "upset"})]
)
@pytest.mark.parametrize("set_ids", [None, range(100)])
# TODO maybe add "plot_confusion matrix_difference", "plot_marker_correlation"
# TODO add further kwargs for each fun
def test_selection_plots(selector, fun, tmp_path, set_ids, kwargs):
    ref_name = f"tests/plotting/test_data/selection_{fun}_{set_ids}_{kwargs}.png"
    fig_name = f"{tmp_path}/selection_{fun}_{set_ids}_{kwargs}.png"
    getattr(selector, fun)(save=fig_name, **kwargs)
    # getattr(selector, fun)(save=ref_name, **kwargs)
    assert compare_images(ref_name, fig_name, 0.001) is None
