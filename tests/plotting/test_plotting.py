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


@pytest.mark.parametrize("metric", ["gene_corr", "forest_clfs"])
def test_plot_evaluations_gene_corr(evaluator, small_probeset, metric, tmp_path):
    ref_name = f"tests/plotting/test_data/plot_evaluations_{metric}.png"
    fig_name = f"{tmp_path}/plot_evaluations_{metric}.png"
    evaluator.plot_evaluations(metrics=[metric], show=False, save=fig_name)
    # evaluator.plot_evaluations(metrics=[metric], show=False, save=ref_name)
    assert compare_images(ref_name, fig_name, 0.001) is None
