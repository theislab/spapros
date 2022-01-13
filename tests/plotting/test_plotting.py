import pytest
from matplotlib.testing.compare import compare_images
from spapros import pl

# Note: The figures depend somehow on the environment!
# Test failes if compared figures derived from different envs eg development env and test env


#############
# selection #
#############


def test_masked_dotplot(small_adata, selector):
    ref_name = "tests/plotting/test_data/masked_dotplot.png"
    fig_name = "tests/plotting/test_data/tmp_masked_dotplot.png"
    pl.masked_dotplot(small_adata, selector, save=fig_name)
    # pl.masked_dotplot(small_adata, selector, save=ref_name)
    assert compare_images(ref_name, fig_name, 0.001) is None


##############
# evaluation #
##############


def test_plot_summary(evaluator):
    ref_name = "tests/plotting/test_data/plot_summary.png"
    fig_name = "tests/plotting/test_data/tmp_plot_summary.png"
    evaluator.plot_summary(show=False, save=fig_name)
    # evaluator.plot_summary(show=False, save=ref_name)
    assert compare_images(ref_name, fig_name, 0.001) is None


@pytest.mark.parametrize("metric", ["gene_corr", "forest_clfs"])
def test_plot_evaluations_gene_corr(evaluator, small_probeset, metric):
    fig_name = f"tests/plotting/test_data/tmp_plot_evaluations_{metric}.png"
    ref_name = f"tests/plotting/test_data/plot_evaluations_{metric}.png"
    evaluator.evaluate_probeset(small_probeset)
    evaluator.plot_evaluations(metrics=[metric], show=False, save=fig_name)
    # evaluator.plot_evaluations(metrics=[metric], show=False, save=ref_name)
    assert compare_images(ref_name, fig_name, 0.001) is None
