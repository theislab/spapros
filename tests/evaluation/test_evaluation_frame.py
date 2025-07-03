"""Test cases for the ProbesetEvaluator."""

import anndata
import pytest


def test_init(raw_evaluator):
    adata = raw_evaluator.adata
    assert isinstance(adata, anndata.AnnData)
    assert raw_evaluator.celltype_key in adata.obs_keys()


def test_compute_shared_results(raw_evaluator):
    raw_evaluator.compute_or_load_shared_results()
    for metric in raw_evaluator.metrics:
        assert metric in raw_evaluator.shared_results


def test_load_shared_results(evaluator_with_dir):
    evaluator_with_dir.compute_or_load_shared_results()
    for metric in evaluator_with_dir.metrics:
        assert metric in evaluator_with_dir.shared_results


def test_computed_metrics(raw_evaluator, small_probeset):
    raw_evaluator.evaluate_probeset(small_probeset, set_id="testset")
    for metric in raw_evaluator.metrics:
        assert metric in raw_evaluator.pre_results
        assert metric in raw_evaluator.results


def test_error_and_repeat(raw_evaluator, small_probeset):
    raw_evaluator.verbosity = 2
    # this line will lead to a TypeError
    raw_evaluator.metrics = 0
    with pytest.raises(TypeError):
        raw_evaluator.evaluate_probeset(small_probeset, set_id="testset")
    # fixing the mistake
    raw_evaluator.metrics = raw_evaluator._get_metrics_of_scheme()
    # now check that restarting the evaluation works (earlier, the progress bars made trouble)
    raw_evaluator.evaluate_probeset(small_probeset, set_id="testset")
    # assert None
