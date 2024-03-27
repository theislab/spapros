"""Test cases for the ProbesetEvaluator."""

import anndata


def test_init(raw_evaluator):
    adata = raw_evaluator.adata
    assert isinstance(adata, anndata.AnnData)
    assert raw_evaluator.celltype_key in adata.obs_keys()


def test_compute_shared_results(raw_evaluator):
    raw_evaluator.compute_or_load_shared_results()
    for metric in raw_evaluator.metrics:
        assert metric in raw_evaluator.shared_results


# TODO: include this test again
# def test_load_shared_results(evaluator_with_dir):
#    evaluator_with_dir.compute_or_load_shared_results()
#    for metric in evaluator_with_dir.metrics:
#        assert metric in evaluator_with_dir.shared_results
#
#
# def test_computed_metrics(raw_evaluator, small_probeset):
#    raw_evaluator.evaluate_probeset(small_probeset, set_id="testset")
#    for metric in raw_evaluator.metrics:
#        assert metric in raw_evaluator.pre_results
#        assert metric in raw_evaluator.results
