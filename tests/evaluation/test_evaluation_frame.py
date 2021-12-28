"""Test cases for the ProbesetEvaluator."""
import anndata


def test_init(raw_evaluator):
    adata = raw_evaluator.adata
    assert type(adata) == anndata.AnnData
    assert raw_evaluator.celltype_key in adata.obs_keys()


def test_shared_computations(raw_evaluator):
    raw_evaluator.compute_or_load_shared_results()
    for metric in raw_evaluator.metrics:
        assert metric in raw_evaluator.shared_results


def test_computed_metrics(raw_evaluator, small_probeset):
    raw_evaluator.evaluate_probeset(small_probeset, set_id="testset")
    for metric in raw_evaluator.metrics:
        assert metric in raw_evaluator.pre_results
        assert metric in raw_evaluator.results
