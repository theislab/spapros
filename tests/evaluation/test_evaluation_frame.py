"""Test cases for the ProbesetEvaluator."""
import anndata
import pytest
from spapros.evaluation.metrics import get_metric_default_parameters
from spapros.evaluation.evaluation import ProbesetEvaluator



def test_init(small_adata):
    evaluator = ProbesetEvaluator(small_adata)
    adata = evaluator.adata
    assert type(adata) == anndata.AnnData
    assert evaluator.celltype_key in adata.obs_keys()


def test_shared_computations(small_adata):
    evaluator = ProbesetEvaluator(small_adata)
    evaluator.compute_or_load_shared_results()
    for metric in evaluator.metrics:
        assert metric in evaluator.shared_results


def test_computed_metrics(small_adata, small_probeset):
    evaluator = ProbesetEvaluator(small_adata, results_dir=None)
    evaluator.evaluate_probeset(small_probeset, set_id="testset")
    for metric in evaluator.metrics:
        assert metric in evaluator.pre_results
        assert metric in evaluator.results




