"""Global fixtures for testing."""
import random
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from spapros import ev
from spapros import se
from spapros.util import util


#############
# selection #
#############


@pytest.fixture()
def small_adata():
    small_adata = sc.read_h5ad("tests/selection/test_data/small_data_raw_counts.h5ad")
    # random.seed(0)
    # adata = adata[random.sample(range(adata.n_obs), 100), :]
    return small_adata


@pytest.fixture()
def tiny_adata(small_adata):
    random.seed(0)
    tiny_adata = small_adata[random.sample(range(small_adata.n_obs), 200), :]
    sc.pp.filter_genes(tiny_adata, min_counts=3)
    # np.isnan(small_adata.X.toarray()).all(axis=1)
    return tiny_adata


@pytest.fixture()
def adata_pbmc3k():
    adata = sc.read_h5ad("tests/selection/test_data/adata_pbmc3k.h5ad")
    # quick fix because somehow "base" gets lost
    adata.uns["log1p"]["base"] = None
    return adata


@pytest.fixture()
def raw_selector(tiny_adata):
    sc.pp.log1p(tiny_adata)
    raw_selector = se.ProbesetSelector(
        tiny_adata,
        n=50,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        verbosity=0,
        save_dir=None,
    )
    return raw_selector


@pytest.fixture()
def selector(raw_selector):
    raw_selector.select_probeset()
    return raw_selector


@pytest.fixture()
def selector_with_marker(tiny_adata):
    sc.pp.log1p(tiny_adata)
    selector = se.ProbesetSelector(
        tiny_adata,
        n=50,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        verbosity=0,
        save_dir=None,
        marker_list="tests/selection/test_data/small_data_marker_list.csv"
    )
    selector.select_probeset()
    return selector


@pytest.fixture()
def selector_with_penalties(tiny_adata, lower_th=1, upper_th=3.5):

    # we don't set fixed expression thresholds. Instead, we introduce smoothness factors (heuristic user choice)
    factor = 1
    var = [factor * 0.1, factor * 0.5]

    # get the expression quantiles
    util.get_expression_quantile(tiny_adata, q=0.99, log1p=False, zeros_to_nan=False, normalise=False)
    # design the penalty kernel
    penalty = util.plateau_penalty_kernel(var=var, x_min=np.array(lower_th), x_max=np.array(upper_th))
    # calcluate the expression penalties
    tiny_adata.var['expression_penalty'] = penalty(tiny_adata.var[f'quantile_0.99'])

    # upper
    util.get_expression_quantile(tiny_adata, q=0.99, log1p=False, zeros_to_nan=False, normalise=False)
    penalty = util.plateau_penalty_kernel(var=var, x_min=None, x_max=upper_th)
    tiny_adata.var['expression_penalty_upper'] = penalty(tiny_adata.var[f'quantile_0.99'])

    # lower
    util.get_expression_quantile(tiny_adata, q=0.9, log1p=False, zeros_to_nan=True, normalise=False)
    penalty = util.plateau_penalty_kernel(var=var, x_min=lower_th, x_max=None)
    tiny_adata.var['expression_penalty_lower'] = penalty(tiny_adata.var[f'quantile_0.9 expr > 0'])

    sc.pp.log1p(tiny_adata)
    selector = se.ProbesetSelector(
        tiny_adata,
        n=50,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        verbosity=0,
        save_dir="tests/selection/test_data/selector_with_penalties",
        # save_dir=None,
        marker_list="tests/selection/test_data/small_data_marker_list.csv",
        pca_penalties=["expression_penalty_upper", "expression_penalty_lower"],
        DE_penalties=["expression_penalty_upper", "expression_penalty_lower"]
    )
    selector.select_probeset()
    return selector


@pytest.fixture()
def ref_probeset(
    adata, n, genes_key, seeds, verbosity, save_dir, request, reference_selections
):
    reference_probesets = se.select_reference_probesets(
        adata,
        n=n,
        genes_key=genes_key,
        seeds=seeds,
        verbosity=verbosity,
        save_dir=None if not save_dir else request.getfixturevalue(save_dir),
        methods=reference_selections,
    )

    return reference_probesets

##############
# evaluation #
##############


@pytest.fixture()
def small_probeset():
    selection = pd.read_csv("tests/evaluation/test_data/selections_genesets_1.csv", index_col="index")
    genes = list(selection.index[selection["genesets_1_0"]])
    # ['ISG15', 'IFI6', 'S100A11', 'S100A9', 'S100A8', 'FCER1G', 'FCGR3A',
    #        'GNLY', 'GPX1', 'IL7R', 'CD74', 'LTB', 'HLA-DPA1', 'HLA-DPB1',
    #        'SAT1', 'LYZ', 'IL32', 'CCL5', 'NKG7', 'LGALS1']
    return genes


@pytest.fixture()
def marker_list():
    return {
        "celltype_1": ["S100A8", "S100A9", "LYZ", "BLVRB"],
        "celltype_6": ["BIRC3", "TMEM116", "CD3D"],
        "celltype_7": ["CD74", "CD79B", "MS4A1"],
        "celltype_2": ["C5AR1"],
        "celltype_5": ["RNASE6"],
        "celltype_4": ["PPBP", "SPARC", "CDKN2D"],
        "celltype_8": ["NCR3"],
        "celltype_9": ["NAPA-AS1"],
    }


@pytest.fixture()
def raw_evaluator(small_adata):
    # random.seed(0)
    # small_adata = small_adata[random.sample(range(small_adata.n_obs), 100), :]
    raw_evaluator = ev.ProbesetEvaluator(small_adata, scheme="full", verbosity=0, results_dir=None)
    return raw_evaluator


@pytest.fixture()
def evaluator_with_dir(small_adata):
    # random.seed(0)
    # small_adata = small_adata[random.sample(range(small_adata.n_obs), 100), :]
    evaluator = ev.ProbesetEvaluator(
        small_adata, scheme="full", verbosity=0, results_dir="tests/evaluation/test_data/evaluation_results_probeset1"
    )
    return evaluator


@pytest.fixture()
def evaluator(evaluator_with_dir, small_probeset):
    evaluator_with_dir.evaluate_probeset(small_probeset)
    return evaluator_with_dir


@pytest.fixture()
def evaluator_4_sets(small_adata, marker_list):
    evaluator = ev.ProbesetEvaluator(
        small_adata,
        scheme="full",
        verbosity=0,
        results_dir="tests/evaluation/test_data/evaluation_results_4_sets",
        marker_list=marker_list
    )
    four_probesets = pd.read_csv("tests/evaluation/test_data/4_probesets_of_20.csv",
                                 index_col=0)
    for set_id in four_probesets:
        evaluator.evaluate_probeset(set_id=set_id, genes=list(four_probesets[set_id]))
    return evaluator


@pytest.fixture()
def selections_info_1(evaluator_4_sets):
    s_info = pd.DataFrame(index=list(evaluator_4_sets.results["knn_overlap"].keys()))
    s_info["groupby"] = ["ref"] * 3 + ["final"]
    return s_info


@pytest.fixture()
def selections_info_2(evaluator_4_sets):
    s_info = pd.DataFrame(index=list(evaluator_4_sets.results["knn_overlap"].keys()))
    s_info["groupby"] = ["ref"] * 3 + ["final"]
    s_info["color"] = ["purple"] * 3 + ["red"]
    s_info["linewidth"] = [1] * 3 + [3]
    s_info["linestyle"] = ["dotted"] * 3 + ["dashed"]
    return s_info
