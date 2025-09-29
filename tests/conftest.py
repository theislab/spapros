"""Global fixtures for testing."""

import random
from pathlib import Path

import pandas as pd
import pytest
import scanpy as sc

from spapros import ev, se

#############
# selection #
#############


@pytest.fixture()
def out_dir():
    out_dir = "tests/_out_dir"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture()
def raw_selector(small_adata):
    random.seed(0)
    sc.pp.log1p(small_adata)
    small_adata = small_adata[random.sample(range(small_adata.n_obs), 100), :]
    raw_selector = se.ProbesetSelector(
        small_adata,
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
def selector_with_marker(small_adata):
    random.seed(0)
    sc.pp.log1p(small_adata)
    small_adata = small_adata[random.sample(range(small_adata.n_obs), 100), :]
    raw_selector = se.ProbesetSelector(
        small_adata,
        n=50,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        verbosity=0,
        save_dir=None,
        marker_list="tests/evaluation/test_data/small_data_marker_list.csv",
    )
    raw_selector.select_probeset()
    return raw_selector


@pytest.fixture()
def small_adata():
    adata = sc.read_h5ad("tests/selection/test_data/small_data_raw_counts.h5ad")
    # random.seed(0)
    # adata = adata[random.sample(range(adata.n_obs), 100), :]
    return adata


@pytest.fixture()
def adata_pbmc3k():
    adata = sc.read_h5ad("tests/selection/test_data/adata_pbmc3k.h5ad")
    # quick fix because somehow "base" gets lost
    adata.uns["log1p"]["base"] = None
    return adata


def ref_probeset(adata_pbmc3k, n, genes_key, seeds, verbosity, save_dir, request, reference_selections):
    se.select_reference_probesets(
        adata_pbmc3k,
        n=n,
        genes_key=genes_key,
        seeds=seeds,
        verbosity=verbosity,
        save_dir=None if not save_dir else request.getfixturevalue(save_dir),
        reference_selections=reference_selections,
    )


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
        marker_list=marker_list,
    )
    four_probesets = pd.read_csv("tests/evaluation/test_data/4_probesets_of_20.csv", index_col=0)
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
