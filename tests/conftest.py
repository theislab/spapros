"""Global fixtures for testing."""
import random

import pandas as pd
import pytest
import scanpy as sc
from spapros import ev
from spapros import se


#############
# selection #
#############


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
        marker_list="/big/st/strasserl/spapros/tests/selection/test_data/small_data_raw_counts.h5ad"
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
        small_adata, scheme="full", verbosity=0, results_dir="tests/evaluation/test_data/evaluation_results"
    )
    return evaluator


@pytest.fixture()
def evaluator(evaluator_with_dir, small_probeset):
    evaluator_with_dir.evaluate_probeset(small_probeset)
    return evaluator_with_dir
