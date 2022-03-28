"""Global fixtures for testing."""
import pandas as pd
import pytest
import scanpy as sc
from spapros import ev
from spapros import se
import random


#############
# selection #
#############

@pytest.fixture()
def raw_selector(small_adata):
    raw_selector = se.ProbesetSelector(small_adata, n=50, celltype_key="celltype", verbosity=0, save_dir=None)
    return raw_selector


@pytest.fixture()
def selector(raw_selector):
    raw_selector.select_probeset()
    return raw_selector


@pytest.fixture()
def small_adata():
    adata = sc.read_h5ad("data/small_data_raw_counts.h5ad")
    # random.seed(0)
    # adata = adata[random.sample(range(adata.n_obs), 100), :]
    return adata


@pytest.fixture()
def adata_pbmc3k():
    adata = sc.datasets.pbmc3k()
    adata_tmp = sc.datasets.pbmc3k_processed()
    adata = adata[adata_tmp.obs_names, adata_tmp.var_names]
    adata_raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4, key_added="size_factors")
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=1000)
    adata.X = adata_raw.X
    sc.pp.log1p(adata)
    adata.obs['celltype'] = adata_tmp.obs['louvain']
    return adata


@pytest.fixture(params=[None, "./probeset_selection"])
def selection_dir(request):
    return request.getfixturevalue(request.param)


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
    raw_evaluator = ev.ProbesetEvaluator(small_adata,
                                         scheme="quick",
                                         verbosity=0,
                                         results_dir=None)
    return raw_evaluator


@pytest.fixture()
def evaluator(raw_evaluator, small_probeset):
    raw_evaluator.evaluate_probeset(small_probeset)
    return raw_evaluator
