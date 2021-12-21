"""Global fixtures for testing."""
import pandas as pd
import pytest
import scanpy as sc
from spapros import ev
from spapros import se




@pytest.fixture()
def small_adata():
    adata = sc.read_h5ad("data/small_data_raw_counts.h5ad")

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
def evaluator(small_adata):
    evaluator = ev.ProbesetEvaluator(small_adata,
                                     scheme="quick",
                                     verbosity=0,
                                     results_dir=None)
    return evaluator


#############
# selection #
#############

@pytest.fixture()
def selector(small_adata):
    selector = se.ProbesetSelector(small_adata, n=50, celltype_key="celltype", verbosity=0)
    return selector
