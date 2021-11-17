"""Global fixtures for testing."""
import pandas as pd
import pytest
import scanpy as sc


@pytest.fixture()
def small_adata():
    adata = sc.read_h5ad("data/small_data_raw_counts.h5ad")

    return adata


@pytest.fixture()
def small_probeset():
    selection = pd.read_csv("tests/evaluation/test_data/selections_genesets_1.csv", index_col="index")
    genes = list(selection.index[selection["genesets_1_0"]])

    return genes
