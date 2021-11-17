"""Global fixtures for testing."""

import pytest
import scanpy as sc
import pandas as pd


@pytest.fixture()
def small_adata():
    adata = sc.read_h5ad("data/small_data_raw_counts.h5ad")

    return adata


@pytest.fixture()
def small_probeset():
    selection = pd.read_csv("results/selections_genesets_1.csv", index_col="index")
    genes = list(selection.index[selection['genesets_1_0']])

    return genes

