import anndata
import pytest
import scanpy as sc
from spapros import ev


def test_selecor_init(raw_selector):
    adata = raw_selector.adata
    assert type(adata) == anndata.AnnData
    assert raw_selector.ct_key in adata.obs_keys()
    assert raw_selector.g_key in adata.var


def test_get_celltypes_with_too_small_test_sets(raw_selector):
    cts_below_min_test_size, counts_below_min_test_size = ev.get_celltypes_with_too_small_test_sets(
        raw_selector.adata[raw_selector.obs],
        raw_selector.ct_key,
        min_test_n=raw_selector.min_test_n,
        split_kwargs={"seed": raw_selector.seed, "split": 4},
    )
    assert all(counts_below_min_test_size) < raw_selector.min_test_n


def test_load_adata():
    adata = sc.datasets.pbmc3k()
    adata_tmp = sc.datasets.pbmc3k_processed()
    adata = adata[adata_tmp.obs_names, adata_tmp.var_names]
    adata_raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4, key_added="size_factors")
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger", n_top_genes=1000)
    adata.X = adata_raw.X
    sc.pp.log1p(adata)
    adata.obs["celltype"] = adata_tmp.obs["louvain"]


def test_error_and_repeat(raw_selector):
    raw_selector.verbosity = 2
    # this line will lead to a TypeError
    raw_selector.n_pca_genes = "string"
    with pytest.raises(TypeError):
        raw_selector.select_probeset()
    # fixing the mistake
    raw_selector.n_pca_genes = 100
    # now check that restarting the evaluation works (earlier, the progress bars made trouble)
    raw_selector.select_probeset()
    # assert None
