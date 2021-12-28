from spapros import se, ev
import anndata


def test_selecor_init(raw_selector):
    adata = raw_selector.adata
    assert type(adata) == anndata.AnnData
    assert raw_selector.ct_key in adata.obs_keys()
    assert raw_selector.g_key in adata.var


def test_get_celltypes_with_too_small_test_sets(raw_selector):
    cts_below_min_test_size, counts_below_min_test_size = ev.get_celltypes_with_too_small_test_sets(
        raw_selector.adata[raw_selector.obs], raw_selector.ct_key, min_test_n=raw_selector.min_test_n, split_kwargs={"seed": raw_selector.seed, "split": 4}
    )
    assert all(counts_below_min_test_size) < raw_selector.min_test_n

# TODO
# - test whether all file paths exists
# -
