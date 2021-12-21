"""Test cases for the metric calculations."""
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from spapros.evaluation.metrics import correlation_matrix
from spapros.evaluation.metrics import gene_set_correlation_matrix
from spapros.evaluation.metrics import max_marker_correlations
from spapros.evaluation.metrics import xgboost_forest_classification
from spapros.evaluation.metrics import mean_overlaps
from spapros.evaluation.metrics import clustering_nmis
from spapros.evaluation.metrics import knns
from spapros.evaluation.metrics import leiden_clusterings
from spapros.evaluation.metrics import marker_correlation_matrix

############################
# test shared computations #
############################


@pytest.mark.skip(reason="succeeds locally but fails on github")
def test_leiden_clustering_shared_comp_equals_ref(small_adata):
    ns = [3, 5]
    annotations = leiden_clusterings(small_adata, ns, start_res=1.0)
    annotations["resolution"] = annotations["resolution"].astype(float).round(5)
    ## to create the reference annotation:
    # annotations.to_csv("tests/evaluation/test_data/annotations.csv", float_format="%.5f")
    annotations_ref = pd.read_csv("tests/evaluation/test_data/annotations.csv", index_col=0, dtype=object)
    annotations_ref["resolution"] = annotations["resolution"].astype(float).round(5)
    assert pd.testing.assert_frame_equal(annotations, annotations_ref) is None


def test_leiden_clustering_shared_comp_minimal():
    minimal_adata = sc.AnnData(X=np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]]))
    annotations = leiden_clusterings(minimal_adata, ns=[2, 2], start_res=1.0)
    # with the initial resolution, immediately a clustering with 2 clusters is found
    assert annotations.shape == (1, 5)
    assert all(annotations["resolution"] == 1.0)
    assert len(annotations.iloc[0, 1:].unique()) == 2


@pytest.mark.parametrize(
    "ns, n_range",
    [
        ([2, 3], [2, 3]),
        # ([20, 23], [20, 21, 22, 23]),  resolution diff < 0.00005 between 19 and 21
        # so it terminates before finding a clustering
        # with n=20
        # ([2, 1], []),  ValueError
        ([1, 1], [1]),
    ],
)
def test_leiden_clustering_shared_comp_each_n(small_adata, ns, n_range):
    annotations = leiden_clusterings(small_adata, ns, start_res=1.0)
    for x in n_range:
        # assert that for each requested n, the output dataframe contains a column
        assert x in annotations.index
        # assert that the clustering found for n, really has n cluster
        assert len(annotations.loc[x, ~annotations.columns.isin(["resolution"])].unique()) == x


@pytest.mark.parametrize(
    "marker_list",
    [
        "data/small_data_marker_list.csv",
        {
            "celltype_1": ["S100A8", "S100A9", "LYZ", "BLVRB"],
            "celltype_6": ["BIRC3", "TMEM116", "CD3D"],
            "celltype_7": ["CD74", "CD79B", "MS4A1"],
            "celltype_2": ["C5AR1"],
            "celltype_5": ["RNASE6"],
            "celltype_4": ["PPBP", "SPARC", "CDKN2D"],
            "celltype_8": ["NCR3"],
            "celltype_9": ["NAPA-AS1"],
        },
    ],
)
def test_marker_correlation_matrix_shared_comp(small_adata, marker_list):
    df = marker_correlation_matrix(small_adata, marker_list)
    if type(marker_list) == str:
        marker_list = pd.read_csv(marker_list)
        marker_list = {x: marker_list[x].to_list() for x in marker_list}
    first_markers = []
    for celltype, markers in marker_list.items():
        for marker in markers:
            if marker not in small_adata.var_names:
                continue
            assert np.testing.assert_almost_equal(df["mean"][marker], np.mean(small_adata[:, marker].X, axis=0)) is None
            assert df["celltype"][marker] == celltype
            assert np.testing.assert_almost_equal(df[marker][marker], 1) is None
            first_markers.append(marker)
    assert df.shape == (len(first_markers), small_adata.n_vars + 2)


@pytest.mark.parametrize("var_names", [None, ["PPBP", "SPARC", "S100A8"]])
def test_correlation_matrix_shared_comp_input_options(small_adata, var_names):
    cor_mat = correlation_matrix(small_adata, var_names)
    n = len(var_names) if var_names is not None else small_adata.n_vars
    assert cor_mat.shape == (n, n)
    # assert diagonal equals one (due to normalization)
    assert all(np.round(cor_mat.iloc[i, i], 0) == 1 for i in range(n))
    # assert symmetry
    assert all(round(cor_mat.iloc[i, j], 5) == round(cor_mat.iloc[j, i], 5) for i in range(n) for j in range(n))


def test_correlation_matrix_shared_comp_minimal():
    minimal_adata = sc.AnnData(X=np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 1]]))
    cor_mat = correlation_matrix(minimal_adata)
    assert cor_mat.shape == (3, 3)
    assert all(cor_mat.iloc[i, i] == 1 for i in range(3))
    assert all(cor_mat.iloc[i, j] == cor_mat.iloc[j, i] for i in range(3) for j in range(3))
    assert round(cor_mat.iloc[1, 0], 5) == -0.81818
    assert round(cor_mat.iloc[2, 0], 5) == -0.52223
    assert round(cor_mat.iloc[2, 1], 5) == 0.87039


# @pytest.mark.parametrize("ks", ["all", ['PPBP', 'SPARC', 'S100A8'], ["PPBP"]])
def test_knns_shared_comp(small_adata):
    df = knns(small_adata, "all")
    # to create the reference dataframe
    # df.to_csv("tests/evaluation/test_data/knn_df.csv")
    ref_df = pd.read_csv("tests/evaluation/test_data/knn_df.csv", index_col=0)
    assert df.equals(ref_df)


############################
# test metric computations #
############################

def test_clustering_nmis(small_adata, small_probeset):
    ns = [2, 3]
    annotations_ref = leiden_clusterings(small_adata, ns, start_res=1.0)
    annotations = leiden_clusterings(small_adata[:, small_probeset], ns, start_res=1.0)
    annotations_perm = leiden_clusterings(small_adata[:, small_probeset[::-1]], ns, start_res=1.0)
    nmis = clustering_nmis(annotations, annotations_ref, ns)
    nmis_sym = clustering_nmis(annotations_ref, annotations, ns)
    nmis_perm = clustering_nmis(annotations_perm, annotations_ref, ns)
    ref = pd.DataFrame({"nmi": [0.940299, 0.946526]}, index=[2, 3])
    # assert equals ref
    assert pd.testing.assert_frame_equal(nmis, ref, check_exact=False) is None
    # assert symmety
    assert pd.testing.assert_frame_equal(nmis, nmis_sym) is None
    # assert independence of permutation
    assert pd.testing.assert_frame_equal(nmis, nmis_perm) is None


def test_clustering_nmis_minimal():
    minimal_adata = sc.AnnData(X=np.array([[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0]]))
    annotations_ref = leiden_clusterings(minimal_adata, ns=[3, 3], start_res=1.0)
    annotations_1 = leiden_clusterings(minimal_adata[:, :2], ns=[2, 2], start_res=1.0)
    annotations_2 = leiden_clusterings(minimal_adata[:, 1:], ns=[2, 2], start_res=1.0)
    nmis_1 = clustering_nmis(annotations_1, annotations_ref, ns=[2, 2])
    nmis_2 = clustering_nmis(annotations_2, annotations_ref, ns=[2, 2])
    assert nmis_1["nmi"][2] == 1.0
    assert nmis_2["nmi"][2].round(6) == 0.478704


def test_mean_overlaps(small_adata, small_probeset):
    knn_df = knns(small_adata, genes=small_probeset)
    ref_knn_df = knns(small_adata, genes="all")
    mean_df = mean_overlaps(knn_df, ref_knn_df, ks=[10, 20])
    mean_ref = pd.DataFrame({"mean": [0.491728, 0.566959]}, index=[10, 20])
    assert pd.testing.assert_frame_equal(mean_df, mean_ref, check_exact=False) is None


def test_xgboost_forest_classification(small_adata, small_probeset):
    dfs = xgboost_forest_classification(small_adata, small_probeset, ct_key="celltype")
    # dfs[0].to_csv("tests/evaluation/test_data/xgboost_forest_classification_0.csv")
    # dfs[1].to_csv("tests/evaluation/test_data/xgboost_forest_classification_1.csv")
    df_0 = pd.read_csv("tests/evaluation/test_data/xgboost_forest_classification_0.csv", index_col=0)
    df_1 = pd.read_csv("tests/evaluation/test_data/xgboost_forest_classification_1.csv", index_col=0)
    assert pd.testing.assert_frame_equal(dfs[0], df_0) is None
    assert pd.testing.assert_frame_equal(dfs[1], df_1) is None


def test_max_marker_correlations(small_adata, marker_list, small_probeset):
    cor_matrix = marker_correlation_matrix(small_adata, marker_list)
    mmc = max_marker_correlations(small_probeset, cor_matrix)
    # mmc.to_csv("tests/evaluation/test_data/max_marker_correlation.csv")
    mmc_ref = pd.read_csv("tests/evaluation/test_data/max_marker_correlation.csv", index_col=0)
    mmc_ref.columns.name = "index"
    assert pd.testing.assert_frame_equal(mmc, mmc_ref) is None


########################
# test summary metrics #
########################

# AUC
# summary_metric_diagonal_confusion_mean
# linear_step
# summary_metric_diagonal_confusion_percentage
# summary_metric_correlation_mean
# summary_metric_correlation_percentage


# TODO discuss
"""
Where I see no need for testing:
- all the getter (get_metric_names, get_metric_parameter_names, get_metric_default_parameters
- compute_clustering_and_update
- summary functions: summary_nmi_AUCs, summary_knn_AUC,
- utils

Questions:
- test for the parameter types and return types? I think no -> type hinting
- are the "...equals_ref" tests usefull or too time consuming? eg. test_leiden_clustering_shared_comp_equals_ref
- smaller anndata than small_adata
- time for all tests currently > 3.5 min
"""
