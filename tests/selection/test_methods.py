import random

import pandas as pd
import pytest

from spapros import se


@pytest.mark.parametrize("genes_key", ["highly_variable", None])
@pytest.mark.parametrize("n", [10, 50])
@pytest.mark.parametrize("preselected_genes", [["IL7R", "CD14", "NKG7"], ["IL7R", "CD14"]])
@pytest.mark.parametrize("prior_genes", [["LST1", "CST3"], ["CD14", "NKG7"]])
@pytest.mark.parametrize("n_pca_genes", [100])
def test_selection_params(
    adata_pbmc3k,
    genes_key,
    n,
    preselected_genes,
    prior_genes,
    n_pca_genes,
):
    random.seed(0)
    selector = se.ProbesetSelector(
        adata_pbmc3k[random.sample(range(adata_pbmc3k.n_obs), 100), :],
        genes_key=genes_key,
        n=n,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        preselected_genes=preselected_genes,
        prior_genes=prior_genes,
        n_pca_genes=n_pca_genes,
        save_dir=None,
    )
    selector.select_probeset()


@pytest.mark.parametrize("celltypes", (["CD4 T cells"], ["Megakaryocytes"]))
@pytest.mark.parametrize(
    "marker_list",
    [
        "evaluation/selection/test_data/pbmc3k_marker_list.csv",
        {
            "CD4 T cells": ["IL7R"],
            "CD14+ Monocytes": ["CD14", "LYZ"],
            "B cells": ["MS4A1"],
            "CD8 T cells": ["CD8A"],
            "NK cells": ["GNLY", "NKG7"],
            "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],
            "3	Dendritic Cells": ["FCER1A", "CST3"],
            "Megakaryocytes": ["NAPA-AS1", "PPBP"],
        },
    ],
)
def test_selection_celltypes(adata_pbmc3k, celltypes, marker_list):
    selector = se.ProbesetSelector(
        adata_pbmc3k,  # [random.sample(range(adata_pbmc3k.n_obs), 100), :],
        genes_key="highly_variable",
        n=50,
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        save_dir=None,
        celltypes=celltypes,
    )
    selector.select_probeset()


@pytest.mark.parametrize("save_dir", [None, "tmp_path"])
@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_selection_verbosity(
    adata_pbmc3k,
    verbosity,
    save_dir,
    request,
):
    random.seed(0)
    selector = se.ProbesetSelector(
        adata_pbmc3k[random.sample(range(adata_pbmc3k.n_obs), 100), :],
        celltype_key="celltype",
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
        verbosity=verbosity,
        save_dir=None if not save_dir else request.getfixturevalue(save_dir),
    )
    selector.select_probeset()


def test_selection_stable(adata_pbmc3k):
    random.seed(0)
    idx = random.sample(range(adata_pbmc3k.n_obs), 100)
    selector_a = se.ProbesetSelector(
        adata_pbmc3k[idx, :],
        verbosity=2,
        n=50,
        celltype_key="celltype",
        seed=0,
        save_dir=None,
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
    )
    selector_a.select_probeset()
    selection1 = selector_a.probeset.copy()
    selector_a.select_probeset()
    selection2 = selector_a.probeset.copy()
    selector_b = se.ProbesetSelector(
        adata_pbmc3k[idx, :],
        verbosity=2,
        n=50,
        celltype_key="celltype",
        seed=0,
        save_dir=None,
        forest_hparams={"n_trees": 10, "subsample": 200, "test_subsample": 400},
    )
    selector_b.select_probeset()
    selection3 = selector_b.probeset.copy()
    assert pd.testing.assert_frame_equal(selection1, selection2) is None
    assert pd.testing.assert_frame_equal(selection2, selection3) is None


@pytest.mark.parametrize(
    "n, " "genes_key, " "seeds, " "verbosity, " "save_dir, " "methods",
    [
        (
            10,
            "highly_variable",
            [0, 202],
            0,
            None,
            {
                "hvg_selection": {"flavor": "cell_ranger"},
                "random_selection": {},
                "pca_selection": {
                    "variance_scaled": False,
                    "absolute": True,
                    "n_pcs": 20,
                    "penalty_keys": [],
                    "corr_penalty": None,
                },
                "DE_selection": {"per_group": "True"},
            },
        ),
        (
            50,
            "highly_variable",
            [],
            1,
            None,
            {
                "hvg_selection": {"flavor": "seurat"},
                "pca_selection": {
                    "variance_scaled": True,
                    "absolute": False,
                    "n_pcs": 10,
                },
                "DE_selection": {"per_group": "False"},
            },
        ),
        (100, "highly_variable", [], 2, "tmp_path", ["PCA", "DE", "HVG", "random"]),
    ],
)
def test_select_reference_probesets(adata_pbmc3k, n, genes_key, seeds, verbosity, save_dir, request, methods):
    se.select_reference_probesets(
        adata_pbmc3k,
        n=n,
        genes_key=genes_key,
        seeds=seeds,
        verbosity=verbosity,
        save_dir=None if not save_dir else request.getfixturevalue(save_dir),
        methods=methods,
    )
