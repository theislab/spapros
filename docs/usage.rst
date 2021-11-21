Usage
=====

.. click:: spapros.__main__:spapros_cli
   :prog: spapros
   :nested: full


Example
-------

Probeset selection
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    spapros selection data/small_data_raw_counts.h5ad


Probeset evaluation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    spapros evaluation data/small_data_raw_counts.h5ad results/selections_genesets_1.csv data/small_data_marker_list.csv genesets_1_0 genesets_1_1 genesets_1_13 --parameters data/parameters.yml


API
---

Import the ehrapy API as follows:

.. code:: python

   import spapros as sp

You can then access the respective modules like:

.. code:: python

   sp.pl.cool_fancy_plot()

.. contents::
    :local:
    :backlinks: none


Selection
~~~~~~~~~

.. currentmodule:: spapros.selection

.. autosummary::
    :toctree: selection

    selection.run_selection


Evaluation
~~~~~~~~~~

.. currentmodule:: spapros.evaluation

The central class for probe set evaluation, comparison, plotting:

.. autosummary::
    :toctree: evaluation/ProbesetEvaluator
    :template: _templates/autosummary/class.rst

    spapros.evaluation.evaluation.ProbesetEvaluator

Other methods in evaluation.py (will be moved or removed)

.. autosummary::
    :toctree: evaluation/evaluation_methods

    evaluation.combine_tree_results
    evaluation.eval_ct_tree_helper
    evaluation.forest_classifications
    evaluation.forest_rank_table
    evaluation.get_celltypes_with_too_small_test_sets
    evaluation.get_outlier_reference_celltypes
    evaluation.get_reference_masks
    evaluation.load_forest
    evaluation.outlier_mask
    evaluation.plot_gene_expressions
    evaluation.plot_nmis
    evaluation.pool_eval_ct_tree_helper
    evaluation.pool_train_ct_tree_helper
    evaluation.save_forest
    evaluation.single_forest_classifications
    evaluation.split_train_test_sets
    evaluation.summarize_specs
    evaluation.train_ct_tree_helper
    evaluation.uniform_samples

The evaluation metrics:

.. autosummary::
    :toctree: evaluation/metrics

    spapros.evaluation.metrics


Plotting
~~~~~~~~

.. currentmodule:: spapros.plotting

.. autosummary:: spapros.plotting.plot
    :toctree: plotting


