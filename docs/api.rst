API
=====

Import the spapros API as follows:

.. code:: python

   import spapros as sp

Selection
-----------

.. module:: spapros.se

.. currentmodule:: spapros

.. rubric:: Main Class

.. autosummary::
  :toctree: se

    se.ProbesetSelector

.. rubric:: Other Functions

.. autosummary::
  :toctree: se

    se.select_reference_probesets
    se.select_pca_genes
    se.select_DE_genes


Evaluation
------------

.. module:: spapros.ev

.. currentmodule:: spapros

.. rubric:: Main Class

.. autosummary::
  :toctree: ev

    ev.ProbesetEvaluator

.. rubric:: Other Functions

.. autosummary::
  :toctree: ev

    ev.get_metric_default_parameters
    ev.forest_classifications
    ev.single_forest_classifications


Utility functions
-------------------

.. module:: spapros.ut

.. currentmodule:: spapros

.. rubric:: Expression Constraints

.. autosummary::
    :toctree: ut

    ut.get_expression_quantile
    ut.transfered_expression_thresholds
    ut.plateau_penalty_kernel

.. rubric:: Spatial Data Analysis

.. autosummary::
    :toctree: ut


Plotting
----------

The sp.pl functions are used directly only in rare cases, instead always try to use the wrapper methods from the
ProbesetEvaluator and ProbesetSelector (see 2nd table below).

.. module:: spapros.pl

.. currentmodule:: spapros

.. autosummary::
    :toctree: pl

    pl.masked_dotplot
    pl.clf_genes_umaps
    pl.selection_histogram
    pl.gene_overlap
    pl.correlation_matrix
    pl.summary_table
    pl.cluster_similarity
    pl.knn_overlap
    pl.confusion_matrix
    pl.marker_correlation


.. tabularcolumns:: |C|C|C|

.. list-table::
   :header-rows: 1
   :widths: 1 1 1
   :align: left

   * - .. centered:: `pl` module
     - .. centered:: :class:`.ProbesetSelector`
     - .. centered:: :class:`.ProbesetEvaluator`

   * - :func:`pl.masked_dotplot`
     - see pl.masked_dotplot for now
     -
   * - :func:`pl.clf_genes_umaps`
     - :meth:`.ProbesetSelector.plot_clf_genes`
     -
   * - :func:`pl.selection_histogram`
     - :meth:`.ProbesetSelector.plot_histogram`
     -
   * - :func:`pl.gene_overlap`
     - :meth:`.ProbesetSelector.plot_gene_overlap`
     - potentially added soon
   * - :func:`pl.correlation_matrix`
     - :meth:`.ProbesetSelector.plot_coexpression`
     - :meth:`.ProbesetEvaluator.plot_coexpression`
   * - :func:`pl.summary_table`
     -
     - :meth:`.ProbesetEvaluator.plot_summary`
   * - :func:`pl.cluster_similarity`
     -
     - :meth:`.ProbesetEvaluator.plot_cluster_similarity`
   * - :func:`pl.knn_overlap`
     -
     - :meth:`.ProbesetEvaluator.plot_knn_overlap`
   * - :func:`pl.confusion_matrix`
     -
     - :meth:`.ProbesetEvaluator.plot_confusion_matrix`
   * - :func:`pl.marker_correlation`
     -
     - :meth:`.ProbesetEvaluator.plot_marker_corr`

