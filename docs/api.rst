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
    se.select_pca


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

Plotting
----------

.. module:: spapros.pl

.. currentmodule:: spapros

.. autosummary::
    :toctree: pl

    pl.confusion_matrix
    pl.correlation_matrix
    pl.clustering_lineplot
    pl.summary_table
    pl.selection_histogram
    pl.gene_overlap
    pl.classification_rule_umaps
    pl.masked_dotplot
