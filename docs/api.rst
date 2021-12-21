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

    pl.correlation_matrix
    pl.confusion_heatmap
    pl.summary_table
    pl.masked_dotplot
    pl.MaskedDotPlot










