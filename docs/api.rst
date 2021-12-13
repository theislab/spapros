API
=====

Import the spapros API as follows:

.. code:: python

   import spapros as sp

You can then access the main classes like this:

Selection
-----------

The central class for probe set selection:

.. code:: python

   selector = sp.se.ProbesetSelector(adata)
   selector.select_probeset()
   selected_probeset = selector.probeset

.. autosummary::
    :toctree: selection

    spapros.se.ProbesetSelector
    spapros.se.ProbesetSelector.select_probesets
    spapros.se.select_reference_probesets

Evaluation
------------

The central class for probe set evaluation and comparison:

.. code:: python

   evaluator = sp.ev.ProbesetEvaluator()
   evaluator.evaluate_probeset(adata, selected_probeset)

.. autosummary::
    :toctree: evaluation

    spapros.ev.ProbesetEvaluator
    spapros.ev.ProbesetEvaluator.evaluate_probeset

Plotting
----------

The plotting module provides several method to visualize the results:

.. code:: python

   sp.pl.cool_fancy_plot()

.. autosummary::
    :toctree: plotting

    spapros.pl








