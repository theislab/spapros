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

   selector = sp.ProbesetSelector(adata)
   selector.select_probeset()

.. autosummary::
    :toctree: selection

    spapros.ProbesetSelector
    spapros.ProbesetSelector.select_probeset

Evaluation
------------

The central class for probe set evaluation and comparison:

.. code:: python

   evaluator = sp.ProbesetEvaluator()
   evaluator.evaluate_probeset(adata, selector.probeset)

.. autosummary::
    :toctree: evaluation

    spapros.ProbesetEvaluator
    spapros.ProbesetEvaluator.evaluate_probeset

Plotting
----------

Subsequently, visualize the results with the plotting module:

.. code:: python

   sp.plot.cool_fancy_plot()

.. autosummary::
    :toctree: plotting

    spapros.plotting.plot








