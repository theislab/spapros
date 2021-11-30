******
Usage
******

Use either the spapors CLI or the spapros API to invoke the pipline:

CLI
====

.. click:: spapros.__main__:spapros_cli
   :prog: spapros
   :nested: full


Example
________

Probeset selection
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    spapros selection data/small_data_raw_counts.h5ad


Probeset evaluation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    spapros evaluation data/small_data_raw_counts.h5ad results/selections_genesets_1.csv data/small_data_marker_list.csv genesets_1_0 genesets_1_1 genesets_1_13 --parameters data/parameters.yml



API
=====

Import the spapros API as follows:

.. code:: python

   import spapros as sp

You can then access the main classes like this:

.. contents::
    :local:
    :backlinks: none

Selection
----------

The central class for probe set selection:

.. code:: python

   selector = sp.ProbesetSelector(adata)
   selector.select_probeset()

.. py:currentmodule:: spapros.selection

.. autosummary::
    :toctree: selection

    selection_procedure.ProbesetSelector
    selection_procedure.ProbesetSelector.select_probeset

Evaluation
-----------

The central class for probe set evaluation and comparison:

.. code:: python

   evaluator = sp.ProbesetEvaluator()
   evaluator.evaluate_probeset(adata, selector.probeset)


.. autosummary::
    :toctree: evaluation

    ProbesetEvaluator
    ProbesetEvaluator.evaluate_probeset

Plotting
----------

Subsequently, visualize the results with the plotting module:

.. code:: python

   sp.pl.cool_fancy_plot()


.. autosummary::
    :toctree: plotting

    pl


