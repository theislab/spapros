******
Usage
******

The pipeline as well as the package is divided in three main parts: Selection, Evaluation and Plotting.
Use either the spapros CLI or the spapros API to invoke the pipline:

.. contents::
    :local:
    :backlinks: none

API
=====

Import the spapros API as follows:

.. code:: python

   import spapros as sp

You can then access the main classes like this:

Selection
------------

The central class for probe set selection:

.. code:: python

   selector = sp.se.ProbesetSelector(adata)
   selector.select_probeset()
   selected_probeset = selector.probeset.index[selector.probeset["selection"]].to_list()


Evaluation
------------

The central class for probe set evaluation and comparison:

.. code:: python

   evaluator = sp.ev.ProbesetEvaluator()
   evaluator.evaluate_probeset(adata, selected_probeset)

Plotting
----------

The plotting module provides several methods to visualize the results:

.. code:: python

   sp.pl.cool_fancy_plot()


CLI
====

Selection
-----------------


.. code-block:: bash

    spapros selection data/small_data_raw_counts.h5ad


Evaluation
--------------

.. code-block:: bash

    spapros evaluation data/small_data_raw_counts.h5ad results/selections_genesets_1.csv data/small_data_marker_list.csv genesets_1_0 genesets_1_1 genesets_1_13 --parameters data/parameters.yml







