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
