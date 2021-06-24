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

    spapros evaluation data/small_data_raw_counts.h5ad results/selections_genesets_1.csv data/small_data_marker_list.csv
