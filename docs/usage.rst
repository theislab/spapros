******
Usage
******

Spapros provides functions for gene set selection, evaluation and several visualization utilities. For any selection and
evaluation an AnnData_ object (``adata``) of a scRNA-seq data set is needed. To combine gene set selection with probe design
(pre-filtering of genes and final probe sequence design) we use the spapros package in combination with our probe design
package oligo-designer-toolsuite_.


Import package
--------------

Import the spapros API as follows:

.. code:: python

   import spapros as sp

You can then access the main classes like this:

Selection
---------

The central class for probe set selection:

.. code:: python

   selector = sp.se.ProbesetSelector(adata)
   selector.select_probeset()
   selected_probeset = selector.probeset.index[selector.probeset["selection"]].to_list()


Evaluation
----------

The central class for probe set evaluation and comparison:

.. code:: python

   evaluator = sp.ev.ProbesetEvaluator()
   evaluator.evaluate_probeset(adata, selected_probeset)


Probe design
------------

Check out our tutorial :doc:`./_tutorials/spapros_tutorial_end_to_end_selection` for a detailed explanation on how to
combine gene panel selection and probe design.


More information
----------------

- The quickest way to learn how to use spapros is to follow our :doc:`tutorials`.
- For an overview of available functions and detailed information on each function check out the :doc:`api`.
- If you run into issues or have further questions check out the
  `issue section on github <https://github.com/theislab/spapros/issues>`__ and please raise your question there if no
  one else has yet.






.. _AnnData: https://anndata.readthedocs.io/en/latest/
.. _oligo-designer-toolsuite: https://github.com/HelmholtzAI-Consultants-Munich/oligo-designer-toolsuite

