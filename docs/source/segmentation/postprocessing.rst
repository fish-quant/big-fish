.. _segmentation_postprocessing overview:

Postprocessing
**************

.. currentmodule:: bigfish.segmentation

Functions used to clean and refine segmentation results.

Label and clean instances
=========================

Label disconnected instances:

* :func:`bigfish.segmentation.label_instances`
* :func:`bigfish.segmentation.merge_labels`

Clean segmentation results:

* :func:`bigfish.segmentation.clean_segmentation`
* :func:`bigfish.segmentation.remove_disjoint`

Match nuclei and cells:

* :func:`bigfish.segmentation.match_nuc_cell`

See an example of application `here <https://github.com/fish-quant/big-fish-
examples/blob/master/notebooks/4%20-%20Segment%20nuclei%20and%20cells.ipynb>`_.

.. autofunction:: label_instances
.. autofunction:: merge_labels
.. autofunction:: clean_segmentation
.. autofunction:: remove_disjoint
.. autofunction:: match_nuc_cell

------------

Compute instance statistics
===========================

Compute statistics for each segmented instance:

* :func:`bigfish.segmentation.compute_mean_diameter`
* :func:`bigfish.segmentation.compute_mean_convexity_ratio`
* :func:`bigfish.segmentation.compute_surface_ratio`
* :func:`bigfish.segmentation.count_instances`

.. autofunction:: compute_mean_diameter
.. autofunction:: compute_mean_convexity_ratio
.. autofunction:: compute_surface_ratio
.. autofunction:: count_instances