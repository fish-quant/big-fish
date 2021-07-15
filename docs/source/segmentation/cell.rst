.. _cell overview:

Cell segmentation
*****************

.. currentmodule:: bigfish.segmentation

Functions used to segment cells.

Apply watershed algorithm
=========================

Main function to segment cells with a watershed algorithm:

* :func:`bigfish.segmentation.cell_watershed`

Our segmentation using watershed algorithm can also be perform with two
separated steps:

* :func:`bigfish.segmentation.get_watershed_relief`
* :func:`bigfish.segmentation.apply_watershed`

.. autofunction:: cell_watershed
.. autofunction:: get_watershed_relief
.. autofunction:: apply_watershed

------------

Apply a Unet-based model (distance map)
=======================================

Load a pretrained model:

* :func:`bigfish.segmentation.unet_distance_edge_double`

Segment cells:

* :func:`bigfish.segmentation.apply_unet_distance_double`
* :func:`bigfish.segmentation.from_distance_to_instances`

See an example of application `here <https://github.com/fish-quant/big-fish-ex
amples/blob/master/notebooks/4%20-%20Segment%20nuclei%20and%20cells.ipynb/>`_.

.. autofunction:: unet_distance_edge_double
.. autofunction:: apply_unet_distance_double
.. autofunction:: from_distance_to_instances
