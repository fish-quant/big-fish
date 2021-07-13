.. _projections overview:

2D Projections
**************

.. currentmodule:: bigfish.stack.projection

Functions used to project 3D images in 2D.

Standard projections
====================

Build a 2D projection by computing the maximum, mean or median values:

* :func:`bigfish.stack.maximum_projection<maximum_projection>`
* :func:`bigfish.stack.mean_projection<mean_projection>`
* :func:`bigfish.stack.median_projection<median_projection>`

.. autofunction:: maximum_projection
.. autofunction:: mean_projection
.. autofunction:: median_projection

------------

In-focus projection
===================

Compute a pixel-wise focus score:

* :func:`bigfish.stack.compute_focus<bigfish.stack.compute_focus>`

Remove the out-of-focus z-slices of a 3D image:

* :func:`bigfish.stack.in_focus_selection<in_focus_selection>`
* :func:`bigfish.stack.get_in_focus_indices<get_in_focus_indices>`

Build a 2D projection by removing the out-of-focus z-slices/pixels:

* :func:`bigfish.stack.focus_projection<focus_projection>`

.. autofunction:: bigfish.stack.compute_focus
.. autofunction:: in_focus_selection
.. autofunction:: get_in_focus_indices
.. autofunction:: focus_projection
