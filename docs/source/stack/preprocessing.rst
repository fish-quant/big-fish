.. _preprocessing overview:

Image preparation
*****************

.. currentmodule:: bigfish.stack

Functions used to normalize, cast, project or filter images.

Normalize images
================

Rescale or contrast pixel intensity:

* :func:`bigfish.stack.rescale`

.. autofunction:: rescale

------------

Cast images
===========

Cast images to a specified dtype (with respect to the image range of values):

* :func:`bigfish.stack.cast_img_uint8`
* :func:`bigfish.stack.cast_img_uint16`
* :func:`bigfish.stack.cast_img_float32`
* :func:`bigfish.stack.cast_img_float64`

.. autofunction:: cast_img_uint8
.. autofunction:: cast_img_uint16
.. autofunction:: cast_img_float32
.. autofunction:: cast_img_float64

------------

Filter images
=============

Apply filtering transformations:

* :func:`bigfish.stack.mean_filter`
* :func:`bigfish.stack.median_filter`
* :func:`bigfish.stack.gaussian_filter`
* :func:`bigfish.stack.maximum_filter`
* :func:`bigfish.stack.minimum_filter`
* :func:`bigfish.stack.dilation_filter`
* :func:`bigfish.stack.erosion_filter`

Use Laplacian of Gaussian (LoG) filter to enhance peak signals and denoise the
rest of the image:

* :func:`bigfish.stack.log_filter`

Use blurring filters with large kernel to estimate and remove background
signal:

* :func:`bigfish.stack.remove_background_mean`
* :func:`bigfish.stack.remove_background_gaussian`

.. autofunction:: mean_filter
.. autofunction:: median_filter
.. autofunction:: gaussian_filter
.. autofunction:: maximum_filter
.. autofunction:: minimum_filter
.. autofunction:: dilation_filter
.. autofunction:: erosion_filter
.. autofunction:: log_filter
.. autofunction:: remove_background_mean
.. autofunction:: remove_background_gaussian

------------

Project images in 2D
====================

Build a 2D projection by computing the maximum, mean or median values:

* :func:`bigfish.stack.maximum_projection`
* :func:`bigfish.stack.mean_projection`
* :func:`bigfish.stack.median_projection`

.. autofunction:: maximum_projection
.. autofunction:: mean_projection
.. autofunction:: median_projection

------------

Clean out-of-focus pixels
=========================

Compute a pixel-wise focus score:

* :func:`bigfish.stack.compute_focus`

Remove the out-of-focus z-slices of a 3D image:

* :func:`bigfish.stack.in_focus_selection`
* :func:`bigfish.stack.get_in_focus_indices`

Build a 2D projection by removing the out-of-focus z-slices/pixels:

* :func:`bigfish.stack.focus_projection`

.. autofunction:: bigfish.stack.compute_focus
.. autofunction:: in_focus_selection
.. autofunction:: get_in_focus_indices
.. autofunction:: focus_projection
