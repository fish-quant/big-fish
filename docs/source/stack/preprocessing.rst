.. _preprocessing overview:

Preprocessing
*************

.. currentmodule:: bigfish.stack.preprocess

Functions used to normalize, cast or filter images.

Normalize images
================

Rescale or contrast pixel intensity:

* :func:`bigfish.stack.rescale<rescale>`

.. autofunction:: rescale

------------

Cast images
===========

Cast images to a specified dtype (with respect to the image range of values):

* :func:`bigfish.stack.cast_img_uint8<cast_img_uint8>`
* :func:`bigfish.stack.cast_img_uint16<cast_img_uint16>`
* :func:`bigfish.stack.cast_img_float32<cast_img_float32>`
* :func:`bigfish.stack.cast_img_float64<cast_img_float64>`

.. autofunction:: cast_img_uint8
.. autofunction:: cast_img_uint16
.. autofunction:: cast_img_float32
.. autofunction:: cast_img_float64

------------

Filter images
=============

.. currentmodule:: bigfish.stack.filter

Apply filtering transformation:

* :func:`bigfish.stack.mean_filter<mean_filter>`
* :func:`bigfish.stack.median_filter<median_filter>`
* :func:`bigfish.stack.gaussian_filter<gaussian_filter>`
* :func:`bigfish.stack.maximum_filter<maximum_filter>`
* :func:`bigfish.stack.minimum_filter<minimum_filter>`
* :func:`bigfish.stack.dilation_filter<dilation_filter>`
* :func:`bigfish.stack.erosion_filter<erosion_filter>`

Use Laplacian of Gaussian (LoG) filter to enhance peak signals and denoise the
rest of the image:

* :func:`bigfish.stack.log_filter<log_filter>`

Use blurring filters with large kernel to estimate and remove background
signal:

* :func:`bigfish.stack.remove_background_mean<remove_background_mean>`
* :func:`bigfish.stack.remove_background_gaussian<remove_background_gaussian>`

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
