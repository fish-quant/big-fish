.. _segmentation_utils overview:

Utils
*****

.. currentmodule:: bigfish.segmentation

Utility functions to compute statistics about segmented instances and prepare
input images.

Segment by thresholding
=======================

Thresholding is the most standard and direct segmentation method:

* :func:`bigfish.segmentation.thresholding`

.. autofunction:: thresholding

------------

Prepare input images
====================

Resize, pad and normalize input image before feeding a deep learning model:

* :func:`bigfish.segmentation.resize_image`
* :func:`bigfish.segmentation.get_marge_padding`
* :func:`bigfish.segmentation.compute_image_standardization`

.. autofunction:: resize_image
.. autofunction:: get_marge_padding
.. autofunction:: compute_image_standardization

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
