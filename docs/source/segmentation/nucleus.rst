.. _nucleus overview:

Nucleus segmentation
********************

.. currentmodule:: bigfish.segmentation

Functions used to segment nuclei.

Apply thresholding
==================

Thresholding is the most standard and direct binary segmentation method:

.. autofunction:: thresholding

Apply a Unet-based model (3-classes)
====================================

Two additional dependencies can be requested for these functions: tensorflow
(>= 2.3.0) and tensorflow-addons (>= 0.12.1).

Load a pretrained model:

* :func:`bigfish.segmentation.unet_3_classes_nuc`

Segment nuclei:

* :func:`bigfish.segmentation.apply_unet_3_classes`
* :func:`bigfish.segmentation.from_3_classes_to_instances`

See an example of application `here <https://github.com/fish-quant/big-fish-
examples/blob/master/notebooks/4%20-%20Segment%20nuclei%20and%20cells.ipynb>`_.

.. autofunction:: unet_3_classes_nuc
.. autofunction:: apply_unet_3_classes
.. autofunction:: from_3_classes_to_instances

------------

Remove segmented nuclei
=======================

Remove segmented nucleus instances to perform a new segmentation on the
residual image:

.. autofunction:: remove_segmented_nuc
