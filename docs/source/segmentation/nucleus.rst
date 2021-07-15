.. _nucleus overview:

Nucleus segmentation
********************

.. currentmodule:: bigfish.segmentation

Functions used to segment nuclei.

Apply a Unet-based model (3-classes)
====================================

Load a pretrained model:

* :func:`bigfish.segmentation.unet_3_classes_nuc`

Segment nuclei:

* :func:`bigfish.segmentation.apply_unet_3_classes`
* :func:`bigfish.segmentation.from_3_classes_to_instances`

See an example of application `here <https://github.com/fish-quant/big-fish-ex
amples/blob/master/notebooks/4%20-%20Segment%20nuclei%20and%20cells.ipynb/>`_.

.. autofunction:: unet_3_classes_nuc
.. autofunction:: apply_unet_3_classes
.. autofunction:: from_3_classes_to_instances

------------

Remove segmented nuclei
=======================

Remove segmented nucleus instances to perform a new segmentation on the
residual image:

.. autofunction:: remove_segmented_nuc
