.. _augmentation overview:

Augmentation
************

.. currentmodule:: bigfish.stack

Functions used to increase and diversify a dataset by duplicating and
transforming images. Available transformations are:

* Identity
* Transpose
* Inverse transpose
* Horizontal flip
* Vertical flip
* 90° rotation
* 180° rotation
* 270° rotation

Apply a random transformation on a 2D image:

* :func:`bigfish.stack.augment_2d`
* :func:`bigfish.stack.augment_2d_function`

Apply all the possible transformations on a 2D image:

* :func:`bigfish.stack.augment_8_times`
* :func:`bigfish.stack.augment_8_times_reversed`

.. autofunction:: augment_2d
.. autofunction:: augment_2d_function
.. autofunction:: augment_8_times
.. autofunction:: augment_8_times_reversed
