.. _utils overview:

Utility functions
*****************

.. currentmodule:: bigfish.stack

Check input quality
===================

* :func:`bigfish.stack.check_array`
* :func:`bigfish.stack.check_df`
* :func:`bigfish.stack.check_parameter`
* :func:`bigfish.stack.check_range_value`

.. autofunction:: check_array
.. autofunction:: check_df
.. autofunction:: check_parameter
.. autofunction:: check_range_value

------------

Get constant values
===================

* :func:`bigfish.stack.get_margin_value`
* :func:`bigfish.stack.get_eps_float32`

.. autofunction:: get_margin_value
.. autofunction:: get_eps_float32

------------

Load and check stored data
==========================

* :func:`bigfish.stack.load_and_save_url`
* :func:`bigfish.stack.check_hash`
* :func:`bigfish.stack.compute_hash`
* :func:`bigfish.stack.check_input_data`

.. autofunction:: load_and_save_url
.. autofunction:: check_hash
.. autofunction:: compute_hash
.. autofunction:: check_input_data

------------

Compute moving average
======================

* :func:`bigfish.stack.moving_average`
* :func:`bigfish.stack.centered_moving_average`

.. autofunction:: moving_average
.. autofunction:: centered_moving_average

------------

Convert pixels and nanometers
=============================

.. currentmodule:: bigfish.detection

* :func:`bigfish.detection.convert_spot_coordinates`
* :func:`bigfish.detection.get_object_radius_pixel`
* :func:`bigfish.detection.get_object_radius_nm`

.. autofunction:: convert_spot_coordinates
.. autofunction:: get_object_radius_pixel
.. autofunction:: get_object_radius_nm

------------

Extract a spot image
====================

* :func:`bigfish.detection.get_spot_volume`
* :func:`bigfish.detection.get_spot_surface`

.. autofunction:: get_spot_volume
.. autofunction:: get_spot_surface

------------

Format and save plots
=====================

.. currentmodule:: bigfish.plot

* :func:`bigfish.plot.save_plot`
* :func:`bigfish.plot.get_minmax_values`
* :func:`bigfish.plot.create_colormap`

.. autofunction:: save_plot
.. autofunction:: get_minmax_values
.. autofunction:: create_colormap