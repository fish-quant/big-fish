.. _dense overview:

Dense regions decomposition
***************************

Functions to detect dense and bright regions (with potential clustered spots),
then use gaussian simulations to correct a misdetection in these regions.

Decompose and simulate dense regions
====================================

.. currentmodule:: bigfish.detection.dense_decomposition

The main function to decompose dense regions is:

* :func:`bigfish.detection.decompose_dense<decompose_dense>`

It is also possible to perform the main steps of this decomposition separately:

* :func:`bigfish.detection.get_dense_region<get_dense_region>`
* :func:`bigfish.detection.simulate_gaussian_mixture<simulate_gaussian_mixture>`

See an example of application `here <https://github.com/fish-quant/big-fish-
examples/blob/master/notebooks/5%20-%20Detect%20spots.ipynb/>`_.

.. autofunction:: decompose_dense
.. autofunction:: get_dense_region
.. autofunction:: simulate_gaussian_mixture

------------

Modelize a reference spot
=========================

.. currentmodule:: bigfish.detection.spot_modeling

To simulate additional spots in the dense regions it is necessary to:

#. Build a reference spot.

    * :func:`bigfish.detection.build_reference_spot<build_reference_spot>`

#. Modelize this reference spot by fitting gaussian parameters.

    * :func:`bigfish.detection.modelize_spot<modelize_spot>`

#. Simulate gaussian signal.

    * :func:`bigfish.detection.precompute_erf<precompute_erf>`
    * :func:`bigfish.detection.initialize_grid<initialize_grid>`
    * :func:`bigfish.detection.gaussian_2d<gaussian_2d>`
    * :func:`bigfish.detection.gaussian_3d<gaussian_3d>`

.. autofunction:: build_reference_spot
.. autofunction:: modelize_spot
.. autofunction:: precompute_erf
.. autofunction:: initialize_grid
.. autofunction:: gaussian_2d
.. autofunction:: gaussian_3d
