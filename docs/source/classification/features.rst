.. _features overview:

Features engineering
********************

.. currentmodule:: bigfish.classification

Prepare input coordinates
=========================

Format input coordinates and compute intermediary results to prepare features
computation:

.. autofunction:: prepare_extracted_data

------------

Compute features
================

Functions to compute features about cell morphology and RNAs localization.
There are two main functions to compute spatial and morphological features are:

* :func:`bigfish.classification.compute_features`
* :func:`bigfish.classification.get_features_name`

Group of features can be computed separately:

* :func:`bigfish.classification.features_distance`
* :func:`bigfish.classification.features_in_out_nucleus`
* :func:`bigfish.classification.features_protrusion`
* :func:`bigfish.classification.features_dispersion`
* :func:`bigfish.classification.features_topography`
* :func:`bigfish.classification.features_foci`
* :func:`bigfish.classification.features_area`
* :func:`bigfish.classification.features_centrosome`

See an example of application `here <https://github.com/fish-quant/big-fish-
examples/blob/master/notebooks/7%20-%20Analyze%20coordinates.ipynb>`_.

.. autofunction:: compute_features
.. autofunction:: get_features_name
.. autofunction:: features_distance
.. autofunction:: features_in_out_nucleus
.. autofunction:: features_protrusion
.. autofunction:: features_dispersion
.. autofunction:: features_topography
.. autofunction:: features_foci
.. autofunction:: features_area
.. autofunction:: features_centrosome