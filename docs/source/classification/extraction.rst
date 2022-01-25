.. _extraction overview:

Single-cell identification
**************************

.. currentmodule:: bigfish.multistack

Functions to exploit detection and segmentation results, by identifying
individual cells and their objects.

Identify and remove transcription sites
=======================================

Define transcription sites as clustered RNAs detected inside nucleus:

* :func:`bigfish.multistack.remove_transcription_site`

More generally, identify detected objects within a specific cellular region:

* :func:`bigfish.multistack.identify_objects_in_region`

.. autofunction:: remove_transcription_site
.. autofunction:: identify_objects_in_region

------------

Define and export single-cell results
=====================================

Extract detection and segmentation results and for every individual cell:

* :func:`bigfish.multistack.extract_cell`
* :func:`bigfish.multistack.extract_spots_from_frame`
* :func:`bigfish.multistack.summarize_extraction_results`

See an example of application `here <https://github.com/fish-quant/big-fish-
examples/blob/master/notebooks/6%20-%20Extract%20cell%20level%20results.ipynb>`_.

.. autofunction:: extract_cell
.. autofunction:: extract_spots_from_frame
.. autofunction:: summarize_extraction_results

------------

Manipulate surfaces, coordinates and boundaries
===============================================

Convert identified surfaces into coordinates, delimit boundaries and
manipulates coordinates:

* :func:`bigfish.multistack.center_mask_coord`
* :func:`bigfish.multistack.from_boundaries_to_surface`
* :func:`bigfish.multistack.from_surface_to_boundaries`
* :func:`bigfish.multistack.from_binary_to_coord`
* :func:`bigfish.multistack.complete_coord_boundaries`
* :func:`bigfish.multistack.from_coord_to_frame`
* :func:`bigfish.multistack.from_coord_to_surface`

.. autofunction:: center_mask_coord
.. autofunction:: from_boundaries_to_surface
.. autofunction:: from_surface_to_boundaries
.. autofunction:: from_binary_to_coord
.. autofunction:: complete_coord_boundaries
.. autofunction:: from_coord_to_frame
.. autofunction:: from_coord_to_surface