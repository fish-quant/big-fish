.. _io overview:

I/O operations
**************

.. currentmodule:: bigfish.stack.io

Functions used to read data from various sources and store them in a numpy
array.

Read files
==========

Read image, video and numerical data as a numpy array:

* :func:`bigfish.stack.read_image<read_image>`
* :func:`bigfish.stack.read_dv<read_dv>`
* :func:`bigfish.stack.read_array<read_array>`

Read a zipped archive of files as a dictionary-like object:

* :func:`bigfish.stack.read_uncompressed<read_uncompressed>`
* :func:`bigfish.stack.read_cell_extracted<read_cell_extracted>`

Read CSV file:

* :func:`bigfish.stack.read_array_from_csv<read_array_from_csv>`
* :func:`bigfish.stack.read_dataframe_from_csv<read_dataframe_from_csv>`

.. autofunction:: read_image
.. autofunction:: read_dv
.. autofunction:: read_array
.. autofunction:: read_uncompressed
.. autofunction:: read_cell_extracted
.. autofunction:: read_array_from_csv
.. autofunction:: read_dataframe_from_csv

------------

Write files
===========

Save numpy array:

* :func:`bigfish.stack.save_image<save_image>`
* :func:`bigfish.stack.save_array<save_array>`

Save cell-level results in a zipped archive of files:

* :func:`bigfish.stack.save_cell_extracted<save_cell_extracted>`

Save tabular data in a CSV file:

* :func:`bigfish.stack.save_data_to_csv<save_data_to_csv>`

.. autofunction:: save_image
.. autofunction:: save_array
.. autofunction:: save_cell_extracted
.. autofunction:: save_data_to_csv
