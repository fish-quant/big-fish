.. big-fish documentation master file, created by
   sphinx-quickstart on Thu Nov 19 22:45:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Big-FISH
********


Overview
========

.. toctree::
   :caption: Overview
   :hidden:

   overview/introduction
   overview/installation
   overview/pipeline
   overview/examples

------------

.. toctree::
   :caption: Preprocessing

   stack/io
   stack/preprocessing
   stack/augmentation

------------

.. toctree::
   :caption: Spot detection

   detection/spots
   detection/dense
   detection/subpixel
   detection/cluster

------------

.. toctree::
   :caption: Segmentation

   segmentation/thresholding
   segmentation/input
   segmentation/nucleus
   segmentation/cell
   segmentation/postprocessing

------------

.. toctree::
   :caption: Analysis

   classification/extraction
   classification/input
   classification/features


------------

.. toctree::
   :caption: Plot

   plot/plot

------------

.. toctree::
   :caption: Utils
   :hidden:

   utils/utils

* :ref:`Utility functions<utils overview>`: Sanity checks, get constant values,
  compute hash, etc...


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
