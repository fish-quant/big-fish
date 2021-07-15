.. _spots overview:

Automated spot detection
************************

.. currentmodule:: bigfish.detection

Functions used to detect spots in a 2D or 3D image. Detection is performed in
three steps:

#. Image is denoised and spots are enhanced by using a Laplacian of Gaussian
   (LoG) filter.
#. Peaks are detected in the filtered image with a local maximum detection
   algorithm.
#. An intensity threshold is applied to discriminate actual spots from noisy
   background.

Detect spots
============

The main function for spot detection is:

* :func:`bigfish.detection.detect_spots`

It is also possible to perform the main steps of the spot detection separately:

* :func:`bigfish.detection.local_maximum_detection`
* :func:`bigfish.detection.spots_thresholding`

See an example of application `here <https://github.com/fish-quant/big-fish-
examples/blob/master/notebooks/5%20-%20Detect%20spots.ipynb/>`_.

.. autofunction:: detect_spots
.. autofunction:: local_maximum_detection
.. autofunction:: spots_thresholding

------------

Find a threshold (automatically)
================================

The need to set an appropriate threshold for each image is a real bottleneck
that limits the possibility to scale a spot detection. Our method includes a
heuristic function to to automatically set this threshold:

* :func:`bigfish.detection.automated_threshold_setting`
* :func:`bigfish.detection.get_elbow_values`

.. autofunction:: automated_threshold_setting
.. autofunction:: get_elbow_values

------------

Compute signal-to-noise ratio
=============================

Compute a signal-to-noise ratio (SNR) for the image:

* :func:`bigfish.detection.compute_snr_spots`

.. autofunction:: compute_snr_spots
