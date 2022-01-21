# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.segmentation subpackage.
"""

import bigfish.stack as stack

import numpy as np

from skimage.measure import regionprops


# TODO make functions compatible with different type of integers

# ### Thresholding method ###

def thresholding(image, threshold):
    """Segment a 2-d image to discriminate object from background applying a
    threshold.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 2-d image to segment with shape (y, x).
    threshold : int or float
        Pixel intensity threshold used to discriminate foreground from
        background.

    Returns
    -------
    image_segmented : np.ndarray, bool
        Binary 2-d image with shape (y, x).

    """
    # check parameters
    stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16])
    stack.check_parameter(threshold=(float, int))

    # discriminate nuclei from background, applying a threshold.
    image_segmented = image >= threshold

    return image_segmented


# ### Instances measures ###

def compute_mean_diameter(image_label):
    """Compute the averaged size of the segmented instances.

    For each instance, we compute the diameter of an object with an equivalent
    surface. Then, we average the diameters.

    Parameters
    ----------
    image_label : np.ndarray, np.int or np.uint
        Labelled image with shape (y, x).

    Returns
    -------
    mean_diameter : float
        Averaged size of the segmented instances.

    """
    # check parameters
    stack.check_array(
        image_label,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64])

    # compute properties of the segmented instances
    props = regionprops(image_label)

    # get equivalent diameter and average it
    diameter = 0
    n = len(props)
    for prop in props:
        diameter += prop.equivalent_diameter
    if n > 0:
        mean_diameter = diameter / n
    else:
        mean_diameter = np.nan

    return mean_diameter


def compute_mean_convexity_ratio(image_label):
    """Compute the averaged convexity ratio of the segmented instances.

    For each instance, we compute the ratio between its area and the area of
    its convex hull. Then, we average the diameters.

    Parameters
    ----------
    image_label : np.ndarray, np.int or np.uint
        Labelled image with shape (y, x).

    Returns
    -------
    mean_convexity_ratio : float
        Averaged convexity ratio of the segmented instances.

    """
    # check parameters
    stack.check_array(
        image_label,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64])

    # compute properties of the segmented instances
    props = regionprops(image_label)

    # get convexity ratio and average it
    n = len(props)
    convexity_ratio = 0
    for prop in props:
        convexity_ratio += prop.area / prop.convex_area
    if n > 0:
        mean_convexity_ratio = convexity_ratio / n
    else:
        mean_convexity_ratio = np.nan

    return mean_convexity_ratio


def compute_surface_ratio(image_label):
    """Compute the averaged surface ratio of the segmented instances.

    We compute the proportion of surface occupied by instances.

    Parameters
    ----------
    image_label : np.ndarray, np.int or np.uint
        Labelled image with shape (y, x).

    Returns
    -------
    surface_ratio : float
        Surface ratio of the segmented instances.

    """
    # check parameters
    stack.check_array(
        image_label,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64])

    # compute surface ratio
    surface_instances = image_label > 0
    area_instances = surface_instances.sum()
    surface_ratio = area_instances / image_label.size

    return surface_ratio


def count_instances(image_label):
    """Count the number of instances annotated in the image.

    Parameters
    ----------
    image_label : np.ndarray, np.int or np.uint
        Labelled image with shape (y, x).

    Returns
    -------
    nb_instances : int
        Number of instances in the image.

    """
    # check parameters
    stack.check_array(
        image_label,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64])

    indices = set(image_label.ravel())
    if 0 in indices:
        nb_instances = len(indices) - 1
    else:
        nb_instances = len(indices)

    return nb_instances
