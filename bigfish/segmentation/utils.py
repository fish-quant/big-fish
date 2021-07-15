# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.segmentation subpackage.
"""

import bigfish.stack as stack

import numpy as np

from skimage.measure import regionprops
from skimage.transform import resize


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
    stack.check_array(image_label,
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
    stack.check_array(image_label,
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
    stack.check_array(image_label,
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
    stack.check_array(image_label,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])

    indices = set(image_label.ravel())
    if 0 in indices:
        nb_instances = len(indices) - 1
    else:
        nb_instances = len(indices)

    return nb_instances


# ### Format and crop images ###

def resize_image(image, output_shape, method="bilinear"):
    """Resize an image with bilinear interpolation or nearest neighbor method.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    output_shape : Tuple[int]
        Shape of the resized image.
    method : str
        Interpolation method to use.

    Returns
    -------
    image_resized : np.ndarray
        Resized image.

    """
    # check parameters
    stack.check_parameter(output_shape=tuple, method=str)
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32])

    # resize image
    if method == "bilinear":
        image_resized = resize(image, output_shape,
                               mode="reflect", preserve_range=True,
                               order=1, anti_aliasing=True)
    elif method == "nearest":
        image_resized = resize(image, output_shape,
                               mode="reflect", preserve_range=True,
                               order=0, anti_aliasing=False)
    else:
        raise ValueError("Method {0} is not available. Choose between "
                         "'bilinear' or 'nearest' instead.".format(method))

    # cast output dtype
    image_resized = image_resized.astype(image.dtype)

    return image_resized


def get_marge_padding(height, width, x):
    """Pad image to make its shape a multiple of `x`.

    Parameters
    ----------
    height : int
        Original height of the image.
    width : int
        Original width of the image.
    x : int
        Padded image have a `height` and `width` multiple of `x`.

    Returns
    -------
    marge_padding : List[List]
        List of lists with the format
        [[`marge_height_t`, `marge_height_b`], [`marge_width_l`,
        `marge_width_r`]].

    """
    # check parameters
    stack.check_parameter(height=int, width=int, x=int)

    # pad height and width to make it multiple of x
    marge_sup_height = x - (height % x)
    marge_sup_height_l = int(marge_sup_height / 2)
    marge_sup_height_r = marge_sup_height - marge_sup_height_l
    marge_sup_width = x - (width % x)
    marge_sup_width_l = int(marge_sup_width / 2)
    marge_sup_width_r = marge_sup_width - marge_sup_width_l
    marge_padding = [[marge_sup_height_l, marge_sup_height_r],
                     [marge_sup_width_l, marge_sup_width_r]]

    return marge_padding


def compute_image_standardization(image):
    """Normalize image by computing its z score.

    Parameters
    ----------
    image : np.ndarray
        Image to normalize with shape (y, x).

    Returns
    -------
    normalized_image : np.ndarray
        Normalized image with shape (y, x).

    """
    # check parameters
    stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16, np.float32])

    # check image is in 2D
    if len(image.shape) != 2:
        raise ValueError("'image' should be a 2-d array. Not {0}-d array"
                         .format(len(image.shape)))

    # compute mean and standard deviation
    m = np.mean(image)
    adjusted_stddev = max(np.std(image), 1.0 / np.sqrt(image.size))

    # normalize image
    normalized_image = (image - m) / adjusted_stddev

    return normalized_image
