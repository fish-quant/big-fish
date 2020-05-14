# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.segmentation subpackage.
"""

import bigfish.stack as stack

import numpy as np
from scipy import ndimage as ndi

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


# ### Labelled images ###

def label_instances(image_binary):
    """Count and label the different instances previously segmented in an
    image.

    Parameters
    ----------
    image_binary : np.ndarray, bool
        Binary segmented image with shape (y, x).

    Returns
    -------
    image_label : np.ndarray, np.int64
        Labelled image. Each instance is characterized by the same pixel value.

    """
    # check parameters
    stack.check_array(image_binary, ndim=2, dtype=bool)

    # label instances
    image_label = label(image_binary)

    return image_label


def merge_labels(image_label_1, image_label_2):
    """Combine two partial labels of the same image.

    To prevent merging conflict, labels should not be rescale.

    Parameters
    ----------
    image_label_1 : np.ndarray, np.int64
        Labelled image with shape (y, x).
    image_label_2 : np.ndarray, np.int64
        Labelled image with shape (y, x).

    Returns
    -------
    image_label : np.ndarray, np.int64
        Labelled image with shape (y, x).

    """
    # check parameters
    stack.check_array(image_label_1, ndim=2, dtype=np.int64)
    stack.check_array(image_label_2, ndim=2, dtype=np.int64)

    # count number of instances
    nb_instances_1 = image_label_1.max()
    nb_instances_2 = image_label_2.max()
    nb_instances = nb_instances_1 + nb_instances_2

    # check if labels can be merged
    if nb_instances > np.iinfo(np.int64).max:
        raise ValueError("Labels can not be merged. There are too many "
                         "instances for a 64 bit image, labels could overlap.")

    # merge labels
    image_label_2[image_label_2 > 0] += image_label_1
    image_label = np.maximum(image_label_1, image_label_2)

    return image_label


# ### Basic segmentation ###

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


def clean_segmentation(image, small_object_size=None, fill_holes=False,
                       smoothness=None, delimit_instance=False):
    """Clean segmentation results (binary masks or integer labels).

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).
    small_object_size : int
        Areas with a smaller surface (in pixels) are removed.
    fill_holes : bool
        Fill holes within a labelled or masked area.
    smoothness : int
        Radius of a median kernel filter. The higher the smoother instance
        boundaries are.
    delimit_instance : bool
        Delimit clearly instances boundaries by preventing contact between each
        others.

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # check parameters
    stack.check_array(image, ndim=2, dtype=[np.int64, bool])
    stack.check_parameter(small_object_size=(int, type(None)),
                          fill_holes=bool,
                          smoothness=(int, type(None)),
                          delimit_instance=bool)

    # initialize cleaned image
    image_cleaned = image.copy()

    # remove small object
    if small_object_size is not None:
        image_cleaned = _remove_small_area(image_cleaned, small_object_size)

    # fill holes
    if fill_holes:
        image_cleaned = _fill_hole(image_cleaned)

    if smoothness:
        image_cleaned = _smooth_instance(image_cleaned, smoothness)

    if delimit_instance:
        image_cleaned = _delimit_instance(image_cleaned)

    return image_cleaned


def _remove_small_area(image, min_size):
    """Remove segmented areas with a small surface.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).
    min_size : int
        Areas with a smaller surface (in pixels) are removed.

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # remove small object
    image_cleaned = remove_small_objects(image, min_size=min_size)

    return image_cleaned


def _fill_hole(image):
    """Fill holes within the segmented areas.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # fill holes in a binary mask
    if image.dtype == bool:
        image_cleaned = ndi.binary_fill_holes(image)

    # fill holes in a labelled image
    else:
        image_cleaned = np.zeros_like(image)
        for i in range(1, image.max() + 1):
            image_binary = image == i
            image_binary = ndi.binary_fill_holes(image_binary)
            image_cleaned[image_binary] = i

    return image_cleaned


def _smooth_instance(image, radius):
    """Apply a median filter to smooth instance boundaries.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).
    radius : int
        Radius of the kernel for the median filter. The higher the smoother.

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # smooth instance boundaries for a binary mask
    if image.dtype == bool:
        image_cleaned = image.astype(np.uint8)
        image_cleaned = stack.median_filter(image_cleaned, "disk", radius)
        image_cleaned = image_cleaned.astype(bool)

    # smooth instance boundaries for a labelled image
    else:
        if image.max() <= 65535 and image.min() >= 0:
            image_cleaned = image.astype(np.uint16)
            image_cleaned = stack.median_filter(image_cleaned, "disk", radius)
            image_cleaned = image_cleaned.astype(np.int64)
        else:
            raise ValueError("Segmentation boundaries can't be smoothed "
                             "because more than 65535 has been detected in "
                             "the image. Smoothing is performed with 16-bit "
                             "unsigned integer images.")

    return image_cleaned


def _delimit_instance(image):
    """Substract an eroded image to a dilated one in order to prevent
    boundaries contact.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # handle 64 bit integer
    original_dtype = image.dtype
    if image.dtype == np.int64:
        image = image.astype(np.float64)

    # erode-dilate mask
    image_dilated = stack.dilation_filter(image, "disk", 1)
    image_eroded = stack.erosion_filter(image, "disk", 1)
    if original_dtype  == bool:
        borders = image_dilated & ~image_eroded
        image_cleaned = image.copy()
        image_cleaned[borders] = False
    else:
        borders = image_dilated - image_eroded
        image_cleaned = image.copy()
        image_cleaned[borders > 0] = 0
        image_cleaned = image_cleaned.astype(original_dtype)

    return image_cleaned


# ### Instances measures ###

def compute_instances_mean_diameter(image_label):
    """Compute the averaged size of the segmented instances.

    For each instance, we compute the diameter of an object with an equivalent
    surface. Then, we average the diameters.

    Parameters
    ----------
    image_label : np.ndarray, np.int64
        Labelled image with shape (y, x).

    Returns
    -------
    mean_diameter : float
        Averaged size of the segmented instances.

    """
    # check parameters
    stack.check_array(image_label, ndim=2, dtype=np.int64)

    # compute properties of the segmented instances
    props = regionprops(image_label)

    # get equivalent diameter and average it
    diameter = []
    for prop in props:
        diameter.append(prop.equivalent_diameter)
    mean_diameter = np.mean(diameter)

    return mean_diameter


# ### Nuclei-cells matching

def match_nuc_cell(nuc_label, cell_label):
    """Match each nucleus instance with the most overlapping cell instance.

    Parameters
    ----------
    nuc_label : np.ndarray, np.int64
        Labelled image of nuclei with shape (y, x).
    cell_label : np.ndarray, np.int64
        Labelled image of cells with shape (y, x).

    Returns
    -------
    new_nuc_label : np.ndarray, np.int64
        Labelled image of nuclei with shape (y, x).
    new_cell_label : np.ndarray, np.int64
        Labelled image of cells with shape (y, x).

    """
    # check parameters
    stack.check_array(nuc_label, ndim=2, dtype=np.int64)
    stack.check_array(cell_label, ndim=2, dtype=np.int64)

    # initialize new labelled images
    new_nuc_label = np.zeros_like(nuc_label)
    new_cell_label = np.zeros_like(cell_label)

    # match nuclei and cells
    for i_nuc in range(1, nuc_label.max() + 1):
        nuc_mask = nuc_label == i_nuc
        i_cell = _get_most_frequent_value(cell_label[nuc_mask])
        if i_cell == 0:
            continue
        cell_mask = cell_label == i_cell
        cell_mask |= nuc_mask
        new_nuc_label[nuc_mask] = i_nuc
        new_cell_label[cell_mask] = i_nuc

    return new_nuc_label, new_cell_label


def _get_most_frequent_value(array):
    """Count the most frequent value in a array.

    Parameters
    ----------
    array : np.ndarray, np.uint or np.int
        Array-like object.

    Returns
    -------
    value : int
        Most frequent integer in the array.

    """
    value = np.argmax(np.bincount(array))

    return value
