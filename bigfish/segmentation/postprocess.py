# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Postprocessing functions functions for bigfish.segmentation subpackage.
"""

import bigfish.stack as stack

import numpy as np
from scipy import ndimage as ndi

from skimage.measure import label
from skimage.morphology import remove_small_objects


# TODO make functions compatible with different type of integers

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

def clean_segmentation(image, small_object_size=None, fill_holes=False,
                       smoothness=None, delimit_instance=False):
    """Clean segmentation results (binary masks or integer labels).

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).
    small_object_size : int or None
        Areas with a smaller surface (in pixels) are removed.
    fill_holes : bool
        Fill holes within a labelled or masked area.
    smoothness : int or None
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
    """Subtract an eroded image to a dilated one in order to prevent
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
    if original_dtype == bool:
        borders = image_dilated & ~image_eroded
        image_cleaned = image.copy()
        image_cleaned[borders] = False
    else:
        borders = image_dilated - image_eroded
        image_cleaned = image.copy()
        image_cleaned[borders > 0] = 0
        image_cleaned = image_cleaned.astype(original_dtype)

    return image_cleaned


def remove_disjoint(image):
    """For each instances with disconnected parts, keep the larger one.

    Parameters
    ----------
    image : np.ndarray, np.int or np.uint
        Labelled image with shape (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int or np.uint
        Cleaned image with shape (y, x).

    """
    # check parameters
    stack.check_array(image,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])

    # initialize cleaned labels
    image_cleaned = np.zeros_like(image)

    # loop over instances
    max_label = image.max()
    for i in range(1, max_label + 1):

        # get instance mask
        mask = image == i

        # check if an instance is labelled with this value
        if mask.sum() == 0:
            continue

        # get an index for each disconnected part of the instance
        labelled_mask = label(mask)
        indices = sorted(list(set(labelled_mask.ravel())))
        if 0 in indices:
            indices = indices[1:]

        # keep the largest part of the instance
        max_area = 0
        mask_instance = None
        for j in indices:
            mask_part_j = labelled_mask == j
            area_j = mask_part_j.sum()
            if area_j > max_area:
                max_area = area_j
                mask_instance = mask_part_j

        # add instance in the final label
        image_cleaned[mask_instance] = i

    return image_cleaned


# ### Nuclei-cells matching

def match_nuc_cell(nuc_label, cell_label, single_nuc, cell_alone):
    """Match each nucleus instance with the most overlapping cell instance.

    Parameters
    ----------
    nuc_label : np.ndarray, np.int or np.uint
        Labelled image of nuclei with shape (y, x).
    cell_label : np.ndarray, np.int or np.uint
        Labelled image of cells with shape (y, x).
    single_nuc : bool
        Authorized only one nucleus in a cell.
    cell_alone : bool
        Authorized cell without nucleus.

    Returns
    -------
    new_nuc_label : np.ndarray, np.int or np.uint
        Labelled image of nuclei with shape (y, x).
    new_cell_label : np.ndarray, np.int or np.uint
        Labelled image of cells with shape (y, x).

    """
    # check parameters
    stack.check_array(nuc_label,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])
    stack.check_array(cell_label,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])

    # initialize new labelled images
    new_nuc_label = np.zeros_like(nuc_label)
    new_cell_label = np.zeros_like(cell_label)
    remaining_cell_label = cell_label.copy()

    # loop over nuclei
    i_instance = 1
    max_nuc_label = nuc_label.max()
    for i_nuc in range(1, max_nuc_label + 1):

        # get nuc mask
        nuc_mask = nuc_label == i_nuc

        # check if a nucleus is labelled with this value
        if nuc_mask.sum() == 0:
            continue

        # check if a cell is labelled with this value
        i_cell = _get_most_frequent_value(cell_label[nuc_mask])
        if i_cell == 0:
            continue

        # get cell mask
        cell_mask = cell_label == i_cell

        # ensure nucleus is totally included in cell
        cell_mask |= nuc_mask
        cell_label[cell_mask] = i_cell
        remaining_cell_label[cell_mask] = i_cell

        # assign cell and nucleus
        new_nuc_label[nuc_mask] = i_instance
        new_cell_label[cell_mask] = i_instance
        i_instance += 1

        # remove pixel already assigned
        remaining_cell_label[cell_mask] = 0

        # if one nucleus per cell only, we remove the cell as candidate
        if single_nuc:
            cell_label[cell_mask] = 0

    # if only cell with nucleus are authorized we stop here
    if not cell_alone:
        return new_nuc_label, new_cell_label

    # loop over remaining cells
    max_remaining_cell_label = remaining_cell_label.max()
    for i_cell in range(1, max_remaining_cell_label + 1):

        # get cell mask
        cell_mask = remaining_cell_label == i_cell

        # check if a cell is labelled with this value
        if cell_mask.sum() == 0:
            continue

        # add cell in the result
        new_cell_label[cell_mask] = i_instance
        i_instance += 1

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
