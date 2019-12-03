# -*- coding: utf-8 -*-

"""
Utilities function for nuclei and cytoplasm segmentation.
"""

import warnings

import bigfish.stack as stack

import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


# TODO homogenize the dtype of masks

# ### Manipulate labels ###

def label_instances(mask):
    """Count and label the different instances previously segmented in an
    image.

    Parameters
    ----------
    mask : np.ndarray, bool
        Binary segmented image with shape (y, x).

    Returns
    -------
    image_label : np.ndarray, np.int64
        Labelled image. Each object is characterized by the same pixel value.
    nb_labels : int
        Number of different instances counted in the image.

    """
    # check parameters
    stack.check_array(mask,
                      ndim=2,
                      dtype=bool)

    # get labels
    image_label, nb_labels = label(mask, return_num=True)
    return image_label, nb_labels


def compute_mean_size_object(image_labelled):
    """Compute the averaged size of the segmented objects.

    For each object, we compute the diameter of an object with an equivalent
    surface. Then, we average the diameters.

    Parameters
    ----------
    image_labelled : np.ndarray, np.uint
        Labelled image with shape (y, x).

    Returns
    -------
    mean_diameter : float
        Averaged size of the segmented objects.

    """
    # check parameters
    stack.check_array(image_labelled,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])

    # compute properties of the segmented object
    props = regionprops(image_labelled)

    # get equivalent diameter and average it
    diameter = []
    for prop in props:
        diameter.append(prop.equivalent_diameter)
    mean_diameter = np.mean(diameter)

    return mean_diameter


def merge_labels(label_1, label_2):
    """Combine two partial labels of the same image.

    To prevent merging conflict, labels should not be rescale.

    Parameters
    ----------
    label_1 : np.ndarray, np.uint or np.int
        Labelled image with shape (y, x).
    label_2 : np.ndarray, np.uint or np.int
        Labelled image with shape (y, x).

    Returns
    -------
    label : np.ndarray, np.int64
        Labelled image with shape (y, x).

    """
    # check parameters
    stack.check_array(label_1,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])
    stack.check_array(label_2,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])

    # count number of label
    nb_label_1 = label_1.max()
    nb_label_2 = label_2.max()

    # clean masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label_1 = remove_small_objects(label_1, 3000)
        label_2 = remove_small_objects(label_2, 3000)

    # cast labels in np.int64
    label_1 = label_1.astype(np.int64)
    label_2 = label_2.astype(np.int64)

    # check if labels can be merged
    if nb_label_1 + nb_label_2 > np.iinfo(nb_label_1.dtype).max:
        raise ValueError("Labels can not be merged (labels could overlapped).")

    # merge labels
    label_2[label_2 > 0] += nb_label_1
    label = np.maximum(label_1, label_2)

    return label


def dilate_erode_labels(label):
    """Substract an eroded label to a dilated one in order to prevent
    boundaries contact.

    Parameters
    ----------
    label : np.ndarray, np.uint or np.int
        Labelled image with shape (y, x).

    Returns
    -------
    label_final : np.ndarray, np.int64
        Labelled image with shape (y, x).

    """
    # check parameters
    stack.check_array(label,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])

    # handle 64 bit integer
    if label.dtype == np.int64:
        label = label.astype(np.uint16)

    # erode-dilate mask
    label_dilated = stack.dilation_filter(label, "disk", 2)
    label_eroded = stack.erosion_filter(label, "disk", 2)
    borders = label_dilated - label_eroded
    label_final = label.copy()
    label_final[borders > 0] = 0
    label_final = label_final.astype(np.int64)

    return label_final
