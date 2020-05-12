# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Class and functions to segment cells.
"""

import bigfish.stack as stack
from .utils import thresholding, clean_segmentation

import numpy as np
from scipy import ndimage as ndi

from skimage.morphology import watershed


def cell_watershed(image, nuc_label, threshold, alpha=0.8):
    """Apply watershed algorithm to segment cell instances.

    Cells are represented as watershed, with a low values to the center and
    maximum values at their boundaries. We flood these watersheds sequentially.
    The flooded area correspond to the cell segmentation.

    A image of cells ('relief') is built as a combination of pixel intensity
    and distance to nuclei, following the equation:

        relief = alpha * pixel_intensity + (1 - alpha) * distance_to_nucleus

    The watershed representation of cells is the opposite of this combined
    image 'relief'.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Cells image with shape (y, x).
    nuc_label : np.ndarray, np.int64
        Result of the nuclei segmentation with shape (y, x) and nuclei
        instances labelled.
    threshold : int or float
        Threshold to discriminate cells surfaces from background.
    alpha : float or int
        Weight of the pixel intensity values to compute the relief.

    Returns
    -------
    cell_label : np.ndarray, np.int64
        Segmentation of cells with shape (y, x).

    """
    # build relief
    relief = get_watershed_relief(image, nuc_label, alpha)

    # TODO improve cell mask methods
    # build cells mask
    if image.ndim == 3:
        image_2d = stack.maximum_projection(image)
    else:
        image_2d = image
    cell_mask = thresholding(image_2d, threshold)
    cell_mask[nuc_label > 0] = True
    cell_mask = clean_segmentation(cell_mask,
                                   small_object_size=5000,
                                   fill_holes=True)

    # segment cells
    cell_label = apply_watershed(relief, nuc_label, cell_mask)

    return cell_label


def get_watershed_relief(image, nuc_label, alpha):
    """Build a representation of cells as watershed.

    Cells are represented as watershed, with a low values to the center and
    maximum values at their boundaries. We flood these watersheds sequentially.
    The flooded area correspond to the cell segmentation.

    A image of cells ('relief') is built as a combination of pixel intensity
    and distance to nuclei, following the equation:

        relief = alpha * pixel_intensity + (1 - alpha) * distance_to_nucleus

    The watershed representation of cells is the opposite of this combined
    image 'relief'.


    Parameters
    ----------
    image : np.ndarray, np.uint
        Cells image with shape (z, y, x) or (y, x).
    nuc_label : np.ndarray, np.int64
        Result of the nuclei segmentation with shape (y, x) and nuclei
        instances labelled.
    alpha : float or int
        Weight of the pixel intensity values to compute the relief.

    Returns
    -------
    relief : np.ndarray, np.uint16
        Watershed representation of cells with shape (y, x).

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16])
    stack.check_array(nuc_label, ndim=2, dtype=np.int64)
    stack.check_parameter(alpha=(int, float))

    # if a 3-d image is provided we sum its pixel values
    image = image.astype(np.int64)
    if image.ndim == 3:
        image = image.sum(axis=0)

    # use pixel intensity of the cells image
    if alpha == 1:
        relief = image.max() - image
        relief[nuc_label > 0] = 0
        relief = np.true_divide(relief, relief.max(), dtype=np.float64)
        relief = stack.cast_img_uint16(relief, catch_warning=True)

    # use distance from the nuclei
    elif alpha == 0:
        nuc_mask = nuc_label > 0
        relief = ndi.distance_transform_edt(~nuc_mask)
        relief = np.true_divide(relief, relief.max(), dtype=np.float64)
        relief = stack.cast_img_uint16(relief, catch_warning=True)

    # use a combination of both previous methods
    elif 0 < alpha < 1:
        relief_pixel = image.max() - image
        relief_pixel[nuc_label > 0] = 0
        relief_pixel = np.true_divide(relief_pixel,
                                      relief_pixel.max(),
                                      dtype=np.float64)
        nuc_mask = nuc_label > 0
        relief_distance = ndi.distance_transform_edt(~nuc_mask)
        relief_distance = np.true_divide(relief_distance,
                                         relief_distance.max(),
                                         dtype=np.float64)
        relief = alpha * relief_pixel + (1 - alpha) * relief_distance
        relief = stack.cast_img_uint16(relief, catch_warning=True)

    else:
        raise ValueError("Parameter 'alpha' is wrong. It must be comprised "
                         "between 0 and 1. Currently 'alpha' is {0}"
                         .format(alpha))

    return relief


def apply_watershed(relief, nuc_label, cell_mask):
    """Apply watershed algorithm to segment cell instances.

    Cells are represented as watershed, with a low values to the center and
    maximum values at their boundaries. We flood these watersheds sequentially.
    The flooded area correspond to the cell segmentation.

    A image of cells ('relief') is built as a combination of pixel intensity
    and distance to nuclei, following the equation:

        relief = alpha * pixel_intensity + (1 - alpha) * distance_to_nucleus

    The watershed representation of cells is the opposite of this combined
    image 'relief'.

    Parameters
    ----------
    relief : np.ndarray, np.uint or np.int
        Watershed representation of cells with shape (y, x).
    nuc_label : np.ndarray, np.int64
        Result of the nuclei segmentation with shape (y, x) and nuclei
        instances labelled.
    cell_mask : np.ndarray, bool
        Binary image of cells surface with shape (y, x).

    Returns
    -------
    cell_label : np.ndarray, np.int64
        Segmentation of cells with shape (y, x).

    """
    # check parameters
    stack.check_array(relief, ndim=2, dtype=[np.uint8, np.uint16, np.int64])
    stack.check_array(nuc_label, ndim=2, dtype=np.int64)
    stack.check_array(cell_mask, ndim=2, dtype=bool)

    # segment cells
    cell_label = watershed(relief, markers=nuc_label, mask=cell_mask)
    cell_label = cell_label.astype(np.int64)

    return cell_label
