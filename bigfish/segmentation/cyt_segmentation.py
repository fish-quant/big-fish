# -*- coding: utf-8 -*-

"""
Class and functions to segment nucleus and cytoplasm in 2-d and 3-d.
"""

import numpy as np

import bigfish.stack as stack

from skimage.morphology import remove_small_objects, remove_small_holes, label
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from scipy import ndimage as ndi


def build_cyt_binary_mask(image_projected, threshold=None):
    """Compute a binary mask of the cytoplasm.

    Parameters
    ----------
    image_projected : np.ndarray, np.uint
        A 2-d projection of the cytoplasm with shape (y, x).
    threshold : int
        Intensity pixel threshold to compute the binary mask. If None, an Otsu
        threshold is computed.

    Returns
    -------
    mask : np.ndarray, bool
        Binary mask of the cytoplasm with shape (y, x).

    """
    # check parameters
    stack.check_array(image_projected,
                      ndim=2,
                      dtype=[np.uint8, np.uint16])
    stack.check_parameter(threshold=(int, type(None)))

    # get a threshold
    if threshold is None:
        threshold = threshold_otsu(image_projected)

    # compute a binary mask
    mask = (image_projected > threshold)
    mask = remove_small_objects(mask, 3000)
    mask = remove_small_holes(mask, 2000)

    return mask


def build_cyt_relief(image_projected, nuc_labelled, mask_cyt, alpha=0.8):
    """Compute a 2-d representation of the cytoplasm to be used by watershed
    algorithm.

    Cells are represented as watershed, with a low values to the center and
    maximum values at their borders.

    The equation used is:
        relief = alpha * relief_pixel + (1 - alpha) * relief_distance

    - 'relief_pixel' exploit the differences in pixel intensity values.
    - 'relief_distance' use the distance from the nuclei.

    Parameters
    ----------
    image_projected : np.ndarray, np.uint
        Projected image of the cytoplasm with shape (y, x).
    nuc_labelled : np.ndarray,
        Result of the nuclei segmentation with shape (y, x).
    mask_cyt : np.ndarray, bool
        Binary mask of the cytoplasm with shape (y, x).
    alpha : float or int
        Weight of the pixel intensity values to compute the relief. A value of
        0 and 1 respectively return 'relief_distance' and 'relief_pixel'.

    Returns
    -------
    relief : np.ndarray, np.uint
        Relief image of the cytoplasm with shape (y, x).

    """
    # check parameters
    stack.check_array(image_projected,
                      ndim=2,
                      dtype=[np.uint8, np.uint16])
    stack.check_array(nuc_labelled,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_array(mask_cyt,
                      ndim=2,
                      dtype=[bool])
    stack.check_parameter(alpha=(float, int))

    # use pixel intensity of the cytoplasm channel to compute the seed.
    if alpha == 1:
        relief = image_projected.copy()
        max_intensity = np.iinfo(image_projected.dtype).max
        relief = max_intensity - relief
        relief[nuc_labelled > 0] = 0
        relief[mask_cyt == 0] = max_intensity
        relief = stack.rescale(relief)

    # use distance from the nuclei
    elif alpha == 0:
        binary_mask_nuc = nuc_labelled > 0
        relief = ndi.distance_transform_edt(~binary_mask_nuc)
        relief[mask_cyt == 0] = relief.max()
        relief = np.true_divide(relief, relief.max(), dtype=np.float32)
        if image_projected.dtype == np.uint8:
            relief = stack.cast_img_uint8(relief)
        else:
            relief = stack.cast_img_uint16(relief)

    # use both previous methods
    elif 0 < alpha < 1:
        relief_pixel = image_projected.copy()
        max_intensity = np.iinfo(image_projected.dtype).max
        relief_pixel = max_intensity - relief_pixel
        relief_pixel[nuc_labelled > 0] = 0
        relief_pixel[mask_cyt == 0] = max_intensity
        relief_pixel = stack.rescale(relief_pixel)
        relief_pixel = stack.cast_img_float32(relief_pixel)
        binary_mask_nuc = nuc_labelled > 0
        relief_distance = ndi.distance_transform_edt(~binary_mask_nuc)
        relief_distance[mask_cyt == 0] = relief_distance.max()
        relief_distance = np.true_divide(relief_distance,
                                         relief_distance.max(),
                                         dtype=np.float32)
        relief = alpha * relief_pixel + (1 - alpha) * relief_distance
        if image_projected.dtype == np.uint8:
            relief = stack.cast_img_uint8(relief)
        else:
            relief = stack.cast_img_uint16(relief)

    else:
        raise ValueError("Parameter 'alpha' is wrong. Must be comprised "
                         "between 0 and 1. Currently 'alpha' is {0}"
                         .format(alpha))

    return relief


def cyt_watershed(relief, nuc_labelled, mask, smooth=None):
    """Apply watershed algorithm on the cytoplasm to segment cell instances.

    Parameters
    ----------
    relief : np.ndarray, np.uint
        Relief image of the cytoplasm with shape (y, x).
    nuc_labelled : np.ndarray, np.int64
        Result of the nuclei segmentation with shape (y, x).
    mask : np.ndarray, bool
        Binary mask of the cytoplasm with shape (y, x).
    smooth : int
        Smooth the final boundaries applying a median filter on the mask
        (kernel_size=smooth).

    Returns
    -------
    cyt_segmented_final : np.ndarray, np.int64
        Segmentation of the cytoplasm with instance differentiation and shape
        (y, x).

    """
    # TODO how to be sure nucleus label corresponds to cell label?
    # check parameters
    stack.check_array(relief,
                      ndim=2,
                      dtype=[np.uint8, np.uint16])
    stack.check_array(nuc_labelled,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64])
    stack.check_array(mask,
                      ndim=2,
                      dtype=[bool])
    stack.check_parameter(smooth=(int, type(None)))

    # get markers
    markers = np.zeros_like(relief)
    for r in regionprops(nuc_labelled):
        markers[tuple(map(int, r.centroid))] = r.label
    markers = markers.astype(np.int64)

    # segment cytoplasm
    cyt_segmented = watershed(relief, markers, mask=mask)

    # smooth boundaries
    if smooth is not None:
        cyt_segmented = stack.median_filter(cyt_segmented.astype(np.uint16),
                                            kernel_shape="disk",
                                            kernel_size=smooth)
        cyt_segmented = remove_small_objects(cyt_segmented, 3000)
        cyt_segmented = cyt_segmented.astype(np.int64)

    # be sure to remove potential small disjoint part of the mask
    cyt_segmented_final = np.zeros_like(cyt_segmented)
    for id_cell in range(1, cyt_segmented.max() + 1):
        cell = cyt_segmented == id_cell
        cell_cc = label(cell)

        # one mask for the cell
        if cell_cc.max() == 1:
            mask = cell

        # multiple masks for the cell - we keep the larger one
        else:
            cell_properties = regionprops(cell_cc)
            m = 0
            mask = np.zeros_like(cyt_segmented).astype(bool)
            for cell_properties_ in cell_properties:
                area = cell_properties_.area
                if area > m:
                    m = area
                    mask = cell_cc == cell_properties_.label

        cyt_segmented_final[mask] = id_cell

    return cyt_segmented_final
