# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Class and functions to segment nucleus.
"""

from bigfish import stack
from .utils import clean_segmentation

import numpy as np

from skimage.morphology.selem import disk
from skimage.morphology import reconstruction


def remove_segmented_nuc(image, nuc_mask, size_nuclei=2000):
    """Remove the nuclei we have already segmented in an image.

    1) We start from the segmented nuclei with a light dilation. The missed
    nuclei and the background are set to 0 and removed from the original image.
    2) We reconstruct the missing nuclei by small dilation. As we used the
    original image to set the maximum allowed value at each pixel, the
    background pixels remain unchanged. However, pixels from the missing
    nuclei are partially reconstructed by the dilation. The reconstructed
    image only differs from the original one where the nuclei have been missed.
    3) We subtract the reconstructed image from the original one.
    4) From the few missing nuclei kept and restored, we build a binary mask
    (dilation, small object removal).
    5) We apply this mask to the original image to get the original pixel
    intensity of the missing nuclei.
    6) We remove pixels with a too low intensity.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Original nuclei image with shape (y, x).
    nuc_mask : np.ndarray,
        Result of the segmentation (with instance differentiation or not).
    size_nuclei : int
        Threshold above which we detect a nuclei.

    Returns
    -------
    image_without_nuc : np.ndarray
        Image with shape (y, x) and the same dtype of the original image.
        Nuclei previously detected in the mask are removed.

    """
    # check parameters
    stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16])
    stack.check_array(nuc_mask, ndim=2, dtype=bool)
    stack.check_parameter(size_nuclei=int)

    # store original dtype
    original_dtype = image.dtype

    # dilate the mask
    mask_dilated = stack.dilation_filter(image, "disk", 10)

    # remove the unsegmented nuclei from the original image
    diff = image.copy()
    diff[mask_dilated == 0] = 0

    # reconstruct the missing nuclei by dilation
    s = disk(1).astype(original_dtype)
    image_reconstructed = reconstruction(diff, image, selem=s)
    image_reconstructed = image_reconstructed.astype(original_dtype)

    # substract the reconstructed image from the original one
    image_filtered = image.copy()
    image_filtered -= image_reconstructed

    # build the binary mask for the missing nuclei
    missing_mask = image_filtered > 0
    missing_mask = clean_segmentation(missing_mask,
                                      small_object_size=size_nuclei,
                                      fill_holes=True)
    missing_mask = stack.dilation_filter(missing_mask, "disk", 20)

    # TODO improve the thresholds
    # get the original pixel intensity of the unsegmented nuclei
    unsegmented_nuclei = image.copy()
    unsegmented_nuclei[missing_mask == 0] = 0
    if original_dtype == np.uint8:
        unsegmented_nuclei[unsegmented_nuclei < 40] = 0
    else:
        unsegmented_nuclei[unsegmented_nuclei < 10000] = 0

    return unsegmented_nuclei
