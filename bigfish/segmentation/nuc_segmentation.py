# -*- coding: utf-8 -*-

"""
Class and functions to segment nucleus and cytoplasm in 2-d and 3-d.
"""

from bigfish import stack

from scipy import ndimage as ndi
import numpy as np

from skimage.morphology.selem import disk
from skimage.morphology import (reconstruction, binary_dilation,
                                remove_small_objects)

# TODO rename functions
# TODO complete documentation methods
# TODO add sanity functions


def filtered_threshold(image, kernel_shape="disk", kernel_size=200,
                       threshold=2, small_object_size=2000):
    """Segment a 2-d image to discriminate object from background.

    1) Compute background noise applying a large mean filter.
    2) remove this background from original image, clipping negative values
    to 0.
    3) Apply a threshold in the image
    4) Remove object with a small pixel area.
    5) Fill in holes in the segmented objects.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 2-d image to segment with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).
    threshold : int
        Pixel intensity threshold used to discriminate background from nuclei.
    small_object_size : int
        Pixel area of small objects removed after segmentation.

    Returns
    -------
    image_segmented : np.ndarray, bool
        Binary 2-d image with shape (y, x).

    """
    # remove background noise from image
    image = stack.remove_background_mean(image,
                                         kernel_shape=kernel_shape,
                                         kernel_size=kernel_size)

    # discriminate nuclei from background, applying a threshold.
    image_segmented = image >= threshold

    # clean the segmented result
    remove_small_objects(image_segmented,
                         min_size=small_object_size,
                         in_place=True)
    image_segmented = ndi.binary_fill_holes(image_segmented)

    return image_segmented


def remove_segmented_nuc(image, mask, nuclei_size=2000):
    """Remove the nuclei we have already segmented in an image.

    1) We only keep the segmented nuclei. The missed ones and the background
    are set to 0 and removed from the original image, using a dilated mask.
    2) We reconstruct the missing nuclei by small dilatation. As we used the
    original image as a mask (the maximum allowed value at each pixel), the
    background pixels remain unchanged. However, pixels from the missing
    nuclei are partially reconstructed by the dilatation. This reconstructed
    image only differs from the original one where the nuclei have been missed.
    3) We subtract the reconstructed image from the original one.
    4) From the few pixels kept and restored from the missing nuclei, we build
    a binary mask (dilatation, small object removal).
    5) We apply this mask to the original image to get the original pixel
    intensity of the missing nuclei.
    6) We remove pixels with a too low intensity (using Otsu thresholding).

    Parameters
    ----------
    image : np.ndarray, np.uint
        Original image with shape (y, x).
    mask : np.ndarray,
        Result of the segmentation (with instance differentiation or not).
    nuclei_size : int
        Threshold above which we detect a nuclei.

    Returns
    -------
    unsegmented_nuclei : np.ndarray
        Image with shape (y, x) and the same dtype of the original image.
        Nuclei previously detected in the mask are removed.

    """
    # TODO fix the dtype of the mask
    # TODO start from the original image to manage the potential rescaling
    # TODO improve the threshold
    # TODO correct the word dilatation -> dilation
    # check parameters
    stack.check_array(image,
                      ndim=2,
                      dtype=[np.uint8, np.uint16])
    stack.check_array(mask,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64, bool])

    # cast mask in np.int64 if it is binary
    if mask.dtype == bool or mask.dtype == np.uint16:
        mask = mask.astype(np.int64)

    # store original dtype
    original_dtype = image.dtype

    # dilate the mask
    s = disk(10, bool)
    dilated_mask = binary_dilation(mask, selem=s)

    # remove the unsegmented nuclei from the original image
    diff = image.copy()
    diff[dilated_mask == 0] = 0

    # reconstruct the missing nuclei by dilation
    s = disk(1)
    image_reconstructed = reconstruction(diff, image, selem=s)
    image_reconstructed = image_reconstructed.astype(original_dtype)

    # substract the reconstructed image from the original one
    image_filtered = image.copy()
    image_filtered -= image_reconstructed

    # build the binary mask for the missing nuclei
    missing_mask = image_filtered > 0
    missing_mask = remove_small_objects(missing_mask, nuclei_size)
    s = disk(20, bool)
    missing_mask = binary_dilation(missing_mask, selem=s)

    # get the original pixel intensity of the unsegmented nuclei
    unsegmented_nuclei = image.copy()
    unsegmented_nuclei[missing_mask == 0] = 0
    if original_dtype == np.uint8:
        unsegmented_nuclei[unsegmented_nuclei < 40] = 0
    else:
        unsegmented_nuclei[unsegmented_nuclei < 10000] = 0

    return unsegmented_nuclei
