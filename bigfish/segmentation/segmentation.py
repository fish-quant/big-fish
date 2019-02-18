# -*- coding: utf-8 -*-

"""
Class and functions to segment nucleus and cytoplasm in 2-d and 3-d.
"""

from bigfish import stack

from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy import ndimage as ndi
import numpy as np


def nuc_segmentation_2d(tensor, r=0, nuc_channel=0, method="threshold",
                        return_label=False):
    """Segment nuclei from a 2d projection.

    Parameters
    ----------
    tensor : nd.ndarray, np.uint8
        Tensor with shape (r, c, z, y, x).
    r : int
        Round index to segment.
    nuc_channel : int
        Channel index of the dapi image.
    method : str
        Method used to segment.
    return_label : bool
        Condition to count and label the instances segmented in the image.

    Returns
    -------
    image_segmented : np.ndarray, bool
        Binary 2-d image with shape (y, x).
    image_labelled : np.ndarray, np.int64
        Image with labelled segmented instances and shape (y, x).
    nb_labels : int
        Number of different instances segmented.
    """
    # get 2D dapi image
    image_2d = stack.projection(tensor, method="mip", r=r, c=nuc_channel)

    # apply segmentation
    image_segmented = None
    if method == "threshold":
        # TODO be able to change the parameters of 'filtered_threshold'
        image_segmented = filtered_threshold(image_2d)

    # labelled and count segmented instances
    if return_label:
        image_labelled, nb_labels = label_instances(image_segmented)
        return image_segmented, image_labelled, nb_labels
    else:
        return image_segmented


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
    image : np.ndarray, np.uint8
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
    image = _remove_background(image,
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


def _remove_background(image, kernel_shape="disk", kernel_size=200):
    """Remove background noise from a 2-d image.

    Parameters
    ----------
    image : np.ndarray, np.uint8
        Image to process. Casting in np.uint8 makes the computation faster.
    kernel_shape : str
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_without_back : np.ndarray, np.uint8
        Image processed.

    """
    # compute background noise with a large mean filter
    background = stack.mean_filter(image,
                                   kernel_shape=kernel_shape,
                                   kernel_size=kernel_size)
    # subtract the background from the original image, clipping negative
    # values to 0
    mask = image > background
    image_without_back = np.subtract(image, background,
                                     out=np.zeros_like(image, dtype=np.uint8),
                                     where=mask)

    return image_without_back


def label_instances(image_segmented):
    """Count and label the different instances previously segmented in an
    image.

    Parameters
    ----------
    image_segmented : np.ndarray, bool
        Binary segmented image with shape (y, x).

    Returns
    -------
    image_label : np.ndarray, np.uint64
        Labelled image. Each object is characterized by the same pixel value.
    nb_labels : int
        Number of different instances counted in the image.

    """
    image_label, nb_labels = label(image_segmented, return_num=True)
    return image_label, nb_labels



