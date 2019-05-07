# -*- coding: utf-8 -*-

"""
Class and functions to segment nucleus and cytoplasm in 2-d and 3-d.
"""

from bigfish import stack
from .nuc_segmentation import nuc_segmentation_2d

from skimage.morphology import remove_small_objects, remove_small_holes
import numpy as np
from skimage.morphology import watershed
from skimage.filters import threshold_otsu
from skimage.measure import regionprops


# TODO rename functions
# TODO complete documentation methods


def cyt_segmentation_2d(tensor, r, c_nuc, c_cyt, segmentation_method):
    # TODO add documentation
    # check tensor dimensions and its dtype
    stack.check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16])

    # apply segmentation
    # TODO validate the pipeline with this cast
    image_segmented = stack.cast_img_uint8(tensor)
    if segmentation_method == "watershed":
        image_segmented = watershed_2d(image_segmented, r, c_nuc, c_cyt)
    else:
        pass
    return image_segmented


def watershed_2d(tensor, r, c_nuc, c_cyt):
    # TODO add documentation
    # TODO better integration with nuclei segmentation
    # nuclei segmentation
    _, nuc_labelled, _ = nuc_segmentation_2d(
        tensor,
        projection_method="mip",
        r=r, c=c_nuc,
        segmentation_method="threshold",
        return_label=True)

    # get source image
    cyt = tensor[r, c_cyt, :, :, :]
    cyt_projected = stack.projection(tensor, method="mip", r=r, c=c_cyt)

    # get a mask for the cytoplasm
    mask = (cyt_projected > threshold_otsu(cyt_projected))
    mask = remove_small_objects(mask, 200)
    mask = remove_small_holes(mask, 200)

    # get image to apply watershed on
    seed = np.sum(cyt, 0)
    seed = seed.max() - seed
    seed[nuc_labelled > 0] = 0

    # get the markers from the nuclei
    markers = np.zeros_like(seed)
    for r in regionprops(nuc_labelled):
        markers[tuple(map(int, r.centroid))] = r.label

    # apply watershed
    cyt_segmented = watershed(seed, markers, mask=mask)

    return cyt_segmented
