# -*- coding: utf-8 -*-

"""
Class and functions to segment nucleus and cytoplasm in 2-d and 3-d.
"""

from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi
from bigfish import stack
from skimage.measure import label


def segnuc_threshold(image, filter_size=200, small_object_size=2000):
    image_filtered = stack.remove_background(image, filter_size)
    image_segmented = image_filtered >= 2
    remove_small_objects(image_segmented,
                         min_size=small_object_size,
                         in_place=True)
    image_segmented = ndi.binary_fill_holes(image_segmented)
    return image_segmented


def label_nucleus(image_segmented):
    image_label, nb_labels = label(image_segmented, return_num=True)
    return image_label, nb_labels



