# -*- coding: utf-8 -*-

"""
Utilities function for nuclei and cytoplasm segmentation.
"""

from skimage.measure import label


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
