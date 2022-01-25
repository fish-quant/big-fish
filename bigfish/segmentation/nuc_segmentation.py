# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Class and functions to segment nucleus.
"""

import numpy as np

import bigfish.stack as stack

from .postprocess import label_instances
from .postprocess import clean_segmentation

from skimage.morphology.selem import disk
from skimage.morphology import reconstruction
from skimage.util import pad


# ### Unet model ###

def unet_3_classes_nuc():
    """Load a pretrained Unet model to predict 3 classes from nucleus images:
    background, edge and foreground.

    Returns
    -------
    model : ``tensorflow.keras.model`` object
        Pretrained Unet model.

    """
    # import  deep_learning subpackage
    import bigfish.deep_learning as dl

    # load model
    model = dl.load_pretrained_model("nuc", "3_classes")

    return model


def apply_unet_3_classes(model, image, target_size=None,
                         test_time_augmentation=False):
    """Segment image with a 3-classes trained model.

    Parameters
    ----------
     model : ``tensorflow.keras.model`` object
        Pretrained Unet model that predicts 3 classes from nucleus or cell
        images (background, edge and foreground).
    image : np.ndarray, np.uint
        Original image to segment with shape (y, x).
    target_size : int
        Resize image before segmentation. A squared image is resize to
        `target_size`. A rectangular image is resize such that its smaller
        dimension equals `target_size`.
    test_time_augmentation : bool
        Apply test time augmentation or not. The image is augmented 8 times
        and the final segmentation is the average result over these
        augmentations.

    Returns
    -------
    image_label_pred : np.ndarray, np.int64
        Labelled image. Each instance is characterized by the same pixel value.

    """
    # check parameters
    stack.check_parameter(
        target_size=(int, type(None)),
        test_time_augmentation=bool)
    stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16])

    # get original shape
    height, width = image.shape

    # resize image if necessary
    if target_size is None:
        # keep the original shape
        image_to_process = image.copy()
        new_height, new_width = height, width

    else:
        # target size should be multiple of 16
        target_size -= 16

        # we resize the images below the target size
        ratio = target_size / min(height, width)
        new_height = int(np.round(height * ratio))
        new_width = int(np.round(width * ratio))
        new_shape = (new_height, new_width)
        image_to_process = stack.resize_image(image, new_shape, "bilinear")

    # get padding marge to make it multiple of 16
    marge_padding = stack.get_marge_padding(new_height, new_width, x=16)
    top, bottom = marge_padding[0]
    left, right = marge_padding[1]
    image_to_process = pad(image_to_process, marge_padding, mode='symmetric')

    # standardize and cast image
    image_to_process = stack.compute_image_standardization(image_to_process)
    image_to_process = image_to_process.astype(np.float32)

    # augment images
    if test_time_augmentation:
        image_to_process = stack.augment_8_times(image_to_process)
        n_augmentations = 8
    else:
        image_to_process = [image_to_process]
        n_augmentations = 1

    # loop over augmentations
    predictions_augmented = []
    for i in range(n_augmentations):

        # get images
        image_to_process_ = image_to_process[i]
        image_to_process_ = image_to_process_[np.newaxis, :, :, np.newaxis]

        # make predictions
        prediction = model.predict(image_to_process_)

        # remove padding
        if i in [0, 1, 2, 6]:
            prediction = prediction[0, top:-bottom, left:-right, :]
        else:
            prediction = prediction[0, left:-right, top:-bottom, :]

        # resize predictions back taking into account a potential deformation
        # from the image augmentation
        if target_size is not None:
            if i in [0, 1, 2, 6]:
                prediction = stack.resize_image(
                    prediction, (height, width), "bilinear")
            else:
                prediction = stack.resize_image(
                    prediction, (width, height), "bilinear")

        # store predictions
        predictions_augmented.append(prediction)

    # reversed image augmentation
    if test_time_augmentation:
        predictions = stack.augment_8_times_reversed(predictions_augmented)
        mean_prediction = np.mean(predictions, axis=0)
    else:
        mean_prediction = predictions_augmented[0]

    # postprocess predictions
    image_label_pred = from_3_classes_to_instances(mean_prediction)

    return image_label_pred


def from_3_classes_to_instances(label_3_classes):
    """Extract instance labels from 3-classes Unet output.

    Parameters
    ----------
    label_3_classes : np.ndarray, np.float32
        Model prediction about the nucleus surface and boundaries, with shape
        (y, x, 3).

    Returns
    -------
    label : np.ndarray, np.int64
        Labelled image. Each instance is characterized by the same pixel value.

    """
    # check parameters
    stack.check_array(label_3_classes, ndim=3, dtype=[np.float32])

    # get classes indices
    label_3_classes = np.argmax(label_3_classes, axis=-1)

    # keep foreground predictions
    mask = label_3_classes > 1

    # instantiate each individual foreground surface predicted
    label = label_instances(mask)

    # dilate label
    label = label.astype(np.float64)
    label = stack.dilation_filter(label, kernel_shape="disk", kernel_size=1)
    label = label.astype(np.int64)

    return label


# ### Utility functions ###

def remove_segmented_nuc(image, nuc_mask, size_nuclei=2000):
    """Remove the nuclei we have already segmented in an image.

    #. We start from the segmented nuclei with a light dilation. The missed
       nuclei and the background are set to 0 and removed from the original
       image.
    #. We reconstruct the missing nuclei by small dilation. As we used the
       original image to set the maximum allowed value at each pixel, the
       background pixels remain unchanged. However, pixels from the missing
       nuclei are partially reconstructed by the dilation. The reconstructed
       image only differs from the original one where the nuclei have been
       missed.
    #. We subtract the reconstructed image from the original one.
    #. From the few missing nuclei kept and restored, we build a binary mask
       (dilation, small object removal).
    #. We apply this mask to the original image to get the original pixel
       intensity of the missing nuclei.
    #. We remove pixels with a too low intensity.

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

    # subtract the reconstructed image from the original one
    image_filtered = image.copy()
    image_filtered -= image_reconstructed

    # build the binary mask for the missing nuclei
    missing_mask = image_filtered > 0
    missing_mask = clean_segmentation(
        missing_mask,
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
