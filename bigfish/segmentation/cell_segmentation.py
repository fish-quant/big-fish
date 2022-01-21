# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Class and functions to segment cells.
"""

import bigfish.stack as stack

from .utils import thresholding
from .postprocess import label_instances
from .postprocess import clean_segmentation

import numpy as np
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.util import pad


# ### Unet models ###

def unet_distance_edge_double():
    """Load a pretrained Unet model to predict foreground and a distance map
    to edge from nucleus and cell images.

    Returns
    -------
    model : ``tensorflow.keras.model`` object
        Pretrained Unet model.

    """
    # import  deep_learning subpackage
    import bigfish.deep_learning as dl

    # load model
    model = dl.load_pretrained_model("double", "distance_edge")

    return model


def apply_unet_distance_double(model, nuc, cell, nuc_label, target_size=None,
                               test_time_augmentation=False):
    """Segment cell with a pretrained model to predict distance map and  use
    it with a watershed algorithm.

    Parameters
    ----------
    model : ``tensorflow.keras.model`` object
        Pretrained Unet model that predict distance to edges and cell surface.
    nuc : np.ndarray, np.uint
        Original nucleus image with shape (y, x).
    cell : np.ndarray, np.uint
        Original cell image to segment with shape (y, x).
    nuc_label : np.ndarray, np.int64
        Labelled nucleus image. Each nucleus is characterized by the same pixel
        value.
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
    cell_label_pred : np.ndarray, np.int64
        Labelled cell image. Each cell is characterized by the same pixel
        value.

    """
    # check parameters
    stack.check_parameter(
        target_size=(int, type(None)),
        test_time_augmentation=bool)
    stack.check_array(nuc, ndim=2, dtype=[np.uint8, np.uint16])
    stack.check_array(cell, ndim=2, dtype=[np.uint8, np.uint16])
    stack.check_array(nuc_label, ndim=2, dtype=np.int64)

    # get original shape
    height, width = cell.shape

    # resize cell image if necessary
    if target_size is None:
        # keep the original shape
        nuc_to_process = nuc.copy()
        cell_to_process = cell.copy()
        nuc_label_to_process = nuc_label.copy()
        new_height, new_width = height, width

    else:
        # target size should be multiple of 16
        target_size -= 16

        # we resize the images below the target size
        ratio = target_size / min(height, width)
        new_height = int(np.round(height * ratio))
        new_width = int(np.round(width * ratio))
        new_shape = (new_height, new_width)
        nuc_to_process = stack.resize_image(nuc, new_shape, "bilinear")
        cell_to_process = stack.resize_image(cell, new_shape, "bilinear")
        nuc_label_to_process = nuc_label.copy()

    # get padding marge to make it multiple of 16
    marge_padding = stack.get_marge_padding(new_height, new_width, x=16)
    top, bottom = marge_padding[0]
    left, right = marge_padding[1]
    nuc_to_process = pad(nuc_to_process, marge_padding, mode='symmetric')
    cell_to_process = pad(cell_to_process, marge_padding, mode='symmetric')

    # standardize and cast cell image
    nuc_to_process = stack.compute_image_standardization(nuc_to_process)
    nuc_to_process = nuc_to_process.astype(np.float32)
    cell_to_process = stack.compute_image_standardization(cell_to_process)
    cell_to_process = cell_to_process.astype(np.float32)

    # augment images
    if test_time_augmentation:
        nuc_to_process = stack.augment_8_times(nuc_to_process)
        cell_to_process = stack.augment_8_times(cell_to_process)
        n_augmentations = 8
    else:
        nuc_to_process = [nuc_to_process]
        cell_to_process = [cell_to_process]
        n_augmentations = 1

    # loop over augmentations
    predictions_cell = []
    predictions_distance = []
    for i in range(n_augmentations):

        # get images
        nuc_to_process_ = nuc_to_process[i]
        nuc_to_process_ = nuc_to_process_[np.newaxis, :, :, np.newaxis]
        cell_to_process_ = cell_to_process[i]
        cell_to_process_ = cell_to_process_[np.newaxis, :, :, np.newaxis]

        # make predictions
        prediction = model.predict([nuc_to_process_, cell_to_process_])

        # remove padding
        if i in [0, 1, 2, 6]:
            prediction_cell = prediction[1][0, top:-bottom, left:-right, 0]
            prediction_distance = prediction[2][0, top:-bottom, left:-right, 0]
        else:
            prediction_cell = prediction[1][0, left:-right, top:-bottom, 0]
            prediction_distance = prediction[2][0, left:-right, top:-bottom, 0]

        # resize predictions back taking into account a potential deformation
        # from the image augmentation
        if target_size is not None:
            if i in [0, 1, 2, 6]:
                prediction_cell = stack.resize_image(
                    prediction_cell, (height, width), "bilinear")
                prediction_distance = stack.resize_image(
                    prediction_distance, (height, width), "bilinear")
            else:
                prediction_cell = stack.resize_image(
                    prediction_cell, (width, height), "bilinear")
                prediction_distance = stack.resize_image(
                    prediction_distance, (width, height), "bilinear")

        # store predictions
        predictions_cell.append(prediction_cell)
        predictions_distance.append(prediction_distance)

    # reversed image augmentation
    if test_time_augmentation:
        predictions_cell = stack.augment_8_times_reversed(predictions_cell)
        predictions_distance = stack.augment_8_times_reversed(
            predictions_distance)
        mean_prediction_cell = np.mean(predictions_cell, axis=0)
        mean_prediction_distance = np.mean(predictions_distance, axis=0)
    else:
        mean_prediction_cell = predictions_cell[0]
        mean_prediction_distance = predictions_distance[0]

    # inverse and format distance map
    min_ = mean_prediction_distance.min()
    max_ = max(1, mean_prediction_distance.max())
    mean_prediction_distance -= min_
    mean_prediction_distance /= max_
    mean_prediction_distance = 1 - mean_prediction_distance
    mean_prediction_distance = np.clip(mean_prediction_distance, 0, 1)
    mean_prediction_distance = stack.cast_img_uint16(
        mean_prediction_distance,
        catch_warning=True)

    # postprocess predictions
    _, cell_label_pred = from_distance_to_instances(
        label_x_nuc=nuc_label_to_process,
        label_2_cell=mean_prediction_cell,
        label_distance=mean_prediction_distance,
        compute_nuc_label=False)

    return cell_label_pred


def from_distance_to_instances(label_x_nuc, label_2_cell, label_distance,
                               nuc_3_classes=False, compute_nuc_label=False):
    """Extract instance labels from a distance map and a binary surface
    prediction with a watershed algorithm.

    Parameters
    ----------
    label_x_nuc : np.ndarray, np.float32
        Model prediction about the nucleus surface (and boundaries), with shape
        (y, x, 1) or (y, x, 3).
    label_2_cell : np.ndarray, np.float32
        Model prediction about cell surface, with shape (y, x, 1).
    label_distance : np.ndarray, np.uint16
        Model prediction about the distance to edges, with shape (y, x, 1).
    nuc_3_classes : bool
        Nucleus image input is an output from a 3-classes Unet.
    compute_nuc_label : bool
        Extract nucleus instance labels.

    Returns
    -------
    nuc_label : np.ndarray, np.int64
        Labelled nucleus image. Each nucleus is characterized by the same pixel
        value.
    cell_label : np.ndarray, np.int64
        Labelled cell image. Each cell is characterized by the same pixel
        value.

    """
    # check parameters
    stack.check_parameter(
        nuc_3_classes=bool,
        compute_nuc_label=bool)
    stack.check_array(label_x_nuc, ndim=2, dtype=[np.float32, np.int64])
    stack.check_array(label_2_cell, ndim=2, dtype=[np.float32])
    stack.check_array(label_distance, ndim=2, dtype=[np.uint16])

    # get nuclei labels
    if nuc_3_classes and compute_nuc_label:
        label_3_nuc = np.argmax(label_x_nuc, axis=-1)
        mask_nuc = label_3_nuc > 1
        nuc_label = label_instances(mask_nuc)
        nuc_label = nuc_label.astype(np.float64)
        nuc_label = stack.dilation_filter(
            nuc_label, kernel_shape="disk", kernel_size=1)
        nuc_label = nuc_label.astype(np.int64)
        mask_nuc = nuc_label > 0
    elif not nuc_3_classes and compute_nuc_label:
        mask_nuc = label_x_nuc > 0.5
        nuc_label = label_instances(mask_nuc)
    else:
        nuc_label = label_x_nuc.copy()
        mask_nuc = nuc_label > 0

    # get cells surfaces
    mask_cell = label_2_cell > 0.5
    mask_cell |= mask_nuc

    # apply watershed algorithm
    cell_label = watershed(label_distance, markers=nuc_label, mask=mask_cell)

    # cast labels in int64
    nuc_label = nuc_label.astype(np.int64)
    cell_label = cell_label.astype(np.int64)

    return nuc_label, cell_label


# ### Watershed ###

def cell_watershed(image, nuc_label, threshold, alpha=0.8):
    """Apply watershed algorithm to segment cell instances.

    In a watershed algorithm we consider cells as watershed to be flooded. The
    watershed relief is inversely proportional to both the pixel intensity and
    the closeness to nuclei. Pixels with a high intensity or close to labelled
    nuclei have a low watershed relief value. They will be flooded in priority.
    Flooding the watersheds allows to propagate nuclei labels through potential
    cytoplasm areas. The lines separating watershed are the final segmentation
    of the cells.

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
        Weight of the pixel intensity values to compute the watershed relief.

    Returns
    -------
    cell_label : np.ndarray, np.int64
        Segmentation of cells with shape (y, x).

    """
    # build relief
    relief = get_watershed_relief(image, nuc_label, alpha)

    # TODO improve cell mask methods
    # TODO add options for clean_segmentation
    # build cells mask
    if image.ndim == 3:
        image_2d = stack.maximum_projection(image)
    else:
        image_2d = image
    cell_mask = thresholding(image_2d, threshold)
    cell_mask[nuc_label > 0] = True
    cell_mask = clean_segmentation(
        cell_mask,
        small_object_size=5000,
        fill_holes=True)

    # segment cells
    cell_label = apply_watershed(relief, nuc_label, cell_mask)

    return cell_label


def get_watershed_relief(image, nuc_label, alpha):
    """Build a representation of cells as watershed.

    In a watershed algorithm we consider cells as watershed to be flooded. The
    watershed relief is inversely proportional to both the pixel intensity and
    the closeness to nuclei. Pixels with a high intensity or close to labelled
    nuclei have a low watershed relief value. They will be flooded in priority.
    Flooding the watersheds allows to propagate nuclei labels through potential
    cytoplasm areas. The lines separating watershed are the final segmentation
    of the cells.

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
    watershed_relief : np.ndarray, np.uint16
        Watershed representation of cells with shape (y, x).

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16])
    stack.check_array(nuc_label, ndim=2, dtype=np.int64)
    stack.check_parameter(alpha=(int, float))

    # use pixel intensity of the cells image
    if alpha == 1:
        # if a 3-d image is provided we sum its pixel values
        image = stack.cast_img_float64(image)
        if image.ndim == 3:
            image = image.sum(axis=0)
        # rescale image
        image = stack.rescale(image)
        # build watershed relief
        watershed_relief = image.max() - image
        watershed_relief[nuc_label > 0] = 0
        watershed_relief = np.true_divide(
            watershed_relief,
            watershed_relief.max(),
            dtype=np.float64)
        watershed_relief = stack.cast_img_uint16(
            watershed_relief,
            catch_warning=True)

    # use distance from the nuclei
    elif alpha == 0:
        # build watershed relief
        nuc_mask = nuc_label > 0
        watershed_relief = ndi.distance_transform_edt(~nuc_mask)
        watershed_relief = np.true_divide(
            watershed_relief,
            watershed_relief.max(),
            dtype=np.float64)
        watershed_relief = stack.cast_img_uint16(
            watershed_relief,
            catch_warning=True)

    # use a combination of both previous methods
    elif 0 < alpha < 1:
        # if a 3-d image is provided we sum its pixel values
        image = stack.cast_img_float64(image)
        if image.ndim == 3:
            image = image.sum(axis=0)
        # rescale image
        image = stack.rescale(image)
        # build watershed relief
        relief_pixel = image.max() - image
        relief_pixel[nuc_label > 0] = 0
        relief_pixel = np.true_divide(
            relief_pixel,
            relief_pixel.max(),
            dtype=np.float64)
        nuc_mask = nuc_label > 0
        relief_distance = ndi.distance_transform_edt(~nuc_mask)
        relief_distance = np.true_divide(
            relief_distance,
            relief_distance.max(),
            dtype=np.float64)
        watershed_relief = alpha * relief_pixel + (1 - alpha) * relief_distance
        watershed_relief = stack.cast_img_uint16(
            watershed_relief,
            catch_warning=True)

    else:
        raise ValueError("Parameter 'alpha' is wrong. It must be comprised "
                         "between 0 and 1. Currently 'alpha' is {0}"
                         .format(alpha))

    return watershed_relief


def apply_watershed(watershed_relief, nuc_label, cell_mask):
    """Apply watershed algorithm to segment cell instances.

    In a watershed algorithm we consider cells as watershed to be flooded. The
    watershed relief is inversely proportional to both the pixel intensity and
    the closeness to nuclei. Pixels with a high intensity or close to labelled
    nuclei have a low watershed relief value. They will be flooded in priority.
    Flooding the watersheds allows to propagate nuclei labels through potential
    cytoplasm areas. The lines separating watershed are the final segmentation
    of the cells.

    Parameters
    ----------
    watershed_relief : np.ndarray, np.uint or np.int
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
    stack.check_array(
        watershed_relief,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64])
    stack.check_array(nuc_label, ndim=2, dtype=np.int64)
    stack.check_array(cell_mask, ndim=2, dtype=bool)

    # segment cells
    cell_label = watershed(watershed_relief, markers=nuc_label, mask=cell_mask)
    cell_label = cell_label.astype(np.int64)

    return cell_label
