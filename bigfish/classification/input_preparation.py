# -*- coding: utf-8 -*-

"""
Functions to prepare input (coordinates or images).
"""

import os
import threading

import bigfish.stack as stack

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from skimage.measure import regionprops
from skimage.draw import polygon_perimeter
from sklearn.preprocessing import LabelEncoder


# TODO define the requirements for 'data'
# TODO add logging
# TODO generalize the use of 'get_offset_value'
# TODO add documentation

# ### Prepare 2-d coordinates in order to compute the hand-crafted features ###

def prepare_coordinate_data(cyt_coord, nuc_coord, rna_coord):
    """

    Parameters
    ----------
    cyt_coord
    nuc_coord
    rna_coord

    Returns
    -------

    """
    # convert coordinates in binary mask surfaces
    mask_cyt, mask_nuc, _, rna_coord = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord,
        external_coord=True)

    # get mask cytoplasm outside nucleus
    mask_cyt_out = mask_cyt.copy()
    mask_cyt_out[mask_nuc] = False

    # compute distance maps for the cytoplasm and the nucleus
    distance_cyt = ndi.distance_transform_edt(mask_cyt)
    distance_nuc_ = ndi.distance_transform_edt(~mask_nuc)
    distance_nuc = mask_cyt * distance_nuc_

    # cast distance map in float32
    distance_cyt = distance_cyt.astype(np.float32)
    distance_nuc = distance_nuc.astype(np.float32)

    # normalize distance maps between 0 and 1
    distance_cyt_normalized = distance_cyt / distance_cyt.max()
    distance_cyt_normalized = stack.cast_img_float32(distance_cyt_normalized)
    distance_nuc_normalized = distance_nuc / distance_nuc.max()
    distance_nuc_normalized = stack.cast_img_float32(distance_nuc_normalized)

    # get rna outside nucleus
    mask_rna_in = mask_nuc[rna_coord[:, 1], rna_coord[:, 2]]
    rna_coord_out = rna_coord[~mask_rna_in]

    # get centroids
    centroid_cyt = _get_centroid_surface(mask_cyt)
    centroid_nuc = _get_centroid_surface(mask_nuc)
    if len(rna_coord) == 0:
        centroid_rna = centroid_cyt.copy()
    else:
        centroid_rna = _get_centroid_rna(rna_coord)
    if len(rna_coord_out) == 0:
        centroid_rna_out = centroid_cyt.copy()
    else:
        centroid_rna_out = _get_centroid_rna(rna_coord_out)

    # get centroid distance maps
    distance_cyt_centroid = _get_centroid_distance_map(centroid_cyt, mask_cyt)
    distance_nuc_centroid = _get_centroid_distance_map(centroid_nuc, mask_cyt)
    distance_rna_out_centroid = _get_centroid_distance_map(centroid_rna_out,
                                                           mask_cyt)

    prepared_inputs = (mask_cyt, mask_nuc, mask_cyt_out,
                       distance_cyt, distance_nuc,
                       distance_cyt_normalized, distance_nuc_normalized,
                       rna_coord_out,
                       centroid_cyt, centroid_nuc,
                       centroid_rna, centroid_rna_out,
                       distance_cyt_centroid, distance_nuc_centroid,
                       distance_rna_out_centroid)

    return prepared_inputs


def _get_centroid_surface(mask):
    # get centroid
    region = regionprops(mask.astype(np.uint8))[0]
    centroid = np.array(region.centroid, dtype=np.int64)

    return centroid


def _get_centroid_rna(rna_coord):
    # get rna centroids
    centroid_rna = np.mean(rna_coord[:, :3], axis=0, dtype=np.int64)
    return centroid_rna


def _get_centroid_distance_map(centroid_coordinate, mask_cyt):
    if centroid_coordinate.size == 3:
        centroid_coordinate_2d = centroid_coordinate[1:]
    else:
        centroid_coordinate_2d = centroid_coordinate.copy()

    # get mask centroid
    mask_centroid = np.zeros_like(mask_cyt).astype(bool)
    mask_centroid[centroid_coordinate_2d[0], centroid_coordinate_2d[1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centroid)
    distance_map[mask_cyt == 0] = 0
    distance_map = distance_map.astype(np.float32)

    return distance_map


# ### Prepare 2-d images for deep learning classification models ###

def build_boundaries_layers(cyt_coord, nuc_coord, rna_coord):
    """

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).

    Returns
    -------
    cyt_boundaries : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing cytoplasm boundaries.
        border.
    nuc_boundaries : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing nucleus boundaries.
    rna_layer : np.ndarray, np.float32
        Binary image of mRNAs localizations with shape (y, x).

    """
    # check parameters
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])

    # build surface binary matrices from coordinates
    cyt_surface, nuc_surface, rna_layer, _ = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord)

    # from surface binary matrices to boundaries binary matrices
    cyt_boundaries = stack.from_surface_to_boundaries(cyt_surface)
    nuc_boundaries = stack.from_surface_to_boundaries(nuc_surface)

    # cast layer in float32
    cyt_boundaries = stack.cast_img_float32(cyt_boundaries)
    nuc_boundaries = stack.cast_img_float32(nuc_boundaries)
    rna_layer = stack.cast_img_float32(rna_layer)

    return cyt_boundaries, nuc_boundaries, rna_layer


def build_surface_layers(cyt_coord, nuc_coord, rna_coord):
    """Compute plain surface layers as input for the model.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).

    Returns
    -------
    cyt_surface : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing cytoplasm surface.
        border.
    nuc_surface : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing nucleus surface.
    rna_layer : np.ndarray, np.float32
        Binary image of mRNAs localizations with shape (y, x).

    """
    # check parameters
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])

    # build surface binary matrices from coordinates
    cyt_surface, nuc_surface, rna_layer, _ = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord)

    # cast layer in float32
    cyt_surface = stack.cast_img_float32(cyt_surface)
    nuc_surface = stack.cast_img_float32(nuc_surface)
    rna_layer = stack.cast_img_float32(rna_layer)

    return cyt_surface, nuc_surface, rna_layer


def build_distance_layers(cyt_coord, nuc_coord, rna_coord, normalized=True):
    """Compute distance layers as input for the model.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).
    normalized : bool
        Normalized the layers between 0 and 1.
    Returns
    -------
    distance_cyt : np.ndarray, np.float32
        A 2-d tensor with shape (y, x) showing distance to the cytoplasm
        border. Normalize between 0 and 1 if 'normalized' True.
    distance_nuc : np.ndarray, np.float32
        A 2-d tensor with shape (y, x) showing distance to the nucleus border.
        Normalize between 0 and 1 if 'normalized' True.

    """
    # check parameters
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])
    stack.check_parameter(normalized=bool)

    # build surface binary matrices from coordinates
    cyt_surface, nuc_surface, rna_layer, _ = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord)

    # compute distances map for cytoplasm and nucleus
    cyt_distance = ndi.distance_transform_edt(cyt_surface)
    nuc_distance_ = ndi.distance_transform_edt(~nuc_surface)
    nuc_distance = cyt_surface * nuc_distance_

    if normalized:
        # cast to np.float32 and normalize it between 0 and 1
        cyt_distance = cyt_distance / cyt_distance.max()
        nuc_distance = nuc_distance / nuc_distance.max()

    # cast layer in float32
    cyt_distance = stack.cast_img_float32(cyt_distance)
    nuc_distance = stack.cast_img_float32(nuc_distance)
    rna_layer = stack.cast_img_float32(rna_layer)

    return cyt_distance, nuc_distance, rna_layer


# ### Input image generator ###

class Generator:

    # TODO add documentation
    # TODO check threading.Lock()
    # TODO add classes
    def __init__(self, filenames, labels, input_directory, batch_size,
                 input_shape, augmentation, with_label, nb_epoch_max=10,
                 shuffle=True, precompute_features=False):

        # make generator threadsafe
        self.lock = threading.Lock()

        # get attributes
        self.filenames = filenames
        self.labels = labels
        self.input_directory = input_directory
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augmentation = augmentation
        self.with_label = with_label
        self.nb_epoch_max = nb_epoch_max
        self.shuffle = shuffle
        self.precompute_features = precompute_features

        # initialize generator
        self.nb_samples = len(self.filenames)
        self.indices = self._get_shuffled_indices()
        self.nb_batch_per_epoch = self._get_batch_per_epoch()
        self.i_batch = 0
        self.i_epoch = 0

        # precompute feature if necessary
        if self.precompute_features and "cell_ID" in self.data.columns:
            unique_cells = list(set(self.data.loc[:, "cell_ID"]))
            self.precomputed_features = self._precompute_features(unique_cells)
        else:
            self.precomputed_features = None

    def __len__(self):
        if self.nb_epoch_max is None:
            raise ValueError("This generator loops indefinitely over the "
                             "dataset. The 'len' method can't be used.")
        else:
            return self.nb_samples * self.nb_epoch_max

    def __bool__(self):
        if self.nb_epoch_max is None or self.nb_epoch_max > 0:
            return True
        else:
            return False

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self._next()

    def _next(self):
        # we reach the end of an epoch
        if self.i_batch == self.nb_batch_per_epoch:
            self.i_epoch += 1

            # the generator loop over the data indefinitely
            if self.nb_epoch_max is None:
                # TODO find something better
                if self.i_epoch == 500:
                    raise StopIteration
                self.i_batch = 0
                self.indices = self._get_shuffled_indices()
                return self._next()

            # we start a new epoch
            elif (self.nb_epoch_max is not None
                  and self.i_epoch < self.nb_epoch_max):
                self.i_batch = 0
                self.indices = self._get_shuffled_indices()
                return self._next()

            # we reach the maximum number of epochs
            elif (self.nb_epoch_max is not None
                  and self.i_epoch == self.nb_epoch_max):
                raise StopIteration

        # we build a new batch
        else:
            if self.with_label:
                batch_data, batch_label = self._build_batch(self.i_batch)
                self.i_batch += 1
                return batch_data, batch_label
            else:
                batch_data = self._build_batch(self.i_batch)
                self.i_batch += 1
                return batch_data

    def _get_shuffled_indices(self):
        # shuffle input data and get their indices
        input_indices_ordered = [i for i in range(self.nb_samples)]
        if self.shuffle:
            np.random.shuffle(input_indices_ordered)
        return input_indices_ordered

    def _get_batch_per_epoch(self):
        # compute the number of batches to generate for the entire epoch
        if self.nb_samples % self.batch_size == 0:
            nb_batch = len(self.indices) // self.batch_size
        else:
            # the last batch can be smaller
            nb_batch = (len(self.indices) // self.batch_size) + 1
        return nb_batch

    def _build_batch(self, i_batch):
        # build a batch
        start_index = i_batch * self.batch_size
        end_index = min((i_batch + 1) * self.batch_size, self.nb_samples)
        indices_batch = self.indices[start_index:end_index]

        # return batch with label
        if self.with_label:
            batch_data, batch_label = build_batch(
                data=self.data,
                indices=indices_batch,
                method=self.method,
                input_shape=self.input_shape,
                augmentation=self.augmentation,
                with_label=self.with_label,
                nb_classes=self.nb_classes,
                precomputed_features=self.precomputed_features)

            return batch_data, batch_label

        # return batch without label
        else:
            batch_data = build_batch(
                data=self.data,
                indices=indices_batch,
                method=self.method,
                input_shape=self.input_shape,
                augmentation=self.augmentation,
                with_label=self.with_label,
                nb_classes=self.nb_classes,
                precomputed_features=self.precomputed_features)

            return batch_data

    def _precompute_features(self, unique_cells):
        """

        Parameters
        ----------
        unique_cells

        Returns
        -------

        """
        # TODO add documentation
        # get a sample for each instance of cell
        d_features = {}
        for cell in unique_cells:
            df_cell = self.data.loc[self.data.cell_ID == cell, :]
            id_cell = df_cell.index[0]
            image_ref = build_image(
                self.data, id_cell,
                image_shape=self.input_shape,
                coord_refinement=True,
                method=self.method,
                augmentation=False)
            d_features[cell] = (image_ref[:, :, 1], image_ref[:, :, 2])

        return d_features

    def reset(self):
        # initialize generator
        self.indices = self._get_shuffled_indices()
        self.nb_batch_per_epoch = self._get_batch_per_epoch()
        self.i_batch = 0
        self.i_epoch = 0


# TODO try to fully vectorize this step
def build_batch(data, indices, method="normal", input_shape=(224, 224),
                augmentation=True, with_label=False, nb_classes=9,
                precomputed_features=None):
    """Build a batch of data.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the data.
    indices : List[int]
        List of indices to use for the batch.
    method : str
        Channels used in the input image.
            - 'normal' for (rna, cyt, nuc)
            - 'distance' for (rna, distance_cyt, distance_nuc)
            - 'surface' for (rna, surface_cyt, surface_nuc)
    input_shape : Tuple[int]
        Shape of the input image.
    augmentation : bool
        Apply a random operator on the image.
    with_label : bool
        Return label of the image as well.
    nb_classes : int
        Number of different classes available.
    precomputed_features : dict
        Some datasets are simulated from a small limited set of background
        cells (cytoplasm and nucleus). In this case, we can precompute and keep
        in memory the related features layers in order to dramatically speed
        up the program. this dict associate the id of the reference cells to
        their computed features layers (cytoplasm, nucleus).

    Returns
    -------
    batch_data : np.ndarray, np.float32
        Tensor with shape (batch_size, x, y, 3).
    batch_label : np.ndarray, np.int64
        Tensor of the encoded label, with shape (batch_size,)

    """
    # initialize the batch
    batch_size = len(indices)
    batch_data = np.zeros((batch_size, input_shape[0], input_shape[1], 3),
                          dtype=np.float32)

    # build each input image of the batch
    if precomputed_features is None:
        for i in range(batch_size):
            id_cell = indices[i]
            image = build_image(
                data, id_cell,
                image_shape=input_shape,
                coord_refinement=True,
                method=method,
                augmentation=augmentation)
            batch_data[i] = image
    else:
        for i in range(batch_size):
            id_cell = indices[i]
            image = build_image_precomputed(
                data, id_cell,
                image_shape=input_shape,
                precomputed_features=precomputed_features,
                augmentation=augmentation)
            batch_data[i] = image

    # return images with one-hot labels
    if with_label:
        labels = np.array(data.loc[indices, "label"], dtype=np.int64)
        batch_label = _one_hot_label(labels, nb_classes)

        return batch_data, batch_label

    # return images only
    else:

        return batch_data
