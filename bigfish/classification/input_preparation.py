# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to prepare input data.
"""

import threading

import numpy as np
from scipy import ndimage as ndi

import bigfish.stack as stack

from skimage.measure import regionprops


# ### Input data ###

def prepare_extracted_data(cell_mask, nuc_mask=None, ndim=None, rna_coord=None,
                           centrosome_coord=None):
    """Prepare data extracted from images.

    Parameters
    ----------
    cell_mask : np.ndarray, np.uint, np.int or bool
        Surface of the cell with shape (y, x).
    nuc_mask: np.ndarray, np.uint, np.int or bool
        Surface of the nucleus with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider (2 or 3). Mandatory if
        'rna_coord' is provided.
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.
    centrosome_coord : np.ndarray, np.int64
        Coordinates of the detected centrosome with shape (nb_elements, 3) or
        (nb_elements, 2). One coordinate per dimension (zyx or yx dimensions).

    Returns
    -------
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    distance_cell : np.ndarray, np.float32
        Distance map from the cell with shape (y, x).
    distance_cell_normalized : np.ndarray, np.float32
        Normalized distance map from the cell with shape (y, x).
    centroid_cell : np.ndarray, np.int64
        Coordinates of the cell centroid with shape (1, 2).
    distance_centroid_cell : np.ndarray, np.float32
        Distance map from the cell centroid with shape (y, x).
    nuc_mask : np.ndarray, bool
        Surface of the nucleus with shape (y, x).
    cell_mask_out_nuc : np.ndarray, bool
        Surface of the cell (outside the nucleus) with shape (y, x).
    distance_nuc : np.ndarray, np.float32
        Distance map from the nucleus with shape (y, x).
    distance_nuc_normalized : np.ndarray, np.float32
        Normalized distance map from the nucleus with shape (y, x).
    centroid_nuc : np.ndarray, np.int64
        Coordinates of the nucleus centroid with shape (1, 2).
    distance_centroid_nuc : np.ndarray, np.float32
        Distance map from the nucleus centroid with shape (y, x).
    rna_coord_out_nuc : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1. Spots detected inside the nucleus are removed.
    centroid_rna : np.ndarray, np.int64
        Coordinates of the rna centroid with shape (1, 2).
    distance_centroid_rna : np.ndarray, np.float32
        Distance map from the rna centroid with shape (y, x).
    centroid_rna_out_nuc : np.ndarray, np.int64
        Coordinates of the rna centroid (outside the nucleus) with shape
        (1, 2).
    distance_centroid_rna_out_nuc : np.ndarray, np.float32
        Distance map from the rna centroid (outside the nucleus) with shape
        (y, x).
    distance_centrosome : np.ndarray, np.float32
        Distance map from the centrosome with shape (y, x).

    """
    # check parameters
    stack.check_parameter(ndim=(int, type(None)))
    if rna_coord is not None and ndim is None:
        raise ValueError("'ndim' should be specified (2 or 3).")

    # check arrays and make masks binary
    stack.check_array(cell_mask, ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64, bool])
    cell_mask = cell_mask.astype(bool)
    if nuc_mask is not None:
        stack.check_array(nuc_mask, ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
        nuc_mask = nuc_mask.astype(bool)
    if rna_coord is not None:
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
    if centrosome_coord is not None:
        stack.check_array(centrosome_coord, ndim=2, dtype=np.int64)

    # build distance map from the cell boundaries
    distance_cell = ndi.distance_transform_edt(cell_mask)
    distance_cell = distance_cell.astype(np.float32)
    distance_cell_normalized = distance_cell / distance_cell.max()

    # get cell centroid and a distance map from its localisation
    centroid_cell = _get_centroid_surface(cell_mask)
    distance_centroid_cell = _get_centroid_distance_map(centroid_cell,
                                                        cell_mask)

    # prepare arrays relative to the nucleus
    if nuc_mask is not None:

        # get cell mask outside nucleus
        cell_mask_out_nuc = cell_mask.copy()
        cell_mask_out_nuc[nuc_mask] = False

        # build distance map from the nucleus
        distance_nuc_ = ndi.distance_transform_edt(~nuc_mask)
        distance_nuc = cell_mask * distance_nuc_
        distance_nuc = distance_nuc.astype(np.float32)
        distance_nuc_normalized = distance_nuc / distance_nuc.max()

        # get nucleus centroid and a distance map from its localisation
        centroid_nuc = _get_centroid_surface(nuc_mask)
        distance_centroid_nuc = _get_centroid_distance_map(centroid_nuc,
                                                           cell_mask)

    else:
        cell_mask_out_nuc = None
        distance_nuc = None
        distance_nuc_normalized = None
        centroid_nuc = None
        distance_centroid_nuc = None

    # prepare arrays relative to the rna
    if rna_coord is not None:

        # get rna centroid
        if len(rna_coord) == 0:
            centroid_rna = centroid_cell.copy()
        else:
            centroid_rna = _get_centroid_rna(rna_coord, ndim)

        # build rna distance map
        distance_centroid_rna = _get_centroid_distance_map(
            centroid_rna, cell_mask)

        # combine rna and nucleus results
        if nuc_mask is not None:

            # get rna outside nucleus
            mask_rna_in_nuc = nuc_mask[rna_coord[:, ndim - 2],
                                       rna_coord[:, ndim - 1]]
            rna_coord_out_nuc = rna_coord[~mask_rna_in_nuc]

            # get rna centroid (outside nucleus)
            if len(rna_coord_out_nuc) == 0:
                centroid_rna_out_nuc = centroid_cell.copy()
            else:
                centroid_rna_out_nuc = _get_centroid_rna(rna_coord_out_nuc,
                                                         ndim)

            # build rna distance map (outside nucleus)
            distance_centroid_rna_out_nuc = _get_centroid_distance_map(
                centroid_rna_out_nuc, cell_mask)

        else:
            rna_coord_out_nuc = None
            centroid_rna_out_nuc = None
            distance_centroid_rna_out_nuc = None

    else:
        centroid_rna = None
        distance_centroid_rna = None
        rna_coord_out_nuc = None
        centroid_rna_out_nuc = None
        distance_centroid_rna_out_nuc = None

    # prepare arrays relative to the centrosome
    if centrosome_coord is not None:

        # build distance map from centroid
        if len(centrosome_coord) == 0:
            distance_centrosome = distance_cell.copy()
        else:
            distance_centrosome = _get_centrosome_distance_map(
                centrosome_coord, cell_mask)

    else:
        distance_centrosome = None

    # gather cell, nucleus, rna and centrosome data
    prepared_inputs = (cell_mask,
                       distance_cell, distance_cell_normalized,
                       centroid_cell, distance_centroid_cell,
                       nuc_mask, cell_mask_out_nuc,
                       distance_nuc, distance_nuc_normalized,
                       centroid_nuc, distance_centroid_nuc,
                       rna_coord_out_nuc,
                       centroid_rna, distance_centroid_rna,
                       centroid_rna_out_nuc, distance_centroid_rna_out_nuc,
                       distance_centrosome)

    return prepared_inputs


def _get_centroid_surface(mask):
    """Get centroid coordinates of a 2-d binary surface.

    Parameters
    ----------
    mask : np.ndarray, bool
        Binary surface with shape (y, x).

    Returns
    -------
    centroid : np.ndarray, np.int64
        Coordinates of the centroid with shape (1, 2).

    """
    # get centroid
    region = regionprops(mask.astype(np.uint8))[0]
    centroid = np.array(region.centroid, dtype=np.int64)

    return centroid


def _get_centroid_rna(rna_coord, ndim):
    """Get centroid coordinates of RNA molecules.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.
    ndim : int
        Number of spatial dimensions to consider (2 or 3).

    Returns
    -------
    centroid_rna : np.ndarray, np.int64
        Coordinates of the rna centroid with shape (1, 2) or (1, 3).

    """
    # get rna centroids
    centroid_rna = np.mean(rna_coord[:, :ndim], axis=0, dtype=np.int64)

    return centroid_rna


def _get_centroid_distance_map(centroid, cell_mask):
    """Build distance map from a centroid localisation.

    Parameters
    ----------
    centroid : np.ndarray, np.int64
        Coordinates of the centroid with shape (1, 2) or (1, 3).
    cell_mask : np.ndarray, bool
        Binary surface of the cell with shape (y, x).

    Returns
    -------
    distance_map : np.ndarray, np.float32
        Distance map from the centroid with shape (y, x).

    """
    if centroid.size == 3:
        centroid_2d = centroid[1:]
    else:
        centroid_2d = centroid.copy()

    # get mask centroid
    mask_centroid = np.zeros_like(cell_mask)
    mask_centroid[centroid_2d[0], centroid_2d[1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centroid)
    distance_map[cell_mask == 0] = 0
    distance_map = distance_map.astype(np.float32)

    return distance_map


def _get_centrosome_distance_map(centrosome_coord, cell_mask):
    """Build distance map from a centrosome localisation.

    Parameters
    ----------
    centrosome_coord : np.ndarray, np.int64
        Coordinates of the detected centrosome with shape (nb_elements, 3) or
        (nb_elements, 2). One coordinate per dimension (zyx or yx dimensions).
    cell_mask : np.ndarray, bool
        Binary surface of the cell with shape (y, x).

    Returns
    -------
    distance_map : np.ndarray, np.float32
        Distance map from the centrosome with shape (y, x).

    """
    if centrosome_coord.size == 3:
        centrosome_coord_2d = centrosome_coord[1:]
    else:
        centrosome_coord_2d = centrosome_coord.copy()

    # get mask centrosome
    mask_centrosome = np.zeros_like(cell_mask)
    mask_centrosome[centrosome_coord_2d[:, 0],
                    centrosome_coord_2d[:, 1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centrosome)
    distance_map[cell_mask == 0] = 0
    distance_map = distance_map.astype(np.float32)

    return distance_map


# ### Input layers ###

# TODO add documentation

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

# TODO define the requirements for 'data'
# TODO add logging

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
