# -*- coding: utf-8 -*-

"""
Functions to prepare the data before feeding a model.
"""

import os
import threading

import numpy as np
from scipy import ndimage as ndi

from .augmentation import augment
from .utils import check_array
from .preprocess import (cast_img_uint8, cast_img_uint16, cast_img_float32,
                         cast_img_float64)

from skimage.transform import resize
from skimage.morphology.selem import square
from skimage.morphology import binary_dilation
from skimage.draw import polygon_perimeter
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder


# TODO define the requirements for 'data'

# ### Split data ###

def split_from_background(data, p_validation=0.2, p_test=0.2, logdir=None):
    """Split dataset between train, validation and test, based on the
    background volume used to simulate the cell.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the simulated data.
    p_validation : float
        Proportion of the validation dataset.
    p_test : float
        Proportion of the test dataset.
    logdir : str
        Path of the log directory used to save the split indices.

    Returns
    -------
    df_train : pandas.DataFrame
        Dataframe with the train dataset.
    df_validation : pandas.DataFrame
        Dataframe with the validation dataset.
    df_test : pandas.DataFrame
        Dataframe with the test dataset.

    """
    # get unique background cell
    background_id = list(set(data["cell_ID"]))
    np.random.shuffle(background_id)

    # split background cell between train, validation and test
    nb_validation = int(len(background_id) * p_validation)
    nb_test = int(len(background_id) * p_test)
    validation_cell = background_id[:nb_validation]
    test_cell = background_id[nb_validation:nb_validation+nb_test]
    train_cell = background_id[nb_validation+nb_test:]

    # split data between train, validation and test
    data_train = data.query("cell_ID in {0}".format(str(train_cell)))
    data_validation = data.query("cell_ID in {0}".format(str(validation_cell)))
    data_test = data.query("cell_ID in {0}".format(str(test_cell)))

    # save indices
    if logdir is not None:
        path = os.path.join(logdir, "indices_split.npz")
        np.savez(path,
                 indices_train=np.array(data_train.index),
                 indices_validation=np.array(data_validation.index),
                 indices_test=np.array(data_test.index))

    # reset index
    data_train.reset_index(drop=True, inplace=True)
    data_validation.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)

    return data_train, data_validation, data_test


# ### Encode labels ###

def encode_labels(data, column_name="pattern_name", classes_to_analyse="all"):
    """Filter classes we want to analyze and encode them from a string format
    to a numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.
    classes_to_analyse : str
        Define the set of classes we want to keep and to encode before training
        a model:
        - 'experimental' to fit with the experimental data (5 classes).
        - '2d' to analyze the 2-d classes only (7 classes).
        - 'all' to analyze all the classes (9 classes).

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # experimental analysis
    if classes_to_analyse == "experimental":
        data, encoder, classes = _encode_label_experimental(data, column_name)
    # 2-d analysis
    elif classes_to_analyse == "2d":
        data, encoder, classes = _encode_label_2d(data, column_name)
    # complete analysis
    elif classes_to_analyse == "all":
        data, encoder, classes = _encode_label_all(data, column_name)
    else:
        raise ValueError("'classes_to_analyse' can only take three values: "
                         "'experimental', '2d' or 'all'.")

    return data, encoder, classes


def _encode_label_experimental(data, column_name):
    """Filter the 5 classes included in the experimental dataset, then encode
    them from a string format to a numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # get classes to use
    classes = ["random", "foci", "cellext", "inNUC", "nuc2D"]

    # fit a label encoder
    encoder = LabelEncoder()
    encoder.fit(classes)

    # filter rows
    query = "{0} in {1}".format(column_name, str(classes))
    data = data.query(query)

    # encode labels
    if column_name == "label":
        data = data.assign(
            label_str=data.loc[:, column_name],
            label_num=encoder.transform(data.loc[:, column_name]))
    else:
        data = data.assign(
            label=encoder.transform(data.loc[:, column_name]))

    return data, encoder, classes


def _encode_label_2d(data, column_name):
    """Filter the 2-d classes, then encode them from a string format to a
    numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # get classes to use
    classes = ["random", "foci", "cellext", "inNUC", "nuc2D", "cell2D",
               "polarized"]

    # fit a label encoder
    encoder = LabelEncoder()
    encoder.fit(classes)

    # filter rows
    query = "{0} in {1}".format(column_name, str(classes))
    data = data.query(query)

    # encode labels
    if column_name == "label":
        data = data.assign(
            label_str=data.loc[:, column_name],
            label_num=encoder.transform(data.loc[:, column_name]))
    else:
        data = data.assign(
            label=encoder.transform(data.loc[:, column_name]))

    return data, encoder, classes


def _encode_label_all(data, column_name):
    """Encode all the classes from a string format to a numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # get classes to use
    classes = ["random", "foci", "cellext", "inNUC", "nuc2D", "cell2D",
               "polarized", "cell3D", "nuc3D"]

    # fit a label encoder
    encoder = LabelEncoder()
    encoder.fit(classes)

    # filter rows
    query = "{0} in {1}".format(column_name, str(classes))
    data = data.query(query)

    # encode labels
    if column_name == "label":
        data = data.assign(
            label_str=data.loc[:, column_name],
            label_num=encoder.transform(data.loc[:, column_name]))
    else:
        data = data.assign(
            label=encoder.transform(data.loc[:, column_name]))

    return data, encoder, classes


def get_map_label(data, column_num="label", columns_str="pattern_name"):
    label_num = list(set(data.loc[:, column_num]))
    label_str = list(set(data.loc[:, columns_str]))
    d = {}
    for i, label_num_ in enumerate(label_num):
        label_str_ = label_str[i]
        d[label_str_] = label_num

    return d


# ### Build images ###

def build_input_image(data, id_cell, channels="normal", input_shape=None,
                      augmentation=False):
    """

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the data.
    id_cell : int
        Index of the targeted cell.
    channels : str
        channels used in the input image.
            - 'normal' for (rna, cyt, nuc)
            - 'distance' for (rna, distance_cyt, distance_nuc)
            - 'surface' for (rna, surface_cyt, surface_nuc)
    input_shape : Tuple[int]
        Shape of the input image.
    augmentation : bool
        Apply a random operator on the image.

    Returns
    -------
    image : np.ndarray, np.float32
        A 3-d tensor with shape (x, y, channels). Values are normalized between
        0 and 1 (binaries values are unchanged and float values are rescaled
        according to their original dtype).

    """
    # TODO improve the resizing of different channels
    # build image from coordinates data
    cyt, nuc, rna = build_cell_2d(data, id_cell)

    # build the required input image
    if channels == "normal":
        image = np.stack((rna, cyt, nuc), axis=-1)
        image = resize_image(image, new_shape=input_shape, binary=True)
    elif channels == "distance":
        distance_cyt, distance_nuc = get_distance_layers(cyt, nuc)
        rna = resize_image(rna, new_shape=input_shape, binary=True)
        distance_cyt = resize_image(distance_cyt, new_shape=input_shape)
        distance_nuc = resize_image(distance_nuc, new_shape=input_shape)
        image = np.stack((rna, distance_cyt, distance_nuc), axis=-1)
    elif channels == "surface":
        surface_cyt, surface_nuc = get_surface_layers(cyt, nuc)
        image = np.stack((rna, surface_cyt, surface_nuc), axis=-1)
        image = resize_image(image, new_shape=input_shape, binary=True)
    else:
        raise ValueError("{0} is an invalid value for parameter 'channels': "
                         "must be 'normal', 'distance' or 'surface'."
                         .format(channels))

    # apply augmentation
    if augmentation:
        image = augment(image)

    return image


def build_input_image_precomputed(data, id_cell, channels="normal",
                                  input_shape=None, augmentation=False,
                                  precomputed_features=None):
    # TODO improve the resizing of different channels
    # TODO add documentation
    # build rna image from coordinates data
    rna = build_rna_2d(data, id_cell)
    rna = resize_image(rna, new_shape=input_shape, binary=True)

    # get precomputed features
    id_cell = data.loc[id_cell, "cell_ID"]
    cyt, nuc = precomputed_features[id_cell]

    # build the required input image
    image = np.stack((rna, cyt, nuc), axis=-1)
    if channels not in ["normal", "distance", "surface"]:
        raise ValueError("{0} is an invalid value for parameter 'channels': "
                         "must be 'normal', 'distance' or 'surface'."
                         .format(channels))

    # apply augmentation
    if augmentation:
        image = augment(image)

    return image


def build_rna_2d(data, id_cell):
    # TODO add documentation
    # get coordinates
    cyt_coord, _, rna_coord = get_coordinates(data, id_cell)

    # TODO manage the case where different spots meet at different heights,
    #  but same xy localization
    # build the dense representation for the rna if available
    max_x = cyt_coord[:, 0].max() + 5
    max_y = cyt_coord[:, 1].max() + 5
    values = [1] * rna_coord.shape[0]
    rna = coo_matrix((values, (rna_coord[:, 0], rna_coord[:, 1])),
                     shape=(max_x, max_y))
    rna = (rna > 0)
    rna = cast_img_float32(rna.todense())

    return rna


def build_cell_2d(data, id_cell):
    """Build 2-d images from data coordinates.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the data.
    id_cell : int
        Index of the targeted cell.

    Returns
    -------
    cyt : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).
    nuc : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).
    rna : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).

    """
    # get coordinates
    cyt_coord, nuc_coord, rna_coord = get_coordinates(data, id_cell)

    # build 2d images
    cyt, nuc, rna = from_coord_to_image(cyt_coord, nuc_coord, rna_coord)

    return cyt, nuc, rna


def get_coordinates(data, id_cell):
    """Get the coordinates a specific cell.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the data.
    id_cell : int
        Index of the targeted cell.

    Returns
    -------
    cyt : np.ndarray, np.int64
        Cytoplasm coordinates with shape (x, y).
    nuc : np.ndarray, np.int64
        Nucleus coordinates with shape (x, y).
    rna : np.ndarray, np.int64
        RNA spots coordinates with shape (x, y, z).

    """
    # get coordinates
    cyt = data.loc[id_cell, "pos_cell"]
    cyt = np.array(cyt, dtype=np.int64)
    nuc = data.loc[id_cell, "pos_nuc"]
    nuc = np.array(nuc, dtype=np.int64)
    rna = data.loc[id_cell, "RNA_pos"]
    rna = np.array(rna, dtype=np.int64)

    # complete cytoplasm and nucleus coordinates
    cyt_x, cyt_y = polygon_perimeter(cyt[:, 0], cyt[:, 1])
    cyt_x = cyt_x[:, np.newaxis]
    cyt_y = cyt_y[:, np.newaxis]
    cyt = np.concatenate((cyt_x, cyt_y), axis=-1)
    nuc_x, nuc_y = polygon_perimeter(nuc[:, 0], nuc[:, 1])
    nuc_x = nuc_x[:, np.newaxis]
    nuc_y = nuc_y[:, np.newaxis]
    nuc = np.concatenate((nuc_x, nuc_y), axis=-1)

    return cyt, nuc, rna


def from_coord_to_image(cyt_coord, nuc_coord, rna_coord=None):
    """Build 2-d images from the coordinates data.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Cytoplasm coordinates in 2-d with shape (x, y).
    nuc_coord : np.ndarray, np.int64
        Nucleus coordinates in 2-d with shape (x, y).
    rna_coord : np.ndarray, np.int64
        RNA spots coordinates in 3-d with shape (x, y, z).

    Returns
    -------
    cyt : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).
    nuc : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).
    rna : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).

    """
    # build the dense representation for the cytoplasm
    values = [1] * cyt_coord.shape[0]
    max_x = cyt_coord[:, 0].max() + 5
    max_y = cyt_coord[:, 1].max() + 5
    cyt = coo_matrix((values, (cyt_coord[:, 0], cyt_coord[:, 1])),
                     shape=(max_x, max_y))
    cyt = (cyt > 0)
    cyt = cast_img_float32(cyt.todense())

    # build the dense representation for the nucleus
    values = [1] * nuc_coord.shape[0]
    nuc = coo_matrix((values, (nuc_coord[:, 0], nuc_coord[:, 1])),
                     shape=(max_x, max_y))
    nuc = (nuc > 0)
    nuc = cast_img_float32(nuc.todense())

    if rna_coord is None:
        return cyt, nuc

    else:
        # TODO manage the case where different spots meet at different heights,
        #  but same xy localization
        # build the dense representation for the rna if available
        values = [1] * rna_coord.shape[0]
        rna = coo_matrix((values, (rna_coord[:, 0], rna_coord[:, 1])),
                         shape=(max_x, max_y))
        rna = (rna > 0)
        rna = cast_img_float32(rna.todense())

        return cyt, nuc, rna


def get_distance_layers(cyt, nuc):
    """Compute distance layers as input for the model.

    Parameters
    ----------
    cyt : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).
    nuc : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).

    Returns
    -------
    distance_cyt : np.ndarray, np.float32
        A 2-d tensor with shape (x, y) showing distance to the cytoplasm
        border.
    distance_nuc : np.ndarray, np.float32
        A 2-d tensor with shape (x, y) showing distance to the nucleus border.

    """
    # compute surfaces from cytoplasm and nucleus
    mask_cyt, mask_nuc = get_surface_layers(cyt, nuc)
    mask_cyt = mask_cyt.astype(np.bool)
    mask_nuc = mask_nuc.astype(np.bool)

    # compute distances from cytoplasm and nucleus
    distance_cyt = ndi.distance_transform_edt(mask_cyt)
    distance_nuc_ = ndi.distance_transform_edt(~mask_nuc)
    distance_nuc = mask_cyt * distance_nuc_

    # cast to np.float32 and normalize it between 0 and 1
    distance_cyt = cast_img_float32(distance_cyt / distance_cyt.max())
    distance_nuc = cast_img_float32(distance_nuc / distance_nuc.max())

    return distance_cyt, distance_nuc


def get_surface_layers(cyt, nuc):
    """Compute plain surface layers as input for the model.

    Sometimes the border is too fragmented to compute the surface. In this
    case, we iteratively apply a dilatation filter (with an increasing kernel
    size) until the boundary is properly connected the boundaries.

    Parameters
    ----------
    cyt : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).
    nuc : np.ndarray, np.float32
        A 2-d binary image with shape (x, y).

    Returns
    -------
    surface_cyt : np.ndarray, np.float32
        A 2-d binary tensor with shape (x, y) showing cytoplasm surface.
        border.
    surface_nuc : np.ndarray, np.float32
        A 2-d binary tensor with shape (x, y) showing nucleus surface.

        """
    # compute surface from cytoplasm and nucleus
    surface_cyt = ndi.binary_fill_holes(cyt)
    surface_nuc = ndi.binary_fill_holes(nuc)

    # cast to np.float32
    surface_cyt = cast_img_float32(surface_cyt)
    surface_nuc = cast_img_float32(surface_nuc)

    return surface_cyt, surface_nuc


def resize_image(image, new_shape=None, binary=False):
    """Resize image.

    If the size is decreased, the image is downsampled using a mean filter. If
    the shape is increased, new pixels' values are interpolated using spline
    method.

    Parameters
    ----------
    image : np.ndarray
        Image the resize with shape (y, x) or (y, x, channel).
    new_shape : Tuple[int]
        Spatial shape used for input images.
    binary : bool
        Keep binaries values after the resizing.

    Returns
    -------
    image_output : np.ndarray
        Resized image with shape (new_y, new_x) or (new_y, new_x, channel).

    """
    # check image dtype
    check_array(image, dtype=[np.uint8, np.uint16,
                              np.float32, np.float64,
                              np.bool])

    # get default output_shape
    if new_shape is None:
        return image

    # resize
    image_dtype = image.dtype
    if binary:
        # TODO use 'order=1' then binarize the image and reduce connected
        #  component.
        image_output = resize(image, new_shape,
                              anti_aliasing=False,
                              mode="constant",
                              cval=0)
        image_output = (image_output > 0)
    else:
        image_output = resize(image, new_shape,
                              anti_aliasing=True,
                              mode="constant",
                              cval=0)

    # cast the image in the original dtype
    if image_dtype == np.bool:
        image_output = (image_output > 0)
    elif image_dtype == np.uint8:
        image_output = cast_img_uint8(image_output)
    elif image_dtype == np.uint16:
        image_output = cast_img_uint16(image_output)
    elif image_dtype == np.float32:
        image_output = cast_img_float32(image_output)
    elif image_dtype == np.float64:
        image_output = cast_img_float64(image_output)

    return image_output


def get_label(data, id_cell):
    """Get the label of a specific cell.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the data.
    id_cell : int
        Index of the targeted cell.

    Returns
    -------
    label : int
        Encoded label of the cell.

    """
    # get encoded label
    label = data.loc[id_cell, "label"]

    return label


# ### Generator ###

class Generator:

    # TODO add documentation
    # TODO check threading.Lock()
    # TODO add classes
    def __init__(self, data, method, batch_size, input_shape, augmentation,
                 with_label, nb_classes, nb_epoch_max=10, shuffle=True,
                 precompute_features=False):
        # make generator threadsafe
        self.lock = threading.Lock()

        # get attributes
        self.data = data
        self.method = method
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augmentation = augmentation
        self.with_label = with_label
        self.nb_classes = nb_classes
        self.nb_epoch_max = nb_epoch_max
        self.shuffle = shuffle
        self.precompute_features = precompute_features

        # initialize generator
        self.nb_samples = self.data.shape[0]
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
                             "data. The 'len' method can't be used.")
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
        input_indices_ordered = list(self.data.index)
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
            cell_ref_if = df_cell.index[0]
            image_ref = build_input_image(self.data, cell_ref_if,
                                          channels=self.method,
                                          input_shape=self.input_shape)
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
    for i in range(batch_size):
        id_cell = indices[i]

        # use precomputed features if available
        if precomputed_features is None:
            image = build_input_image(data, id_cell, method, input_shape,
                                      augmentation)
        else:
            image = build_input_image_precomputed(data, id_cell, method,
                                                  input_shape, augmentation,
                                                  precomputed_features)

        batch_data[i] = image

    # return images with one-hot labels
    if with_label:
        labels = np.array(data.loc[indices, "label"], dtype=np.int64)
        batch_label = one_hot_label(labels, nb_classes)

        return batch_data, batch_label

    # return images only
    else:

        return batch_data


def one_hot_label(labels, nb_classes):
    """Binarize labels in a one-vs-all fashion.

    Parameters
    ----------
    labels : np.ndarray, np.int64
        Vector of labels with shape (nb_sample,).
    nb_classes : int
        Number of different classes available.

    Returns
    -------
    label_one_hot : np.ndarray, np.float32
        One-hot label (binary) with shape (nb_samples, nb_classes).

    """
    # binarize labels
    label_one_hot = np.eye(nb_classes, dtype=np.float32)[labels]

    return label_one_hot
