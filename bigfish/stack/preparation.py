# -*- coding: utf-8 -*-

"""
Functions to prepare the data before feeding a model.
"""

import os
import threading

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .utils import get_offset_value
from .augmentation import augment
from .preprocess import cast_img_float32
from .filter import mean_filter

from skimage.draw import polygon_perimeter
from sklearn.preprocessing import LabelEncoder


# TODO define the requirements for 'data'
# TODO add logging
# TODO generalize the use of 'get_offset_value'
# TODO move the script to the classification submodule

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


# ### Filter data ###

def filter_data(data, proportion_to_exclude=0.2):
    # TODO add documentation

    if (isinstance(proportion_to_exclude, float)
            and 0 <= proportion_to_exclude <= 1):
        p = int(proportion_to_exclude * 10)
    elif (isinstance(proportion_to_exclude, int)
          and 0 <= proportion_to_exclude <= 100):
        p = proportion_to_exclude // 10
    else:
        raise ValueError("'proportion' must be a float between 0 and 1 or an "
                         "integer between 0 and 100.")

    # filter inNUC, nuc2D, cell3D, "cell2D" and nuc3D
    l = ['p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p70', 'p80', 'p90', 'p100']
    level_kept = l[:p]
    query = "pattern_level not in {0}".format(str(level_kept))
    data_filtered = data.query(query)

    # filter foci
    l = ['p50', 'p60', 'p70', 'p80', 'p90', 'p100', 'p110', 'p120', 'p130',
         'p140', 'p150']
    level_kept = l[:p]
    query = "pattern_level not in {0} or pattern_name != 'foci'".format(
        str(level_kept))
    data_filtered = data_filtered.query(query)

    # reset index
    data_filtered.reset_index(drop=True, inplace=True)

    return data_filtered


# ### Balance data ###

def balance_data(data, column_to_balance, verbose=0):
    # TODO add documentation
    # TODO make it consistent for int values
    values = list(data.loc[:, column_to_balance].value_counts().index)
    frequencies = list(data.loc[:, column_to_balance].value_counts())

    max_frequency = max(frequencies)
    diff_frequency = [max_frequency - frequency for frequency in frequencies]

    for i, value in enumerate(values):
        n = diff_frequency[i]
        if verbose > 0:
            print("add {0} new samples {1} to balance the dataset..."
                  .format(n, value))
        df = data.query("{0} == '{1}'".format(column_to_balance, value))
        df = df.sample(n, replace=True, random_state=13)
        data = pd.concat([data, df])
    if verbose > 0:
        print()

    # reset index
    data.reset_index(drop=True, inplace=True)

    return data


# ### Encode labels and genes ###

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
    # get label encoder
    encoder, classes = get_label_encoder(classes_to_analyze=classes_to_analyse)

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

    # reset index
    data.loc[:, "original_index"] = data.index
    data.reset_index(drop=True, inplace=True)

    return data, encoder, classes


def get_label_encoder(classes_to_analyze="all"):
    # TODO add documentation
    # get set of classes to analyze
    if classes_to_analyze == "experimental":
        classes = ["random", "foci", "cellext", "inNUC", "nuc2D"]
    elif classes_to_analyze == "2d":
        classes = ["random", "foci", "cellext", "inNUC", "nuc2D", "cell2D",
                   "polarized"]
    elif classes_to_analyze == "all":
        classes = ["random", "foci", "cellext", "inNUC", "nuc2D", "cell2D",
                   "polarized", "cell3D", "nuc3D"]
    else:
        raise ValueError("'classes_to_analyse' can only take three values: "
                         "'experimental', '2d' or 'all'.")

    # fit a label encoder
    encoder = LabelEncoder()
    encoder.fit(classes)

    return encoder, classes


def get_map_label(data, column_num="label", columns_str="pattern_name"):
    # TODO add documentation
    # TODO redo with encoder
    label_num = list(set(data.loc[:, column_num]))
    label_str = list(set(data.loc[:, columns_str]))
    d = {}
    for i, label_num_ in enumerate(label_num):
        label_str_ = label_str[i]
        d[label_str_] = label_num

    return d


def get_gene_encoder(genes_str):
    # encode genes
    encoder_gene = LabelEncoder()
    encoder_gene.fit(genes_str)

    return encoder_gene


# ### Build images from coordinates ###

def build_image(data, id_cell, image_shape=None, coord_refinement=True,
                method="normal", augmentation=False):
    """

    Parameters
    ----------
    data
    id_cell
    image_shape
    coord_refinement
    method
    augmentation

    Returns
    -------

    """
    # TODO add documentation
    # TODO add sanity check for precomputation
    # get coordinates
    rna_coord, cyt_coord, nuc_coord = get_coordinates(data, id_cell,
                                                      image_shape,
                                                      coord_refinement)

    # build matrices
    if image_shape is None:
        max_x = cyt_coord[:, 0].max() + get_offset_value()
        max_y = cyt_coord[:, 1].max() + get_offset_value()
        image_shape = (max_x, max_y)
    rna = np.zeros(image_shape, dtype=np.float32)
    rna[rna_coord[:, 0], rna_coord[:, 1]] = 1.0
    cyt = np.zeros(image_shape, dtype=np.float32)
    cyt[cyt_coord[:, 0], cyt_coord[:, 1]] = 1.0
    nuc = np.zeros(image_shape, dtype=np.float32)
    nuc[nuc_coord[:, 0], nuc_coord[:, 1]] = 1.0

    # get features
    if method == "normal":
        pass
    elif method == "surface":
        cyt, nuc = get_surface_layers(cyt, nuc)
    elif method == "distance":
        cyt, nuc = get_distance_layers(cyt, nuc)
    else:
        raise ValueError(
            "{0} is an invalid value for parameter 'channels': must be "
            "'normal', 'distance' or 'surface'.".format(method))

    # stack image
    image = np.stack((rna, cyt, nuc), axis=-1)

    # augment
    if augmentation:
        image = augment(image)

    return image


def build_image_precomputed(data, id_cell, image_shape=None,
                            precomputed_features=None, augmentation=False):
    """

    Parameters
    ----------
    data
    id_cell
    image_shape
    precomputed_features
    augmentation

    Returns
    -------

    """
    # TODO add documentation
    # TODO add sanity check for precomputation

    # build rna image from coordinates data
    rna = _build_rna(data, id_cell, image_shape)

    # get precomputed features
    id_cell = data.loc[id_cell, "cell_ID"]
    cyt, nuc = precomputed_features[id_cell]

    # build the required input image
    image = np.stack((rna, cyt, nuc), axis=-1)

    # apply augmentation
    if augmentation:
        image = augment(image)

    return image


def _build_rna(data, id_cell, output_shape=None):
    # TODO add documentation
    # TODO check if 'polygone_perimeter' changes the input shape
    # get coordinates
    rna_coord = data.loc[id_cell, "RNA_pos"]
    rna_coord = np.array(rna_coord, dtype=np.int64)

    # get current shape
    cyt_coord = data.loc[id_cell, "pos_cell"]
    cyt_coord = np.array(cyt_coord, dtype=np.int64)
    max_x = cyt_coord[:, 0].max() + get_offset_value()
    max_y = cyt_coord[:, 1].max() + get_offset_value()
    input_shape = (max_x, max_y)

    if output_shape is not None:
        # compute resizing factor
        factor = _compute_resizing_factor(input_shape, output_shape)

        # resize coordinates directly
        rna_coord = _resize_coord(rna_coord, factor)

    else:
        output_shape = input_shape

    # build rna image
    rna = np.zeros(output_shape, dtype=np.float32)
    rna[rna_coord[:, 0], rna_coord[:, 1]] = 1.0

    return rna


def get_coordinates(data, id_cell, output_shape=None, coord_refinement=True):
    """

    Parameters
    ----------
    data
    id_cell
    output_shape
    coord_refinement

    Returns
    -------

    """
    # TODO add documentation
    # get coordinates
    rna_coord = data.loc[id_cell, "RNA_pos"]
    rna_coord = np.array(rna_coord, dtype=np.int64)
    cyt_coord = data.loc[id_cell, "pos_cell"]
    cyt_coord = np.array(cyt_coord, dtype=np.int64)
    nuc_coord = data.loc[id_cell, "pos_nuc"]
    nuc_coord = np.array(nuc_coord, dtype=np.int64)

    # resize coordinates
    if output_shape is not None:
        max_x = cyt_coord[:, 0].max() + 5
        max_y = cyt_coord[:, 1].max() + 5
        input_shape = (max_x, max_y)
        factor = _compute_resizing_factor(input_shape, output_shape)
        rna_coord = _resize_coord(rna_coord, factor)
        cyt_coord = _resize_coord(cyt_coord, factor[:, :2])
        nuc_coord = _resize_coord(nuc_coord, factor[:, :2])

    # complete cytoplasm and nucleus coordinates
    if coord_refinement:
        # TODO use util.complete_coordinates_2d
        cyt_x, cyt_y = polygon_perimeter(cyt_coord[:, 0], cyt_coord[:, 1])
        cyt_x = cyt_x[:, np.newaxis]
        cyt_y = cyt_y[:, np.newaxis]
        cyt_coord = np.concatenate((cyt_x, cyt_y), axis=-1)
        nuc_x, nuc_y = polygon_perimeter(nuc_coord[:, 0], nuc_coord[:, 1])
        nuc_x = nuc_x[:, np.newaxis]
        nuc_y = nuc_y[:, np.newaxis]
        nuc_coord = np.concatenate((nuc_x, nuc_y), axis=-1)

    return rna_coord, cyt_coord, nuc_coord


def _compute_resizing_factor(input_shape, output_shape):
    # compute factor
    delta_x = output_shape[0] / input_shape[0]
    delta_y = output_shape[1] / input_shape[1]
    factor = np.array([delta_x, delta_y, 1], dtype=np.float32)[np.newaxis, :]

    return factor


def _resize_coord(coord, factor):
    # resize coordinates directly
    coord = np.round(coord * factor).astype(np.int64)

    return coord


def get_distance_layers(cyt, nuc, normalized=True):
    """Compute distance layers as input for the model.

    Parameters
    ----------
    cyt : np.ndarray, np.float32
        A 2-d binary image with shape (y, x).
    nuc : np.ndarray, np.float32
        A 2-d binary image with shape (y, x).
    normalized : bool
        Normalized it between 0 and 1.

    Returns
    -------
    distance_cyt : np.ndarray, np.float32
        A 2-d tensor with shape (y, x) showing distance to the cytoplasm
        border. Normalize between 0 and 1 if 'normalized' True.
    distance_nuc : np.ndarray, np.float32
        A 2-d tensor with shape (y, x) showing distance to the nucleus border.
        Normalize between 0 and 1 if 'normalized' True.

    """
    # TODO can return NaN
    # compute surfaces from cytoplasm and nucleus
    mask_cyt, mask_nuc = get_surface_layers(cyt, nuc, cast_float=False)

    # compute distances from cytoplasm and nucleus
    distance_cyt = ndi.distance_transform_edt(mask_cyt)
    distance_nuc_ = ndi.distance_transform_edt(~mask_nuc)
    distance_nuc = mask_cyt * distance_nuc_

    if normalized:
        # cast to np.float32 and normalize it between 0 and 1
        distance_cyt = cast_img_float32(distance_cyt / distance_cyt.max())
        distance_nuc = cast_img_float32(distance_nuc / distance_nuc.max())

    return distance_cyt.astype(np.float32), distance_nuc.astype(np.float32)


def get_surface_layers(cyt, nuc, cast_float=True):
    """Compute plain surface layers as input for the model.

    Sometimes the border is too fragmented to compute the surface. In this
    case, we iteratively apply a dilatation filter (with an increasing kernel
    size) until the boundary is properly connected the boundaries.

    Parameters
    ----------
    cyt : np.ndarray, np.float32
        A 2-d binary image with shape (y, x).
    nuc : np.ndarray, np.float32
        A 2-d binary image with shape (y, x).
    cast_float : bool
        Cast output in np.float32.

    Returns
    -------
    surface_cyt : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing cytoplasm surface.
        border.
    surface_nuc : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing nucleus surface.

        """
    # compute surface from cytoplasm and nucleus
    surface_cyt = ndi.binary_fill_holes(cyt)
    surface_nuc = ndi.binary_fill_holes(nuc)

    # cast to np.float32
    if cast_float:
        surface_cyt = cast_img_float32(surface_cyt)
        surface_nuc = cast_img_float32(surface_nuc)

    return surface_cyt, surface_nuc


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


def _one_hot_label(labels, nb_classes):
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


# ### Experimental data ###

def format_experimental_data(data, label_encoder=None):
    # TODO add documentation
    # initialize the formatted dataset
    data_formatted = data.copy(deep=True)

    # format coordinates
    data_formatted.loc[:, 'pos_cell'] = data_formatted.apply(
        lambda row: _decompose_experimental_coordinate(row["pos"].T)[0],
        axis=1)
    data_formatted.loc[:, 'pos_nuc'] = data_formatted.apply(
        lambda row: _decompose_experimental_coordinate(row["pos"].T)[1],
        axis=1)
    data_formatted.loc[:, 'RNA_pos'] = data_formatted.apply(
        lambda row: _decompose_experimental_coordinate(row["pos"].T)[2],
        axis=1)

    # format cell indices
    data_formatted.loc[:, 'cell_ID'] = data_formatted.index

    # format RNA count
    data_formatted.loc[:, 'nb_rna'] = data_formatted.apply(
        lambda row: len(row["RNA_pos"]),
        axis=1)

    # format label
    if label_encoder is not None:
        pattern_level = [None] * data_formatted.shape[0]
        data_formatted.loc[:, 'pattern_level'] = pattern_level
        data_formatted.loc[:, 'pattern_name'] = data_formatted.apply(
            lambda row: _label_experimental_num_to_str_(row["labels"]),
            axis=1)
        data_formatted.loc[:, 'label'] = data_formatted.apply(
            lambda row: label_encoder.transform([row["pattern_name"]])[0],
            axis=1)

    # remove useless columns
    if label_encoder is not None:
        features_to_keep = ['gene', 'pos_nuc', 'pos_cell', 'RNA_pos', 'cell_ID',
                            'nb_rna', 'pattern_level', 'pattern_name', 'label']
    else:
        features_to_keep = ['gene', 'pos_nuc', 'pos_cell', 'RNA_pos',
                            'cell_ID', 'nb_rna']
    data_formatted = data_formatted.loc[:, features_to_keep]

    return data_formatted


def _decompose_experimental_coordinate(positions):
    # TODO add documentation
    # get coordinate for each element of the cell
    nuc_coord = positions[positions[:, 2] == 0]
    nuc_coord = nuc_coord[:, :2].astype(np.int64)
    cyt_coord = positions[positions[:, 2] == 1]
    cyt_coord = cyt_coord[:, :2].astype(np.int64)
    rna_coord = positions[positions[:, 2] == 2]
    rna_coord = rna_coord.astype(np.int64)
    rna_coord[:, 2] = np.zeros((rna_coord.shape[0],), dtype=np.int64)

    return cyt_coord, nuc_coord, rna_coord


def _label_experimental_num_to_str_(label_num):
    # TODO add documentation
    if label_num == 1:
        label_str = "foci"
    elif label_num == 2:
        label_str = "cellext"
    elif label_num == 3:
        label_str = "inNUC"
    elif label_num == 4:
        label_str = "nuc2D"
    elif label_num == 5:
        label_str = "random"
    else:
        raise ValueError("Label value should be comprised between 1 and 5.")

    return label_str


def remove_transcription_site_bis(data, threshold):
    # TODO add documentation
    # TODO vectorize it
    data_corrected = data.copy(deep=True)
    for index, row in data_corrected.iterrows():
        id_cell = row['cell_ID']
        image = build_image(data, id_cell,
                            coord_refinement=True,
                            method="surface")
        rna, cyt, nuc = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        rna_in = np.copy(rna)
        rna_in[nuc == 0] = 0
        rna_out = np.copy(rna)
        rna_out[nuc > 0] = 0
        rna_in = 255 * rna_in.astype(np.uint8)
        density_img = mean_filter(rna_in, kernel_shape="disk", kernel_size=4)
        density_img = cast_img_float32(density_img)
        transcription_site = density_img > threshold
        rna_in[transcription_site] = 0

        rna = rna_in + rna_out

        rna_pos = np.nonzero(rna)
        rna_pos = np.column_stack(rna_pos).astype(np.int64)
        rna_pos = np.concatenate(
            [rna_pos, np.zeros((rna_pos.shape[0], 1), dtype=np.int64)],
            axis=1)
        data_corrected.at[index, 'RNA_pos'] = rna_pos

    return data_corrected
