# -*- coding: utf-8 -*-

"""
Functions to prepare the data before feeding a model.
"""

import threading
import numpy as np

from .preprocess import (cast_img_uint8, cast_img_uint16, cast_img_float32,
                         cast_img_float64)
from .augmentation import augment
from .utils import check_array

from skimage.transform import resize
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder

from scipy import ndimage as ndi


# TODO define the requirements for 'data'

# ### Split and subset data ###

def split_from_background(data, p_validation=0.2, p_test=0.2):
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
    data_train.reset_index(drop=True, inplace=True)
    data_validation = data.query("cell_ID in {0}".format(str(validation_cell)))
    data_validation.reset_index(drop=True, inplace=True)
    data_test = data.query("cell_ID in {0}".format(str(test_cell)))
    data_test.reset_index(drop=True, inplace=True)

    return data_train, data_validation, data_test


def subset_data(data, classes_name=None):
    # choose classes to keep
    if classes_name is None:
        classes_name = list(set(data["pattern_name"]))

    # keep specific classes
    query = "pattern_name in {0}".format(str(classes_name))
    data = data.query(query)

    # encode the label
    le = LabelEncoder()
    data = data.assign(label=le.fit_transform(data["pattern_name"]))

    return data


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

    if augmentation:
        image = augment(image)

    return image


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
    # compute distances from cytoplasm and nucleus
    mask_cyt = ndi.binary_fill_holes(cyt)
    mask_nuc = ndi.binary_fill_holes(nuc)
    distance_cyt = ndi.distance_transform_edt(ndi.binary_fill_holes(cyt))
    distance_nuc_ = ndi.distance_transform_edt(~mask_nuc)
    distance_nuc = mask_cyt * distance_nuc_

    # cast to np.float32 and normalize it between 0 and 1
    distance_cyt = cast_img_float32(distance_cyt / distance_cyt.max())
    distance_nuc = cast_img_float32(distance_nuc / distance_nuc.max())

    return distance_cyt, distance_nuc


def get_surface_layers(cyt, nuc):
    """Compute plain surface layers as input for the model.

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

class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    https://gist.github.com/platdrag/e755f3947552804c42633a99ffd325d4
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


class Generator:

    # TODO add documentation
    # TODO check threading.Lock()
    # TODO add classes
    def __init__(self, data, method, batch_size, input_shape, augmentation,
                 with_label, nb_classes, nb_epoch_max=10, shuffle=True):
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

        # initialize generator
        self.nb_samples = self.data.shape[0]
        self.indices = self._get_shuffled_indices()
        self.nb_batch_per_epoch = self._get_batch_per_epoch()
        self.i_batch = 0
        self.i_epoch = 0

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

            # the generator loop over the data indefinitely
            if self.nb_epoch_max is None:
                if self.i_epoch == 500:
                    raise StopIteration
                self.i_epoch += 1
                self.i_batch = 0
                self.indices = self._get_shuffled_indices()
                return self._next()

            # we start a new epoch
            elif (self.nb_epoch_max is not None
                  and self.i_epoch < self.nb_epoch_max):
                self.i_epoch += 1
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
                nb_classes=self.nb_classes)

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
                nb_classes=self.nb_classes)

            return batch_data

    def reset(self):
        # initialize generator
        self.indices = self._get_shuffled_indices()
        self.nb_batch_per_epoch = self._get_batch_per_epoch()
        self.i_batch = 0
        self.i_epoch = 0


def generate_images(data, method, batch_size, input_shape, augmentation,
                    with_label, nb_classes):
    """Generate batches of images.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the data.
    method : str
        Channels used in the input image.
            - 'normal' for (rna, cyt, nuc)
            - 'distance' for (rna, distance_cyt, distance_nuc)
            - 'surface' for (rna, surface_cyt, surface_nuc)
    batch_size : int
        Size of the batch.
    input_shape : Tuple[int]
        Shape of the input image.
    augmentation : bool
        Apply a random operator on the image.
    with_label : bool
        Return label of the image as well.
    nb_classes : int
        Number of different classes available.

    Returns
    -------
    batch_data: np.ndarray, np.float32
        Tensor with shape (batch_size, x, y, 3).
    batch_label : np.ndarray, np.int64
        Tensor of the encoded label, with shape (batch_size,)

    """
    # TODO make it loop indefinitely
    # shuffle input data and get their indices
    input_indices_ordered = list(data.index)
    np.random.shuffle(input_indices_ordered)
    nb_samples = len(input_indices_ordered)

    # compute the number of batches to generate for the entire epoch
    if nb_samples % batch_size == 0:
        nb_batch = len(input_indices_ordered) // batch_size
    else:
        # the last batch can be smaller
        nb_batch = (len(input_indices_ordered) // batch_size) + 1

    # build batches
    for i_batch in range(nb_batch):
        start_index = i_batch * batch_size
        end_index = min((i_batch + 1) * batch_size, nb_samples)
        indices_batch = input_indices_ordered[start_index:end_index]

        # return batch with label
        if with_label:
            batch_data, batch_label = build_batch(data, indices_batch, method,
                                                  input_shape, augmentation,
                                                  with_label, nb_classes)

            yield batch_data, batch_label

        # return batch without label
        else:
            batch_data = build_batch(data, indices_batch, method, input_shape,
                                     augmentation, with_label, nb_classes)

            yield batch_data


def build_batch(data, indices, method="normal", input_shape=(224, 244),
                augmentation=True, with_label=False, nb_classes=9):
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

    Returns
    -------
    batch_data : np.ndarray, np.float32
        Tensor with shape (batch_size, x, y, 3).
    batch_label : np.ndarray, np.int64
        Tensor of the encoded label, with shape (batch_size,)

    """
    # TODO try to fully vectorize this step
    # initialize the batch
    batch_size = len(indices)
    batch_data = np.zeros((batch_size, input_shape[0], input_shape[1], 3),
                          dtype=np.float32)

    # build each input image of the batch
    for i in range(batch_size):
        id_cell = indices[i]
        image = build_input_image(data, id_cell, method, input_shape,
                                  augmentation)
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
