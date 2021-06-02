# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Segmentation models.
"""

import os

import bigfish.stack as stack

import tensorflow as tf

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Softmax
from tensorflow.python.keras.engine.training import Model

from .utils_models import EncoderDecoder
from .utils_models import SameConv


# ### Pre-trained models ###

def load_pretrained_model(model_name, channel):
    """Build and compile a model, then load its pretrained weights.

    Parameters
    ----------
    model_name : str
        Name of the model used ('3-classes' or 'distance_edge').
    channel : str
        Input channel for the model ('nuc' or 'cell').

    Returns
    -------
    model : tensorflow.keras.model object
        Pretrained Unet model.

    """
    # check parameters
    stack.check_parameter(model_name=str,
                          channel=str)

    # unet 3-classes for nucleus segmentation
    if model_name == "3_classes" and channel == "nuc":
        model = build_compile_3_classes_model()

    # unet 3-classes for cell segmentation
    elif model_name == "3_classes" and channel == "cell":
        model = build_compile_3_classes_model()

    # unet distance map to edge for cell segmentation
    elif model_name == "distance_edge" and channel == "cell":
        model = build_compile_distance_model()

    else:
        raise ValueError("Model name and channel to segment are not "
                         "consistent: {0} - {1}.".format(model_name, channel))

    # load weights
    path_pretrained_directory = check_pretrained_weights(model_name, channel)
    path_checkpoint = os.path.join(
        path_pretrained_directory, "checkpoint")
    model.load_weights(path_checkpoint)

    return model


def check_pretrained_weights(model_name, channel):
    """Check pretrained weights exist and download them if necessary.

    Parameters
    ----------
    model_name : str
        Name of the model used ('3-classes' or 'distance_edge').
    channel : str
        Input channel for the model ('nuc' or 'cell').

    Returns
    -------
    directory_checkpoint : str
        Path of the checkpoint directory.

    """
    # check parameters
    stack.check_parameter(model_name=str,
                          channel=str)

    # get path checkpoint
    path_weights_directory = _get_weights_directory()
    pretrained_directory = "_".join([channel, model_name])
    path_directory = os.path.join(path_weights_directory, pretrained_directory)
    download = False

    # get url and hash
    if model_name == "3_classes" and channel == "nuc":
        url_checkpoint = ""
        hash_checkpoint = ""
        url_data = ""
        hash_data = ""
        url_index = ""
        hash_index = ""
    elif model_name == "3_classes" and channel == "cell":
        url_checkpoint = ""
        hash_checkpoint = ""
        url_data = ""
        hash_data = ""
        url_index = ""
        hash_index = ""
    elif model_name == "distance_edge" and channel == "cell":
        url_checkpoint = ""
        hash_checkpoint = ""
        url_data = ""
        hash_data = ""
        url_index = ""
        hash_index = ""
    else:
        raise ValueError("Model name and channel to segment are not "
                         "consistent: {0} - {1}.".format(model_name, channel))

    # case where pretrained directory exists
    if os.path.isdir(path_directory):

        # paths
        path_checkpoint = os.path.join(path_directory, "checkpoint")
        path_data = os.path.join(path_directory,
                                 "checkpoint.data-00000-of-00001")
        path_index = os.path.join(path_directory, "checkpoint.index")

        # checkpoint available and not corrupted
        if os.path.exists(path_checkpoint):
            try:
                stack.check_hash(path_checkpoint, hash_checkpoint)
            except IOError:
                print("{0} seems corrupted.".format(path_checkpoint))
                download = True
        else:
            download = True

        # data available and not corrupted
        if os.path.exists(path_data):
            try:
                stack.check_hash(path_data, hash_data)
            except IOError:
                print("{0} seems corrupted.".format(path_data))
                download = True
        else:
            download = True

        # index available and not corrupted
        if os.path.exists(path_index):
            try:
                stack.check_hash(path_index, hash_index)
            except IOError:
                print("{0} seems corrupted.".format(path_index))
                download = True
        else:
            download = True

    # case where pretrained directory does not exist
    else:
        os.mkdir(path_directory)
        download = True

    # download checkpoint files
    if download:
        print("downloading checkpoint files...")
        path = stack.load_and_save_url(
            url_checkpoint, path_directory, "checkpoint")
        stack.check_hash(path, hash_checkpoint)
        path = stack.load_and_save_url(
            url_data, path_directory, "checkpoint.data-00000-of-00001")
        stack.check_hash(path, hash_data)
        path = stack.load_and_save_url(
            url_index, path_directory, "checkpoint.index")
        stack.check_hash(path, hash_index)

    return path_directory


def _get_weights_directory():
    path = os.path.realpath(__file__)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.dirname(path)
    path = os.path.join(path, "weights")

    return path


# ### Models 3-classes ###

def build_compile_3_classes_model():
    """Build and compile a Unet model to predict 3 classes from nucleus or
    cell images: background, edge and foreground.

    Returns
    -------
    model_3_classes : tensorflow.keras.model object
        Compiled Unet model.

    """
    # define inputs
    inputs_image = Input(
        shape=(None, None, 1), dtype="float32", name="image")

    # define model
    outputs = _get_3_classes_model(inputs_image)
    model_3_classes = Model(
        inputs_image,
        outputs,
        name="3ClassesModel")

    # losses
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # metrics
    accuracy = tf.metrics.SparseCategoricalAccuracy(name="accuracy")

    # compile model
    model_3_classes.compile(
        optimizer='adam',
        loss=loss,
        metrics=accuracy)

    return model_3_classes


def _get_3_classes_model(inputs):
    """Build Unet architecture that return a 3-classes prediction.

    Parameters
    ----------
    inputs : tensorflow.keras.Input object
        Input layer with shape (B, H, W, 1) or (H, W, 1).

    Returns
    -------
    output : tensorflow.keras.layers object
        Output layer (softmax) with shape (B, H, W, 3) or (H, W, 3).

    """
    # compute feature map
    features_core = EncoderDecoder(
        name="encoder_decoder")(inputs)  # (B, H, W, 32)

    # compute 3-classes output
    features_3_classes = SameConv(
        filters=3,
        kernel_size=(1, 1),
        activation="linear",
        name="final_conv")(features_core)  # (B, H, W, 3)
    output = Softmax(
        axis=-1,
        name="label_3")(features_3_classes)  # (B, H, W, 3)

    return output


# ### Models distance map ###

def build_compile_distance_model():
    """Build and compile a Unet model to predict foreground and a distance map
    from nucleus or cell images.

    Returns
    -------
    model_distance : Tensorflow model
        Compiled Unet model.

    """
    # define inputs
    inputs_image = Input(
        shape=(None, None, 1), dtype="float32", name="image")

    # define model
    output_surface, output_distance = _get_distance_model(inputs_image)
    outputs = [output_surface, output_distance]
    model_distance = Model(
        inputs_image,
        outputs,
        name="DistanceModel")

    # losses
    loss_surface = tf.keras.losses.BinaryCrossentropy()
    loss_distance = tf.keras.losses.MeanAbsoluteError()
    losses = [[loss_surface], [loss_distance]]
    losses_weight = [[1.0], [1.0]]

    # metrics
    metric_surface = tf.metrics.BinaryAccuracy(name="accuracy")
    metric_distance = tf.metrics.MeanAbsoluteError(name="mae")
    metrics = [[metric_surface], [metric_distance]]

    # compile model
    model_distance.compile(
        optimizer='adam',
        loss=losses,
        loss_weights=losses_weight,
        metrics=metrics)

    return model_distance


def _get_distance_model(inputs):
    """Build Unet architecture that return a distance map.

    Parameters
    ----------
    inputs : tensorflow.keras.Input object
        Input layer with shape (B, H, W, 1) or (H, W, 1).

    Returns
    -------
    output_surface : tensorflow.keras.layers object
        Output layer for the foreground/background prediction with shape
        (B, H, W, 1) or (H, W, 1).
    output_distance : tensorflow.keras.layers object
        Output layer for the distance map with shape (B, H, W, 1) or (H, W, 1).

    """
    # compute feature map
    features_core = EncoderDecoder(
        name="encoder_decoder")(inputs)  # (B, H, W, 32)

    # compute surface output
    output_surface = SameConv(
        filters=1,
        kernel_size=(1, 1),
        activation="sigmoid",
        name="label_2")(features_core)  # (B, H, W, 1)

    # compute distance output
    output_distance = SameConv(
        filters=1,
        kernel_size=(1, 1),
        activation="relu",
        name="label_distance")(features_core)  # (B, H, W, 1)

    return output_surface, output_distance
