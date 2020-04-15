# -*- coding: utf-8 -*-

"""
Models based on U-net.

Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
Authors: Ronneberger, Olaf
         Fischer, Philipp
         Brox, Thomas
Year: 2015

Page: Deconvolution and Checkerboard Artifacts
Authors: Odena, Augustus
         Dumoulin, Vincent
         Olah, Chris
Year: 2016
Link: http://doi.org/10.23915/distill.00003
"""

import os

import tensorflow as tf
import numpy as np

#from .base import BaseModel, get_optimizer

from tensorflow.python.keras.backend import function, learning_phase
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import (Conv2D, Concatenate, MaxPooling2D,
                                            Dropout, GlobalAveragePooling2D,
                                            Add, Input, Activation,
                                            ZeroPadding2D, BatchNormalization,
                                            Cropping2D)

# TODO add logging routines
# TODO add cache routines
# TODO manage multiprocessing
# TODO improve logging
# ### 2D models ###


# ### Architecture functions ###

def unet_network(input_tensor, nb_classes):
    """Original architecture of the network.

    Parameters
    ----------
    input_tensor : Keras tensor, float32
        Input tensor with shape (batch_size, ?, ?, 1).
    nb_classes : int
        Number of final classes.

    Returns
    -------
    tensor : Keras tensor, float32
        Output tensor with shape (batch_size, ?, ?, nb_classes)

    """
    # contraction 1
    conv1 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        name='conv1')(
        input_tensor)  # (batch_size, ?, ?, 64)
    conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        name='conv2')(
        conv1)  # (batch_size, ?, ?, 64)
    crop2 = Cropping2D(
        cropping=((88, 88), (88, 88)),
        name="crop2")(
        conv2)  # (batch_size, ?, ?, 64)
    maxpool2 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool2")(
        conv2)  # (batch_size, ?, ?, 64)

    # contraction 2
    conv3 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        name='conv3')(
        maxpool2)  # (batch_size, ?, ?, 128)
    conv4 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        name='conv4')(
        conv3)  # (batch_size, ?, ?, 128)
    crop4 = Cropping2D(
        cropping=((40, 40), (40, 40)),
        name="crop4")(
        conv4)  # (batch_size, ?, ?, 128)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool4")(
        conv4)  # ((batch_size, ?, ?, 128)

    # contraction 3
    conv5 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='relu',
        name='conv5')(
        maxpool4)  # (batch_size, ?, ?, 256)
    conv6 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='relu',
        name='conv6')(
        conv5)  # (batch_size, ?, ?, 256)
    crop6 = Cropping2D(
        cropping=((16, 16), (16, 16)),
        name="crop6")(
        conv6)  # (batch_size, ?, ?, 256)
    maxpool6 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool6")(
        conv6)  # (batch_size, ?, ?, 256)

    # contraction 4
    conv7 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        name='conv7')(
        maxpool6)  # (batch_size, ?, ?, 512)
    conv8 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        name='conv8')(
        conv7)  # (batch_size, ?, ?, 512)
    crop8 = Cropping2D(
        cropping=((4, 4), (4, 4)),
        name="crop8")(
        conv8)  # (batch_size, ?, ?, 512)
    maxpool8 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool8")(
        conv8)  # (batch_size, ?, ?, 512)

    # bottom
    conv9 = Conv2D(
        filters=1024,
        kernel_size=(3, 3),
        activation='relu',
        name='conv9')(
        maxpool8)  # (batch_size, ?, ?, 1024)
    conv10 = Conv2D(
        filters=1024,
        kernel_size=(3, 3),
        activation='relu',
        name='conv10')(
        conv9)  # (batch_size, ?, ?, 1024)

    # expansion 1
    upconv11 = up_conv_2d(
        input_tensor=conv10,
        nb_filters=512,
        name='upconv11')  # (batch_size, ?, ?, 512)
    concat11 = tf.concat(
        values=[crop8, upconv11],
        axis=-1,
        name='concat11')  # (batch_size, ?, ?, 1024)
    conv12 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        name='conv12')(
        concat11)  # (batch_size, ?, ?, 512)
    conv13 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        activation='relu',
        name='conv13')(
        conv12)  # (batch_size, ?, ?, 512)

    # expansion 2
    upconv14 = up_conv_2d(
        input_tensor=conv13,
        nb_filters=256,
        name='upconv14')  # (batch_size, ?, ?, 256)
    concat14 = tf.concat(
        values=[crop6, upconv14],
        axis=-1,
        name='concat14')  # (batch_size, ?, ?, 512)
    conv15 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='relu',
        name='conv15')(
        concat14)  # (batch_size, ?, ?, 256)
    conv16 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='relu',
        name='conv16')(
        conv15)  # (batch_size, ?, ?, 256)

    # expansion 3
    upconv17 = up_conv_2d(
        input_tensor=conv16,
        nb_filters=128,
        name='upconv17')  # (batch_size, ?, ?, 128)
    concat17 = tf.concat(
        values=[crop4, upconv17],
        axis=-1,
        name='concat17')  # (batch_size, ?, ?, 256)
    conv18 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        name='conv18')(
        concat17)  # (batch_size, ?, ?, 128)
    conv19 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        name='conv19')(
        conv18)  # (batch_size, ?, ?, 128)

    # expansion 4
    upconv20 = up_conv_2d(
        input_tensor=conv19,
        nb_filters=64,
        name='upconv20')  # (batch_size, ?, ?, 64)
    concat20 = tf.concat(
        values=[crop2, upconv20],
        axis=-1,
        name='concat20')  # (batch_size, ?, ?, 128)
    conv21 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        name='conv21')(
        concat20)  # (batch_size, ?, ?, 64)
    conv22 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        name='conv22')(
        conv21)  # (batch_size, ?, ?, 64)
    conv23 = Conv2D(
        filters=nb_classes,
        kernel_size=(1, 1),
        activation='sigmoid',
        name='conv23')(
        conv22)  # (batch_size, ?, ?, nb_classes)

    return conv23


#norm10 = BatchNormalization(
#        name="batchnorm10")(
#        conv10)  # (batch_size, 13, 13, nb_classes)

#dropout10 = Dropout(
#        rate=0.5,
#        name="dropout10")(
#        fire9)


def up_conv_2d(input_tensor, nb_filters, name):
    """Fire module.

    1) Tensor is resized by a factor 2 using nearest neighbors.
    2) Tensor is padded with a symmetric mode to avoid boundary artifacts.
    3) A 2-d convolution with a 3x3 filter is applied. In the original article
    the convolution has a 2x2 filter.

    Parameters
    ----------
    input_tensor : Keras tensor, float32
        Input tensor with shape (batch_size, height, width, channels).
    nb_filters : int
        Number of filters of the convolution layer.
    name : str
        Name of these layers.

    Returns
    -------
    output_layer : Keras tensor, float32
        Output tensor with shape (batch_size, 2 * height, 2 * width, channels).

    """
    resize = UpSampling2D(size=(2, 2), interpolation='nearest')(input_tensor)
    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    resize = tf.pad(resize, paddings, "SYMMETRIC")
    output_layer = Conv2D(
        filters=nb_filters,
        kernel_size=(3, 3),
        activation='relu',
        name=name)(
        resize)

    return output_layer


def get_input_size_unet(bottom_size):
    """Compute the input size required to have a specific bottom size.

    Parameters
    ----------
    bottom_size : int
        Tensor size at the bottom of the U-net model.

    Returns
    -------
    input_size : int
        Input size required to get the specified bottom size.

    """
    # compute the relation between the input size and the bottom size
    input_size = 4 + 2 * (4 + 2 * (4 + 2 * (4 + 2 * bottom_size)))

    return input_size



########################################




def depthwise_softmax(x):
    exp_tensor = K.exp(x - K.max(x, axis=-1, keepdims=True))
    # softmax_tensor = exp_tensor / K.sum(exp_tensor, axis=-1, keepdims=True)

    return exp_tensor / K.sum(exp_tensor, axis=-1, keepdims=True)


def channelwise_structure(radiuses):
    np_structure = numpy.ones(
        (2 * max(radiuses) + 1, 2 * max(radiuses) + 1, len(radiuses)))
    structures = []
    np_structure = numpy.stack([erosion(disk(radius), disk(radius)),
                                erosion(disk(radius), disk(radius)),
                                disk(radius)], axis=-1)
    structure = tf.constant(np_structure, dtype='float32')
    return structure


def binary_closing(input, structure):
    dilated = tf.nn.dilation2d(input, structure, [1, 1, 1, 1], [1, 1, 1, 1],
                               padding="SAME")

    eroded = tf.nn.erosion2d(dilated, structure, [1, 1, 1, 1], [1, 1, 1, 1],
                             padding="SAME")

    return eroded

