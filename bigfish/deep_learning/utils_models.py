# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Segmentation models (utility functions).
"""

import tensorflow_addons as tfa

from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.engine.training import Model


# ### Convolution blocks ###

class SameConv(Model):

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 normalization=True,
                 activation="relu",
                 **kwargs):
        super(SameConv, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation

        # define layers
        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same')
        if self.normalization:
            self.norm = tfa.layers.InstanceNormalization()
        else:
            self.norm = None
        self.act = Activation(
            activation=self.activation)

    def call(self, inputs, training=False, mask=None):
        # compute layers
        x = self.conv(inputs)
        if self.normalization:
            x = self.norm(x)
        x = self.act(x)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation}

        return config


class UpConv(Model):

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 normalization=True,
                 activation="relu",
                 interpolation="bilinear",
                 **kwargs):
        super(UpConv, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        self.interpolation = interpolation

        # define layers
        self.resize = UpSampling2D(
            size=(2, 2),
            interpolation=self.interpolation)
        self.conv = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)

    def call(self, inputs, training=False, mask=None):
        # resize and convolve image
        x = self.resize(inputs)
        x = self.conv(x)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation,
            'interpolation': self.interpolation}

        return config


class DownBlock(Model):

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 normalization=True,
                 activation="relu",
                 pool_size=(2, 2),
                 **kwargs):
        super(DownBlock, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        self.pool_size = pool_size

        # define layers
        self.conv_1 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)
        self.conv_2 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)
        self.residual = SameConv(
            filters=self.filters,
            kernel_size=(1, 1),
            normalization=self.normalization,
            activation=self.activation)
        self.add = Add()
        self.conv_3 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)
        self.conv_4 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)
        self.pool = MaxPooling2D(
            pool_size=self.pool_size)

    def call(self, inputs, training=False, mask=None):
        # two convolution layers
        x = self.conv_1(inputs)
        x = self.conv_2(x)

        # residual connection
        residual = self.residual(inputs)
        residual = self.add([x, residual])

        # two convolution layers
        x = self.conv_3(residual)
        x = self.conv_4(x)

        # pooling layer
        x = self.pool(x)

        return x, residual

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation,
            'pool_size': self.pool_size}

        return config


class UpBlock(Model):

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 normalization=True,
                 activation="relu",
                 interpolation="bilinear",
                 **kwargs):
        super(UpBlock, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        self.interpolation = interpolation

        # define layers
        self.upsampling = UpConv(
            self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation,
            interpolation=self.interpolation)
        self.add_1 = Add()
        self.conv_1 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)
        self.conv_2 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)
        self.add_2 = Add()
        self.conv_3 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)
        self.conv_4 = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)

    def call(self, inputs, training=False, mask=None):
        # get inputs
        x, residual = inputs

        # upsampling
        x = self.upsampling(x)

        # residual connection
        residual = self.add_1([x, residual])

        # two convolution layers
        x = self.conv_1(residual)
        x = self.conv_2(x)

        # residual connection
        x = self.add_2([x, residual])

        # two convolution layers
        x = self.conv_3(x)
        x = self.conv_4(x)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation,
            'interpolation': self.interpolation}

        return config


class Encoder(Model):

    def __init__(self,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # define layers
        self.down_block_1 = DownBlock(filters=32)
        self.down_block_2 = DownBlock(filters=64)
        self.down_block_3 = DownBlock(filters=128)
        self.down_block_4 = DownBlock(filters=256)
        self.conv_1 = SameConv(filters=256)
        self.conv_2 = SameConv(filters=256)

    def call(self, inputs, training=False, mask=None):
        # (batch, H/2, W/2, 32), (batch, H, W, 32)
        x, residual_1 = self.down_block_1(inputs)

        # (batch, H/4, W/4, 64), (batch, H/2, W/2, 64)
        x, residual_2 = self.down_block_2(x)

        # (batch, H/8, W/8, 128), (batch, H/4, W/4, 128)
        x, residual_3 = self.down_block_3(x)

        # (batch, H/16, W/16, 256), (batch, H/8, W/8, 256)
        x, residual_4 = self.down_block_4(x)

        # bottom
        x = self.conv_1(x)
        x = self.conv_2(x)  # (batch, H/16, W/16, 256)

        return x, residual_1, residual_2, residual_3, residual_4

    def get_config(self):
        config = dict()

        return config


class Decoder(Model):

    def __init__(self,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        # define layers
        self.up_block_4 = UpBlock(filters=256)
        self.up_block_3 = UpBlock(filters=128)
        self.up_block_2 = UpBlock(filters=64)
        self.up_block_1 = UpBlock(filters=32)

    def call(self, inputs, training=False, mask=None):
        # get inputs
        x, residual_1, residual_2, residual_3, residual_4 = inputs

        # expansions
        x = self.up_block_4([x, residual_4])  # (batch, H/8, W/8, 256)
        x = self.up_block_3([x, residual_3])  # (batch, H/4, W/4, 128)
        x = self.up_block_2([x, residual_2])  # (batch, H/2, W/2, 64)
        x = self.up_block_1([x, residual_1])  # (batch, H, W, 32)

        return x

    def get_config(self):
        config = dict()

        return config


class EncoderDecoder(Model):

    def __init__(self,
                 **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)

        # initialize parameters
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, training=False, mask=None):
        # encode
        (x, residual_1, residual_2,
         residual_3, residual_4) = self.encoder(inputs)

        # decode
        x = self.decoder([x, residual_1, residual_2, residual_3, residual_4])

        return x

    def get_config(self):
        config = dict()

        return config
