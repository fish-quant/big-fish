# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Layers and blocks of layers used for deep learning models.
"""

import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.engine.training import Model

# TODO add references (instance norm, upsampling, conv block)


# ### Convolution layers ###

class SameConv(Model):
    """Padded 2D convolution with normalization and activation.

    Parameters
    ----------
    filters : int
        The number of output filters.
    kernel_size : int or tuple, default=(3, 3)
        Height and width  of the 2D convolution window. If integer, the same
        value is applied for both spatial dimensions.
    normalization : bool, default=True
        Apply normalization (instance normalization) to the convolution output.
    activation : str, default='relu'
        Activation function applied at the end (after a potential
        normalization).

    Input shape
    -----------
    One 4D tensor with shape :
        (B, H, W, D)

    Output shape
    ------------
    One 4D tensor with shape :
        (B, H, W, filters)

    """

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
        x = self.act(x)  # (B, H, W, filters)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation}

        return config


class UpConv(Model):
    """Upsampling layer followed by a padded 2D convolution with
    normalization and activation.

    Parameters
    ----------
    filters : int
        The number of output filters.
    kernel_size : int or tuple, default=(3, 3)
        Height and width  of the 2D convolution window. If integer, the same
        value is applied for both spatial dimensions.
    normalization : bool, default=True
        Apply normalization (instance normalization) to the convolution output.
    activation : str, default='relu'
        Activation function applied at the end (after a potential
        normalization).
    up_size : int or tuple, default=(2, 2)
        Upsampling factors. If one integer, the same value is applied for both
        spatial dimensions.
    interpolation : str, default='bilinear'
        Interpolation method used to upscale.

    Input shape
    -----------
    One 4D tensor with shape :
        (B, H, W, D)

    Output shape
    ------------
    One 4D tensor with shape :
        (B, 2*H, 2*W, filters)

    """

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 normalization=True,
                 activation="relu",
                 up_size=(2, 2),
                 interpolation="bilinear",
                 **kwargs):
        super(UpConv, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        self.up_size = up_size
        self.interpolation = interpolation

        # define layers
        self.resize = UpSampling2D(
            size=self.up_size,
            interpolation=self.interpolation)
        self.conv = SameConv(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation)

    def call(self, inputs, training=False, mask=None):
        # resize and convolve image
        x = self.resize(inputs)  # (B, up_size * H, up_size * W, D)
        x = self.conv(x)  # (B, up_size * H, up_size * W, filters)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation,
            'up_size': self.up_size,
            'interpolation': self.interpolation}

        return config


# ### Convolution blocks ###

class SameBlock(Model):
    """Convolution block.

    Convolution block includes 4 padded convolutions and 1 residual connection.

    Parameters
    ----------
    filters : int
        The number of output filters.
    kernel_size : int or tuple, default=(3, 3)
        Height and width  of the 2D convolution window. If one integer, the
        same value is applied for both spatial dimensions.
    normalization : bool, default=True
        Apply normalization (instance normalization) to the convolution output.
    activation : str, default='relu'
        Activation function applied at the end (after a potential
        normalization).
    return_residual : bool, default=False
        Return in a 4D residual tensor.

    Input shape
    -----------
    One 4D tensor with shape :
        (B, H, W, D)

    Output shape
    ------------
    One or two 4D tensors with shape :
        (B, H, W, filters)
        (B, H, W, filters) if return_residual=True

    """

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 normalization=True,
                 activation="relu",
                 return_residual=False,
                 **kwargs):
        super(SameBlock, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        self.return_residual = return_residual

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

    def call(self, inputs, training=False, mask=None):
        # two convolution layers
        x = self.conv_1(inputs)  # (B, H, W, filters)
        x = self.conv_2(x)  # (B, H, W, filters)

        # residual connection
        residual = self.residual(inputs)
        residual = self.add([x, residual])  # (B, H, W, filters)

        # two convolution layers
        x = self.conv_3(residual)  # (B, H, W, filters)
        x = self.conv_4(x)  # (B, H, W, filters)

        if self.return_residual:
            return x, residual

        else:
            return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation,
            'return_residual': self.return_residual}

        return config


class DownBlock(Model):
    """Convolution block followed by a maximum pooling layer.

    Convolution block includes 4 padded convolutions and 1 residual connection.

    Parameters
    ----------
    filters : int
        The number of output filters.
    kernel_size : int or tuple, default=(3, 3)
        Height and width  of the 2-dimensional convolution window. If one
        integer, the same value is applied for both spatial dimensions.
    normalization : bool, default=True
        Apply normalization (instance normalization) to the convolution output.
    activation : str, default='relu'
        Activation function applied at the end (after a potential
        normalization).
    pool_size : int or tuple, default=(2, 2)
        Window size for the maximum pooling function. If one integer, the same
        value is applied for both spatial dimensions.

    Input shape
    -----------
    One 4D tensor with shape :
        (B, H, W, D)

    Output shape
    ------------
    Two 4D tensors with shape :
        (B, H/pool_size, W/pool_size, filters)
        (B, H, W, filters)

    """
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
        self.conv_block = SameBlock(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation,
            return_residual=True)
        self.pool = MaxPooling2D(
            pool_size=self.pool_size)

    def call(self, inputs, training=False, mask=None):
        # convolution block with residual (B, H, W, filters) (B, H, W, filters)
        x, residual = self.conv_block(inputs)

        # pooling layer
        x = self.pool(x)  # (B, H/pool_size, W/pool_size, filters)

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
    """Convolution block with an upsampling layer.

    Input signal is upsampled and add to a residual layer. Then a convolution
    block follows, with 4 padded convolutions and 1 residual connection.

    Parameters
    ----------
    filters : int
        The number of output filters.
    kernel_size : int or tuple, default=(3, 3)
        Height and width  of the 2-dimensional convolution window. If one
        integer, the same value is applied for both spatial dimensions.
    normalization : bool, default=True
        Apply normalization (instance normalization) to the convolution output.
    activation : str, default='relu'
        Activation function applied at the end (after a potential
        normalization).
    up_size : int or tuple, default=(2, 2)
        Upsampling factors. If one integer, the same value is applied for both
        spatial dimensions.
    interpolation : str, default='bilinear'
        Interpolation method used to upscale.

    Input shape
    -----------
    Two 4D tensors with shape :
        (B, H, W, D)
        (B, up_size*H, up_size*W, filters)

    Output shape
    ------------
    One 4D tensor with shape :
        (B, up_size*H, up_size*W, filters)

    """

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 normalization=True,
                 activation="relu",
                 up_size=(2, 2),
                 interpolation="bilinear",
                 **kwargs):
        super(UpBlock, self).__init__(**kwargs)

        # initialize parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.normalization = normalization
        self.activation = activation
        self.up_size = up_size
        self.interpolation = interpolation

        # define layers
        self.upsampling = UpConv(
            self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation,
            up_size=self.up_size,
            interpolation=self.interpolation)
        self.add = Add()
        self.conv_block = SameBlock(
            filters=self.filters,
            kernel_size=self.kernel_size,
            normalization=self.normalization,
            activation=self.activation,
            return_residual=False)

    def call(self, inputs, training=False, mask=None):
        # get inputs
        x, residual = inputs

        # upsampling
        x = self.upsampling(x)

        # residual connection
        residual = self.add([x, residual])

        # convolution block
        x = self.conv_block(residual)

        return x

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'normalization': self.normalization,
            'activation': self.activation,
            'up_size': self.up_size,
            'interpolation': self.interpolation}

        return config


# ### Network architectures ###

class Encoder(Model):
    """Encoder architecture.

    The encoder includes 4 convolutions blocks with maximum pooling and two
    additional convolution blocks with the smallest spatial resolutions.

    Input shape
    -----------
    One 4D tensor with shape :
        (B, H, W, D)

    Output shape
    ------------
    Five 4D tensors with shape :
        (B, H/16, W/16, 256)
        (B, H, W, 32)
        (B, H/2, W/2, 64)
        (B, H/4, W/4, 128)
        (B, H/8, W/8, 256)

    """

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
        # (B, H/2, W/2, 32), (B, H, W, 32)
        x, residual_1 = self.down_block_1(inputs)

        # (B, H/4, W/4, 64), (B, H/2, W/2, 64)
        x, residual_2 = self.down_block_2(x)

        # (B, H/8, W/8, 128), (B, H/4, W/4, 128)
        x, residual_3 = self.down_block_3(x)

        # (B, H/16, W/16, 256), (B, H/8, W/8, 256)
        x, residual_4 = self.down_block_4(x)

        # bottom
        x = self.conv_1(x)
        x = self.conv_2(x)  # (B, H/16, W/16, 256)

        return x, residual_1, residual_2, residual_3, residual_4

    def get_config(self):
        config = dict()

        return config


class Decoder(Model):
    """Decoder architecture.

    The decoder includes 4 convolutions blocks with upsampling layers.

    Input shape
    -----------
    Five 4D tensors with shape :
        (B, H/16, W/16, 256)
        (B, H, W, 32)
        (B, H/2, W/2, 64)
        (B, H/4, W/4, 128)
        (B, H/8, W/8, 256)

    Output shape
    ------------
    One 4D tensor with shape :
        (B, H, W, 32)

    """
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
        x = self.up_block_4([x, residual_4])  # (B, H/8, W/8, 256)
        x = self.up_block_3([x, residual_3])  # (B, H/4, W/4, 128)
        x = self.up_block_2([x, residual_2])  # (B, H/2, W/2, 64)
        x = self.up_block_1([x, residual_1])  # (B, H, W, 32)

        return x

    def get_config(self):
        config = dict()

        return config


class EncoderDecoder(Model):
    """Encoder-decoder architecture.

    Both encoder and decoder includes 4 convolutions blocks with respectively
    maximum pooling layers and upsampling layers. Two additional convolution
    blocks are used with the smallest spatial resolutions.

    Input shape
    -----------
    One 4D tensor with shape :
        (B, H, W, D)

    Output shape
    ------------
    One 4D tensor with shape :
        (B, H, W, 32)

    """

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
