# -*- coding: utf-8 -*-

"""
Models based on SqueezeNet.

Paper: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
        and <0.5MB model size"
Authors: Iandola, Forrest N
         Han, Song
         Moskewicz, Matthew W
         Ashraf, Khalid
         Dally, William J
         Keutzer, Kurt
Year: 2016
Version: 1.1 (see github https://github.com/DeepScale/SqueezeNet)
"""

from .base import BaseModel

import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D, Concatenate


# ### 2D models ###

class SqueezeNet(BaseModel):

    def __init__(self):
        super().__init__()
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


# ### Functions ###

def squeezenet_network(input_tensor):

    # first convolution block

    tensor = Conv2D(
        filters=96,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='valid',
        activation='relu',
        name='conv_0')(
        input_)

    # fire modules

    fire_module(input_tensor, nb_filters_squeeze, nb_filters_expand_1x1,
                nb_filters_expand_3x3, name)

    # last convolution block



    return


__init__(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)


def fire_module(input_tensor, nb_filters_squeeze, nb_filters_expand_1x1,
                nb_filters_expand_3x3, name):
    """Fire module.

    A first convolution 2-d 1x1 reduces the depth of the input tensor (squeeze
    layer). It then allows us to 1) replace 3x3 filters by 1x1 filters and 2)
    decrease the number of input channels to 3x3 filters (expand layer). To
    define a convolution step with different kernel size (1x1 and 3x3), we use
    two different convolution layers, then we concatenate their results along
    the channel dimension (output layer).

    Parameters
    ----------
    input_tensor :
        Input tensor with shape (batch_size, height, width, channels).
    nb_filters_squeeze : int
        Number of filters of the squeeze layer (1x1 Conv2D).
    nb_filters_expand_1x1 : int
        Number of filters of the expand layer (1x1 Conv2D).
    nb_filters_expand_3x3 : int
        Number of filters of the expand layer (3x3 Conv2D).
    name : str
        Name of these layers.

    Returns
    -------
    output_layer :
        Output tensor with shape (batch_size, height, width,
        nb_filters_expand_1x1 + nb_filters_expand_3x3)).

    """
    # squeeze layer to reduce depth
    squeeze_layer = Conv2D(
        filters=nb_filters_squeeze,
        kernel_size=(1, 1),
        activation="relu",
        name="{0}_squeeze_layer".format(name))(
        input_tensor)

    # expand layer
    expand_layer_1x1 = Conv2D(
        filters=nb_filters_expand_1x1,
        kernel_size=(1, 1),
        activation="relu",
        name="{0}_expand_layer_1x1".format(name))(
        squeeze_layer)
    expand_layer_3x3 = Conv2D(
        filters=nb_filters_expand_3x3,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="{0}_expand_layer_3x3".format(name))(
        squeeze_layer)

    # output layer
    output_layer = Concatenate(
        axis=-1,
        name="{0}_output_layer".format(name))(
        [expand_layer_1x1, expand_layer_3x3])

    return output_layer




def SqueezeNetOutput(input_, num_classes=4, bypass=None):
    valid = [None, 'simple', 'complex']
    if bypass not in valid:
        raise UserWarning('"bypass" argument must be one of %s.' % ', '.join(map(str, valid)))

    conv_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv_0', activation='relu')(input_)
    mxp_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_0')(conv_0)

    # Block 1
    fm_2 = fire_module(id=2, squeeze=16, expand=64)(mxp_0)
    fm_3 = fire_module(id=3, squeeze=16, expand=64)(fm_2)
    input_fm_4_ = fm_3
    if bypass == 'simple':
        input_fm_4_ = Add()([fm_2, fm_3])
    fm_4 = fire_module(id=4, squeeze=32, expand=128)(input_fm_4_)
    mxp_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_1')(fm_4)

    # Block 2
    fm_5 = fire_module(id=5, squeeze=32, expand=128)(mxp_1)
    input_fm_6_ = fm_5
    if bypass == 'simple':
        input_fm_6_ = Add()([mxp_1, fm_5])
    fm_6 = fire_module(id=6, squeeze=48, expand=192)(input_fm_6_)
    fm_7 = fire_module(id=7, squeeze=48, expand=192)(fm_6)
    input_fm_8_ = fm_7
    if bypass == 'simple':
        input_fm_8_ = Add()([fm_6, fm_7])
    fm_8 = fire_module(id=8, squeeze=64, expand=256)(input_fm_8_)
    mxp_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool_2')(fm_8)

    # Block 3
    fm_9 = fire_module(id=9, squeeze=64, expand=256)(mxp_2)
    input_conv_10_ = fm_9
    if bypass == 'simple':
        input_conv_10_ = Add()([mxp_2, fm_9])
    # embedding = GlobalAveragePooling2D(name='embedding_layer')(input_conv_10_)
    dropped = Dropout(0.5, name='Dropout')(input_conv_10_)
    conv_10 = Conv2D(num_classes, (1, 1), padding='valid', name='conv10', activation='relu')(dropped)
    normalized = BatchNormalization(name='batch_normalization')(conv_10)

    # Predictions
    avgp_0 = GlobalAveragePooling2D(name='globalaveragepooling')(normalized)
    probas = Activation('softmax', name='probabilities')(avgp_0)

    return probas


input_ = Input(shape=next(train_generator)[0].shape[1:])
        if self.model.lower() == 'squeezenet':
            output_ = SqueezeNetOutput(input_, num_classes, bypass='simple')


model = Model(input_, output_, name=self.model)
# model = multi_gpu_model(model, gpus=len(
#                 gpus), cpu_merge=False, cpu_relocation=False)

adam = Adam(lr=1e-4)
logdir = LOCAL + self.logdir

if not os.path.exists(logdir):
    os.makedirs(logdir)
else:
    try:
        print('Picking up checkpoint')
        model.load_weights(logdir + '/model-ckpt')
    except OSError:
        pass



        model.compile(loss='categorical_crossentropy' if self.model != 'ae' else 'binary_crossentropy',
                      optimizer=adam,
                      metrics=['acc'],
                      options=run_options,
                      run_metadata=run_metadata
                      )

        # Fit on generator
        # with K.tf.device('/gpu:0'):
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_dataset.shape[0] // BATCH_SIZE,
            callbacks=[tb, checkpointer, reduce_lr, earl],
            validation_data=test_generator,
            validation_steps=test_dataset.shape[0] // BATCH_SIZE,
            epochs=50,
            verbose=1,
            max_queue_size=5,
            workers=1,
            use_multiprocessing=False,
            class_weight=class_weights
        )