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
Version: 1.0 and 1.1 (see github https://github.com/DeepScale/SqueezeNet)
"""

import os

import tensorflow as tf
import numpy as np

from .base import BaseModel, get_optimizer

from tensorflow.python.keras.backend import function, learning_phase
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import (Conv2D, Concatenate, MaxPooling2D,
                                            Dropout, GlobalAveragePooling2D,
                                            Add, Input, Activation,
                                            ZeroPadding2D, BatchNormalization)


# TODO add logging routines
# TODO add cache routines
# TODO manage multiprocessing
# TODO improve logging
# TODO use last version of the model
# ### 2D models ###

class SqueezeNet0(BaseModel):
    # TODO add documentation

    def __init__(self, nb_classes, bypass=False, optimizer="adam",
                 logdir=None):
        # get model's attributes
        super().__init__()
        self.nb_classes = nb_classes
        self.logdir = logdir

        # initialize model
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
        self.model = None
        self.trained = False
        self.history = None

        # build model
        self._build_model(bypass, optimizer)

    def fit(self, train_data, train_label, validation_data, validation_label,
            batch_size, nb_epochs):
        # TODO exploit 'sample_weight'
        # TODO implement resumed training with 'initial_epoch'
        # TODO add documentation

        callbacks = []

        # define checkpoints
        if self.logdir is not None:
            # create checkpoint callback
            checkpoint_path = os.path.join(self.logdir, "cp-{epoch}.ckpt")
            cp_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1)
            callbacks.append(cp_callback)

        # TODO debug early stopping
        # define early stopping
        early_stop = EarlyStopping(
            monitor="val_categorical_accuracy",
            min_delta=0,
            patience=5,
            verbose=2)
        callbacks.append(early_stop)

        # fit model
        self.history = self.model.fit(
            x=train_data,
            y=train_label,
            batch_size=batch_size,
            epochs=nb_epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=(validation_data, validation_label),
            shuffle=True,
            sample_weight=None,
            initial_epoch=0)

        # update model attribute
        self.trained = True

        return

    def fit_generator(self, train_generator, validation_generator, nb_epochs,
                      nb_workers=1, multiprocessing=False):
        # TODO implement multiprocessing
        # TODO exploit an equivalent of 'sample_weight'
        # TODO implement resumed training with 'initial_epoch'
        # TODO add documentation
        # TODO check distribution strategy during compilation
        # TODO check callbacks parameters
        # check generators
        if train_generator.nb_epoch_max is not None:
            Warning("Train generator must loop indefinitely over the data. "
                    "The parameter 'nb_epoch_max' is set to None.")
            train_generator.nb_epoch_max = None
        if validation_generator.nb_epoch_max is not None:
            Warning("Validation generator must loop indefinitely over the "
                    "data. The parameter 'nb_epoch_max' is set to None.")
            validation_generator.nb_epoch_max = None

        callbacks = []

        # define checkpoints
        if self.logdir is not None:
            # create checkpoint callback
            checkpoint_path = os.path.join(self.logdir, "cp-{epoch}.ckpt")
            cp_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1)
            callbacks.append(cp_callback)

        # define early stopping
        early_stop = EarlyStopping(
            monitor='val_categorical_accuracy',
            min_delta=0,
            patience=5,
            verbose=2)
        callbacks.append(early_stop)

        # fit model from generator
        steps_per_epoch = train_generator.nb_batch_per_epoch
        self.history = self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=validation_generator.nb_batch_per_epoch,
            max_queue_size=10,
            workers=nb_workers,
            use_multiprocessing=multiprocessing,
            initial_epoch=0)

        # update model attribute
        self.trained = True

        return

    def predict(self, data, return_probability=False):
        # compute probabilities
        probability = self.predict_probability(data=data)

        # make prediction
        prediction = np.argmax(probability, axis=-1)

        if return_probability:
            return prediction, probability
        else:
            return prediction

    def predict_probability(self, data):
        # compute probabilities
        probability = self.model.predict(x=data)

        return probability

    def predict_generator(self, generator, return_probability=False,
                          nb_workers=1, multiprocessing=False, verbose=0):
        # compute probabilities
        probability = self.predict_probability_generator(
            generator=generator,
            nb_workers=nb_workers,
            multiprocessing=multiprocessing,
            verbose=verbose)

        # make prediction
        prediction = np.argmax(probability, axis=-1)

        if return_probability:
            return prediction, probability
        else:
            return prediction

    def predict_probability_generator(self, generator, nb_workers=1,
                                      multiprocessing=False, verbose=0):
        # TODO add multiprocessing
        # compute probabilities
        probability = self.model.predict_generator(
            generator=generator,
            steps=generator.nb_batch_per_epoch,
            workers=nb_workers,
            max_queue_size=1,
            use_multiprocessing=multiprocessing,
            verbose=verbose)

        return probability

    def evaluate(self, data, label, verbose=0):
        # evaluate model
        loss, accuracy = self.model.evaluate(x=data, y=label)
        if verbose > 0:
            print("Loss: {0:.3f} | Accuracy: {1:.3f}"
                  .format(loss, 100 * accuracy))

        return loss, accuracy

    def evaluate_generator(self, generator, nb_workers=1,
                           multiprocessing=False, verbose=0):
        # TODO check the outcome 'loss' and 'accuracy'
        # evaluate model
        loss, accuracy = self.model.evaluate_generator(
            generator=generator,
            steps=generator.nb_batch_per_epoch,
            workers=nb_workers,
            max_queue_size=1,
            use_multiprocessing=multiprocessing,
            verbose=verbose)
        if verbose > 0:
            print("Loss: {0:.3f} | Accuracy: {1:.3f}"
                  .format(loss, 100 * accuracy))

        return loss, accuracy

    def _build_model(self, bypass, optimizer):
        # build model architecture
        input_ = Input(shape=(224, 224, 3),
                       name="input",
                       dtype="float32")
        logit_ = squeezenet_network_v0(input_tensor=input_,
                                       nb_classes=self.nb_classes,
                                       bypass=bypass)
        output_ = squeezenet_classifier(logit=logit_)

        self.model = Model(inputs=input_,
                           outputs=output_,
                           name="SqueezeNet_v0")

        # get optimizer
        self.optimizer = get_optimizer(optimizer_name=optimizer)

        # compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"])

    def print_model(self):
        print(self.model.summary(), "\n")

    def get_weight(self, latest=True, checkpoint_name="cp.ckpt"):
        # TODO fix the loose of the optimizer state
        # load weights from a training checkpoint if it exists
        if self.logdir is not None and os.path.isdir(self.logdir):
            # the last one...
            if latest:
                checkpoint_path = tf.train.latest_checkpoint(self.logdir)
            # ...or a specific one
            else:
                checkpoint_path = os.path.join(self.logdir, checkpoint_name)

            # load weights
            self.model.load_weights(checkpoint_path)
            self.trained = True

        else:
            raise ValueError("Impossible to load pre-trained weights. The log "
                             "directory is not specified or does not exist.")

    def save_training_history(self):
        """Save the loss and accuracy of the train and validation data over
        the different epochs.

        Returns
        -------

        """
        if self.logdir is not None:
            path = os.path.join(self.logdir, "history.npz")
            np.savez(path,
                     loss=self.history.history["loss"],
                     categorical_accuracy=self.history.history["loss"],
                     val_loss=self.history.history["loss"],
                     val_categorical_accuracy=self.history.history["loss"])

        return

    def get_feature_map(self, generator, after_average_pooling=True):
        # TODO add documentation
        # get input layer
        input_ = self.model.input

        # get embedding layer
        if after_average_pooling:
            output_ = self.model.layers[-2].output
        else:
            output_ = self.model.layers[-3].output

        # define the steps to compute the feature map
        features_map = function([input_, learning_phase()], [output_])

        # compute the feature map
        if generator.with_label:
            embedding = [features_map([batch, 0])[0]
                         for (batch, _) in generator]
        else:
            embedding = [features_map([batch, 0])[0]
                         for batch in generator]
        embedding = np.array(embedding)
        embedding = np.concatenate(embedding, axis=0)

        if not after_average_pooling:
            a, b, c, d = embedding.shape
            embedding = np.reshape(embedding, (a, b * c * d))

        return embedding


# ### Architecture functions ###

def squeezenet_network_v0(input_tensor, nb_classes, bypass=False):
    """Original architecture of the network.

    Parameters
    ----------
    input_tensor : Keras tensor, float32
        Input tensor with shape (batch_size, 224, 224, 3).
    nb_classes : int
        Number of final classes.
    bypass : bool
        Use residual bypasses.

    Returns
    -------
    tensor : Keras tensor, float32
        Output tensor with shape (batch_size, nb_classes)

    """
    # first convolution block
    padding1 = ZeroPadding2D(
        padding=((2, 2), (2, 2)),
        name="padding1")(
        input_tensor)  # (batch_size, 228, 228, 3)
    conv1 = Conv2D(
        filters=96,
        kernel_size=(7, 7),
        strides=(2, 2),
        activation='relu',
        name='conv1')(
        padding1)  # (batch_size, 111, 111, 96)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool1")(
        conv1)  # (batch_size, 55, 55, 96)

    # fire modules
    fire2 = fire_module(
        input_tensor=maxpool1,
        nb_filters_s1x1=16,
        nb_filters_e1x1=64,
        nb_filters_e3x3=64,
        name="fire2")  # (batch_size, 55, 55, 128)
    fire3 = fire_module(
        input_tensor=fire2,
        nb_filters_s1x1=16,
        nb_filters_e1x1=64,
        nb_filters_e3x3=64,
        name="fire3")  # (batch_size, 55, 55, 128)
    if bypass:
        fire3 = Add()([fire2, fire3])
    fire4 = fire_module(
        input_tensor=fire3,
        nb_filters_s1x1=32,
        nb_filters_e1x1=128,
        nb_filters_e3x3=128,
        name="fire4")  # (batch_size, 55, 55, 256)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool4")(
        fire4)  # (batch_size, 27, 27, 256)
    fire5 = fire_module(
        input_tensor=maxpool4,
        nb_filters_s1x1=32,
        nb_filters_e1x1=128,
        nb_filters_e3x3=128,
        name="fire5")  # (batch_size, 27, 27, 256)
    if bypass:
        fire5 = Add()([maxpool4, fire5])
    fire6 = fire_module(
        input_tensor=fire5,
        nb_filters_s1x1=48,
        nb_filters_e1x1=192,
        nb_filters_e3x3=192,
        name="fire6")  # (batch_size, 27, 27, 384)
    fire7 = fire_module(
        input_tensor=fire6,
        nb_filters_s1x1=48,
        nb_filters_e1x1=192,
        nb_filters_e3x3=192,
        name="fire7")  # (batch_size, 27, 27, 384)
    if bypass:
        fire7 = Add()([fire6, fire7])
    fire8 = fire_module(
        input_tensor=fire7,
        nb_filters_s1x1=64,
        nb_filters_e1x1=256,
        nb_filters_e3x3=256,
        name="fire8")  # (batch_size, 27, 27, 512)
    maxpool8 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool3")(
        fire8)  # (batch_size, 13, 13, 512)
    fire9 = fire_module(
        input_tensor=maxpool8,
        nb_filters_s1x1=64,
        nb_filters_e1x1=256,
        nb_filters_e3x3=256,
        name="fire9")  # (batch_size, 13, 13, 512)
    if bypass:
        fire9 = Add()([maxpool8, fire9])

    # last convolution block
    dropout10 = Dropout(
        rate=0.5,
        name="dropout10")(
        fire9)
    conv10 = Conv2D(
        filters=nb_classes,
        kernel_size=(1, 1),
        activation='relu',
        name='conv10')(
        dropout10)  # (batch_size, 13, 13, nb_classes)
    norm10 = BatchNormalization(
        name="batchnorm10")(
        conv10)  # (batch_size, 13, 13, nb_classes)
    avgpool10 = GlobalAveragePooling2D(
        name="avgpool10")(
        norm10)  # (batch_size, nb_classes)

    return avgpool10


def squeezenet_network_v1(input_tensor, nb_classes, bypass=False):
    """A lighter architecture of the network.

    Parameters
    ----------
    input_tensor : Keras tensor, float32
        Input tensor with shape (batch_size, 224, 224, 3).
    nb_classes : int
        Number of final classes.
    bypass : bool
        Use residual bypasses.

    Returns
    -------
    tensor : Keras tensor, float32
        Output tensor with shape (batch_size, nb_classes)

    """
    # first convolution block
    conv1 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        activation='relu',
        name='conv1')(
        input_tensor)  # (batch_size, 111, 111, 64)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool1")(
        conv1)  # (batch_size, 55, 55, 64)

    # fire modules
    fire2 = fire_module(
        input_tensor=maxpool1,
        nb_filters_s1x1=16,
        nb_filters_e1x1=64,
        nb_filters_e3x3=64,
        name="fire2")  # (batch_size, 55, 55, 128)
    fire3 = fire_module(
        input_tensor=fire2,
        nb_filters_s1x1=16,
        nb_filters_e1x1=64,
        nb_filters_e3x3=64,
        name="fire3")  # (batch_size, 55, 55, 128)
    if bypass:
        fire3 = Add()([fire2, fire3])
    maxpool3 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool3")(
        fire3)  # (batch_size, 27, 27, 128)
    fire4 = fire_module(
        input_tensor=maxpool3,
        nb_filters_s1x1=32,
        nb_filters_e1x1=128,
        nb_filters_e3x3=128,
        name="fire4")  # (batch_size, 27, 27, 256)
    fire5 = fire_module(
        input_tensor=fire4,
        nb_filters_s1x1=32,
        nb_filters_e1x1=128,
        nb_filters_e3x3=128,
        name="fire5")  # (batch_size, 27, 27, 256)
    if bypass:
        fire5 = Add()([fire4, fire5])
    maxpool5 = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        name="maxpool5")(
        fire5)  # (batch_size, 13, 13, 256)
    fire6 = fire_module(
        input_tensor=maxpool5,
        nb_filters_s1x1=48,
        nb_filters_e1x1=192,
        nb_filters_e3x3=192,
        name="fire6")  # (batch_size, 13, 13, 384)
    fire7 = fire_module(
        input_tensor=fire6,
        nb_filters_s1x1=48,
        nb_filters_e1x1=192,
        nb_filters_e3x3=192,
        name="fire7")  # (batch_size, 13, 13, 384)
    if bypass:
        fire7 = Add()([fire6, fire7])
    fire8 = fire_module(
        input_tensor=fire7,
        nb_filters_s1x1=64,
        nb_filters_e1x1=256,
        nb_filters_e3x3=256,
        name="fire8")  # (batch_size, 13, 13, 512)
    fire9 = fire_module(
        input_tensor=fire8,
        nb_filters_s1x1=64,
        nb_filters_e1x1=256,
        nb_filters_e3x3=256,
        name="fire9")  # (batch_size, 13, 13, 512)
    if bypass:
        fire9 = Add()([fire8, fire9])

    # last convolution block
    dropout10 = Dropout(
        rate=0.5,
        name="dropout10")(
        fire9)
    conv10 = Conv2D(
        filters=nb_classes,
        kernel_size=(1, 1),
        activation='relu',
        name='conv10')(
        dropout10)  # (batch_size, 13, 13, nb_classes)
    avgpool10 = GlobalAveragePooling2D(
        name="avgpool10")(
        conv10)  # (batch_size, nb_classes)

    return avgpool10


def fire_module(input_tensor, nb_filters_s1x1, nb_filters_e1x1,
                nb_filters_e3x3, name):
    """Fire module.

    A first convolution 2-d 1x1 reduces the depth of the input tensor (squeeze
    layer). It then allows us to 1) replace 3x3 filters by 1x1 filters and 2)
    decrease the number of input channels to 3x3 filters (expand layer). To
    define a convolution step with different kernel size (1x1 and 3x3), we use
    two different convolution layers, then we concatenate their results along
    the channel dimension (output layer).

    Parameters
    ----------
    input_tensor : Keras tensor, float32
        Input tensor with shape (batch_size, height, width, channels).
    nb_filters_s1x1 : int
        Number of filters of the squeeze layer (1x1 Conv2D).
    nb_filters_e1x1 : int
        Number of filters of the expand layer (1x1 Conv2D).
    nb_filters_e3x3 : int
        Number of filters of the expand layer (3x3 Conv2D).
    name : str
        Name of these layers.

    Returns
    -------
    output_layer : Keras tensor, float32
        Output tensor with shape
        (batch_size, height, width, nb_filters_e1x1 + nb_filters_e3x3)).

    """
    # squeeze layer
    squeeze_layer = Conv2D(
        filters=nb_filters_s1x1,
        kernel_size=(1, 1),
        activation="relu",
        name="{0}_s1x1".format(name))(
        input_tensor)

    # expand layer
    expand_layer_1x1 = Conv2D(
        filters=nb_filters_e1x1,
        kernel_size=(1, 1),
        activation="relu",
        name="{0}_e1x1".format(name))(
        squeeze_layer)
    expand_layer_3x3 = Conv2D(
        filters=nb_filters_e3x3,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="{0}_e3x3".format(name))(
        squeeze_layer)

    # output layer
    output_layer = Concatenate(
        axis=-1,
        name="{0}_output".format(name))(
        [expand_layer_1x1, expand_layer_3x3])

    return output_layer


def squeezenet_classifier(logit):
    """Normalized logit using softmax function.

    Parameters
    ----------
    logit : Keras tensor, float32
        Output layer of the network.

    Returns
    -------
    normalized_logit : Keras tensor, float32
        Normalized output of the network, between 0 and 1.

    """
    # softmax
    normalized_logit = Activation(activation="softmax", name="softmax")(logit)

    return normalized_logit


#from keras import backend as K
#import numpy as np


#nS = 100 # number of Monte Carlo samples
#MC_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
#learning_phase = True  # use dropout at test time
#MC_samples = [MC_output([x_test, learning_phase])[0] for _ in range(nS)]
#MC_samples = np.array(MC_samples)
## print(MC_samples.shape)

#predictions = np.mean(MC_samples,axis=0)
#y_preds = np.argmax(predictions, axis=1)
#nberr_S = np.where(y_preds != y_test, 1.0, 0.0).sum()
#print("nb errors MC dropout="+str(nberr_S))

#np.save("MC_samples_dropout", MC_samples)