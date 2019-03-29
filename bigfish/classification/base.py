# -*- coding: utf-8 -*-

"""
General classes and methods to use the models.
"""

from abc import ABCMeta, abstractmethod

from tensorflow.python.keras.optimizers import (Adam, Adadelta, Adagrad,
                                                Adamax, SGD)


# ### General models ###

class BaseModel(object, metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train_data, train_label, validation_data, validation_label,
            batch_size, nb_epochs):
        pass

    @abstractmethod
    def fit_generator(self, train_generator, validation_generator, nb_epochs,
                      nb_workers=1, multiprocessing=False):
        pass

    @abstractmethod
    def predict(self, data, return_probability=False):
        pass

    @abstractmethod
    def predict_generator(self, generator, return_probability=False,
                          nb_workers=1, multiprocessing=False):
        pass

    @abstractmethod
    def predict_probability(self, data):
        pass

    @abstractmethod
    def predict_probability_generator(self, generator,
                                      nb_workers=1, multiprocessing=False):
        pass

    @abstractmethod
    def evaluate(self, data, label):
        pass

    @abstractmethod
    def evaluate_generator(self, generator, nb_workers=1,
                           multiprocessing=False):
        pass


# ### optimizer ###

def get_optimizer(optimizer_name="adam", **kwargs):
    """Instantiate the optimizer.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer to use.

    Returns
    -------
    optimizer : tf.keras.optimizers
        Optimizer instance used in the model.

    """
    # TODO use tensorflow optimizer
    if optimizer_name == "adam":
        optimizer = Adam(**kwargs)
    elif optimizer_name == "adadelta":
        optimizer = Adadelta(**kwargs)
    elif optimizer_name == "adagrad":
        optimizer = Adagrad(**kwargs)
    elif optimizer_name == "adamax":
        optimizer = Adamax(**kwargs)
    elif optimizer_name == "sgd":
        optimizer = SGD(**kwargs)
    else:
        raise ValueError("Instead of {0}, optimizer must be chosen among "
                         "['adam', 'adadelta', 'adagrad', adamax', sgd']."
                         .format(optimizer_name))

    return optimizer




#print(globals())
#print()
#print(globals()["BaseModel"])
#print()
#print(locals())
#print()
#print(BaseModel.__subclasses__())
