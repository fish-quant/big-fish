# -*- coding: utf-8 -*-

"""
General classes and methods to use the models. to classify the localization patterns of an cell image.
"""

from abc import ABCMeta, abstractmethod


# ### Load models ###

# ### General models ###

class BaseModel(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


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
