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
"""

import tensorflow as tf
#from tensorflow.keras import layer
#from tensorflow.keras.layers import Dense, Conv2D

print(tf.VERSION)
print(tf.keras.__version__)


from collections import Iterator, Generator
import unittest

class Test(unittest.TestCase):
    def test_Fib(self):
        f = Fib()
        self.assertEqual(next(f), 0)
        self.assertEqual(next(f), 1)
        self.assertEqual(next(f), 1)
        self.assertEqual(next(f), 2) #etc...
    def test_Fib_is_iterator(self):
        f = Fib()
        self.assertIsInstance(f, Iterator)
    def test_Fib_is_generator(self):
        f = Fib()
        self.assertIsInstance(f, Generator)