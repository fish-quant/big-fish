# -*- coding: utf-8 -*-

"""
Test if the code use GPU device"""

import os
import tensorflow as tf

if __name__ == '__main__':
    print()
    print("Running {0} file...". format(os.path.basename(__file__)), "\n")

    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        print(sess.run(c))
