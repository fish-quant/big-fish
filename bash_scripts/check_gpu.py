# -*- coding: utf-8 -*-

"""
Test if the code use GPU device"""

import os
import time
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == '__main__':
    print()
    print("Running {0} file...". format(os.path.basename(__file__)), "\n")

    print("--- DEVICES ---", "\n")

    # creates a graph
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
    c = tf.matmul(a, b)

    # run a session with 'log_device_placement'
    config = tf.ConfigProto(log_device_placement=True)
    session = tf.Session(config=config)
    print(session.run(c))
    session.close()
    print()
    time.sleep(2)

    print("--- GPU ACCESS ---", "\n")

    # creates a graph assigning the devices
    with tf.device("/cpu:0"):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
    with tf.device("/gpu:0"):
        c = tf.matmul(a, b)

    # run a session with 'log_device_placement'
    config = tf.ConfigProto(log_device_placement=True)
    session = tf.Session(config=config)
    print(session.run(c))
    session.close()
    print()
    time.sleep(2)

    print("--- GPU GROWTH ---", "\n")

    # creates a graph assigning the devices
    with tf.device("/cpu:0"):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
    with tf.device("/gpu:0"):
        c = tf.matmul(a, b)

    # run a session with 'log_device_placement'
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)
    print(session.run(c))
    session.close()
    print()
    time.sleep(2)

    print("--- SOFT PLACEMENT ---", "\n")

    # creates a graph assigning the devices
    with tf.device("/cpu:0"):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
    with tf.device("/gpu:0"):
        c = tf.matmul(a, b)

    # run a session with 'log_device_placement'
    config = tf.ConfigProto(log_device_placement=True,
                            allow_soft_placement=True)
    session = tf.Session(config=config)
    print(session.run(c))
    session.close()
    print()
    time.sleep(2)

    print("--- MULTI-GPU ACCESS ---", "\n")

    # creates a graph assigning the devices
    c = []
    for d in ["/gpu:0", "/gpu:1"]:
        with tf.device(d):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
            c.append(tf.matmul(a, b))
    with tf.device("/cpu:0"):
        s = tf.add_n(c)

    # run a session with 'log_device_placement'
    config = tf.ConfigProto(log_device_placement=True)
    session = tf.Session(config=config)
    print(session.run(s))
    session.close()
    print()
