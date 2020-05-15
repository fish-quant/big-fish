# -*- coding: utf-8 -*-

"""
Functions to augment the data (images or coordinates).
"""

import numpy as np


def identity(image):
    """don't apply any operation to the image.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image with shape (x, y, channels).

    Returns
    -------
    image : np.ndarray, np.float32
        Image with shape (x, y, channels).

    """
    return image


def flip_h(image):
    """Flip an image horizontally.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to flip with shape (x, y, channels).

    Returns
    -------
    image_flipped : np.ndarray, np.float32
        Image flipped with shape (x, y, channels).

    """
    image_flipped = np.flip(image, axis=0)

    return image_flipped


def flip_v(image):
    """Flip an image vertically.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to flip with shape (x, y, channels).

    Returns
    -------
    image_flipped : np.ndarray, np.float32
        Image flipped with shape (x, y, channels).

    """
    image_flipped = np.flip(image, axis=1)

    return image_flipped


def transpose(image):
    """Transpose an image.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to transpose with shape (x, y, channels).

    Returns
    -------
    image_transposed : np.ndarray, np.float32
        Image transposed with shape (x, y, channels).

    """
    image_transposed = np.transpose(image, axes=(1, 0, 2))

    return image_transposed


def rotation_90(image):
    """Rotate an image with 90 degrees.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to rotate with shape (x, y, channels).

    Returns
    -------
    image_rotated : np.ndarray, np.float32
        Image rotated with shape (x, y, channels).

    """
    image_rotated = flip_h(image)
    image_rotated = transpose(image_rotated)

    return image_rotated


def rotation_180(image):
    """Rotate an image with 90 degrees.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to rotate with shape (x, y, channels).

    Returns
    -------
    image_rotated : np.ndarray, np.float32
        Image rotated with shape (x, y, channels).

    """
    image_rotated = flip_v(image)
    image_rotated = flip_h(image_rotated)

    return image_rotated


def rotation_270(image):
    """Rotate an image with 90 degrees.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to rotate with shape (x, y, channels).

    Returns
    -------
    image_rotated : np.ndarray, np.float32
        Image rotated with shape (x, y, channels).

    """
    image_rotated = flip_v(image)
    image_rotated = transpose(image_rotated)

    return image_rotated


def transpose_inverse(image):
    """Transpose an image from the other diagonal.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to transpose with shape (x, y, channels).

    Returns
    -------
    image_transposed : np.ndarray, np.float32
        Image transposed with shape (x, y, channels).

    """
    image_transposed = rotation_270(image)
    image_transposed = transpose(image_transposed)

    return image_transposed


def augment(image):
    """Augment an image applying a random operation.

    Parameters
    ----------
    image : np.ndarray, np.float32
        Image to augment with shape (x, y, channels).

    Returns
    -------
    image_augmented : np.ndarray, np.float32
        Image augmented with shape (x, y, channels).

    """
    # randomly choose an operator
    operations = [identity,
                  flip_h, flip_v,
                  transpose, transpose_inverse,
                  rotation_90, rotation_180, rotation_270]
    random_operation = np.random.choice(operations)

    # augment the image
    image_augmented = random_operation(image)

    return image_augmented
