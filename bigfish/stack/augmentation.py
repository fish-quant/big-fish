# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to augment the data (images or coordinates).
"""

import numpy as np
from .preprocess import check_parameter
from .preprocess import check_array


def augment_2d(image):
    """Augment an image applying a random operation.

    Parameters
    ----------
    image : np.ndarray
        Image to augment with shape (y, x, channels).

    Returns
    -------
    image_augmented : np.ndarray
        Image augmented with shape (y, x, channels).

    """
    # check input image
    check_parameter(image=np.ndarray)
    check_array(image, ndim=[2, 3])

    # randomly choose an operator
    operations = [_identity,
                  _flip_h, _flip_v,
                  _transpose, _transpose_inverse,
                  _rotation_90, _rotation_180, _rotation_270]
    random_operation = np.random.choice(operations)

    # augment the image
    image_augmented = random_operation(image)

    return image_augmented


def _identity(image):
    """Do not apply any operation to the image.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (x, y, channels).

    Returns
    -------
    image : np.ndarray
        Image with shape (x, y, channels).

    """
    return image


def _flip_h(image):
    """Flip an image horizontally.

    Parameters
    ----------
    image : np.ndarray
        Image to flip with shape (y, x, channels).

    Returns
    -------
    image_flipped : np.ndarray
        Image flipped with shape (y, x, channels).

    """
    image_flipped = np.flip(image, axis=0)

    return image_flipped


def _flip_v(image):
    """Flip an image vertically.

    Parameters
    ----------
    image : np.ndarray
        Image to flip with shape (y, x, channels).

    Returns
    -------
    image_flipped : np.ndarray
        Image flipped with shape (y, x, channels).

    """
    image_flipped = np.flip(image, axis=1)

    return image_flipped


def _transpose(image):
    """Transpose an image (flip it along the top left - bottom right diagonal).

    Parameters
    ----------
    image : np.ndarray
        Image to transpose with shape (y, x, channels).

    Returns
    -------
    image_transposed : np.ndarray
        Image transposed with shape (y, x, channels).

    """
    if image.ndim == 3:
        image_transposed = np.transpose(image, axes=(1, 0, 2))
    else:
        image_transposed = np.transpose(image)

    return image_transposed


def _rotation_90(image):
    """Rotate an image with 90 degrees (clockwise).

    Parameters
    ----------
    image : np.ndarray
        Image to rotate with shape (y, x, channels).

    Returns
    -------
    image_rotated : np.ndarray
        Image rotated with shape (y, x, channels).

    """
    image_rotated = _flip_h(image)
    image_rotated = _transpose(image_rotated)

    return image_rotated


def _rotation_180(image):
    """Rotate an image with 180 degrees.

    Parameters
    ----------
    image : np.ndarray
        Image to rotate with shape (y, x, channels).

    Returns
    -------
    image_rotated : np.ndarray
        Image rotated with shape (y, x, channels).

    """
    image_rotated = _flip_v(image)
    image_rotated = _flip_h(image_rotated)

    return image_rotated


def _rotation_270(image):
    """Rotate an image with 270 degrees (clockwise).

    Parameters
    ----------
    image : np.ndarray
        Image to rotate with shape (y, x, channels).

    Returns
    -------
    image_rotated : np.ndarray
        Image rotated with shape (y, x, channels).

    """
    image_rotated = _flip_v(image)
    image_rotated = _transpose(image_rotated)

    return image_rotated


def _transpose_inverse(image):
    """Flip an image along the bottom left - top right diagonal.

    Parameters
    ----------
    image : np.ndarray
        Image to transpose with shape (y, x, channels).

    Returns
    -------
    image_transposed : np.ndarray
        Image transposed with shape (y, x, channels).

    """
    image_transposed = _transpose(image)
    image_transposed = _rotation_180(image_transposed)

    return image_transposed
