# -*- coding: utf-8 -*-

"""Illumination correction functions."""

import numpy as np

from .utils import check_array, check_parameter
from .filter import gaussian_filter


# ### Illumination surface ###

def compute_illumination_surface(stacks, sigma=None):
    """Compute the illumination surface of a specific experiment.

    Parameters
    ----------
    stacks : np.ndarray, np.uint
        Concatenated 5-d tensors along the z-dimension with shape
        (r, c, z, y, x). They represent different images acquired during a
        same experiment.
    sigma : float, int, Tuple(float, int) or List(float, int)
        Sigma of the gaussian filtering used to smooth the illumination
        surface.

    Returns
    -------
    illumination_surfaces : np.ndarray, np.float
        A 4-d tensor with shape (r, c, y, x) approximating the average
        differential of illumination in our stack of images, for each channel
        and each round.

    """
    # check parameters
    check_array(stacks, ndim=5, dtype=[np.uint8, np.uint16], allow_nan=False)
    check_parameter(sigma=(float, int, tuple, list, type(None)))

    # initialize illumination surfaces
    r, c, z, y, x = stacks.shape
    illumination_surfaces = np.zeros((r, c, y, x))

    # compute mean over the z-dimension
    mean_stacks = np.mean(stacks, axis=2)

    # separate the channels and the rounds
    for i_round in range(r):
        for i_channel in range(c):
            illumination_surface = mean_stacks[i_round, i_channel, :, :]

            # smooth the surface
            if sigma is not None:
                illumination_surface = gaussian_filter(illumination_surface,
                                                       sigma=sigma,
                                                       allow_negative=False)

            illumination_surfaces[i_round, i_channel] = illumination_surface

    return illumination_surfaces


def correct_illumination_surface(tensor, illumination_surfaces):
    """Correct a tensor with uneven illumination.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 5-d tensor with shape (r, c, z, y, x).
    illumination_surfaces : np.ndarray, np.float
        A 4-d tensor with shape (r, c, y, x) approximating the average
        differential of illumination in our stack of images, for each channel
        and each round.

    Returns
    -------
    tensor_corrected : np.ndarray, np.float
        A 5-d tensor with shape (r, c, z, y, x).

    """
    # check parameters
    check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16], allow_nan=False)
    check_array(illumination_surfaces, ndim=4, dtype=[np.float32, np.float64],
                allow_nan=False)

    # initialize corrected tensor
    tensor_corrected = np.zeros_like(tensor)

    # TODO control the multiplication and the division
    # correct each round/channel independently
    r, c, _, _, _ = tensor.shape
    for i_round in range(r):
        for i_channel in range(c):
            image_3d = tensor[i_round, i_channel, ...]
            s = illumination_surfaces[i_round, i_channel]
            tensor_corrected[i_round, i_channel] = image_3d * np.mean(s) / s

    return tensor_corrected
