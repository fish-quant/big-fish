# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.detection subpackage.
"""

import numpy as np
import bigfish.stack as stack


def convert_spot_coordinates(spots, voxel_size):
    """Convert spots coordinates from pixel to nanometer.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.

    Returns
    -------
    spots_nanometer : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 3), in nanometer.

    """
    # check parameters
    stack.check_parameter(
        voxel_size=(int, float, tuple, list))
    stack.check_array(spots, ndim=2, dtype=[np.float64, np.int64])

    # check consistency between parameters
    ndim = spots.shape[1]
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError("'voxel_size' must be a scalar or a sequence "
                             "with {0} elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim

    # convert spots coordinates in nanometer
    spots_nanometer = spots.copy()
    if ndim == 3:
        spots_nanometer[:, 0] *= voxel_size[0]
        spots_nanometer[:, 1:] *= voxel_size[-1]

    else:
        spots_nanometer *= voxel_size[-1]

    return spots_nanometer


def get_object_radius_pixel(voxel_size_nm, object_radius_nm, ndim):
    """Convert the object radius in pixel.

    When the object considered is a spot this value can be interpreted as the
    standard deviation of the spot PSF, in pixel. For any object modelled with
    a gaussian signal, this value can be interpreted as the standard deviation
    of the gaussian.

    Parameters
    ----------
    voxel_size_nm : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    object_radius_nm : int, float, Tuple(int, float) or List(int, float)
        Radius of the object, in nanometer. One value per spatial dimension
        (zyx or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions.
    ndim : int
        Number of spatial dimension to consider.

    Returns
    -------
    object_radius_px : Tuple[float]
        Radius of the object in pixel, one element per dimension (zyx or yx
        dimensions).

    """
    # check parameters
    stack.check_parameter(
        voxel_size_nm=(int, float, tuple, list),
        object_radius_nm=(int, float, tuple, list),
        ndim=int)

    # check consistency between parameters
    if isinstance(voxel_size_nm, (tuple, list)):
        if len(voxel_size_nm) != ndim:
            raise ValueError("'voxel_size_nm' must be a scalar or a sequence "
                             "with {0} elements.".format(ndim))
    else:
        voxel_size_nm = (voxel_size_nm,) * ndim
    if isinstance(object_radius_nm, (tuple, list)):
        if len(object_radius_nm) != ndim:
            raise ValueError("'object_radius_nm' must be a scalar or a "
                             "sequence with {0} elements.".format(ndim))
    else:
        object_radius_nm = (object_radius_nm,) * ndim

    # get radius in pixel
    object_radius_px = [b / a for a, b in zip(voxel_size_nm, object_radius_nm)]
    object_radius_px = tuple(object_radius_px)

    return object_radius_px


def get_object_radius_nm(voxel_size_nm, object_radius_px, ndim):
    """Convert the object radius in nanometer.

    When the object considered is a spot this value can be interpreted as the
    standard deviation of the spot PSF, in nanometer. For any object modelled
    with a gaussian signal, this value can be interpreted as the standard
    deviation of the gaussian.

    Parameters
    ----------
    voxel_size_nm : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    object_radius_px : int, float, Tuple(int, float) or List(int, float)
        Radius of the object, in pixel. One value per spatial dimension
        (zyx or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions.
    ndim : int
        Number of spatial dimension to consider.

    Returns
    -------
    object_radius_nm : Tuple[float]
        Radius of the object in nanometer, one element per dimension (zyx or yx
        dimensions).

    """
    # check parameters
    stack.check_parameter(
        voxel_size_nm=(int, float, tuple, list),
        object_radius_px=(int, float, tuple, list),
        ndim=int)

    # check consistency between parameters
    if isinstance(voxel_size_nm, (tuple, list)):
        if len(voxel_size_nm) != ndim:
            raise ValueError("'voxel_size_nm' must be a scalar or a sequence "
                             "with {0} elements.".format(ndim))
    else:
        voxel_size_nm = (voxel_size_nm,) * ndim
    if isinstance(object_radius_px, (tuple, list)):
        if len(object_radius_px) != ndim:
            raise ValueError("'object_radius_px' must be a scalar or a "
                             "sequence with {0} elements.".format(ndim))
    else:
        object_radius_px = (object_radius_px,) * ndim

    # get radius in pixel
    object_radius_nm = [a * b for a, b in zip(voxel_size_nm, object_radius_px)]
    object_radius_nm = tuple(object_radius_nm)

    return object_radius_nm
