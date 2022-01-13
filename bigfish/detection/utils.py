# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.detection subpackage.
"""


def convert_spot_coordinates(spots, voxel_size_z, voxel_size_yx):
    """Convert spots coordinates from pixel to nanometer.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.

    Returns
    -------
    spots_nanometer : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 3), in nanometer.

    """
    # convert spots coordinates in nanometer
    spots_nanometer = spots.copy()
    if spots.shape[1] == 3:
        spots_nanometer[:, 0] *= voxel_size_z
        spots_nanometer[:, 1:] *= voxel_size_yx

    else:
        spots_nanometer *= voxel_size_yx

    return spots_nanometer
