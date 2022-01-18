# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to detect colocalized spots in 2-d and 3-d.
"""

import numpy as np

import bigfish.stack as stack

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from bigfish.detection.utils import convert_spot_coordinates

# TODO complete documentation
# TODO process multiple images together

# ### Main function ###


def detect_spots_colocalization(spots_1, spots_2, voxel_size_z=None,
                                voxel_size_yx=100, threshold=None,
                                return_indices=False, return_threshold=False):
    """

    Parameters
    ----------
    spots_1
    spots_2
    voxel_size_z
    voxel_size_yx
    threshold
    return_indices
    return_threshold

    Returns
    -------

    """
    # check parameters
    stack.check_parameter(threshold=(float, int, type(None)),
                          return_threshold=bool,
                          voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float))

    # check spots coordinates
    stack.check_array(spots_1, ndim=2, dtype=[np.float64, np.int64])
    stack.check_array(spots_2, ndim=2, dtype=[np.float64, np.int64])

    # convert spots coordinates in nanometer
    spots_1_nanometer = convert_spot_coordinates(
        spots=spots_1,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx)
    spots_2_nanometer = convert_spot_coordinates(
        spots=spots_2,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx)

    # compute distance matrix between spots
    distance_matrix = cdist(spots_1_nanometer, spots_2_nanometer)

    # assign spots based on their euclidean distance
    indices_1, indices_2 = linear_sum_assignment(distance_matrix)

    # get distance between colocalized spots
    distances = distance_matrix[indices_1, indices_2]

    # keep colocalized spots under a specific threshold
    if threshold is None:
        threshold = _automated_threshold_setting_colocalization(distances)

    # keep colocalized spots within a specific distance
    mask = distances <= threshold
    indices_1 = indices_1[mask]
    indices_2 = indices_2[mask]
    distances = distances[mask]

    # get colocalized spots
    spots_1_colocalized = spots_1[indices_1, ...]
    spots_2_colocalized = spots_2[indices_2, ...]

    # return indices and threshold or not
    if return_indices and return_threshold:
        return (spots_1_colocalized, spots_2_colocalized, distances, indices_1,
                indices_2, threshold)
    elif return_indices and not return_threshold:
        return (spots_1_colocalized, spots_2_colocalized, distances, indices_1,
                indices_2)
    elif not return_indices and return_threshold:
        return spots_1_colocalized, spots_2_colocalized, distances, threshold
    else:
        return spots_1_colocalized, spots_2_colocalized, distances


def _automated_threshold_setting_colocalization(distances):
    """

    Parameters
    ----------
    distances

    Returns
    -------

    """
    # get threshold values we want to test
    min_threshold = distances.min()
    max_threshold = distances.max()
    n_candidates = min(int(max_threshold - min_threshold), 10000)
    thresholds = np.linspace(min_threshold, max_threshold, num=n_candidates)

    # get colocalized spots count
    nb_colocalized = []
    for threshold in thresholds:
        mask = distances <= threshold
        n = mask.sum()
        nb_colocalized.append(n)
    nb_colocalized = np.array(nb_colocalized)

    # select threshold where the break of the distribution is located
    x = thresholds.copy()
    y = -nb_colocalized.copy() + nb_colocalized.max()
    y_smooth = savgol_filter(y, 501, 3)
    optimal_threshold, _, _ = _get_breaking_point(x, y_smooth)

    # TODO manage failures

    return optimal_threshold


def _get_breaking_point(x, y):
    """Select the x-axis value where a L-curve has a kink.

    Assuming a L-curve from A to B, the 'breaking_point' is the more distant
    point to the segment [A, B].

    Parameters
    ----------
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.

    Returns
    -------
    breaking_point : float
        X-axis value at the kink location.
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.

    """
    # select threshold where curve break
    slope = (y[-1] - y[0]) / len(y)
    y_grad = np.gradient(y)
    m = list(y_grad >= slope)
    j = m.index(False)
    m = m[j:]
    x = x[j:]
    y = y[j:]
    if True in m:
        i = m.index(True)
    else:
        i = -1
    breaking_point = float(x[i])

    return breaking_point, x, y


def get_elbow_value_colocalized(spots_1, spots_2, voxel_size_z=None,
                                voxel_size_yx=100):
    """

    Parameters
    ----------
    spots_1
    spots_2
    voxel_size_z
    voxel_size_yx

    Returns
    -------

    """
    # check parameters
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float))

    # check spots coordinates
    stack.check_array(spots_1, ndim=2, dtype=[np.float64, np.int64])
    stack.check_array(spots_2, ndim=2, dtype=[np.float64, np.int64])

    # convert spots coordinates in nanometer
    spots_1_nanometer = convert_spot_coordinates(
        spots=spots_1,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx)
    spots_2_nanometer = convert_spot_coordinates(
        spots=spots_2,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx)

    # compute distance matrix between spots
    distance_matrix = cdist(spots_1_nanometer, spots_2_nanometer)

    # assign spots based on their euclidean distance
    indices_1, indices_2 = linear_sum_assignment(distance_matrix)

    # get distance between colocalized spots
    distances = distance_matrix[indices_1, indices_2]

    # get candidate thresholds
    min_threshold = distances.min()
    max_threshold = distances.max()
    n_candidates = min(int(max_threshold - min_threshold), 10000)
    thresholds = np.linspace(min_threshold, max_threshold, num=n_candidates)

    # get colocalized spots count
    nb_colocalized = []
    for threshold in thresholds:
        mask = distances <= threshold
        n = mask.sum()
        nb_colocalized.append(n)
    nb_colocalized = np.array(nb_colocalized)

    # select threshold where the break of the distribution is located
    x = thresholds.copy()
    y = -nb_colocalized.copy() + nb_colocalized.max()
    y_smooth = savgol_filter(y, 501, 3)
    optimal_threshold, _, _ = _get_breaking_point(x, y_smooth)

    return thresholds, nb_colocalized, optimal_threshold
