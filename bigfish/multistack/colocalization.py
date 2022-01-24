# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to detect colocalized spots in 2-d and 3-d.
"""

import numpy as np

import bigfish.stack as stack
import bigfish.detection as detection

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter


# TODO process multiple images together

# ### Main function ###

def detect_spots_colocalization(spots_1, spots_2, voxel_size, threshold=None,
                                return_indices=False, return_threshold=False):
    """Detect colocalized spots between two arrays of spot coordinates
    'spots_1' and 'spots_2'. Pairs of spots below a specific threshold are
    defined as colocalized.

    Parameters
    ----------
    spots_1 : np.ndarray
        Coordinates of the spots 1 with shape (nb_spots_1, 3) or
        (nb_spots_1, 2).
    spots_2 : np.ndarray
        Coordinates of the spots 2 with shape (nb_spots_2, 3) or
        (nb_spots_2, 2).
    voxel_size : int, float, Tuple(int, float), or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    threshold : int, float or None
        A threshold to discriminate colocalized spots from distant ones. If
        None, an optimal threshold is selected automatically.
    return_indices : bool
        Return the indices of the colocalized spots within 'spots_1' and
        'spots_2'.
    return_threshold : bool
        Return the threshold used to detect colocalized spots.

    Returns
    -------
    spots_1_colocalized : np.ndarray
        Coordinates of the colocalized spots from 'spots_1' with shape
        (nb_colocalized_spots,).
    spots_2_colocalized : np.ndarray
        Coordinates of the colocalized spots from 'spots_2'with shape
        (nb_colocalized_spots,).
    distances : np.ndarray, np.float64
        Distance matrix between spots with shape (nb_colocalized_spots,).
    indices_1 : np.ndarray, np.int64
        Indices of the colocalized spots in 'spots_1' with shape
        (nb_colocalized_spots,). Optional.
    indices_2 : np.ndarray, np.int64
        Indices of the colocalized spots in 'spots_2' with shape
        (nb_colocalized_spots,). Optional.
    threshold : int or float
        Threshold used to discriminate colocalized spots from distant ones.
        Optional.

    """
    # check parameters
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        threshold=(float, int, type(None)),
        return_indices=bool,
        return_threshold=bool)

    # check spots coordinates
    stack.check_array(spots_1, ndim=2, dtype=[np.float64, np.int64])
    stack.check_array(spots_2, ndim=2, dtype=[np.float64, np.int64])

    # convert spots coordinates in nanometer
    spots_1_nanometer = detection.convert_spot_coordinates(
        spots=spots_1,
        voxel_size=voxel_size)
    spots_2_nanometer = detection.convert_spot_coordinates(
        spots=spots_2,
        voxel_size=voxel_size)

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
    """Automatically set the optimal threshold to detect colocalized spots
    between two arrays of spot coordinates 'spots_1' and 'spots_2'.

    Parameters
    ----------
    distances : np.ndarray, np.float64
        Distance matrix between spots with shape
        (min(nb_spots_1, nb_spots_2),).

    Returns
    -------
    optimal_threshold : int
        Optimal threshold to discriminate distant spots and colocalized ones.

    """
    # get threshold values we want to test
    min_threshold = distances.min()
    max_threshold = distances.max() + 10
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
    y_smooth = savgol_filter(y, 501, 3, mode="nearest")
    if y_smooth.size > 0:
        optimal_threshold, _, _ = detection.get_breaking_point(x, y_smooth)

    # case where no spots were detected
    else:
        optimal_threshold = None

    return optimal_threshold


def get_elbow_value_colocalized(spots_1, spots_2, voxel_size):
    """Get values to plot the elbow curve used to automatically set the
    threshold to detect colocalized spots.

    Parameters
    ----------
    spots_1 : np.ndarray
        Coordinates of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    spots_2 : np.ndarray
        Coordinates of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    voxel_size : int, float, Tuple(int, float), or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.

    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    nb_colocalized : np.ndarray, np.float64
        Colocalized spots count.
    optimal_threshold : float or None
        Threshold automatically set.

    """
    # check parameters
    stack.check_parameter(voxel_size=(int, float, tuple, list))
    stack.check_array(spots_1, ndim=2, dtype=[np.float64, np.int64])
    stack.check_array(spots_2, ndim=2, dtype=[np.float64, np.int64])

    # check consistency between parameters
    ndim = spots_1.shape[1]
    if ndim not in [2, 3]:
        raise ValueError("Spot coordinates should be in 2 or 3 dimensions, "
                         "not {0}.".format(ndim))
    if spots_2.shape[1] != ndim:
        raise ValueError("Spot coordinates should have the same number of "
                         "dimensions.")
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim

    # convert spots coordinates in nanometer
    spots_1_nanometer = detection.convert_spot_coordinates(
        spots=spots_1,
        voxel_size=voxel_size)
    spots_2_nanometer = detection.convert_spot_coordinates(
        spots=spots_2,
        voxel_size=voxel_size)

    # compute distance matrix between spots
    distance_matrix = cdist(spots_1_nanometer, spots_2_nanometer)

    # assign spots based on their euclidean distance
    indices_1, indices_2 = linear_sum_assignment(distance_matrix)

    # get distance between colocalized spots
    distances = distance_matrix[indices_1, indices_2]

    # get candidate thresholds
    min_threshold = distances.min()
    max_threshold = distances.max() + 10
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
    y_smooth = savgol_filter(y, 501, 3, mode="nearest")
    optimal_threshold, _, _ = detection.get_breaking_point(x, y_smooth)

    return thresholds, nb_colocalized, optimal_threshold
