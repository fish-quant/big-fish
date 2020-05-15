# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to detected aggregated spots and foci.
"""

import numpy as np
import bigfish.stack as stack
from sklearn.cluster import DBSCAN


# ### Detect foci ###

def detect_foci(spots, voxel_size_z=None, voxel_size_yx=100, radius=350,
                nb_min_spots=4):
    """Detect clustered spots we can define as foci.

    1) If two spots are distant within a specific radius, we consider they are
    related to each other.
    2) A minimum number spots related to each others defines a foci.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    radius : int
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Radius expressed in nanometer.
    nb_min_spots : int
        The number of spots in a neighborhood for a point to be considered as
        a core point (from which a cluster is expanded). This includes the
        point itself.

    Returns
    -------
    clustered_spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension for the foci centroid (zyx or yx coordinates), the number of
        spots detected in the foci and its index.

    """
    # check parameters
    stack.check_array(spots, ndim=2, dtype=np.int64)
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          radius=int,
                          nb_min_spots=int)

    # check number of dimensions
    ndim = spots.shape[1]
    if ndim not in [2, 3]:
        raise ValueError("Spot coordinates should be in 2 or 3 dimensions, "
                         "not {0}.".format(ndim))
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided spot coordinates has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 2:
        voxel_size_z = None

    # cluster spots
    clustered_spots = _cluster_spots(
        spots, voxel_size_z, voxel_size_yx, radius, nb_min_spots)

    # extract and shape foci information
    foci = _extract_information(clustered_spots)

    return clustered_spots, foci


def _convert_spot_coordinates(spots, voxel_size_z, voxel_size_yx):
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


def _cluster_spots(spots, voxel_size_z, voxel_size_yx, radius, nb_min_spots):
    """Assign a cluster to each spot.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    radius : int
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Radius expressed in nanometer.
    nb_min_spots : int
        The number of spots in a neighborhood for a point to be considered as
        a core point (from which a cluster is expanded). This includes the
        point itself.

    Returns
    -------
    clustered_spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.

    """
    # convert spots coordinates in nanometer
    spots_nanometer = _convert_spot_coordinates(spots=spots,
                                                voxel_size_z=voxel_size_z,
                                                voxel_size_yx=voxel_size_yx)

    # fit a DBSCAN clustering algorithm with a specific radius
    dbscan = DBSCAN(eps=radius, min_samples=nb_min_spots)
    dbscan.fit(spots_nanometer)
    labels = dbscan.labels_
    labels = labels[:, np.newaxis]

    # assign a cluster to each spot if possible
    clustered_spots = spots.copy()
    clustered_spots = np.concatenate((clustered_spots, labels), axis=1)

    return clustered_spots


def _extract_information(clustered_spots):
    """Extract foci information from clustered spots.

    Parameters
    ----------
    clustered_spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.

    Returns
    -------
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension for the foci centroid (zyx or yx coordinates), the number of
        spots detected in the foci and its index.

    """
    # extract information for 3-d foci...
    if clustered_spots.shape[1] == 4:

        # get 3-d foci labels
        labels_foci = np.unique(
            clustered_spots[clustered_spots[:, 3] != -1, 3])
        if labels_foci.size == 0:
            foci = np.array([], dtype=np.int64).reshape((0, 5))
            return foci

        # shape information
        foci = []
        for label in labels_foci:
            spots_in_foci = clustered_spots[clustered_spots[:, 3] == label, :3]
            z_foci, y_foci, x_foci = spots_in_foci.mean(axis=0)
            nb_spots_foci = len(spots_in_foci)
            foci.append([z_foci, y_foci, x_foci, nb_spots_foci, label])
        foci = np.array(foci, dtype=np.int64)

    # ... and 2-d foci
    else:

        # get 2-d foci labels
        labels_foci = np.unique(
            clustered_spots[clustered_spots[:, 2] != -1, 2])
        if labels_foci.size == 0:
            foci = np.array([], dtype=np.int64).reshape((0, 4))
            return foci

        # shape information
        foci = []
        for label in labels_foci:
            spots_in_foci = clustered_spots[clustered_spots[:, 2] == label, :2]
            y_foci, x_foci = spots_in_foci.mean(axis=0)
            nb_spots_foci = len(spots_in_foci)
            foci.append([y_foci, x_foci, nb_spots_foci, label])
        foci = np.array(foci, dtype=np.int64)

    return foci
