# -*- coding: utf-8 -*-

"""
Functions to fit gaussian functions to the detected RNA spots, especially in
clustered regions.
"""

import numpy as np

from sklearn.cluster import DBSCAN


# ### Spots clustering ###

def convert_spot_coordinates(spots, resolution_z, resolution_yx):
    """
    Convert spots coordinates in nanometer.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3).
    resolution_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : int or float
        Size of a voxel on the yx plan, in nanometer.

    Returns
    -------
    spots_nanometer : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3), in
        nanometer.

    """
    # convert spots coordinates in nanometer, for each dimension, according to
    # the pixel size of the image
    spots_nanometer = spots.copy()
    spots_nanometer[:, 0] *= resolution_z
    spots_nanometer[:, 1:] *= resolution_yx

    return spots_nanometer


def cluster_spots(spots, resolution_z, resolution_yx, radius, nb_min_spots):
    """
    Assign a cluster to each spot.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3).
    resolution_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    radius : int
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Radius in nanometer.
    nb_min_spots : int
        The number of spots in a neighborhood for a point to be considered as
        a core point (from which a cluster is expanded). This includes the
        point itself.

    Returns
    -------
    clustered_spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4). The last
        column is the cluster assigned to the spot. If no cluster was assigned,
        value is -1.

    """
    # convert spots coordinates in nanometer
    spots_nanometer = convert_spot_coordinates(spots=spots,
                                               resolution_z=resolution_z,
                                               resolution_yx=resolution_yx)

    # fit a DBSCAN clustering algorithm with a specific radius
    dbscan = DBSCAN(eps=radius, min_samples=nb_min_spots)
    dbscan.fit(spots_nanometer)
    labels = dbscan.labels_
    labels = labels[:, np.newaxis]

    # assign a cluster to each spot if possible
    clustered_spots = spots.copy()
    clustered_spots = np.concatenate((clustered_spots, labels), axis=1)

    return clustered_spots


# ### Detect foci ###

def extract_foci(clustered_spots):
    """
    Extract foci information from clustered spots.

    Parameters
    ----------
    clustered_spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4). The last
        column is the cluster assigned to the spot. If no cluster was assigned,
        value is -1.

    Returns
    -------
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of spots detected in the
        foci and its index.

    """
    # get foci labels
    labels_foci = np.unique(clustered_spots[clustered_spots[:, 3] != -1, 3])
    if labels_foci.size == 0:
        foci = np.array([], dtype=np.int64).reshape((0, 5))
        return foci

    # get foci's information
    foci = []
    for label in labels_foci:
        spots_in_foci = clustered_spots[clustered_spots[:, 3] == label, :3]
        z_foci, y_foci, x_foci = spots_in_foci.mean(axis=0)
        nb_spots_foci = len(spots_in_foci)
        foci.append([z_foci, y_foci, x_foci, nb_spots_foci, label])
    foci = np.array(foci, dtype=np.int64)

    return foci
