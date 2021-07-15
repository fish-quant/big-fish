# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Function to cluster spots in point cloud and detect relevant aggregated
structures.
"""

import numpy as np
import bigfish.stack as stack
from sklearn.cluster import DBSCAN


# ### Detect clusters ###

def detect_clusters(spots, voxel_size_z=None, voxel_size_yx=100, radius=350,
                    nb_min_spots=4):
    """Cluster spots and detect relevant aggregated structures.

    #. If two spots are distant within a specific radius, we consider they are
       related to each other.
    #. A minimum number spots related to each others defines a cluster.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, spots are
        considered in 2-d.
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
    clusters : np.ndarray, np.int64
        Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate
        per dimension for the clusters centroid (zyx or yx coordinates), the
        number of spots detected in the clusters and its index.

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

    # case where no spot were detected
    if spots.size == 0:
        clustered_spots = np.array([], dtype=np.int64).reshape((0, ndim + 1))
        clusters = np.array([], dtype=np.int64).reshape((0, ndim + 2))
        return clustered_spots, clusters

    # cluster spots
    clustered_spots = _cluster_spots(
        spots, voxel_size_z, voxel_size_yx, radius, nb_min_spots)

    # extract and shape clusters information
    clusters = _extract_information(clustered_spots)

    return clustered_spots, clusters


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
    """Extract clusters information from clustered spots.

    Parameters
    ----------
    clustered_spots : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.

    Returns
    -------
    clusters : np.ndarray, np.int64
        Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate
        per dimension for the cluster centroid (zyx or yx coordinates), the
        number of spots detected in the cluster and its index.

    """
    # extract information for 3-d cluster...
    if clustered_spots.shape[1] == 4:

        # get 3-d cluster labels
        labels_clusters = np.unique(
            clustered_spots[clustered_spots[:, 3] != -1, 3])
        if labels_clusters.size == 0:
            clusters = np.array([], dtype=np.int64).reshape((0, 5))
            return clusters

        # shape information
        clusters = []
        for label in labels_clusters:
            spots_in_cluster = clustered_spots[clustered_spots[:, 3] == label,
                                               :3]
            z_cluster, y_cluster, x_cluster = spots_in_cluster.mean(axis=0)
            nb_spots_cluster = len(spots_in_cluster)
            clusters.append([z_cluster, y_cluster, x_cluster,
                             nb_spots_cluster, label])
        clusters = np.array(clusters, dtype=np.int64)

    # ... and 2-d cluster
    else:

        # get 2-d cluster labels
        labels_clusters = np.unique(
            clustered_spots[clustered_spots[:, 2] != -1, 2])
        if labels_clusters.size == 0:
            clusters = np.array([], dtype=np.int64).reshape((0, 4))
            return clusters

        # shape information
        clusters = []
        for label in labels_clusters:
            spots_in_cluster = clustered_spots[clustered_spots[:, 2] == label,
                                               :2]
            y_cluster, x_cluster = spots_in_cluster.mean(axis=0)
            nb_spots_cluster = len(spots_in_cluster)
            clusters.append([y_cluster, x_cluster, nb_spots_cluster, label])
        clusters = np.array(clusters, dtype=np.int64)

    return clusters
