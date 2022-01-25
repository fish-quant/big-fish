# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Function to cluster spots in point cloud and detect relevant aggregated
structures.
"""

import numpy as np

import bigfish.stack as stack

from .utils import convert_spot_coordinates

from sklearn.cluster import DBSCAN


# ### Detect clusters ###

def detect_clusters(spots, voxel_size, radius=350, nb_min_spots=4):
    """Cluster spots and detect relevant aggregated structures.

    #. If two spots are distant within a specific radius, we consider they are
       related to each other.
    #. A minimum number spots related to each others defines a cluster.

    Parameters
    ----------
    spots : np.ndarray
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    radius : int
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Radius expressed in nanometer.
    nb_min_spots : int
        The number of spots in a neighborhood for a point to be considered as
        a core point (from which a cluster is expanded). This includes the
        point itself.

    Returns
    -------
    clustered_spots : np.ndarray
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.
    clusters : np.ndarray
        Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate
        per dimension for the clusters centroid (zyx or yx coordinates), the
        number of spots detected in the clusters and its index.

    """
    # TODO check that the behavior is the same with float64 and int64
    #  coordinates
    # check parameters
    stack.check_array(spots, ndim=2, dtype=[np.float64, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        radius=int,
        nb_min_spots=int)

    # check consistency between parameters
    dtype = spots.dtype
    ndim = spots.shape[1]
    if ndim not in [2, 3]:
        raise ValueError("Spot coordinates should be in 2 or 3 dimensions, "
                         "not {0}.".format(ndim))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim

    # case where no spot were detected
    if spots.size == 0:
        clustered_spots = np.array([], dtype=dtype).reshape((0, ndim + 1))
        clusters = np.array([], dtype=dtype).reshape((0, ndim + 2))
        return clustered_spots, clusters

    # cluster spots
    clustered_spots = _cluster_spots(
        spots, voxel_size, radius, nb_min_spots)

    # extract and shape clusters information
    clusters = _extract_information(clustered_spots)

    return clustered_spots, clusters


def _cluster_spots(spots, voxel_size, radius, nb_min_spots):
    """Assign a cluster to each spot.

    Parameters
    ----------
    spots : np.ndarray
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size : Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions).
    radius : int
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Radius expressed in nanometer.
    nb_min_spots : int
        The number of spots in a neighborhood for a point to be considered as
        a core point (from which a cluster is expanded). This includes the
        point itself.

    Returns
    -------
    clustered_spots : np.ndarray
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.

    """
    # convert spots coordinates in nanometer
    spots_nanometer = convert_spot_coordinates(
        spots=spots, voxel_size=voxel_size)

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
    clustered_spots : np.ndarray
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.

    Returns
    -------
    clusters : np.ndarray
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
