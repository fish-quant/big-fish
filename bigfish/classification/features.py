# -*- coding: utf-8 -*-

"""
Functions to craft features.
"""

from bigfish import stack

import numpy as np
from scipy import ndimage as ndi

from skimage.measure import regionprops
from skimage.morphology import binary_opening
from skimage.morphology.selem import disk

from scipy.spatial import distance_matrix
from scipy.stats import spearmanr


def from_coord_to_matrix(cyt_coord, nuc_coord, rna_coord):
    """

    Parameters
    ----------
    cyt_coord
    nuc_coord
    rna_coord

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # get size of the frame
    max_y = cyt_coord[:, 0].max() + 1
    max_x = cyt_coord[:, 1].max() + 1
    image_shape = (max_y, max_x)

    # cytoplasm
    cyt = np.zeros(image_shape, dtype=bool)
    cyt[cyt_coord[:, 0], cyt_coord[:, 1]] = True

    # nucleus
    nuc = np.zeros(image_shape, dtype=bool)
    nuc[nuc_coord[:, 0], nuc_coord[:, 1]] = True

    # rna
    rna = np.zeros(image_shape, dtype=bool)
    rna[rna_coord[:, 0], rna_coord[:, 1]] = True

    return cyt, nuc, rna


def get_centroid(mask):
    """

    Parameters
    ----------
    mask

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # get centroid
    region = regionprops(mask.astype(np.uint8))[0]
    centroid = np.array(region.centroid, dtype=np.int64)

    return centroid


def get_centroid_distance_map(centroid_coordinate, mask_cyt):
    """

    Parameters
    ----------
    centroid_coordinate
    mask_cyt

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # get mask centroid
    mask_centroid = np.zeros_like(mask_cyt)
    mask_centroid[centroid_coordinate[0], centroid_coordinate[1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centroid)
    distance_map = distance_map.astype(np.float32)

    return distance_map


def features_distance(mask_rna, distance_cyt, distance_nuc,
                      distance_cyt_centroid, distance_nuc_centroid):
    """

    Parameters
    ----------
    mask_rna
    distance_cyt
    distance_nuc
    distance_cyt_centroid
    distance_nuc_centroid

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # compute average distances to cytoplasm and quantiles
    factor = distance_cyt[distance_cyt > 0].mean()
    mean_distance_cyt = distance_cyt[mask_rna].mean() / factor
    quantile_5_distance_cyt = np.percentile(distance_cyt[mask_rna], 5)
    quantile_5_distance_cyt /= factor
    quantile_10_distance_cyt = np.percentile(distance_cyt[mask_rna], 10)
    quantile_10_distance_cyt /= factor
    quantile_20_distance_cyt = np.percentile(distance_cyt[mask_rna], 20)
    quantile_20_distance_cyt /= factor
    quantile_50_distance_cyt = np.percentile(distance_cyt[mask_rna], 50)
    quantile_50_distance_cyt /= factor

    # compute average distances to cytoplasm centroid
    factor = distance_cyt_centroid[distance_cyt > 0].mean()
    mean_distance_cyt_centroid = distance_cyt_centroid[mask_rna].mean()
    mean_distance_cyt_centroid /= factor

    # compute average distances to nucleus
    factor = distance_nuc[distance_cyt > 0].mean()
    mean_distance_nuc = distance_nuc[mask_rna].mean() / factor

    # compute average distances to nucleus centroid
    factor = distance_nuc_centroid[distance_cyt > 0].mean()
    mean_distance_nuc_centroid = distance_nuc_centroid[mask_rna].mean()
    mean_distance_nuc_centroid /= factor

    features = [mean_distance_cyt, quantile_5_distance_cyt,
                quantile_10_distance_cyt, quantile_20_distance_cyt,
                quantile_50_distance_cyt, mean_distance_cyt_centroid,
                mean_distance_nuc, mean_distance_nuc_centroid]

    return features


def feature_in_out_nucleus(mask_nuc, mask_rna):
    """

    Parameters
    ----------
    mask_nuc
    distance_nuc
    mask_rna

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # compute the ratio between rna in and out nucleus
    rna_in = mask_rna[mask_nuc].sum()
    nb_rna = mask_rna.sum()
    feature = rna_in / nb_rna

    return feature


def features_opening(opening_sizes, mask_cyt, mask_rna):
    """

    Parameters
    ----------
    opening_sizes
    mask_cyt
    mask_rna

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # get number of rna
    nb_rna = mask_rna.sum()

    # apply opening operator and count the loss of rna
    features = []
    for size in opening_sizes:
        s = disk(size, dtype=bool)
        mask_cyt_transformed = binary_opening(mask_cyt, selem=s)
        nb_rna__after_opening = mask_rna[mask_cyt_transformed > 0].sum()
        diff_opening = (nb_rna - nb_rna__after_opening) / nb_rna
        features.append(diff_opening)

    return features


def ripley_values(radii, mask_cyt, rna_coord, mask_rna):
    """

    Parameters
    ----------
    radii
    mask_cyt
    rna_coord
    mask_rna

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # sort rna coordinates
    sorted_indices = np.lexsort((rna_coord[:, 1], rna_coord[:, 0]))
    rna_coord = rna_coord[sorted_indices]

    # compute distance matrix between rna and rna density
    distances = distance_matrix(rna_coord, rna_coord, p=2)
    factor = len(rna_coord) ** 2 / mask_cyt.sum()

    # cast cytoplasm mask in np.uint8
    mask_cyt_8bit = stack.cast_img_uint8(mask_cyt)

    # for each radius, get neighbors and weight
    values = []
    for r in radii:
        mask_distance = distances.copy()
        mask_distance = mask_distance <= r
        nb_neighbors = np.sum(mask_distance, axis=0) - 1
        weights = stack.mean_filter(mask_cyt_8bit, kernel_shape="disk",
                                    kernel_size=r)
        weights = weights.astype(np.float32) / 255.
        rna_weights = weights[mask_rna]
        nb_neighbors_weighted = np.multiply(nb_neighbors, rna_weights)
        value = nb_neighbors_weighted.sum() / factor
        values.append(value)
    values = np.array(values, dtype=np.float32)
    values_corrected = np.sqrt(values / np.pi) - np.array(radii)

    return values_corrected


def moving_average(a, n=4):
    """

    Parameters
    ----------
    a
    n

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    res = np.cumsum(a, dtype=np.float32)
    res[n:] = res[n:] - res[:-n]
    averaged_array = res[n - 1:] / n

    return averaged_array


def features_ripley(radii, cyt_coord, mask_cyt, rna_coord, mask_rna):
    """

    Parameters
    ----------
    radii
    cyt_coord
    mask_cyt
    rna_coord
    mask_rna

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # compute corrected Ripley values for different radii
    values = ripley_values(radii, mask_cyt, rna_coord, mask_rna)

    # smooth them using moving average
    smoothed_values = moving_average(values, n=4)

    # compute the gradients of these values
    gradients = np.gradient(smoothed_values)

    # compute features
    index_max = np.argmax(smoothed_values)
    max_value = smoothed_values[index_max]
    if index_max == 0:
        max_gradient = gradients[0]
    else:
        max_gradient = max(gradients[:index_max])
    if index_max == len(gradients) - 1:
        min_gradient = gradients[-1]
    else:
        min_gradient = min(gradients[index_max:])
    monotony, _ = spearmanr(smoothed_values, radii[2:-1])
    distances_cell = distance_matrix(cyt_coord, cyt_coord, p=2)
    max_size_cell = np.max(distances_cell)
    big_radius = int(max_size_cell / 4)
    big_value = ripley_values([big_radius], mask_cyt, rna_coord, mask_rna)[0]
    features = [max_value, max_gradient, min_gradient, monotony, big_value]

    return features


def feature_polarization(distance_cyt, distance_cyt_centroid, centroid_rna):
    """

    Parameters
    ----------
    distance_cyt
    distance_cyt_centroid
    rna_coord
    centroid_rna

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # compute polarization index
    factor = np.mean(distance_cyt_centroid[distance_cyt > 0])
    distance_rna_cell = distance_cyt_centroid[centroid_rna[0], centroid_rna[1]]
    feature = distance_rna_cell / factor

    return feature


def feature_dispersion(mask_cyt, rna_coord, centroid_rna):
    """

    Parameters
    ----------
    mask_cyt
    rna_coord
    centroid_rna

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # TODO correct the formula
    # get coordinates of each pixel of the cell
    mask_cyt_coord = np.nonzero(mask_cyt)
    mask_cyt_coord = np.column_stack(mask_cyt_coord)

    # compute dispersion index
    sigma_rna = np.sum((rna_coord - centroid_rna) ** 2, axis=0)
    sigma_rna = np.sum(sigma_rna / len(rna_coord))
    sigma_cell = np.sum((mask_cyt_coord - centroid_rna) ** 2, axis=0)
    sigma_cell = np.sum(sigma_cell / len(mask_cyt_coord))
    feature = sigma_rna / sigma_cell

    return feature


def feature_area(mask_cyt, mask_nuc):
    """

    Parameters
    ----------
    mask_cyt
    mask_nuc

    Returns
    -------

    """
    # TODO add sanity check functions
    # TODO add documentation
    # get area of the cytoplasm and the nucleus
    area_cyt = mask_cyt.sum()
    area_nuc = mask_nuc.sum()

    # compute relative area of the nucleus
    relative_area_nuc = area_nuc / area_cyt

    # return features
    features = [relative_area_nuc, area_cyt, area_nuc]

    return features


def get_features(cyt_coord, nuc_coord, rna_coord):
    """Compute cell features.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Coordinate yx of the cytoplasm boundary with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Coordinate yx of the cytoplasm boundary with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Coordinate yx of the detected rna with shape (nb_rna, 2).

    Returns
    -------
    features : List[float]
        List of features (cf. features.get_features_name()).

    """
    # TODO add sanity check functions
    # TODO add documentation
    # TODO filter features
    # get a binary representation of the coordinates
    cyt, nuc, mask_rna = from_coord_to_matrix(cyt_coord, nuc_coord, rna_coord)

    # fill in masks
    mask_cyt, mask_nuc = stack.get_surface_layers(cyt, nuc, cast_float=False)

    # compute distance maps for the cytoplasm and the nucleus
    distance_cyt, distance_nuc = stack.get_distance_layers(cyt, nuc)

    # get centroids
    centroid_cyt = get_centroid(mask_cyt)
    centroid_nuc = get_centroid(mask_nuc)
    centroid_rna = np.mean(rna_coord, axis=0, dtype=np.int64)

    # get centroid distance maps
    distance_cyt_centroid = get_centroid_distance_map(centroid_cyt, mask_cyt)
    distance_nuc_centroid = get_centroid_distance_map(centroid_nuc, mask_cyt)

    # compute features
    a = features_distance(mask_rna, distance_cyt, distance_nuc,
                          distance_cyt_centroid, distance_nuc_centroid)
    b = feature_in_out_nucleus(mask_nuc, mask_rna)
    opening_sizes = [15, 30, 45, 60]
    c = features_opening(opening_sizes, mask_cyt, mask_rna)
    radii = [r for r in range(40)]
    d = features_ripley(radii, cyt_coord, mask_cyt, rna_coord, mask_rna)
    e = feature_polarization(distance_cyt, distance_cyt_centroid, centroid_rna)
    f = feature_dispersion(mask_cyt, rna_coord, centroid_rna)
    g = feature_area(mask_cyt, mask_nuc)
    features = np.array(a + [b] + c + d + [e] + [f] + g, dtype=np.float32)

    return features


def get_features_name():
    """Return the current list of features names.

    Returns
    -------
    features_name : List[str]
        List of features name returned by features.get_features().

    """
    # TODO add sanity check functions
    # TODO add documentation
    # TODO filter features
    features_name = ["average_dist_cyt", "quantile_5_dist_cyt",
                     "quantile_10_dist_cyt", "quantile_20_dist_cyt",
                     "quantile_50_dist_cyt", "average_dist_cyt_centroid",
                     "average_dist_nuc", "average_dist_nuc_centroid",
                     "ratio_in_nuc", "diff_opening_15", "diff_opening_30",
                     "diff_opening_45", "diff_opening_60", "ripley_max",
                     "ripley_max_gradient", "ripley_min_gradient",
                     "ripley_monotony", "ripley_large", "polarization_index",
                     "dispersion_index", "ratio_area_nuc", "area_cyt",
                     "area_nuc"]

    return features_name
