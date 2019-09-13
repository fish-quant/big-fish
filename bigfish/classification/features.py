# -*- coding: utf-8 -*-

"""
Functions to craft features.
"""

import bigfish.stack as stack
import bigfish.detection as detection

import numpy as np
from scipy import ndimage as ndi

from skimage.measure import regionprops
from skimage.morphology import binary_opening
from skimage.morphology.selem import disk

from scipy.spatial import distance_matrix
from scipy.stats import spearmanr

# TODO add sanity check functions
# TODO add documentation
# TODO check centroid cyt has a yx format


def get_features(cyt_coord, nuc_coord, rna_coord, features_aubin=True,
                 features_no_aubin=False):
    """Compute cell features.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Coordinate yx of the cytoplasm boundary with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Coordinate yx of the cytoplasm boundary with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Coordinate yx of the detected rna with shape (nb_rna, 2).
    features_aubin : bool
        Compute features from Aubin paper.
    features_no_aubin : bool
        Compute features that are not present in Aubin paper.

    Returns
    -------
    features : List[float]
        List of features (cf. features.get_features_name()).

    """
    features = []

    # get a binary representation of the coordinates
    cyt, nuc = from_coord_to_matrix(cyt_coord, nuc_coord)
    rna_coord = rna_coord + stack.get_offset_value()

    # fill in masks
    mask_cyt, mask_nuc = stack.get_surface_layers(cyt, nuc, cast_float=False)

    # compute distance maps for the cytoplasm and the nucleus
    distance_cyt, distance_nuc = stack.get_distance_layers(cyt, nuc)

    # get rna outside nucleus
    mask_rna_in = mask_nuc[rna_coord[:, 1], rna_coord[:, 2]]
    rna_coord_out = rna_coord[~mask_rna_in]

    # get centroids
    centroid_cyt = get_centroid_surface(mask_cyt)
    centroid_nuc = get_centroid_surface(mask_nuc)
    centroid_rna = get_centroid_rna(rna_coord)
    if len(rna_coord_out) == 0:
        centroid_rna_out = centroid_cyt.copy()
    else:
        centroid_rna_out = get_centroid_rna(rna_coord_out)

    # get centroid distance maps
    distance_cyt_centroid = get_centroid_distance_map(centroid_cyt, mask_cyt)
    distance_nuc_centroid = get_centroid_distance_map(centroid_nuc, mask_cyt)
    distance_rna_out_centroid = get_centroid_distance_map(centroid_rna_out,
                                                          mask_cyt)
    # Aubin's features
    if features_aubin:

        # compute features
        a = features_distance_aubin(rna_coord, distance_cyt, distance_nuc,
                                    distance_cyt_centroid,
                                    distance_nuc_centroid)
        b = feature_in_out_nucleus_aubin(rna_coord, mask_nuc)
        opening_sizes = [15, 30, 45, 60]
        c = features_opening_aubin(opening_sizes, rna_coord, mask_cyt)
        radii = [r for r in range(40)]
        d = features_ripley_aubin(radii, rna_coord, cyt_coord, mask_cyt)
        e = feature_polarization_aubin(distance_cyt, distance_cyt_centroid,
                                       centroid_rna)
        f = feature_dispersion_aubin(rna_coord, mask_cyt, centroid_rna)

        # gather features
        features_to_add = a + [b] + c + d + [e] + [f]
        features += features_to_add

    # other features
    if features_no_aubin:

        # compute features
        aa = features_distance(rna_coord_out, distance_cyt, distance_nuc)
        bb = feature_in_out_nucleus(rna_coord, mask_nuc)
        opening_sizes = [15, 30, 45, 60]
        cc = features_protrusion(opening_sizes, rna_coord_out, mask_cyt)
        radii = [r for r in range(40)]
        dd = features_ripley(radii, rna_coord_out, mask_cyt)
        ee = feature_polarization(centroid_rna_out, centroid_cyt,
                                  distance_cyt_centroid)
        ff = feature_dispersion(rna_coord_out, distance_rna_out_centroid,
                                mask_cyt)
        gg = feature_peripheral_dispersion(rna_coord_out,
                                           distance_cyt_centroid,
                                           mask_cyt)
        hh = features_topography(rna_coord, mask_cyt, mask_nuc)
        ii = features_foci(rna_coord_out, distance_cyt, distance_nuc, mask_cyt,
                           mask_nuc)
        jj = feature_area(mask_cyt, mask_nuc)

        # gather features
        features_to_add = aa + [bb] + cc + dd + ee + [ff] + [gg] + hh + ii + jj
        features += features_to_add

    features = np.array(features, dtype=np.float32)

    return features


def get_features_name(features_aubin=True, features_no_aubin=False):
    """Return the current list of features names.

    Parameters
    ----------
    features_aubin : bool
        Compute features from Aubin paper.
    features_no_aubin : bool
        Compute features that are not present in Aubin paper.

    Returns
    -------
    features_name : List[str]
        A list of features name.

    """
    features_name = []
    if features_aubin:
        features_to_add = ["aubin_average_dist_cyt",
                           "aubin_quantile_5_dist_cyt",
                           "aubin_quantile_10_dist_cyt",
                           "aubin_quantile_20_dist_cyt",
                           "aubin_quantile_50_dist_cyt",
                           "aubin_average_dist_cyt_centroid",
                           "aubin_average_dist_nuc",
                           "aubin_average_dist_nuc_centroid",
                           "aubin_ratio_in_nuc",
                           "aubin_diff_opening_15",
                           "aubin_diff_opening_30",
                           "aubin_diff_opening_45",
                           "aubin_diff_opening_60",
                           "aubin_ripley_max",
                           "aubin_ripley_max_gradient",
                           "aubin_ripley_min_gradient",
                           "aubin_ripley_monotony",
                           "aubin_ripley_mid_cell",
                           "aubin_ripley_max_radius",
                           "aubin_polarization_index",
                           "aubin_dispersion_index"]
        features_name += features_to_add

    if features_no_aubin:
        features_to_add = ["mean_distance_cyt",
                           "median_distance_cyt",
                           "std_distance_cyt",
                           "mean_distance_nuc",
                           "median_distance_nuc",
                           "std_distance_nuc",
                           "proportion_in_nuc",
                           "diff_opening_15",
                           "diff_opening_30",
                           "diff_opening_45",
                           "diff_opening_60",
                           "nb_rna_opening_15",
                           "nb_rna_opening_30",
                           "nb_rna_opening_45",
                           "nb_rna_opening_60",
                           "ripley_max",
                           "ripley_min",
                           "ripley_max_gradient",
                           "ripley_min_gradient",
                           "ripley_monotony",
                           "ripley_max_radius",
                           "polarization_score",
                           "polarization_score_normalized",
                           "dispersion_index",
                           "peripheral_dispersion_index",
                           "rna_nuc_edge",
                           "rna_nuc_10_20",
                           "rna_nuc_20_30",
                           "rna_cyt_0_10",
                           "rna_cyt_10_20",
                           "rna_cyt_20_30",
                           "nb_low_density_foci",
                           "ratio_rna_foci_0_10",
                           "ratio_rna_foci_10_20",
                           "foci_mean_distance_cyt",
                           "foci_median_distance_cyt",
                           "foci_std_distance_cyt",
                           "foci_mean_distance_nuc",
                           "foci_median_distance_nuc",
                           "foci_std_distance_nuc",
                           "relative_area_nuc",
                           "area_cyt",
                           "area_nuc"]
        features_name += features_to_add

    return features_name


# ### Prepare the data ###

def from_coord_to_matrix(cyt_coord, nuc_coord):
    # get size of the frame
    max_y = cyt_coord[:, 0].max() + stack.get_offset_value() * 2
    max_x = cyt_coord[:, 1].max() + stack.get_offset_value() * 2
    image_shape = (max_y, max_x)

    # cytoplasm
    cyt = np.zeros(image_shape, dtype=bool)
    cyt[cyt_coord[:, 0] + stack.get_offset_value(),
        cyt_coord[:, 1] + stack.get_offset_value()] = True

    # nucleus
    nuc = np.zeros(image_shape, dtype=bool)
    nuc[nuc_coord[:, 0] + stack.get_offset_value(),
        nuc_coord[:, 1] + stack.get_offset_value()] = True

    return cyt, nuc


def get_centroid_surface(mask):
    # get centroid
    region = regionprops(mask.astype(np.uint8))[0]
    centroid = np.array(region.centroid, dtype=np.int64)

    return centroid


def get_centroid_rna(rna_coord):
    # get rna centroids
    centroid_rna = np.mean(rna_coord[:, :3], axis=0, dtype=np.int64)
    return centroid_rna


def get_centroid_distance_map(centroid_coordinate, mask_cyt):
    if centroid_coordinate.size == 3:
        centroid_coordinate_2d = centroid_coordinate[1:]
    else:
        centroid_coordinate_2d = centroid_coordinate.copy()

    # get mask centroid
    mask_centroid = np.zeros_like(mask_cyt)
    mask_centroid[centroid_coordinate_2d[0], centroid_coordinate_2d[1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centroid)
    distance_map[mask_cyt == 0] = 0
    distance_map /= distance_map.max()
    distance_map = distance_map.astype(np.float32)

    return distance_map


# ### Aubin's features ###

def features_distance_aubin(rna_coord, distance_cyt, distance_nuc,
                            distance_cyt_centroid, distance_nuc_centroid):
    rna_coord_2d = rna_coord[:, 1:3]

    # compute average distances to cytoplasm and quantiles
    factor = distance_cyt[distance_cyt > 0].mean()
    distance_rna_cyt = distance_cyt[rna_coord_2d[:, 0], rna_coord_2d[:, 1]]
    mean_distance_cyt = distance_rna_cyt.mean() / factor
    quantile_5_distance_cyt = np.percentile(distance_rna_cyt, 5)
    quantile_5_distance_cyt /= factor
    quantile_10_distance_cyt = np.percentile(distance_rna_cyt, 10)
    quantile_10_distance_cyt /= factor
    quantile_20_distance_cyt = np.percentile(distance_rna_cyt, 20)
    quantile_20_distance_cyt /= factor
    quantile_50_distance_cyt = np.percentile(distance_rna_cyt, 50)
    quantile_50_distance_cyt /= factor

    # compute average distances to cytoplasm centroid
    factor = distance_cyt_centroid[distance_cyt > 0].mean()
    distance_rna_cyt_centroid = distance_cyt_centroid[rna_coord_2d[:, 0],
                                                      rna_coord_2d[:, 1]]
    mean_distance_cyt_centroid = distance_rna_cyt_centroid.mean()
    mean_distance_cyt_centroid /= factor

    # compute average distances to nucleus
    factor = distance_nuc[distance_cyt > 0].mean()
    distance_rna_nuc = distance_nuc[rna_coord_2d[:, 0], rna_coord_2d[:, 1]]
    mean_distance_nuc = distance_rna_nuc.mean() / factor

    # compute average distances to nucleus centroid
    factor = distance_nuc_centroid[distance_cyt > 0].mean()
    distance_rna_nuc_centroid = distance_nuc_centroid[rna_coord_2d[:, 0],
                                                      rna_coord_2d[:, 1]]
    mean_distance_nuc_centroid = distance_rna_nuc_centroid.mean()
    mean_distance_nuc_centroid /= factor

    features = [mean_distance_cyt, quantile_5_distance_cyt,
                quantile_10_distance_cyt, quantile_20_distance_cyt,
                quantile_50_distance_cyt, mean_distance_cyt_centroid,
                mean_distance_nuc, mean_distance_nuc_centroid]

    return features


def feature_in_out_nucleus_aubin(rna_coord, mask_nuc):
    # compute the ratio between rna in and out nucleus
    mask_rna_in = mask_nuc[rna_coord[:, 1], rna_coord[:, 2]]
    rna_in = rna_coord[mask_rna_in]
    rna_out = rna_coord[~mask_rna_in]
    feature = len(rna_in) / max(len(rna_out), 1)

    return feature


def features_opening_aubin(opening_sizes, rna_coord, mask_cyt):
    # get number of rna
    nb_rna = len(rna_coord)

    # apply opening operator and count the loss of rna
    features = []
    for size in opening_sizes:
        s = disk(size, dtype=bool)
        mask_cyt_transformed = binary_opening(mask_cyt, selem=s)
        mask_rna = mask_cyt_transformed[rna_coord[:, 1], rna_coord[:, 2]]
        rna_after_opening = rna_coord[mask_rna]

        nb_rna_after_opening = len(rna_after_opening)
        diff_opening = (nb_rna - nb_rna_after_opening) / nb_rna
        features.append(diff_opening)

    return features


def features_ripley_aubin(radii, rna_coord, cyt_coord, mask_cyt):
    # compute corrected Ripley values for different radii
    values = _ripley_values_2d(radii, rna_coord, mask_cyt)

    # smooth them using moving average
    smoothed_values = _moving_average(values, n=4)

    # compute the gradients of these values
    gradients = np.gradient(smoothed_values)

    # compute features
    index_max = np.argmax(smoothed_values)
    max_radius = radii[index_max]
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
    big_value = _ripley_values_2d([big_radius],rna_coord, mask_cyt)[0]
    features = [max_value, max_gradient, min_gradient, monotony, big_value,
                max_radius]

    return features


def _ripley_values_2d(radii, rna_coord, mask_cyt):
    rna_coord_2d = rna_coord[:, 1:3]

    # sort rna coordinates
    sorted_indices = np.lexsort((rna_coord_2d[:, 1], rna_coord_2d[:, 0]))
    rna_coord_2d_sorted = rna_coord_2d[sorted_indices]

    # compute distance matrix between rna and rna density
    distances = distance_matrix(rna_coord_2d_sorted, rna_coord_2d_sorted, p=2)
    factor = len(rna_coord_2d_sorted) ** 2 / mask_cyt.sum()

    # cast cytoplasm mask in np.uint8
    mask_cyt_8bit = stack.cast_img_uint8(mask_cyt)

    # for each radius, get neighbors and weight
    values = []
    for r in radii:
        mask_distance = distances.copy()
        mask_distance = mask_distance <= r
        nb_neighbors = np.sum(mask_distance, axis=0) - 1
        weights = stack.mean_filter(mask_cyt_8bit,
                                    kernel_shape="disk",
                                    kernel_size=r)
        weights = weights.astype(np.float32) / 255.
        rna_weights = weights[rna_coord_2d_sorted[:, 0],
                              rna_coord_2d_sorted[:, 1]]
        nb_neighbors_weighted = np.multiply(nb_neighbors, rna_weights)
        value = nb_neighbors_weighted.sum() / factor
        values.append(value)
    values = np.array(values, dtype=np.float32)
    values_corrected = np.sqrt(values / np.pi) - np.array(radii)

    return values_corrected


def _moving_average(a, n=4):
    res = np.cumsum(a, dtype=np.float32)
    res[n:] = res[n:] - res[:-n]
    averaged_array = res[n - 1:] / n

    return averaged_array


def feature_polarization_aubin(distance_cyt, distance_cyt_centroid,
                               centroid_rna):
    # compute polarization index
    factor = np.mean(distance_cyt_centroid[distance_cyt > 0])
    distance_rna_cell = distance_cyt_centroid[centroid_rna[1], centroid_rna[2]]
    feature = distance_rna_cell / factor

    return feature


def feature_dispersion_aubin(rna_coord, mask_cyt, centroid_rna):
    rna_coord_2d = rna_coord[:, 1:3]
    centroid_rna_2d = centroid_rna[1:]

    # get coordinates of each pixel of the cell
    mask_cyt_coord = np.nonzero(mask_cyt)
    mask_cyt_coord = np.column_stack(mask_cyt_coord)

    # compute dispersion index
    sigma_rna = np.sum((rna_coord_2d - centroid_rna_2d) ** 2, axis=0)
    sigma_rna = np.sum(sigma_rna / len(rna_coord_2d))
    sigma_cell = np.sum((mask_cyt_coord - centroid_rna_2d) ** 2, axis=0)
    sigma_cell = np.sum(sigma_cell / len(mask_cyt_coord))
    feature = sigma_rna / sigma_cell

    return feature


# ### Other features ###

def features_distance(rna_coord_out, distance_cyt, distance_nuc):
    rna_coord_out_2d = rna_coord_out[:, 1:3]
    if len(rna_coord_out_2d) == 0:
        features = [1., 1., 1., 1., 1., 1.]
        return features

    # compute statistics from distance to cytoplasm
    distance_rna_cyt = distance_cyt[rna_coord_out_2d[:, 0],
                                    rna_coord_out_2d[:, 1]]
    factor = np.mean(distance_cyt[distance_nuc > 0])
    mean_distance_cyt = np.mean(distance_rna_cyt) / factor
    factor = np.median(distance_cyt[distance_nuc > 0])
    median_distance_cyt = np.median(distance_rna_cyt) / factor
    factor = np.std(distance_cyt[distance_nuc > 0])
    std_distance_cyt = np.std(distance_rna_cyt) / factor

    # compute statistics from distance to nucleus
    distance_rna_nuc = distance_nuc[rna_coord_out_2d[:, 0],
                                    rna_coord_out_2d[:, 1]]
    factor = np.mean(distance_nuc[distance_nuc > 0])
    mean_distance_nuc = np.mean(distance_rna_nuc) / factor
    factor = np.median(distance_nuc[distance_nuc > 0])
    median_distance_nuc = np.median(distance_rna_nuc) / factor
    factor = np.std(distance_nuc[distance_nuc > 0])
    std_distance_nuc = np.std(distance_rna_nuc) / factor

    features = [mean_distance_cyt, median_distance_cyt, std_distance_cyt,
                mean_distance_nuc, median_distance_nuc, std_distance_nuc]

    return features


def feature_in_out_nucleus(rna_coord, mask_nuc):
    # compute the proportion of rna in the nucleus
    mask_rna_in = mask_nuc[rna_coord[:, 1], rna_coord[:, 2]]
    rna_in = rna_coord[mask_rna_in]
    feature = len(rna_in) / len(rna_coord)

    return feature


def features_protrusion(opening_sizes, rna_coord_out, mask_cyt):
    # get number of rna outside nucleus
    nb_rna_out = len(rna_coord_out)

    # case where we do not detect any rna outside the nucleus
    if nb_rna_out == 0:
        features = [0. for _ in opening_sizes] * 2
        return features

    # apply opening operator and count the loss of rna outside the nucleus
    features_opening = []
    features_count = []
    for size in opening_sizes:
        s = disk(size, dtype=bool)
        mask_cyt_transformed = binary_opening(mask_cyt, selem=s)
        mask_rna = mask_cyt_transformed[rna_coord_out[:, 1],
                                        rna_coord_out[:, 2]]
        rna_after_opening = rna_coord_out[mask_rna]
        nb_rna_out_after_opening = len(rna_after_opening)
        diff_opening = (nb_rna_out - nb_rna_out_after_opening) / nb_rna_out
        features_opening.append(diff_opening)
        nb_rna_protrusion = nb_rna_out - nb_rna_out_after_opening
        features_count.append(nb_rna_protrusion)

    # gather features
    features = features_opening + features_count

    return features


def features_ripley(radii, rna_coord_out, mask_cyt):
    # case where we do not detect any rna outside the nucleus
    if len(rna_coord_out) == 0:
        features = [0., 0., 0., 0., 0., 0.]
        return features

    # compute corrected Ripley values for different radii
    values = _ripley_values_3d(radii, rna_coord_out, mask_cyt)

    # smooth them using moving average
    smoothed_values = _moving_average(values, n=4)

    # compute the gradients of these values
    gradients = np.gradient(smoothed_values)

    # compute features
    index_max = np.argmax(smoothed_values)
    max_radius = radii[index_max]
    max_value = smoothed_values.max()
    min_value = smoothed_values.min()
    if index_max == 0:
        max_gradient = gradients[0]
    else:
        max_gradient = max(gradients[:index_max])
    if index_max == len(gradients) - 1:
        min_gradient = gradients[-1]
    else:
        min_gradient = min(gradients[index_max:])
    monotony, _ = spearmanr(smoothed_values, radii[2:-1])

    features = [max_value, min_value, max_gradient, min_gradient,
                monotony, max_radius]

    return features


def _ripley_values_3d(radii, rna_coord_out, mask_cyt):
    rna_coord_out_3d = rna_coord_out[:, :3]

    # sort rna coordinates
    sorted_indices = np.lexsort((rna_coord_out_3d[:, 0],
                                 rna_coord_out_3d[:, 2],
                                 rna_coord_out_3d[:, 1]))
    rna_coord_out_3d = rna_coord_out_3d[sorted_indices]

    # compute distance matrix between rna and rna density
    distances = distance_matrix(rna_coord_out_3d, rna_coord_out_3d, p=2)
    factor = len(rna_coord_out_3d) ** 2 / mask_cyt.sum()

    # cast cytoplasm mask in np.uint8
    mask_cyt_8bit = stack.cast_img_uint8(mask_cyt)

    # for each radius, get neighbors and weight
    values = []
    for r in radii:
        mask_distance = distances.copy()
        mask_distance = mask_distance <= r
        nb_neighbors = np.sum(mask_distance, axis=0) - 1
        weights = stack.mean_filter(mask_cyt_8bit,
                                    kernel_shape="disk",
                                    kernel_size=r)
        weights = weights.astype(np.float32) / 255.
        rna_weights = weights[rna_coord_out_3d[:, 1], rna_coord_out_3d[:, 2]]
        nb_neighbors_weighted = np.multiply(nb_neighbors, rna_weights)
        value = nb_neighbors_weighted.sum() / factor
        values.append(value)
    values = np.array(values, dtype=np.float32)
    values_corrected = np.sqrt(values / np.pi) - np.array(radii)

    return values_corrected


def feature_polarization(centroid_rna_out, centroid_cyt,
                         distance_cyt_centroid):
    centroid_rna_out_2d = centroid_rna_out[1:]

    # compute polarization index
    polarization_index = np.linalg.norm(centroid_rna_out_2d - centroid_cyt)
    factor = distance_cyt_centroid.max()
    polarization_index_normalized = polarization_index / factor
    feature = [polarization_index, polarization_index_normalized]

    return feature


def feature_dispersion(rna_coord_out, distance_rna_centroid, mask_cyt):
    if len(rna_coord_out) == 0:
        return 1.

    # get coordinates of each pixel of the cell
    all_cell_coord = np.nonzero(mask_cyt)
    all_cell_coord = np.column_stack(all_cell_coord)

    # compute dispersion index
    a = distance_rna_centroid[rna_coord_out[:, 1], rna_coord_out[:, 2]]
    b = distance_rna_centroid[all_cell_coord[:, 0], all_cell_coord[:, 1]]
    feature = a.mean() / b.mean()

    return feature


def feature_peripheral_dispersion(rna_coord_out, distance_cyt_centroid,
                                  mask_cyt):
    if len(rna_coord_out) == 0:
        return 1.

    # get coordinates of each pixel of the cell
    all_cell_coord = np.nonzero(mask_cyt)
    all_cell_coord = np.column_stack(all_cell_coord)

    # compute dispersion index
    a = distance_cyt_centroid[rna_coord_out[:, 1], rna_coord_out[:, 2]]
    b = distance_cyt_centroid[all_cell_coord[:, 0], all_cell_coord[:, 1]]
    feature = a.mean() / b.mean()

    return feature


def features_topography(rna_coord, mask_cyt, mask_nuc):
    mask_cyt_bool = mask_cyt > 0
    mask_cyt_bool[:, 0] = False
    mask_cyt_bool[0, :] = False
    mask_nuc_bool = mask_nuc > 0
    mask_nuc_bool[:, 0] = False
    mask_nuc_bool[0, :] = False

    # build nucleus topography
    distance_map_nuc_out = ndi.distance_transform_edt(~mask_nuc_bool)
    mask_cyt_without_nuc = mask_cyt_bool.copy()
    mask_cyt_without_nuc[mask_nuc_bool] = 0
    distance_map_nuc_in = ndi.distance_transform_edt(~mask_cyt_without_nuc)
    distance_map_nuc = distance_map_nuc_out + distance_map_nuc_in
    distance_map_nuc[~mask_cyt_bool] = 0
    distance_map_nuc_edge = distance_map_nuc < 10
    distance_map_nuc_edge[~mask_cyt_bool] = False
    distance_map_nuc_10_20 = distance_map_nuc < 20
    distance_map_nuc_10_20[mask_nuc_bool] = False
    distance_map_nuc_10_20[distance_map_nuc_edge] = False
    distance_map_nuc_10_20[~mask_cyt_bool] = False
    distance_map_nuc_20_30 = distance_map_nuc < 30
    distance_map_nuc_20_30[mask_nuc_bool] = False
    distance_map_nuc_20_30[distance_map_nuc_edge] = False
    distance_map_nuc_20_30[distance_map_nuc_10_20] = False
    distance_map_nuc_20_30[~mask_cyt_bool] = False

    # build cytoplasm topography
    distance_map_cyt = ndi.distance_transform_edt(mask_cyt_bool)
    distance_map_cyt_0_10 = distance_map_cyt < 10
    distance_map_cyt_0_10[~mask_cyt_bool] = False
    distance_map_cyt_10_20 = distance_map_cyt < 20
    distance_map_cyt_10_20[~mask_cyt_bool] = False
    distance_map_cyt_10_20[distance_map_cyt_0_10] = False
    distance_map_cyt_20_30 = distance_map_cyt < 30
    distance_map_cyt_20_30[~mask_cyt_bool] = False
    distance_map_cyt_20_30[distance_map_cyt_0_10] = False
    distance_map_cyt_20_30[distance_map_cyt_10_20] = False

    # count rna for each topographic level
    cell_area = mask_cyt_bool.sum()
    nb_rna = len(rna_coord)

    factor = nb_rna * distance_map_nuc_edge.sum() / cell_area
    mask_rna = distance_map_nuc_edge[rna_coord[:, 1], rna_coord[:, 2]]
    rna_nuc_edge = len(rna_coord[mask_rna]) / factor

    factor = nb_rna * distance_map_nuc_10_20.sum() / cell_area
    mask_rna = distance_map_nuc_10_20[rna_coord[:, 1], rna_coord[:, 2]]
    rna_nuc_10_20 = len(rna_coord[mask_rna]) / factor

    factor = nb_rna * distance_map_nuc_20_30.sum() / cell_area
    mask_rna = distance_map_nuc_20_30[rna_coord[:, 1], rna_coord[:, 2]]
    rna_nuc_20_30 = len(rna_coord[mask_rna]) / factor

    factor = nb_rna * distance_map_cyt_0_10.sum() / cell_area
    mask_rna = distance_map_cyt_0_10[rna_coord[:, 1], rna_coord[:, 2]]
    rna_cyt_0_10 = len(rna_coord[mask_rna]) / factor

    factor = nb_rna * distance_map_cyt_10_20.sum() / cell_area
    mask_rna = distance_map_cyt_10_20[rna_coord[:, 1], rna_coord[:, 2]]
    rna_cyt_10_20 = len(rna_coord[mask_rna]) / factor

    factor = nb_rna * distance_map_cyt_20_30.sum() / cell_area
    mask_rna = distance_map_cyt_20_30[rna_coord[:, 1], rna_coord[:, 2]]
    rna_cyt_20_30 = len(rna_coord[mask_rna]) / factor

    features = [rna_nuc_edge, rna_nuc_10_20, rna_nuc_20_30,
                rna_cyt_0_10, rna_cyt_10_20, rna_cyt_20_30]

    return features


def features_foci(rna_coord_out, distance_cyt, distance_nuc, mask_cyt,
                  mask_nuc):
    if len(rna_coord_out) == 0:
        return [0., 1., 1., 1., 1., 1., 1., 1., 1.]

    # detect low density foci
    clustered_spots = detection.cluster_spots(spots=rna_coord_out[:, :3],
                                              resolution_z=300,
                                              resolution_yx=103,
                                              radius=650,
                                              nb_min_spots=5)
    foci = detection.extract_foci(clustered_spots=clustered_spots)
    nb_low_density_foci = len(foci)

    # get regular foci id
    rna_coord_out_foci = rna_coord_out[rna_coord_out[:, 3] != -1, :]
    if len(rna_coord_out_foci) == 0:
        return [nb_low_density_foci, 0., 0., 1., 1., 1., 1., 1., 1.]
    l_id_foci = list(set(rna_coord_out_foci[:, 3]))

    # count foci neighbors
    rna_foci_0_10 = []
    rna_foci_10_20 = []
    foci_coord = []
    for id_foci in l_id_foci:
        rna_foci = rna_coord_out_foci[rna_coord_out_foci[:, 3] == id_foci, :3]
        foci = np.mean(rna_foci, axis=0).reshape(1, 3)
        foci_coord.append(foci)
        distance = distance_matrix(rna_coord_out_foci[:, :3], foci)
        mask_distance_0_10 = distance < 10
        mask_distance_10_20 = distance < 20
        mask_distance_10_20 &= ~mask_distance_0_10
        nb_rna_foci_0_10 = mask_distance_0_10.sum()
        nb_rna_foci_10_20 = mask_distance_10_20.sum()
        rna_foci_0_10.append(nb_rna_foci_0_10)
        rna_foci_10_20.append(nb_rna_foci_10_20)

    # compute expected ratio
    area_0_10 = np.pi * 10 ** 2
    area_0_20 = np.pi * 20 ** 2
    area_10_20 = area_0_20 - area_0_10
    area_cyt_no_nuc = mask_cyt.sum() - mask_nuc.sum()
    factor_0_10 = len(rna_coord_out) * area_0_10 / area_cyt_no_nuc
    factor_10_20 = len(rna_coord_out) * area_10_20 / area_cyt_no_nuc
    ratio_rna_foci_0_10 = np.mean(rna_foci_0_10) / factor_0_10
    ratio_rna_foci_10_20 = np.mean(rna_foci_10_20) / factor_10_20

    # get foci coordinates
    foci_coord = np.array(foci_coord, dtype=np.int64)
    foci_coord = np.squeeze(foci_coord, axis=1)
    foci_coord_2d = foci_coord[:, 1:3]

    # compute statistics from distance to cytoplasm
    distance_foci_cyt = distance_cyt[foci_coord_2d[:, 0],
                                     foci_coord_2d[:, 1]]
    factor = np.mean(distance_cyt[distance_nuc > 0])
    foci_mean_distance_cyt = np.mean(distance_foci_cyt) / factor
    factor = np.median(distance_cyt[distance_nuc > 0])
    foci_median_distance_cyt = np.median(distance_foci_cyt) / factor
    factor = np.std(distance_cyt[distance_nuc > 0])
    foci_std_distance_cyt = np.std(distance_foci_cyt) / factor

    # compute statistics from distance to nucleus
    distance_foci_nuc = distance_nuc[foci_coord_2d[:, 0],
                                     foci_coord_2d[:, 1]]
    factor = np.mean(distance_nuc[distance_nuc > 0])
    foci_mean_distance_nuc = np.mean(distance_foci_nuc) / factor
    factor = np.median(distance_nuc[distance_nuc > 0])
    foci_median_distance_nuc = np.median(distance_foci_nuc) / factor
    factor = np.std(distance_nuc[distance_nuc > 0])
    foci_std_distance_nuc = np.std(distance_foci_nuc) / factor

    features = [nb_low_density_foci,
                ratio_rna_foci_0_10, ratio_rna_foci_10_20,
                foci_mean_distance_cyt, foci_median_distance_cyt,
                foci_std_distance_cyt, foci_mean_distance_nuc,
                foci_median_distance_nuc, foci_std_distance_nuc]

    return features


def feature_area(mask_cyt, mask_nuc):
    # get area of the cytoplasm and the nucleus
    area_cyt = mask_cyt.sum()
    area_nuc = mask_nuc.sum()

    # compute relative area of the nucleus
    relative_area_nuc = area_nuc / area_cyt

    # return features
    features = [relative_area_nuc, area_cyt, area_nuc]

    return features
