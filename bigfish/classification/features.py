# -*- coding: utf-8 -*-

"""
Functions to craft features.
"""

import bigfish.stack as stack

import numpy as np
from scipy import ndimage as ndi

from skimage.measure import regionprops
from skimage.morphology import binary_opening
from skimage.morphology.selem import disk

# TODO add sanity check functions
# TODO add documentation
# TODO allow to return intermediate results (distance map, etc.)
# TODO round float results


def get_features(cyt_coord, nuc_coord, rna_coord,
                 compute_distance=True,
                 compute_intranuclear=True,
                 compute_protrusion=True,
                 compute_dispersion=True,
                 compute_topography=True,
                 compute_foci=True,
                 compute_area=True):
    """Compute cell features.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Coordinate yx of the cytoplasm boundary with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Coordinate yx of the cytoplasm boundary with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Coordinate zyx of the detected rna, plus the index of a potential foci.
        Shape (nb_rna, 4).
    compute_distance : bool
        Compute features related to distances from nucleus or cytoplasmic
        membrane.
    compute_intranuclear : bool
        Compute features related to intranuclear pattern.
    compute_protrusion : bool
        Compute features related to protrusion pattern.
    compute_dispersion : bool
        Compute features to quantify mRNAs dispersion within the cell.
    compute_topography : bool
        Compute topographic features of the cell.
    compute_foci : bool
        Compute features related to foci pattern.
    compute_area : bool
        Compute features related to area of the cell.

    Returns
    -------
    features : List[float]
        List of features (cf. features.get_features_name()).

    """
    features = []

    # prepare input data
    (mask_cyt, mask_nuc, mask_cyt_out,
     distance_cyt, distance_nuc,
     distance_cyt_normalized, distance_nuc_normalized,
     rna_coord_out,
     centroid_cyt, centroid_nuc,
     centroid_rna, centroid_rna_out,
     distance_cyt_centroid, distance_nuc_centroid,
     distance_rna_out_centroid) = prepare_coordinate_data(cyt_coord,
                                                          nuc_coord,
                                                          rna_coord)

    # distances related features
    if compute_distance:
        aa = features_distance(rna_coord_out,
                               distance_cyt,
                               distance_nuc,
                               mask_cyt_out)

        features += aa

    # intranuclear related features
    if compute_intranuclear:
        bb = features_in_out_nucleus(rna_coord,
                                     rna_coord_out)

        features += bb

    # intranuclear related features
    if compute_protrusion:
        cc = features_protrusion(rna_coord_out,
                                 mask_cyt,
                                 mask_nuc,
                                 mask_cyt_out)

        features += cc

    # dispersion measures
    if compute_dispersion:
        dd = features_polarization(centroid_rna_out,
                                   centroid_cyt,
                                   centroid_nuc,
                                   distance_cyt_centroid,
                                   distance_nuc_centroid)
        ee = features_dispersion(rna_coord_out,
                                 distance_rna_out_centroid,
                                 mask_cyt_out)
        ff = features_peripheral_dispersion(rna_coord_out,
                                            distance_cyt_centroid,
                                            mask_cyt_out)

        features += dd + ee + ff

    # topographic features
    if compute_topography:
        gg = features_topography(rna_coord, rna_coord_out, mask_cyt, mask_nuc,
                                 mask_cyt_out)

        features += gg

    # foci related features
    if compute_foci:
        hh = features_foci(rna_coord_out,
                           distance_cyt,
                           distance_nuc,
                           mask_cyt_out)

        features += hh

    # area related features
    if compute_area:
        ii = features_area(mask_cyt, mask_nuc, mask_cyt_out)

        features += ii

    features = np.array(features, dtype=np.float32)
    features = np.round(features, decimals=2)

    return features


def get_features_name(names_features_distance=True,
                      names_features_intranuclear=True,
                      names_features_protrusion=True,
                      names_features_dispersion=True,
                      names_features_topography=True,
                      names_features_foci=True,
                      names_features_area=True):
    """Return the current list of features names.

    Parameters
    ----------
    names_features_distance : bool
        Return names of features related to distances from nucleus or
        cytoplasmic membrane.
    names_features_intranuclear : bool
        Return names of features related to intranuclear pattern.
    names_features_protrusion : bool
        Return names of features related to protrusion pattern.
    names_features_dispersion : bool
        Return names of features used to quantify mRNAs dispersion within the
        cell.
    names_features_topography : bool
        Return names of topographic features of the cell.
    names_features_foci : bool
        Return names of features related to foci pattern.
    names_features_area : bool
        Return names of features related to area of the cell.

    Returns
    -------
    features_name : List[str]
        A list of features name.

    """
    features_name = []

    if names_features_distance:
        features_name += ["index_mean_distance_cyt",
                          "index_median_distance_cyt",
                          "index_mean_distance_nuc",
                          "index_median_distance_nuc"]

    if names_features_intranuclear:
        features_name += ["proportion_rna_in_nuc",
                          "nb_rna_out",
                          "nb_rna_in"]

    if names_features_protrusion:
        features_name += ["index_rna_opening_30",
                          "proportion_rna_opening_30",
                          "area_opening_30"]

    if names_features_dispersion:
        features_name += ["score_polarization_cyt",
                          "score_polarization_nuc",
                          "index_dispersion",
                          "index_peripheral_dispersion"]

    if names_features_topography:
        features_name += ["index_rna_nuc_edge",
                          "proportion_rna_nuc_edge"]

        a = 5
        for b in range(10, 31, 5):
            features_name += ["index_rna_nuc_radius_{}_{}".format(a, b),
                              "proportion_rna_nuc_radius_{}_{}".format(a, b)]
            a = b

        a = 0
        for b in range(5, 31, 5):
            features_name += ["index_rna_cyt_radius_{}_{}".format(a, b),
                              "proportion_rna_cyt_radius_{}_{}".format(a, b)]
            a = b

    if names_features_foci:
        features_name += ["proportion_rna_in_foci",
                          "index_foci_mean_distance_cyt",
                          "index_foci_median_distance_cyt",
                          "index_foci_mean_distance_nuc",
                          "index_foci_median_distance_nuc"]

    if names_features_area:
        features_name += ["proportion_nuc_area",
                          "area_cyt",
                          "area_nuc",
                          "area_cyt_out"]

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
    distance_map = distance_map.astype(np.float32)

    return distance_map


def prepare_coordinate_data(cyt_coord, nuc_coord, rna_coord):
    # get a binary representation of the coordinates
    cyt, nuc = from_coord_to_matrix(cyt_coord, nuc_coord)
    rna_coord[:, 1:3] += stack.get_offset_value()

    # fill in masks
    mask_cyt, mask_nuc = stack.get_surface_layers(cyt, nuc, cast_float=False)

    # get mask cytoplasm outside nucleus
    mask_cyt_out = mask_cyt.copy()
    mask_cyt_out[mask_nuc] = False

    # compute distance maps for the cytoplasm and the nucleus
    distance_cyt, distance_nuc = stack.get_distance_layers(cyt, nuc,
                                                           normalized=False)

    # normalize distance maps between 0 and 1
    distance_cyt_normalized = distance_cyt / distance_cyt.max()
    distance_cyt_normalized = stack.cast_img_float32(distance_cyt_normalized)
    distance_nuc_normalized = distance_nuc / distance_nuc.max()
    distance_nuc_normalized = stack.cast_img_float32(distance_nuc_normalized)

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

    prepared_inputs = (mask_cyt, mask_nuc, mask_cyt_out,
                       distance_cyt, distance_nuc,
                       distance_cyt_normalized, distance_nuc_normalized,
                       rna_coord_out,
                       centroid_cyt, centroid_nuc,
                       centroid_rna, centroid_rna_out,
                       distance_cyt_centroid, distance_nuc_centroid,
                       distance_rna_out_centroid)

    return prepared_inputs


# ### Other features ###

def features_distance(rna_coord_out, distance_cyt, distance_nuc, mask_cyt_out):
    # initialization
    rna_coord_out_2d = rna_coord_out[:, 1:3]

    if len(rna_coord_out_2d) == 0:
        features = [1., 1., 1., 1.]
        return features
    features = []

    # compute statistics from distance to cytoplasm
    distance_rna_cyt = distance_cyt[rna_coord_out_2d[:, 0],
                                    rna_coord_out_2d[:, 1]]
    factor = np.mean(distance_cyt[mask_cyt_out])
    index_mean_distance_cyt = np.mean(distance_rna_cyt) / factor
    factor = np.median(distance_cyt[mask_cyt_out])
    index_median_distance_cyt = np.median(distance_rna_cyt) / factor

    features += [index_mean_distance_cyt,
                 index_median_distance_cyt]

    # compute statistics from distance to nucleus
    distance_rna_nuc = distance_nuc[rna_coord_out_2d[:, 0],
                                    rna_coord_out_2d[:, 1]]
    factor = np.mean(distance_nuc[mask_cyt_out])
    index_mean_distance_nuc = np.mean(distance_rna_nuc) / factor
    factor = np.median(distance_nuc[mask_cyt_out])
    index_median_distance_nuc = np.median(distance_rna_nuc) / factor

    features += [index_mean_distance_nuc,
                 index_median_distance_nuc]

    return features


def features_in_out_nucleus(rna_coord, rna_coord_out):
    # number of mRNAs outside and inside nucleus
    nb_rna_out = len(rna_coord_out)
    nb_rna_in = len(rna_coord) - nb_rna_out

    # compute the proportion of rna in the nucleus
    proportion_rna_in = nb_rna_in / len(rna_coord)

    features = [proportion_rna_in, nb_rna_out, nb_rna_in]

    return features


def features_protrusion(rna_coord_out, mask_cyt, mask_nuc, mask_cyt_out):
    # get number of rna outside nucleus and cell area
    nb_rna_out = len(rna_coord_out)
    area_nuc = mask_nuc.sum()
    area_cyt_out = mask_cyt_out.sum()

    # apply opening operator and count the loss of rna outside the nucleus
    features = []
    for size in [30]:
        s = disk(size, dtype=bool)
        mask_cyt_transformed = binary_opening(mask_cyt, selem=s)
        mask_cyt_transformed[mask_nuc] = True
        new_area_cell_out = mask_cyt_transformed.sum() - area_nuc
        area_protrusion = area_cyt_out - new_area_cell_out

        # case where we do not detect any rna outside the nucleus
        if nb_rna_out == 0:
            features += [0., 0., area_protrusion]
            continue

        if area_protrusion > 0:
            factor = nb_rna_out * area_protrusion / area_cyt_out
            mask_rna = mask_cyt_transformed[rna_coord_out[:, 1],
                                            rna_coord_out[:, 2]]
            rna_after_opening = rna_coord_out[mask_rna]
            nb_rna_protrusion = nb_rna_out - len(rna_after_opening)
            index_rna_opening = nb_rna_protrusion / factor
            proportion_rna_opening = nb_rna_protrusion / nb_rna_out

            features += [index_rna_opening,
                         proportion_rna_opening,
                         area_protrusion]
        else:
            features += [0., 0., 0.]

    return features


def features_polarization(centroid_rna_out, centroid_cyt, centroid_nuc,
                          distance_cyt_centroid, distance_nuc_centroid):
    centroid_rna_out_2d = centroid_rna_out[1:]

    # compute polarization index from cytoplasm centroid
    polarization_distance = np.linalg.norm(centroid_rna_out_2d - centroid_cyt)
    factor = distance_cyt_centroid.max()
    feature_cyt = polarization_distance / factor

    # compute polarization index from nucleus centroid
    polarization_distance = np.linalg.norm(centroid_rna_out_2d - centroid_nuc)
    factor = distance_nuc_centroid.max()
    feature_nuc = polarization_distance / factor

    # gather features
    features = [feature_cyt,
                feature_nuc]

    return features


def features_dispersion(rna_coord_out, distance_rna_centroid, mask_cyt_out):
    # initialization
    if len(rna_coord_out) == 0:
        features = [1.]
        return features

    # get number of rna outside nucleus and cell area
    if mask_cyt_out.sum() == 0:
        features = [1.]
        return features

    # get coordinates of each pixel of the cell
    cell_outside_nuc_coord = np.nonzero(mask_cyt_out)
    cell_outside_nuc_coord = np.column_stack(cell_outside_nuc_coord)

    # compute dispersion index
    a = distance_rna_centroid[rna_coord_out[:, 1], rna_coord_out[:, 2]]
    b = distance_rna_centroid[cell_outside_nuc_coord[:, 0],
                              cell_outside_nuc_coord[:, 1]]
    index_dispersion = a.mean() / b.mean()

    features = [index_dispersion]

    return features


def features_peripheral_dispersion(rna_coord_out, distance_cyt_centroid,
                                   mask_cyt_out):
    # initialization
    if len(rna_coord_out) == 0:
        features = [1.]
        return features

    # get number of rna outside nucleus and cell area
    if mask_cyt_out.sum() == 0:
        features = [1.]
        return features

    # get coordinates of each pixel of the cell
    cell_outside_nuc_coord = np.nonzero(mask_cyt_out)
    cell_outside_nuc_coord = np.column_stack(cell_outside_nuc_coord)

    # compute dispersion index
    a = distance_cyt_centroid[rna_coord_out[:, 1], rna_coord_out[:, 2]]
    b = distance_cyt_centroid[cell_outside_nuc_coord[:, 0],
                              cell_outside_nuc_coord[:, 1]]
    index_peripheral_dispersion = a.mean() / b.mean()

    features = [index_peripheral_dispersion]

    return features


def features_topography(rna_coord, rna_coord_out, mask_cyt, mask_nuc,
                        mask_cyt_out):
    # initialization
    features = []
    cell_area = mask_cyt.sum()
    nb_rna = len(rna_coord)
    nb_rna_out = len(rna_coord_out)

    # case where no mRNAs outside the nucleus are detected
    if nb_rna_out == 0:
        features = [0., 0.]
        features += [0., 0.] * 5
        features += [0., 0.] * 6
        return features

    # build a distance map from nucleus border and from cytoplasm membrane
    distance_map_nuc_out = ndi.distance_transform_edt(~mask_nuc)
    distance_map_nuc_in = ndi.distance_transform_edt(~mask_cyt_out)
    distance_map_nuc = distance_map_nuc_out + distance_map_nuc_in
    distance_map_nuc[~mask_cyt] = 0
    distance_map_cyt = ndi.distance_transform_edt(mask_cyt)

    # count mRNAs along nucleus edge (-5 to 5 pixels)
    mask_nuc_edge = distance_map_nuc < 5
    mask_nuc_edge[~mask_cyt] = False
    factor = nb_rna * max(mask_nuc_edge.sum(), 1) / cell_area
    mask_rna = mask_nuc_edge[rna_coord[:, 1], rna_coord[:, 2]]
    nb_rna_nuc_edge = len(rna_coord[mask_rna])
    index_rna_nuc_edge = nb_rna_nuc_edge / factor
    proportion_rna_nuc_edge = nb_rna_nuc_edge / nb_rna

    features += [index_rna_nuc_edge,
                 proportion_rna_nuc_edge]

    # count mRNAs in specific regions around nucleus (5-10, 10-15, 15-20,
    # 20-25, 25-30)
    mask_cumulated_radius = mask_nuc_edge.copy()
    for radius in range(10, 31, 5):
        mask_nuc_radius = distance_map_nuc < radius
        mask_nuc_radius[~mask_cyt] = False
        mask_nuc_radius[mask_nuc] = False
        mask_nuc_radius[mask_cumulated_radius] = False
        mask_cumulated_radius |= mask_nuc_radius
        factor = nb_rna * max(mask_nuc_radius.sum(), 1) / cell_area
        mask_rna = mask_nuc_radius[rna_coord[:, 1], rna_coord[:, 2]]
        nb_rna_nuc_radius = len(rna_coord[mask_rna])
        index_rna_nuc_radius = nb_rna_nuc_radius / factor
        proportion_rna_nuc_radius = nb_rna_nuc_radius / nb_rna

        features += [index_rna_nuc_radius,
                     proportion_rna_nuc_radius]

    # count mRNAs in specific regions around cytoplasmic membrane (0-5, 5-10,
    # 10-15, 15-20, 20-25, 25-30)
    mask_cumulated_radius = np.zeros_like(mask_nuc_edge)
    for radius in range(5, 31, 5):
        mask_cyt_radius = distance_map_cyt < radius
        mask_cyt_radius[~mask_cyt] = False
        mask_cyt_radius[mask_nuc] = False
        mask_cyt_radius[mask_cumulated_radius] = False
        mask_cumulated_radius |= mask_cyt_radius
        factor = nb_rna * max(mask_cyt_radius.sum(), 1) / cell_area
        mask_rna = mask_cyt_radius[rna_coord[:, 1], rna_coord[:, 2]]
        nb_rna_cyt_radius = len(rna_coord[mask_rna])
        index_rna_cyt_radius = nb_rna_cyt_radius / factor
        proportion_rna_cyt_radius = nb_rna_cyt_radius / nb_rna

        features += [index_rna_cyt_radius,
                     proportion_rna_cyt_radius]

    return features


def features_foci(rna_coord_out, distance_cyt, distance_nuc, mask_cyt_out):
    # case where no mRNAs outside the nucleus are detected
    if len(rna_coord_out) == 0:
        features = [0., 1., 1., 1., 1]
        return features

    # case where no default foci are detected
    rna_coord_out_foci = rna_coord_out[rna_coord_out[:, 3] != -1, :]
    if len(rna_coord_out_foci) == 0:
        features = [0., 1., 1., 1., 1]
        return features

    # compute proportion of mRNAs in foci
    nb_rna_in_foci = len(rna_coord_out_foci)
    nb_rna = len(rna_coord_out)
    proportion_rna_in_foci = nb_rna_in_foci / nb_rna

    features = [proportion_rna_in_foci]

    # get regular foci id
    l_id_foci = list(set(rna_coord_out_foci[:, 3]))

    # get foci coordinates
    foci_coord = []
    for i in l_id_foci:
        rna_foci_i = rna_coord_out_foci[rna_coord_out_foci[:, 3] == i, :3]
        foci = np.mean(rna_foci_i, axis=0)
        foci = np.round(foci).astype(np.int64)
        foci_coord.append(foci.reshape(1, 3))
    foci_coord = np.array(foci_coord, dtype=np.int64)
    foci_coord = np.squeeze(foci_coord, axis=1)
    foci_coord_2d = foci_coord[:, 1:3]

    # compute statistics from distance to cytoplasm
    distance_foci_cyt = distance_cyt[foci_coord_2d[:, 0], foci_coord_2d[:, 1]]
    factor = np.mean(distance_cyt[mask_cyt_out])
    index_foci_mean_distance_cyt = np.mean(distance_foci_cyt) / factor
    factor = np.median(distance_cyt[mask_cyt_out])
    index_foci_med_distance_cyt = np.median(distance_foci_cyt) / factor

    features += [index_foci_mean_distance_cyt,
                 index_foci_med_distance_cyt]

    # compute statistics from distance to nucleus
    distance_foci_nuc = distance_nuc[foci_coord_2d[:, 0],
                                     foci_coord_2d[:, 1]]
    factor = np.mean(distance_nuc[mask_cyt_out])
    index_foci_mean_distance_nuc = np.mean(distance_foci_nuc) / factor
    factor = np.median(distance_nuc[mask_cyt_out])
    index_foci_med_distance_nuc = np.median(distance_foci_nuc) / factor

    features += [index_foci_mean_distance_nuc,
                 index_foci_med_distance_nuc]

    return features


def features_area(mask_cyt, mask_nuc, mask_cyt_out):
    # get area of the cytoplasm and the nucleus
    area_cyt = mask_cyt.sum()
    area_nuc = mask_nuc.sum()

    # compute relative area of the nucleus
    relative_area_nuc = area_nuc / area_cyt

    # compute area of the cytoplasm outside nucleus
    area_cyt_out = mask_cyt_out.sum()

    # return features
    features = [relative_area_nuc, area_cyt, area_nuc, area_cyt_out]

    return features
