# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to craft features.
"""

import warnings
import numpy as np
from scipy import ndimage as ndi

from skimage.morphology import binary_opening
from skimage.morphology.selem import disk

import bigfish.stack as stack
from .input_preparation import prepare_extracted_data


# ### Main functions ###

def compute_features(cell_mask, nuc_mask, ndim, rna_coord, voxel_size_yx=None,
                     centrosome_coord=None, compute_distance=True,
                     compute_intranuclear=True, compute_protrusion=True,
                     compute_dispersion=True, compute_topography=True,
                     compute_foci=True, compute_area=True,
                     compute_centrosome=True):
    """Compute requested features.

    Parameters
    ----------
    cell_mask : np.ndarray, np.uint, np.int or bool
        Surface of the cell with shape (y, x).
    nuc_mask: np.ndarray, np.uint, np.int or bool
        Surface of the nucleus with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider (2 or 3).
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1. If cluster id is not provided foci related
        features are not computed.
    voxel_size_yx : int, float or None
        Size of a voxel on the yx plan, in nanometer.
    centrosome_coord : np.ndarray, np.int64
        Coordinates of the detected centrosome with shape (nb_elements, 3) or
        (nb_elements, 2). One coordinate per dimension (zyx or yx dimensions).
        These coordinates are mandatory to compute centrosome related features.
    compute_distance : bool
        Compute distance related features.
    compute_intranuclear : bool
        Compute nucleus related features.
    compute_protrusion : bool
        Compute protrusion related features.
    compute_dispersion : bool
        Compute dispersion indices.
    compute_topography : bool
        Compute topographic features.
    compute_foci : bool
        Compute foci related features.
    compute_area : bool
        Compute area related features.
    compute_centrosome : bool
        Compute centrosome related features.

    Returns
    -------
    features : np.ndarray, np.float32
        Array of features.

    """
    # check parameters
    stack.check_parameter(voxel_size_yx=(int, float, type(None)),
                          compute_distance=bool,
                          compute_intranuclear=bool,
                          compute_protrusion=bool,
                          compute_dispersion=bool,
                          compute_topography=bool,
                          compute_foci=bool,
                          compute_area=bool,
                          compute_centrosome=bool)

    # prepare input data
    (cell_mask,
     distance_cell, distance_cell_normalized,
     centroid_cell, distance_centroid_cell,
     nuc_mask, cell_mask_out_nuc,
     distance_nuc, distance_nuc_normalized,
     centroid_nuc, distance_centroid_nuc,
     rna_coord_out_nuc,
     centroid_rna, distance_centroid_rna,
     centroid_rna_out_nuc, distance_centroid_rna_out_nuc,
     distance_centrosome) = prepare_extracted_data(
        cell_mask, nuc_mask, ndim, rna_coord, centrosome_coord)

    # initialization
    features = ()

    # distance related features
    if compute_distance:
        features += features_distance(
            rna_coord, distance_cell, distance_nuc, cell_mask, ndim, False)

    # nucleus related features
    if compute_intranuclear:
        features += features_in_out_nucleus(
            rna_coord, rna_coord_out_nuc, False)

    # protrusion related features
    # if compute_protrusion:
    #     features += features_protrusion(
    #         rna_coord_out_nuc, cell_mask, nuc_mask, cell_mask_out_nuc, ndim,
    #         False)

    # dispersion indices
    if compute_dispersion:
        features += features_polarization(
            centroid_rna, centroid_cell, centroid_nuc,
            distance_centroid_cell, distance_centroid_nuc, ndim, False)
        dispersion_index = features_dispersion(
            rna_coord, distance_centroid_rna, cell_mask, ndim, False)
        peripheral_dispersion_index = features_peripheral_dispersion(
            rna_coord, distance_centroid_cell, cell_mask, ndim, False)
        features += (dispersion_index, peripheral_dispersion_index)

    # topographic features
    if compute_topography and voxel_size_yx is not None:
        features += features_topography(
            rna_coord, cell_mask, nuc_mask, cell_mask_out_nuc, ndim,
            voxel_size_yx, False)
    elif compute_topography and voxel_size_yx is None:
        raise ValueError("Topographic features can't be computed because "
                         "'voxel_size_yx' is not provided.")

    # foci related features
    # if compute_foci:
    #     features += features_foci(
    #         rna_coord_out_nuc, distance_cell, distance_nuc, cell_mask_out_nuc,
    #         ndim, False)

    # area related features
    if compute_area:
        features += features_area(
            cell_mask, nuc_mask, cell_mask_out_nuc, False)

    # centrosome related features
    if (compute_centrosome and centrosome_coord is not None
            and voxel_size_yx is not None):
        features += features_centrosome(
            rna_coord, distance_centrosome, cell_mask, ndim, voxel_size_yx,
            False)
    elif compute_centrosome and centrosome_coord is None:
        raise ValueError("Centrosome related features can't be computed "
                         "because 'centrosome_coord' is not provided.")
    elif (compute_centrosome and voxel_size_yx is None):
        raise ValueError("Centrosome related features can't be computed "
                         "because 'voxel_size_yx' is not provided.")

    # format features
    features = np.array(features, dtype=np.float32)
    features = np.round(features, decimals=2)

    return features


def get_features_name(names_features_distance=True,
                      names_features_intranuclear=True,
                      names_features_protrusion=True,
                      names_features_dispersion=True,
                      names_features_topography=True,
                      names_features_foci=True,
                      names_features_area=True,
                      names_features_centrosome=True):
    """Return the current list of features names.

    Parameters
    ----------
    names_features_distance : bool
        Return names of features related to distances from nucleus or cell
        membrane.
    names_features_intranuclear : bool
        Return names of features related to nucleus.
    names_features_protrusion : bool
        Return names of features related to protrusions.
    names_features_dispersion : bool
        Return names of features used to quantify mRNAs dispersion within the
        cell.
    names_features_topography : bool
        Return names of topographic features of the cell.
    names_features_foci : bool
        Return names of features related to foci.
    names_features_area : bool
        Return names of features related to area of the cell.
    names_features_centrosome : bool
        Return names of features related to centrosome.

    Returns
    -------
    features_name : List[str]
        A list of features name.

    """
    # check parameters
    stack.check_parameter(names_features_distance=bool,
                          names_features_intranuclear=bool,
                          names_features_protrusion=bool,
                          names_features_dispersion=bool,
                          names_features_topography=bool,
                          names_features_foci=bool,
                          names_features_area=bool,
                          names_features_centrosome=bool)

    # initialization
    features_name = []

    # get feature names
    if names_features_distance:
        features_name += ["index_mean_distance_cell",
                          "index_median_distance_cell",
                          "index_mean_distance_nuc",
                          "index_median_distance_nuc"]

    if names_features_intranuclear:
        features_name += ["proportion_rna_in_nuc",
                          "nb_rna_out_nuc",
                          "nb_rna_in_nuc"]

    #if names_features_protrusion:
    #    features_name += ["index_rna_opening_30",
    #                      "proportion_rna_opening_30",
    #                      "area_opening_30"]

    if names_features_dispersion:
        features_name += ["score_polarization_cell",
                          "score_polarization_nuc",
                          "index_dispersion",
                          "index_peripheral_dispersion"]

    if names_features_topography:
        features_name += ["index_rna_nuc_edge",
                          "proportion_rna_nuc_edge"]

        a = 500
        for b in range(1000, 3001, 500):
            features_name += ["index_rna_nuc_radius_{}_{}".format(a, b),
                              "proportion_rna_nuc_radius_{}_{}".format(a, b)]
            a = b

        a = 0
        for b in range(500, 3001, 500):
            features_name += ["index_rna_cell_radius_{}_{}".format(a, b),
                              "proportion_rna_cell_radius_{}_{}".format(a, b)]
            a = b

    #if names_features_foci:
    #    features_name += ["proportion_rna_in_foci",
    #                      "index_foci_mean_distance_cyt",
    #                      "index_foci_median_distance_cyt",
    #                      "index_foci_mean_distance_nuc",
    #                      "index_foci_median_distance_nuc"]

    if names_features_area:
        features_name += ["proportion_nuc_area",
                          "cell_area",
                          "nuc_area",
                          "cell_area_out_nuc"]

    if names_features_centrosome:
        features_name += ["index_mean_distance_centrosome",
                          "index_median_distance_centrosome",
                          "index_rna_centrosome",
                          "proportion_rna_centrosome"]

    return features_name


# ### Features functions ###

def features_distance(rna_coord, distance_cell, distance_nuc, cell_mask, ndim,
                      check_input=True):
    """Compute distance related features.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    distance_cell : np.ndarray, np.float32
        Distance map from the cell with shape (y, x).
    distance_nuc : np.ndarray, np.float32
        Distance map from the nucleus with shape (y, x).
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_mean_dist_cell : float
        Normalized mean distance of RNAs to the cell membrane.
    index_median_dist_cell : float
        Normalized median distance of RNAs to the cell membrane.
    index_mean_dist_nuc : float
        Normalized mean distance of RNAs to the nucleus.
    index_median_dist_nuc : float
        Normalized median distance of RNAs to the nucleus.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
        stack.check_array(distance_cell, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])
        stack.check_array(distance_nuc, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask, ndim=2, dtype=bool)

    # case where no mRNAs are detected
    if len(rna_coord) == 0:
        features = (1., 1., 1., 1.)
        return features

    # compute mean and median distance to cell membrane
    rna_distance_cell = distance_cell[rna_coord[:, ndim - 2],
                                      rna_coord[:, ndim - 1]]
    expected_distance = np.mean(distance_cell[cell_mask])
    index_mean_dist_cell = np.mean(rna_distance_cell) / expected_distance
    expected_distance = np.median(distance_cell[cell_mask])
    index_median_dist_cell = np.median(rna_distance_cell) / expected_distance

    features = (index_mean_dist_cell, index_median_dist_cell)

    # compute mean and median distance to nucleus
    rna_distance_nuc = distance_nuc[rna_coord[:, ndim - 2],
                                    rna_coord[:, ndim - 1]]
    expected_distance = np.mean(distance_nuc[cell_mask])
    index_mean_dist_nuc = np.mean(rna_distance_nuc) / expected_distance
    expected_distance = np.median(distance_nuc[cell_mask])
    index_median_dist_nuc = np.median(rna_distance_nuc) / expected_distance

    features += (index_mean_dist_nuc, index_median_dist_nuc)

    return features


def features_in_out_nucleus(rna_coord, rna_coord_out_nuc, check_input=True):
    """Compute nucleus related features.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    rna_coord_out_nuc : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns. Spots detected inside the nucleus are removed.
    check_input : bool
        Check input validity.

    Returns
    -------
    proportion_rna_in_nuc : float
        Proportion of RNAs detected inside the nucleus.
    nb_rna_out_nuc : float
        Number of RNAs detected outside the nucleus.
    nb_rna_in_nuc : float
        Number of RNAs detected inside the nucleus.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
        stack.check_array(rna_coord_out_nuc, ndim=2, dtype=np.int64)

    # count total number of rna
    nb_rna = float(len(rna_coord))

    # case where no rna are detected
    if nb_rna == 0:
        features = (0., 0., 0.)
        return features

    # number of rna outside and inside nucleus
    nb_rna_out_nuc = float(len(rna_coord_out_nuc))
    nb_rna_in_nuc = nb_rna - nb_rna_out_nuc

    # compute the proportion of rna in the nucleus
    proportion_rna_in_nuc = nb_rna_in_nuc / nb_rna

    features = (proportion_rna_in_nuc, nb_rna_out_nuc, nb_rna_in_nuc)

    return features


# TODO refactor protrusion feature
def features_protrusion(rna_coord_out_nuc, cell_mask, nuc_mask,
                        cell_mask_out_nuc, ndim, check_input=True):
    """Compute protrusion related features.

    Parameters
    ----------
    rna_coord_out_nuc : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns. Spots detected inside the nucleus are removed.
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    nuc_mask : np.ndarray, bool
        Surface of the nucleus with shape (y, x).
    cell_mask_out_nuc : np.ndarray, bool
        Surface of the cell (outside the nucleus) with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------

    """
    # TODO fin a better feature for the protrusion (idea: dilate region from
    #  centroid and stop when a majority of new pixels do not belong to the
    #  cell).

    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord_out_nuc, ndim=2, dtype=np.int64)
        stack.check_array(cell_mask, ndim=2, dtype=bool)
        stack.check_array(nuc_mask, ndim=2, dtype=bool)
        stack.check_array(cell_mask_out_nuc, ndim=2, dtype=bool)

    # get number of rna outside nucleus and cell area
    nb_rna_out_nuc = len(rna_coord_out_nuc)
    area_nuc = nuc_mask.sum()
    area_cyt_out = cell_mask_out_nuc.sum()

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


def features_polarization(centroid_rna, centroid_cell, centroid_nuc,
                          distance_centroid_cell, distance_centroid_nuc, ndim,
                          check_input=True):
    """Compute polarization related features.

    # TODO add reference to RDI paper

    Parameters
    ----------
    centroid_rna : np.ndarray, np.int64
        Coordinates of the rna centroid with shape (1, 2).
    centroid_cell : np.ndarray, np.int64
        Coordinates of the cell centroid with shape (1, 2).
    centroid_nuc : np.ndarray, np.int64
        Coordinates of the nucleus centroid with shape (1, 2).
    distance_centroid_cell : np.ndarray, np.float32
        Distance map from the cell centroid with shape (y, x).
    distance_centroid_nuc : np.ndarray, np.float32
        Distance map from the nucleus centroid with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_polarization_cell : float
        Polarization index around the cell centroid.
    index_polarization_nuc : float
        Polarization index around the nucleus centroid.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(centroid_rna, ndim=2, dtype=np.int64)
        stack.check_array(centroid_cell, ndim=2, dtype=np.int64)
        stack.check_array(centroid_nuc, ndim=2, dtype=np.int64)
        stack.check_array(distance_centroid_cell, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])
        stack.check_array(distance_centroid_nuc, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])

    # initialization
    if ndim == 3:
        centroid_rna_2d = centroid_rna[1:]
    else:
        centroid_rna_2d = centroid_rna.copy()

    # compute polarization index from cell centroid
    centroid_distance = np.linalg.norm(centroid_rna_2d - centroid_cell)
    maximum_distance = distance_centroid_cell.max()
    index_polarization_cell = centroid_distance / maximum_distance

    # compute polarization index from nucleus centroid
    centroid_distance = np.linalg.norm(centroid_rna_2d - centroid_nuc)
    maximum_distance = distance_centroid_nuc.max()
    index_polarization_nuc = centroid_distance / maximum_distance

    # gather features
    features = (index_polarization_cell, index_polarization_nuc)

    return features


def features_dispersion(rna_coord, distance_centroid_rna, cell_mask, ndim,
                        check_input=True):
    """Compute a dispersion index.

    # TODO add reference to RDI paper

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    distance_centroid_rna : np.ndarray, np.float32
        Distance map from the rna centroid with shape (y, x).
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_dispersion : float
        Dispersion index.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
        stack.check_array(distance_centroid_rna, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask, ndim=2, dtype=bool)

    # case where no mRNAs are detected
    if len(rna_coord) == 0:
        features = 1.
        return features

    # get coordinates of each pixel of the cell
    cell_coord = np.nonzero(cell_mask)
    cell_coord = np.column_stack(cell_coord)

    # compute dispersion index
    a = distance_centroid_rna[rna_coord[:, ndim - 2],
                              rna_coord[:, ndim - 1]]
    b = distance_centroid_rna[cell_coord[:, 0],
                              cell_coord[:, 1]]
    index_dispersion = a.mean() / b.mean()

    return index_dispersion


def features_peripheral_dispersion(rna_coord, distance_centroid_cell,
                                   cell_mask, ndim, check_input=True):
    """Compute a peripheral dispersion index.

    # TODO add reference to RDI paper

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    distance_centroid_cell : np.ndarray, np.float32
        Distance map from the cell centroid with shape (y, x).
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_peripheral_dispersion : float
        Peripheral dispersion index.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
        stack.check_array(distance_centroid_cell, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask, ndim=2, dtype=bool)

    # case where no mRNAs are detected
    if len(rna_coord) == 0:
        features = 1.
        return features

    # get coordinates of each pixel of the cell
    cell_coord = np.nonzero(cell_mask)
    cell_coord = np.column_stack(cell_coord)

    # compute peripheral dispersion index
    a = distance_centroid_cell[rna_coord[:, ndim - 2],
                               rna_coord[:, ndim - 1]]
    b = distance_centroid_cell[cell_coord[:, 0],
                               cell_coord[:, 1]]
    index_peripheral_dispersion = a.mean() / b.mean()

    return index_peripheral_dispersion


def features_topography(rna_coord, cell_mask, nuc_mask, cell_mask_out_nuc,
                        ndim, voxel_size_yx,
                        check_input=True):
    """Compute topographic features.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    nuc_mask : np.ndarray, bool
        Surface of the nucleus with shape (y, x).
    cell_mask_out_nuc : np.ndarray, bool
        Surface of the cell (outside the nucleus) with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_rna_nuc_marge : float
        Number of RNAs detected in a specific region around nucleus and
        normalized by the expected number of RNAs under random distribution.
        Six regions are targeted (less than 500nm, 500-1000nm, 1000-1500nm,
        1500-2000nm, 2000-2500nm and 2500-3000nm from the nucleus boundary).
    proportion_rna_nuc_marge : float
        Proportion of RNAs detected in a specific region around nucleus. Six
        regions are targeted (less than 500nm, 500-1000nm, 1000-1500nm,
        1500-2000nm, 2000-2500nm and 2500-3000nm from the nucleus boundary).
    index_rna_cell_marge : float
        Number of RNAs detected in a specific region around cell membrane and
        normalized by the expected number of RNAs under random distribution.
        Six regions are targeted (0-500nm, 500-1000nm, 1000-1500nm,
        1500-2000nm, 2000-2500nm and 2500-3000nm from the cell membrane).
    proportion_rna_cell_marge : float
        Proportion of RNAs detected in a specific region around around cell
        membrane. Six regions are targeted (0-500nm, 500-1000nm, 1000-1500nm,
        1500-2000nm, 2000-2500nm and 2500-3000nm from the cell membrane).

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int,
                              voxel_size_yx=(int, float))
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
        stack.check_array(cell_mask, ndim=2, dtype=bool)
        stack.check_array(nuc_mask, ndim=2, dtype=bool)
        stack.check_array(cell_mask_out_nuc, ndim=2, dtype=bool)

    # initialization
    cell_area = cell_mask.sum()
    nb_rna = len(rna_coord)
    marge = int(500 / voxel_size_yx)
    if marge < 1:
        warnings.warn(UserWarning, "'voxel_size_yx' is greater than 500 "
                                   "nanometers ({0}). Topographic features "
                                   "use contour lines every 500 nanometers so "
                                   "they can't be computed here."
                      .format(voxel_size_yx))
        features = (0., 0.)
        features += (0., 0.) * 5
        features += (0., 0.) * 6
        return features

    # case where no mRNAs are detected
    if nb_rna == 0:
        features = (0., 0.)
        features += (0., 0.) * 5
        features += (0., 0.) * 6
        return features

    # build a distance map from nucleus boundary and from cell membrane
    distance_map_nuc_out = ndi.distance_transform_edt(~nuc_mask)
    distance_map_nuc_in = ndi.distance_transform_edt(~cell_mask_out_nuc)
    distance_map_nuc = distance_map_nuc_out + distance_map_nuc_in
    distance_map_nuc[~cell_mask] = 0.
    distance_map_cell = ndi.distance_transform_edt(cell_mask)

    # count mRNAs along nucleus edge (less than 500nm to the nucleus boundary)
    mask_nuc_marge = distance_map_nuc < marge
    mask_nuc_marge[~cell_mask] = False
    marge_area = max(mask_nuc_marge.sum(), 1)
    expected_rna_nuc_marge = nb_rna * marge_area / cell_area
    mask_rna = mask_nuc_marge[rna_coord[:, ndim-2], rna_coord[:, ndim-1]]
    nb_rna_nuc_marge = len(rna_coord[mask_rna])
    index_rna_nuc_marge = nb_rna_nuc_marge / expected_rna_nuc_marge
    proportion_rna_nuc_marge = nb_rna_nuc_marge / nb_rna

    features = (index_rna_nuc_marge, proportion_rna_nuc_marge)

    # count mRNAs in specific regions around nucleus (500-1000nm, 1000-1500nm,
    # 1500-2000nm, 2000-2500nm, 2500-3000nm)
    mask_cumulated_marge = mask_nuc_marge.copy()
    for i in range(2, 7):
        mask_nuc_marge = distance_map_nuc < i * marge
        mask_nuc_marge[~cell_mask] = False
        mask_nuc_marge[nuc_mask] = False
        mask_nuc_marge[mask_cumulated_marge] = False
        mask_cumulated_marge |= mask_nuc_marge
        marge_area = max(mask_nuc_marge.sum(), 1)
        expected_rna_nuc_marge = nb_rna * marge_area / cell_area
        mask_rna = mask_nuc_marge[rna_coord[:, ndim-2], rna_coord[:, ndim-1]]
        nb_rna_nuc_marge = len(rna_coord[mask_rna])
        index_rna_nuc_marge = nb_rna_nuc_marge / expected_rna_nuc_marge
        proportion_rna_nuc_marge = nb_rna_nuc_marge / nb_rna

        features += (index_rna_nuc_marge, proportion_rna_nuc_marge)

    # count mRNAs in specific regions around cell membrane (0-500nm,
    # 500-1000nm, 1000-1500nm, 1500-2000nm, 2000-2500nm, 2500-3000nm)
    mask_cumulated_marge = np.zeros_like(cell_mask)
    for i in range(1, 7):
        mask_cell_marge = distance_map_cell < i * marge
        mask_cell_marge[~cell_mask] = False
        mask_cell_marge[nuc_mask] = False
        mask_cell_marge[mask_cumulated_marge] = False
        mask_cumulated_marge |= mask_cell_marge
        marge_area = max(mask_cell_marge.sum(), 1)
        expected_rna_cell_marge = nb_rna * marge_area / cell_area
        mask_rna = mask_cell_marge[rna_coord[:, ndim-2], rna_coord[:, ndim-1]]
        nb_rna_cell_marge = len(rna_coord[mask_rna])
        index_rna_cell_marge = nb_rna_cell_marge / expected_rna_cell_marge
        proportion_rna_cell_marge = nb_rna_cell_marge / nb_rna

        features += (index_rna_cell_marge, proportion_rna_cell_marge)

    return features


# TODO refactor foci feature
def features_foci(foci_coord, distance_cell, distance_nuc, cell_mask_out_nuc,
                  ndim, check_input=True):
    """Compute foci related features.

    Parameters
    ----------
    foci_coord : np.ndarray, np.int64
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension for the foci centroid (zyx or yx coordinates), the number of
        spots detected in the foci and its index.
    rna_coord_out_nuc : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1. Spots detected inside the nucleus are removed.
    distance_cell : np.ndarray, np.float32
        Distance map from the cell with shape (y, x).
    distance_nuc : np.ndarray, np.float32
        Distance map from the nucleus with shape (y, x).
    cell_mask_out_nuc : np.ndarray, bool
        Surface of the cell (outside the nucleus) with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    check_input : bool
        Check input validity.

    Returns
    -------
    proportion_rna_in_foci : float
        Proportion of RNAs detected in a foci.
    index_foci_mean_distance_cyt : float

    index_foci_med_distance_cyt : float

    index_foci_mean_distance_nuc : float

    index_foci_med_distance_nuc : float

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord_out_nuc, ndim=2, dtype=np.int64)
        if rna_coord_out_nuc.shape[1] != ndim + 1:
            raise ValueError("'rna_coord_out_nuc' should have {0} spatial "
                             "dimensions and an extra value to assign a "
                             "potential foci, not {1}."
                             .format(ndim, rna_coord_out_nuc.shape[1]))
        stack.check_array(distance_cell, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])
        stack.check_array(distance_nuc, ndim=2,
                          dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask_out_nuc, ndim=2, dtype=bool)

    # case where no mRNAs outside the nucleus are detected
    if len(rna_coord_out_nuc) == 0:
        features = (0., 1., 1., 1., 1)
        return features

    # case where no default foci are detected
    mask_rna_out_nuc_in_foci = rna_coord_out_nuc[:, ndim] != -1
    rna_coord_out_nuc_in_foci = rna_coord_out_nuc[mask_rna_out_nuc_in_foci, :]
    if len(rna_coord_out_nuc_in_foci) == 0:
        features = (0., 1., 1., 1., 1)
        return features

    # compute proportion of mRNAs in foci
    nb_rna_in_foci = len(rna_coord_out_nuc_in_foci)
    nb_rna = len(rna_coord_out_nuc)
    proportion_rna_in_foci = nb_rna_in_foci / nb_rna

    # get regular foci id
    l_id_foci = list(set(rna_coord_out_nuc_in_foci[:, ndim]))

    # get foci coordinates
    foci_coord = []
    for id_foci in l_id_foci:
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

    features = (proportion_rna_in_foci, index_foci_mean_distance_cyt,
                index_foci_med_distance_cyt, index_foci_mean_distance_nuc,
                index_foci_med_distance_nuc)

    return features


def features_area(cell_mask, nuc_mask, cell_mask_out_nuc, check_input=True):
    """Compute area related features.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    nuc_mask : np.ndarray, bool
        Surface of the nucleus with shape (y, x).
    cell_mask_out_nuc : np.ndarray, bool
        Surface of the cell (outside the nucleus) with shape (y, x).
    check_input : bool
        Check input validity.

    Returns
    -------
    nuc_relative_area : float
        Proportion of nucleus area in the cell.
    cell_area : float
        Cell area (in pixels).
    nuc_area : float
        Nucleus area (in pixels).
    cell_area_out_nuc : float
        Cell area outside the nucleus (in pixels).

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_array(cell_mask, ndim=2, dtype=bool)
        stack.check_array(nuc_mask, ndim=2, dtype=bool)
        stack.check_array(cell_mask_out_nuc, ndim=2, dtype=bool)

    # get area of the cell and the nucleus
    cell_area = float(cell_mask.sum())
    nuc_area = float(nuc_mask.sum())

    # compute relative area of the nucleus
    nuc_relative_area = nuc_area / cell_area

    # compute area of the cell outside nucleus
    cell_area_out_nuc = float(cell_mask_out_nuc.sum())

    # return features
    features = (nuc_relative_area, cell_area, nuc_area, cell_area_out_nuc)

    return features


def features_centrosome(rna_coord, distance_centrosome, cell_mask, ndim,
                        voxel_size_yx, check_input=True):
    """Compute centrosome related features.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns.
    distance_centrosome : np.ndarray, np.float32
        Distance map from the centrosome with shape (y, x).
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    check_input : bool
        Check input validity.

    Returns
    -------
    index_mean_dist_cent : float
        Normalized mean distance of RNAs to the closest centrosome.
    index_median_dist_cent : float
        Normalized median distance of RNAs to the closest centrosome.
    index_rna_centrosome : float
        Number of RNAs within a 2000nm radius from a centrosome, normalized by
        the expected number of RNAs under random distribution.
    proportion_rna_centrosome : float
        Proportion of RNAs within a 2000nm radius from a centrosome.

    """
    # check parameters
    stack.check_parameter(check_input=bool)
    if check_input:
        stack.check_parameter(ndim=int)
        if ndim not in [2, 3]:
            raise ValueError("'ndim' should be 2 or 3, not {0}.".format(ndim))
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
        if distance_centrosome is not None:
            stack.check_array(distance_centrosome, ndim=2,
                              dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask, ndim=2, dtype=bool)

    # case where no centrosome are detected
    if distance_centrosome is None:
        features = (1., 1., 1., 0.)
        return features

    # case where no mRNAs are detected
    if len(rna_coord) == 0:
        features = (1., 1., 1., 0.)
        return features

    # initialization
    nb_rna = len(rna_coord)
    cell_area = cell_mask.sum()

    # compute mean and median distances from the centrosomes
    rna_distance_cent = distance_centrosome[rna_coord[:, ndim - 2],
                                            rna_coord[:, ndim - 1]]
    expected_distance = np.mean(distance_centrosome[cell_mask])
    index_mean_dist_cent = np.mean(rna_distance_cent) / expected_distance
    expected_distance = np.median(distance_centrosome[cell_mask])
    index_median_dist_cent = np.median(rna_distance_cent) / expected_distance

    features = (index_mean_dist_cent, index_median_dist_cent)

    # compute proportion of mRNAs next to the centrosomes (<2000nm)
    radius = int(2000 / voxel_size_yx)
    mask_centrosome = distance_centrosome < radius
    mask_centrosome[~cell_mask] = False
    centrosome_area = max(mask_centrosome.sum(), 1)
    expected_nb_rna = nb_rna * centrosome_area / cell_area
    mask_rna = mask_centrosome[rna_coord[:, ndim - 2], rna_coord[:, ndim - 1]]
    nb_rna_centrosome = len(rna_coord[mask_rna])
    index_rna_centrosome = nb_rna_centrosome / expected_nb_rna
    proportion_rna_centrosome = nb_rna_centrosome / len(rna_coord)

    features += (index_rna_centrosome, proportion_rna_centrosome)

    return features
