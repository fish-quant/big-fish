# -*- coding: utf-8 -*-

"""
Functions to prepare input (coordinates or images).
"""

import os
import threading

import bigfish.stack as stack

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from skimage.measure import regionprops
from skimage.draw import polygon_perimeter
from sklearn.preprocessing import LabelEncoder


# TODO define the requirements for 'data'
# TODO add logging
# TODO generalize the use of 'get_offset_value'
# TODO add documentation

# ### Prepare 2-d coordinates in order to compute the hand-crafted features ###

def prepare_coordinate_data(cyt_coord, nuc_coord, rna_coord):
    """

    Parameters
    ----------
    cyt_coord
    nuc_coord
    rna_coord

    Returns
    -------

    """
    # convert coordinates in binary mask surfaces
    mask_cyt, mask_nuc, _, rna_coord = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord,
        external_coord=True)

    # get mask cytoplasm outside nucleus
    mask_cyt_out = mask_cyt.copy()
    mask_cyt_out[mask_nuc] = False

    # compute distance maps for the cytoplasm and the nucleus
    distance_cyt = ndi.distance_transform_edt(mask_cyt)
    distance_nuc_ = ndi.distance_transform_edt(~mask_nuc)
    distance_nuc = mask_cyt * distance_nuc_

    # cast matrices in float32
    mask_cyt = stack.cast_img_float32(mask_cyt)
    mask_nuc = stack.cast_img_float32(mask_nuc)
    mask_cyt_out = stack.cast_img_float32(mask_cyt_out)
    distance_cyt = distance_cyt.astype(np.float32)
    distance_nuc = distance_nuc.astype(np.float32)

    # normalize distance maps between 0 and 1
    distance_cyt_normalized = distance_cyt / distance_cyt.max()
    distance_cyt_normalized = stack.cast_img_float32(distance_cyt_normalized)
    distance_nuc_normalized = distance_nuc / distance_nuc.max()
    distance_nuc_normalized = stack.cast_img_float32(distance_nuc_normalized)

    # get rna outside nucleus
    mask_rna_in = mask_nuc[rna_coord[:, 1], rna_coord[:, 2]].astype(bool)
    rna_coord_out = rna_coord[~mask_rna_in]

    # get centroids
    centroid_cyt = _get_centroid_surface(mask_cyt)
    centroid_nuc = _get_centroid_surface(mask_nuc)
    centroid_rna = _get_centroid_rna(rna_coord)
    if len(rna_coord_out) == 0:
        centroid_rna_out = centroid_cyt.copy()
    else:
        centroid_rna_out = _get_centroid_rna(rna_coord_out)

    # get centroid distance maps
    distance_cyt_centroid = _get_centroid_distance_map(centroid_cyt, mask_cyt)
    distance_nuc_centroid = _get_centroid_distance_map(centroid_nuc, mask_cyt)
    distance_rna_out_centroid = _get_centroid_distance_map(centroid_rna_out,
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


def _get_centroid_surface(mask):
    # get centroid
    region = regionprops(mask.astype(np.uint8))[0]
    centroid = np.array(region.centroid, dtype=np.int64)

    return centroid


def _get_centroid_rna(rna_coord):
    # get rna centroids
    centroid_rna = np.mean(rna_coord[:, :3], axis=0, dtype=np.int64)
    return centroid_rna


def _get_centroid_distance_map(centroid_coordinate, mask_cyt):
    if centroid_coordinate.size == 3:
        centroid_coordinate_2d = centroid_coordinate[1:]
    else:
        centroid_coordinate_2d = centroid_coordinate.copy()

    # get mask centroid
    mask_centroid = np.zeros_like(mask_cyt).astype(bool)
    mask_centroid[centroid_coordinate_2d[0], centroid_coordinate_2d[1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centroid)
    distance_map[mask_cyt == 0] = 0
    distance_map = distance_map.astype(np.float32)

    return distance_map


# ### Prepare 2-d images for deep learning classification models ###

def build_boundaries_layers(cyt_coord, nuc_coord, rna_coord):
    """

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).

    Returns
    -------
    cyt_boundaries : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing cytoplasm boundaries.
        border.
    nuc_boundaries : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing nucleus boundaries.
    rna_layer : np.ndarray, np.float32
        Binary image of mRNAs localizations with shape (y, x).

    """
    # check parameters
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])

    # build surface binary matrices from coordinates
    cyt_surface, nuc_surface, rna_layer, _ = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord)

    # from surface binary matrices to boundaries binary matrices
    cyt_boundaries = stack.from_surface_to_boundaries(cyt_surface)
    nuc_boundaries = stack.from_surface_to_boundaries(nuc_surface)

    # cast layer in float32
    cyt_boundaries = stack.cast_img_float32(cyt_boundaries)
    nuc_boundaries = stack.cast_img_float32(nuc_boundaries)
    rna_layer = stack.cast_img_float32(rna_layer)

    return cyt_boundaries, nuc_boundaries, rna_layer


def build_surface_layers(cyt_coord, nuc_coord, rna_coord):
    """Compute plain surface layers as input for the model.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).

    Returns
    -------
    cyt_surface : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing cytoplasm surface.
        border.
    nuc_surface : np.ndarray, np.float32
        A 2-d binary tensor with shape (y, x) showing nucleus surface.
    rna_layer : np.ndarray, np.float32
        Binary image of mRNAs localizations with shape (y, x).

    """
    # check parameters
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])

    # build surface binary matrices from coordinates
    cyt_surface, nuc_surface, rna_layer, _ = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord)

    # cast layer in float32
    cyt_surface = stack.cast_img_float32(cyt_surface)
    nuc_surface = stack.cast_img_float32(nuc_surface)
    rna_layer = stack.cast_img_float32(rna_layer)

    return cyt_surface, nuc_surface, rna_layer


def build_distance_layers(cyt_coord, nuc_coord, rna_coord, normalized=True):
    """Compute distance layers as input for the model.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).
    normalized : bool
        Normalized the layers between 0 and 1.
    Returns
    -------
    distance_cyt : np.ndarray, np.float32
        A 2-d tensor with shape (y, x) showing distance to the cytoplasm
        border. Normalize between 0 and 1 if 'normalized' True.
    distance_nuc : np.ndarray, np.float32
        A 2-d tensor with shape (y, x) showing distance to the nucleus border.
        Normalize between 0 and 1 if 'normalized' True.

        """
    # check parameters
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])
    stack.check_parameter(normalized=bool)

    # build surface binary matrices from coordinates
    cyt_surface, nuc_surface, rna_layer, _ = stack.from_coord_to_surface(
        cyt_coord=cyt_coord,
        nuc_coord=nuc_coord,
        rna_coord=rna_coord)

    # compute distances map for cytoplasm and nucleus
    cyt_distance = ndi.distance_transform_edt(cyt_surface)
    nuc_distance_ = ndi.distance_transform_edt(~nuc_surface)
    nuc_distance = cyt_surface * nuc_distance_

    if normalized:
        # cast to np.float32 and normalize it between 0 and 1
        cyt_distance = cyt_distance / cyt_distance.max()
        nuc_distance = nuc_distance / nuc_distance.max()

    # cast layer in float32
    cyt_distance = stack.cast_img_float32(cyt_distance)
    nuc_distance = stack.cast_img_float32(nuc_distance)
    rna_layer = stack.cast_img_float32(rna_layer)

    return cyt_distance, nuc_distance, rna_layer
