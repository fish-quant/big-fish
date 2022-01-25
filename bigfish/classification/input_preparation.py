# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to prepare input data.
"""

import numpy as np
from scipy import ndimage as ndi

import bigfish.stack as stack

from skimage.measure import regionprops


# ### Input data ###

def prepare_extracted_data(cell_mask, nuc_mask=None, ndim=None, rna_coord=None,
                           centrosome_coord=None):
    """Prepare data extracted from images.

    Parameters
    ----------
    cell_mask : np.ndarray, np.uint, np.int or bool
        Surface of the cell with shape (y, x).
    nuc_mask: np.ndarray, np.uint, np.int or bool
        Surface of the nucleus with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider (2 or 3). Mandatory if
        `rna_coord` is provided.
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.
    centrosome_coord : np.ndarray, np.int64
        Coordinates of the detected centrosome with shape (nb_elements, 3) or
        (nb_elements, 2). One coordinate per dimension (zyx or yx dimensions).

    Returns
    -------
    cell_mask : np.ndarray, bool
        Surface of the cell with shape (y, x).
    distance_cell : np.ndarray, np.float32
        Distance map from the cell with shape (y, x), in pixels.
    distance_cell_normalized : np.ndarray, np.float32
        Normalized distance map from the cell with shape (y, x).
    centroid_cell : np.ndarray, np.int64
        Coordinates of the cell centroid with shape (2,).
    distance_centroid_cell : np.ndarray, np.float32
        Distance map from the cell centroid with shape (y, x), in pixels.
    nuc_mask : np.ndarray, bool
        Surface of the nucleus with shape (y, x).
    cell_mask_out_nuc : np.ndarray, bool
        Surface of the cell (outside the nucleus) with shape (y, x).
    distance_nuc : np.ndarray, np.float32
        Distance map from the nucleus with shape (y, x), in pixels.
    distance_nuc_normalized : np.ndarray, np.float32
        Normalized distance map from the nucleus with shape (y, x).
    centroid_nuc : np.ndarray, np.int64
        Coordinates of the nucleus centroid with shape (2,).
    distance_centroid_nuc : np.ndarray, np.float32
        Distance map from the nucleus centroid with shape (y, x), in pixels.
    rna_coord_out_nuc : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1. Spots detected inside the nucleus are removed.
    centroid_rna : np.ndarray, np.int64
        Coordinates of the rna centroid with shape (2,) or (3,).
    distance_centroid_rna : np.ndarray, np.float32
        Distance map from the rna centroid with shape (y, x), in pixels.
    centroid_rna_out_nuc : np.ndarray, np.int64
        Coordinates of the rna centroid (outside the nucleus) with shape (2,)
        or (3,).
    distance_centroid_rna_out_nuc : np.ndarray, np.float32
        Distance map from the rna centroid (outside the nucleus) with shape
        (y, x), in pixels.
    distance_centrosome : np.ndarray, np.float32
        Distance map from the centrosome with shape (y, x), in pixels.

    """
    # TODO allow RNA coordinates in float64 and int64
    # check parameters
    stack.check_parameter(ndim=(int, type(None)))
    if rna_coord is not None and ndim is None:
        raise ValueError("'ndim' should be specified (2 or 3).")

    # check arrays and make masks binary
    stack.check_array(
        cell_mask,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, bool])
    cell_mask = cell_mask.astype(bool)
    if nuc_mask is not None:
        stack.check_array(
            nuc_mask,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64, bool])
        nuc_mask = nuc_mask.astype(bool)
    if rna_coord is not None:
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
    if centrosome_coord is not None:
        stack.check_array(centrosome_coord, ndim=2, dtype=np.int64)

    # build distance map from the cell boundaries
    distance_cell = ndi.distance_transform_edt(cell_mask)
    distance_cell = distance_cell.astype(np.float32)
    distance_cell_normalized = distance_cell / distance_cell.max()

    # get cell centroid and a distance map from its localisation
    centroid_cell = _get_centroid_surface(cell_mask)
    distance_centroid_cell = _get_centroid_distance_map(
        centroid_cell,
        cell_mask)

    # prepare arrays relative to the nucleus
    if nuc_mask is not None:

        # get cell mask outside nucleus
        cell_mask_out_nuc = cell_mask.copy()
        cell_mask_out_nuc[nuc_mask] = False

        # build distance map from the nucleus
        distance_nuc_ = ndi.distance_transform_edt(~nuc_mask)
        distance_nuc = cell_mask * distance_nuc_
        distance_nuc = distance_nuc.astype(np.float32)
        distance_nuc_normalized = distance_nuc / distance_nuc.max()

        # get nucleus centroid and a distance map from its localisation
        centroid_nuc = _get_centroid_surface(nuc_mask)
        distance_centroid_nuc = _get_centroid_distance_map(
            centroid_nuc,
            cell_mask)

    else:
        cell_mask_out_nuc = None
        distance_nuc = None
        distance_nuc_normalized = None
        centroid_nuc = None
        distance_centroid_nuc = None

    # prepare arrays relative to the rna
    if rna_coord is not None:

        # get rna centroid
        if len(rna_coord) == 0:
            centroid_rna = np.array([0] * ndim, dtype=np.int64)
        else:
            centroid_rna = _get_centroid_rna(rna_coord, ndim)

        # build rna distance map
        distance_centroid_rna = _get_centroid_distance_map(
            centroid_rna, cell_mask)

        # combine rna and nucleus results
        if nuc_mask is not None:

            # get rna outside nucleus
            mask_rna_in_nuc = nuc_mask[rna_coord[:, ndim - 2],
                                       rna_coord[:, ndim - 1]]
            rna_coord_out_nuc = rna_coord[~mask_rna_in_nuc]

            # get rna centroid (outside nucleus)
            if len(rna_coord_out_nuc) == 0:
                centroid_rna_out_nuc = np.array([0] * ndim, dtype=np.int64)
            else:
                centroid_rna_out_nuc = _get_centroid_rna(
                    rna_coord_out_nuc,
                    ndim)

            # build rna distance map (outside nucleus)
            distance_centroid_rna_out_nuc = _get_centroid_distance_map(
                centroid_rna_out_nuc,
                cell_mask)

        else:
            rna_coord_out_nuc = None
            centroid_rna_out_nuc = None
            distance_centroid_rna_out_nuc = None

    else:
        centroid_rna = None
        distance_centroid_rna = None
        rna_coord_out_nuc = None
        centroid_rna_out_nuc = None
        distance_centroid_rna_out_nuc = None

    # prepare arrays relative to the centrosome
    if centrosome_coord is not None:

        # build distance map from centroid
        if len(centrosome_coord) == 0:
            distance_centrosome = distance_cell.copy()
        else:
            distance_centrosome = _get_centrosome_distance_map(
                centrosome_coord,
                cell_mask)

    else:
        distance_centrosome = None

    # gather cell, nucleus, rna and centrosome data
    prepared_inputs = (
        cell_mask,
        distance_cell,
        distance_cell_normalized,
        centroid_cell,
        distance_centroid_cell,
        nuc_mask,
        cell_mask_out_nuc,
        distance_nuc,
        distance_nuc_normalized,
        centroid_nuc,
        distance_centroid_nuc,
        rna_coord_out_nuc,
        centroid_rna,
        distance_centroid_rna,
        centroid_rna_out_nuc,
        distance_centroid_rna_out_nuc,
        distance_centrosome)

    return prepared_inputs


def _get_centroid_surface(mask):
    """Get centroid coordinates of a 2-d binary surface.

    Parameters
    ----------
    mask : np.ndarray, bool
        Binary surface with shape (y, x).

    Returns
    -------
    centroid : np.ndarray, np.int64
        Coordinates of the centroid with shape (2,).

    """
    # get centroid
    region = regionprops(mask.astype(np.uint8))[0]
    centroid = np.array(region.centroid, dtype=np.int64)

    return centroid


def _get_centroid_rna(rna_coord, ndim):
    """Get centroid coordinates of RNA molecules.

    Parameters
    ----------
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1.
    ndim : int
        Number of spatial dimensions to consider (2 or 3).

    Returns
    -------
    centroid_rna : np.ndarray, np.int64
        Coordinates of the rna centroid with shape (2,) or (3,).

    """
    # get rna centroids
    centroid_rna = np.mean(rna_coord[:, :ndim], axis=0, dtype=np.int64)

    return centroid_rna


def _get_centroid_distance_map(centroid, cell_mask):
    """Build distance map from a centroid localisation.

    Parameters
    ----------
    centroid : np.ndarray, np.int64
        Coordinates of the centroid with shape (2,) or (3,).
    cell_mask : np.ndarray, bool
        Binary surface of the cell with shape (y, x).

    Returns
    -------
    distance_map : np.ndarray, np.float32
        Distance map from the centroid with shape (y, x).

    """
    if centroid.size == 3:
        centroid_2d = centroid[1:]
    else:
        centroid_2d = centroid.copy()

    # get mask centroid
    mask_centroid = np.zeros_like(cell_mask)
    mask_centroid[centroid_2d[0], centroid_2d[1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centroid)
    distance_map[cell_mask == 0] = 0
    distance_map = distance_map.astype(np.float32)

    return distance_map


def _get_centrosome_distance_map(centrosome_coord, cell_mask):
    """Build distance map from a centrosome localisation.

    Parameters
    ----------
    centrosome_coord : np.ndarray, np.int64
        Coordinates of the detected centrosome with shape (nb_elements, 3) or
        (nb_elements, 2). One coordinate per dimension (zyx or yx dimensions).
    cell_mask : np.ndarray, bool
        Binary surface of the cell with shape (y, x).

    Returns
    -------
    distance_map : np.ndarray, np.float32
        Distance map from the centrosome with shape (y, x).

    """
    if centrosome_coord.size == 3:
        centrosome_coord_2d = centrosome_coord[1:]
    else:
        centrosome_coord_2d = centrosome_coord.copy()

    # get mask centrosome
    mask_centrosome = np.zeros_like(cell_mask)
    mask_centrosome[centrosome_coord_2d[:, 0],
                    centrosome_coord_2d[:, 1]] = True

    # compute distance map
    distance_map = ndi.distance_transform_edt(~mask_centrosome)
    distance_map[cell_mask == 0] = 0
    distance_map = distance_map.astype(np.float32)

    return distance_map
