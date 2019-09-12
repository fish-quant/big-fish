# -*- coding: utf-8 -*-

"""
Functions used to format and clean any input loaded in bigfish.
"""

import numpy as np

from .utils import check_array, check_parameter

from skimage.measure import regionprops, find_contours


# ### Transcription sites ###

def remove_transcription_site(mask_nuc, spots_in_foci, foci):
    """We define a transcription site as a foci detected in the nucleus.

    Parameters
    ----------
    mask_nuc : np.ndarray, bool
        Binary mask of the nuclei with shape (y, x).
    spots_in_foci : np.ndarray, np.int64
        Coordinate of the spots detected inside foci, with shape (nb_spots, 4).
        One coordinate per dimension (zyx coordinates) plus the index of the
        foci.
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index.

    Returns
    -------
    spots_in_foci_cleaned : np.ndarray, np.int64
        Coordinate of the spots detected inside foci, with shape (nb_spots, 4).
        One coordinate per dimension (zyx coordinates) plus the index of the
        foci. Transcription sites are removed.
    foci_cleaned : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index. Transcription sites are removed.

    """
    # check parameters
    check_array(mask_nuc,
                ndim=2,
                dtype=[bool],
                allow_nan=False)
    check_array(spots_in_foci,
                ndim=2,
                dtype=[np.int64],
                allow_nan=False)
    check_array(foci,
                ndim=2,
                dtype=[np.int64],
                allow_nan=False)

    # remove foci inside nuclei
    mask_transcription_site = mask_nuc[foci[:, 1], foci[:, 2]]
    foci_cleaned = foci[~mask_transcription_site]

    # filter spots in transcription sites
    spots_to_keep = foci_cleaned[:, 4]
    mask_spots_to_keep = np.isin(spots_in_foci[:, 3], spots_to_keep)
    spots_in_foci_cleaned = spots_in_foci[mask_spots_to_keep]

    return spots_in_foci_cleaned, foci_cleaned


# ### Cell extraction ###

def extract_spots_from_frame(spots, z_lim=None, y_lim=None, x_lim=None):
    """Get spots coordinates within a given frame.

    Parameters
    ----------
    spots : np.ndarray, np.int64
        Coordinate of the spots detected inside foci, with shape (nb_spots, 3)
        or (nb_spots, 4). One coordinate per dimension (zyx coordinates) plus
        the index of the foci if necessary.
    z_lim : tuple[int, int]
        Minimum and maximum coordinate of the frame along the z axis.
    y_lim : tuple[int, int]
        Minimum and maximum coordinate of the frame along the y axis.
    x_lim : tuple[int, int]
        Minimum and maximum coordinate of the frame along the x axis.

    Returns
    -------
    extracted_spots : np.ndarray, np.int64
        Coordinate of the spots detected inside foci, with shape (nb_spots, 3)
        or (nb_spots, 4). One coordinate per dimension (zyx coordinates) plus
        the index of the foci if necessary.

    """
    # check parameters
    check_array(spots,
                ndim=2,
                dtype=[np.int64],
                allow_nan=False)
    check_parameter(z_lim=(tuple, type(None)),
                    y_lim=(tuple, type(None)),
                    x_lim=(tuple, type(None)))

    # extract spots
    extracted_spots = spots.copy()
    if z_lim is not None:
        extracted_spots = extracted_spots[extracted_spots[:, 0] < z_lim[1]]
        extracted_spots = extracted_spots[z_lim[0] < extracted_spots[:, 0]]
        extracted_spots[:, 0] -= z_lim[0]
    if y_lim is not None:
        extracted_spots = extracted_spots[extracted_spots[:, 1] < y_lim[1]]
        extracted_spots = extracted_spots[y_lim[0] < extracted_spots[:, 1]]
        extracted_spots[:, 1] -= y_lim[0]
    if x_lim is not None:
        extracted_spots = extracted_spots[extracted_spots[:, 2] < x_lim[1]]
        extracted_spots = extracted_spots[x_lim[0] < extracted_spots[:, 2]]
        extracted_spots[:, 2] -= x_lim[0]

    return extracted_spots


def extract_coordinates_image(cyt_labelled, nuc_labelled, spots_out, spots_in,
                              foci):
    """Extract relevant coordinates from an image, based on segmentation and
    detection results.

    For each cell in an image we return the coordinates of the cytoplasm, the
    nucleus, the RNA spots and information about the detected foci. We extract
    2-d coordinates for the cell and 3-d coordinates for the spots and foci.

    Parameters
    ----------
    cyt_labelled : np.ndarray, np.uint or np.int
        Labelled cytoplasms image with shape (y, x).
    nuc_labelled : np.ndarray, np.uint or np.int
        Labelled nuclei image with shape (y, x).
    spots_out : np.ndarray, np.int64
        Coordinate of the spots detected outside foci, with shape
        (nb_spots, 4). One coordinate per dimension (zyx coordinates) plus a
        default index (-1 for mRNAs spotted outside a foci).
    spots_in : np.ndarray, np.int64
        Coordinate of the spots detected inside foci, with shape (nb_spots, 4).
        One coordinate per dimension (zyx coordinates) plus the index of the
        foci.
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index.

    Returns
    -------
    results : List[(cyt_coord, nuc_coord, rna_coord, cell_foci, cell)]
        - cyt_coord : np.ndarray, np.int64
            Coordinates of the cytoplasm border with shape (nb_points, 2).
        - nuc_coord : np.ndarray, np.int64
            Coordinates of the nuclei border with shape (nb_points, 2).
        - rna_coord : np.ndarray, np.int64
            Coordinates of the RNA spots with shape (nb_spots, 4). One
            coordinate per dimension (zyx dimension), plus the index of a
            potential foci.
        - cell_foci : np.ndarray, np.int64
            Array with shape (nb_foci, 5). One coordinate per dimension for the
            foci centroid (zyx coordinates), the number of RNAs detected in the
            foci and its index.
        - cell : Tuple[int]
            Box coordinate of the cell in the original image (min_y, min_x,
            max_y and max_x).

    """
    # check parameters
    check_array(cyt_labelled,
                ndim=2,
                dtype=[np.uint8, np.uint16, np.int64],
                allow_nan=True)
    check_array(nuc_labelled,
                ndim=2,
                dtype=[np.uint8, np.uint16, np.int64],
                allow_nan=True)
    check_array(spots_out,
                ndim=2,
                dtype=[np.int64],
                allow_nan=False)
    check_array(spots_in,
                ndim=2,
                dtype=[np.int64],
                allow_nan=False)
    check_array(foci,
                ndim=2,
                dtype=[np.int64],
                allow_nan=False)

    # initialize results
    results = []
    borders = np.zeros(cyt_labelled.shape, dtype=bool)
    borders[:, 0] = True
    borders[0, :] = True
    borders[:, cyt_labelled.shape[1]-1] = True
    borders[cyt_labelled.shape[0]-1, :] = True
    cells = regionprops(cyt_labelled)
    for cell in cells:

        # get information about the cell
        label = cell.label
        (min_y, min_x, max_y, max_x) = cell.bbox

        # get masks of the cell
        cyt = cyt_labelled.copy()
        cyt = (cyt == label)
        nuc = nuc_labelled.copy()
        nuc = (nuc == label)

        # check if cell is not cropped by the borders
        if _check_cropped_cell(cyt, borders):
            continue

        # check if nucleus is in the cytoplasm
        if not _check_nucleus_in_cell(cyt, nuc):
            continue

        # get boundaries coordinates
        cyt_coord, nuc_coord = _get_boundaries_coordinates(cyt, nuc)

        # filter foci
        foci_cell, spots_in_foci_cell = _extract_foci(foci, spots_in, cyt)

        # get rna coordinates
        spots_out_foci_cell = _extract_spots_outside_foci(cyt, spots_out)
        rna_coord = np.concatenate([spots_out_foci_cell, spots_in_foci_cell],
                                   axis=0)

        # filter cell without enough spots
        if len(rna_coord) < 30:
            continue

        # initialize cell coordinates
        cyt_coord[:, 0] -= min_y
        cyt_coord[:, 1] -= min_x
        nuc_coord[:, 0] -= min_y
        nuc_coord[:, 1] -= min_x
        rna_coord[:, 1] -= min_y
        rna_coord[:, 2] -= min_x
        foci_cell[:, 1] -= min_y
        foci_cell[:, 2] -= min_x

        results.append((cyt_coord, nuc_coord, rna_coord, foci_cell, cell.bbox))

    return results


def _check_cropped_cell(cell_cyt_mask, border_frame):
    """
    Check if a cell is cropped by the border frame.

    Parameters
    ----------
    cell_cyt_mask : np.ndarray, bool
        Binary mask of the cell cytoplasm.

    border_frame : np.ndarray, bool
        Binary mask of the border frame.

    Returns
    -------
    _ : bool
        True if cell is cropped.

    """
    # check cell is not cropped by the borders
    crop = cell_cyt_mask & border_frame
    if np.any(crop):
        return True
    else:
        return False


def _check_nucleus_in_cell(cell_cyt_mask, cell_nuc_mask):
    """
    Check if the nucleus is properly contained in the cell cytoplasm.

    Parameters
    ----------
    cell_cyt_mask : np.ndarray, bool
        Binary mask of the cell cytoplasm.

    cell_nuc_mask : np.ndarray, bool
        Binary mask of the nucleus cytoplasm.

    Returns
    -------
    _ : bool
        True if the nucleus is in the cell.

    """
    diff = cell_cyt_mask | cell_nuc_mask
    if np.any(diff != cell_cyt_mask):
        return False
    else:
        return True


def _get_boundaries_coordinates(cell_cyt_mask, cell_nuc_mask):
    """
    Find boundaries coordinates for cytoplasm and nucleus.

    Parameters
    ----------
    cell_cyt_mask : np.ndarray, bool
        Mask of the cell cytoplasm.
    cell_nuc_mask : np.ndarray, bool
        Mask of the cell nucleus.

    Returns
    -------
    cyt_coord : np.ndarray, np.int64
        Coordinates of the cytoplasm in 2-d (yx dimension).
    nuc_coord : np.ndarray, np.int64
        Coordinates of the nucleus in 2-d (yx dimension).

    """
    cyt_coord = np.array([], dtype=np.int64).reshape((0, 2))
    nuc_coord = np.array([], dtype=np.int64).reshape((0, 2))

    # cyt coordinates
    cell_cyt_coord = find_contours(cell_cyt_mask, level=0)
    if len(cell_cyt_coord) == 0:
        pass
    elif len(cell_cyt_coord) == 1:
        cyt_coord = cell_cyt_coord[0].astype(np.int64)
    else:
        m = 0
        for coord in cell_cyt_coord:
            if len(coord) > m:
                m = len(coord)
                cyt_coord = coord.astype(np.int64)

    # nuc coordinates
    cell_nuc_coord = find_contours(cell_nuc_mask, level=0)
    if len(cell_nuc_coord) == 0:
        pass
    elif len(cell_nuc_coord) == 1:
        nuc_coord = cell_nuc_coord[0].astype(np.int64)
    else:
        m = 0
        for coord in cell_nuc_coord:
            if len(coord) > m:
                m = len(coord)
                nuc_coord = coord.astype(np.int64)

    return cyt_coord, nuc_coord


def _extract_foci(foci, spots_in_foci, cell_cyt_mask):
    """
    Extract foci and related spots detected in a specific cell.

    Parameters
    ----------
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index.

    spots_in_foci : : np.ndarray, np.int64
        Coordinate of the spots detected inside foci, with shape (nb_spots, 4).
        One coordinate per dimension (zyx coordinates) plus the index of the
        foci.
    cell_cyt_mask : np.ndarray, bool
        Binary mask of the cell with shape (y, x).

    Returns
    -------
    spots_in_foci_cell : np.ndarray, np.int64
        Coordinate of the spots detected inside foci in the cell, with shape
        (nb_spots, 4). One coordinate per dimension (zyx coordinates) plus the
        index of the foci.
    foci_cell : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index.

    """
    # filter foci
    mask_foci_cell = cell_cyt_mask[foci[:, 1], foci[:, 2]]
    if mask_foci_cell.sum() == 0:
        foci_cell = np.array([], dtype=np.int64).reshape((0, 5))
        spots_in_foci_cell = np.array([], dtype=np.int64).reshape((0, 4))
        return foci_cell, spots_in_foci_cell

    foci_cell = foci[mask_foci_cell]

    # filter spots in foci
    spots_to_keep = foci_cell[:, 4]
    mask_spots_to_keep = np.isin(spots_in_foci[:, 3], spots_to_keep)
    spots_in_foci_cell = spots_in_foci[mask_spots_to_keep]

    return foci_cell, spots_in_foci_cell


def _extract_spots_outside_foci(cell_cyt_mask, spots_out_foci):
    """
    Extract spots detected outside foci, in a specific cell.

    Parameters
    ----------
    cell_cyt_mask : np.ndarray, bool
        Binary mask of the cell with shape (y, x).
    spots_out_foci : np.ndarray, np.int64
        Coordinate of the spots detected outside foci, with shape
        (nb_spots, 4). One coordinate per dimension (zyx coordinates) plus a
        default index (-1 for mRNAs spotted outside a foci).

    Returns
    -------
    spots_out_foci_cell : np.ndarray, np.int64
        Coordinate of the spots detected outside foci in the cell, with shape
        (nb_spots, 4). One coordinate per dimension (zyx coordinates) plus the
        index of the foci.

    """
    # get coordinates of rna outside foci
    mask_spots_to_keep = cell_cyt_mask[spots_out_foci[:, 1],
                                       spots_out_foci[:, 2]]
    spots_out_foci_cell = spots_out_foci[mask_spots_to_keep]

    return spots_out_foci_cell
