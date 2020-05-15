# -*- coding: utf-8 -*-

"""
Functions used to format and clean any intermediate results loaded in or
returned by a bigfish method.
"""

import numpy as np
from scipy import ndimage as ndi

from .utils import check_array, check_parameter, get_offset_value

from skimage.measure import regionprops, find_contours
from skimage.draw import polygon_perimeter


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
    borders[:, cyt_labelled.shape[1] - 1] = True
    borders[cyt_labelled.shape[0] - 1, :] = True
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


# ### Segmentation postprocessing ###

# TODO add from_binary_surface_to_binary_boundaries

def center_binary_mask(cyt, nuc=None, rna=None):
    """Center a 2-d binary mask (surface or boundaries) and pad it.

    One mask should be at least provided ('cyt'). If others masks are provided
    ('nuc' and 'rna'), they will be transformed like the main mask. All the
    provided masks should have the same shape. If others coordinates are
    provided, the values will be transformed, but an array of coordinates with
    the same format is returned

    Parameters
    ----------
    cyt : np.ndarray, np.uint or np.int or bool
        Binary image of cytoplasm with shape (y, x).
    nuc : np.ndarray, np.uint or np.int or bool
        Binary image of nucleus with shape (y, x) or array of nucleus
        coordinates with shape (nb_points, 2).
    rna : np.ndarray, np.uint or np.int or bool
        Binary image of mRNAs localization with shape (y, x) or array of mRNAs
        coordinates with shape (nb_points, 2) or (nb_points, 3).

    Returns
    -------
    cyt_centered : np.ndarray, np.uint or np.int or bool
        Centered binary image of cytoplasm with shape (y, x).
    nuc_centered : np.ndarray, np.uint or np.int or bool
        Centered binary image of nucleus with shape (y, x).
    rna_centered : np.ndarray, np.uint or np.int or bool
        Centered binary image of mRNAs localizations with shape (y, x).

    """
    # check parameters
    check_array(cyt,
                ndim=2,
                dtype=[np.uint8, np.uint16, np.int64, bool])
    if nuc is not None:
        check_array(nuc,
                    ndim=2,
                    dtype=[np.uint8, np.uint16, np.int64, bool])
    if rna is not None:
        check_array(rna,
                    ndim=2,
                    dtype=[np.uint8, np.uint16, np.int64, bool])

    # initialize parameter
    nuc_centered, rna_centered = None, None
    marge = get_offset_value()

    # center the binary mask of the cell
    coord = np.nonzero(cyt)
    coord = np.column_stack(coord)
    min_y, max_y = coord[:, 0].min(), coord[:, 0].max()
    min_x, max_x = coord[:, 1].min(), coord[:, 1].max()
    shape_y = max_y - min_y + 1
    shape_x = max_x - min_x + 1
    cyt_centered_shape = (shape_y + 2 * marge, shape_x + 2 * marge)
    cyt_centered = np.zeros(cyt_centered_shape, dtype=bool)
    crop = cyt[min_y:max_y + 1, min_x:max_x + 1]
    cyt_centered[marge:shape_y + marge, marge:shape_x + marge] = crop

    # center the binary mask of the nucleus with the same transformation
    if nuc is not None:
        if nuc.shape == 2:
            nuc_centered = nuc.copy()
            nuc_centered[:, 0] = nuc_centered[:, 0] - min_y + marge
            nuc_centered[:, 1] = nuc_centered[:, 1] - min_x + marge

        elif nuc.shape == cyt.shape:
            nuc_centered = np.zeros(cyt_centered_shape, dtype=bool)
            crop = nuc[min_y:max_y + 1, min_x:max_x + 1]
            nuc_centered[marge:shape_y + marge, marge:shape_x + marge] = crop

        else:
            raise ValueError("mRNAs mask should have the same shape than "
                             "cytoplasm mask and coordinates should be in 2-d")

    # center the binary mask of the mRNAs with the same transformation
    if rna is not None:
        if rna.shape[1] == 3:
            rna_centered = rna.copy()
            rna_centered[:, 1] = rna_centered[:, 1] - min_y + marge
            rna_centered[:, 2] = rna_centered[:, 2] - min_x + marge

        elif rna.shape[1] == 2:
            rna_centered = rna.copy()
            rna_centered[:, 0] = rna_centered[:, 0] - min_y + marge
            rna_centered[:, 1] = rna_centered[:, 1] - min_x + marge

        elif rna.shape == cyt.shape:
            rna_centered = np.zeros(cyt_centered_shape, dtype=bool)
            crop = rna[min_y:max_y + 1, min_x:max_x + 1]
            rna_centered[marge:shape_y + marge, marge:shape_x + marge] = crop

        else:
            raise ValueError("mRNAs mask should have the same shape than "
                             "cytoplasm mask and coordinates should be in 2-d "
                             "or 3-d")

    return cyt_centered, nuc_centered, rna_centered


def from_surface_to_coord(binary_surface):
    """Extract coordinates from a 2-d binary matrix.

    The resulting coordinates represent the external boundaries of the object.

    Parameters
    ----------
    binary_surface : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x).

    Returns
    -------
    coord : np.ndarray, np.int64
        Array of boundaries coordinates with shape (nb_points, 2).

    """
    # check parameters
    check_array(binary_surface,
                ndim=2,
                dtype=[np.uint8, np.uint16, np.int64, bool])

    # from binary surface to 2D coordinates boundaries
    coord = find_contours(binary_surface, level=0)[0].astype(np.int64)

    return coord


def complete_coord_boundaries(coord):
    """Complete a 2-d coordinates array, by generating/interpolating missing
    points.

    Parameters
    ----------
    coord : np.ndarray, np.int64
        Array of coordinates to complete, with shape (nb_points, 2).

    Returns
    -------
    coord_completed : np.ndarray, np.int64
        Completed coordinates arrays, with shape (nb_points, 2).

    """
    # check parameters
    check_array(coord,
                ndim=2,
                dtype=[np.int64])

    # for each array in the list, complete its coordinates using the scikit
    # image method 'polygon_perimeter'
    coord_y, coord_x = polygon_perimeter(coord[:, 0], coord[:, 1])
    coord_y = coord_y[:, np.newaxis]
    coord_x = coord_x[:, np.newaxis]
    coord_completed = np.concatenate((coord_y, coord_x), axis=-1)

    return coord_completed


def _from_coord_to_boundaries(coord_cyt, coord_nuc=None, coord_rna=None):
    """Convert 2-d coordinates to a binary matrix with the boundaries of the
    object.

    As we manipulate the coordinates of the external boundaries, the relative
    binary matrix has two extra pixels in each dimension. We compensate by
    reducing the marge by one in order to keep the same shape for the frame.
    If others coordinates are provided, the relative binary matrix is build
    with the same shape as the main coordinates.

    Parameters
    ----------
    coord_cyt : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    coord_nuc : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    coord_rna : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).

    Returns
    -------
    cyt : np.ndarray, np.uint or np.int or bool
        Binary image of cytoplasm boundaries with shape (y, x).
    nuc : np.ndarray, np.uint or np.int or bool
        Binary image of nucleus boundaries with shape (y, x).
    rna : np.ndarray, np.uint or np.int or bool
        Binary image of mRNAs localizations with shape (y, x).

    """
    # initialize parameter
    nuc, rna = None, None
    marge = get_offset_value()
    marge -= 1

    # from 2D coordinates boundaries to binary boundaries
    max_y = coord_cyt[:, 0].max()
    max_x = coord_cyt[:, 1].max()
    min_y = coord_cyt[:, 0].min()
    min_x = coord_cyt[:, 1].min()
    shape_y = max_y - min_y + 1
    shape_x = max_x - min_x + 1
    image_shape = (shape_y + 2 * marge, shape_x + 2 * marge)
    coord_cyt[:, 0] = coord_cyt[:, 0] - min_y + marge
    coord_cyt[:, 1] = coord_cyt[:, 1] - min_x + marge
    cyt = np.zeros(image_shape, dtype=bool)
    cyt[coord_cyt[:, 0], coord_cyt[:, 1]] = True

    # transform nucleus coordinates with the same parameters
    if coord_nuc is not None:
        nuc = np.zeros(image_shape, dtype=bool)
        coord_nuc[:, 0] = coord_nuc[:, 0] - min_y + marge
        coord_nuc[:, 1] = coord_nuc[:, 1] - min_x + marge
        nuc[coord_nuc[:, 0], coord_nuc[:, 1]] = True

    # transform mRNAs coordinates with the same parameters
    if coord_rna is not None:
        rna = np.zeros(image_shape, dtype=bool)
        if coord_rna.shape[1] == 3:
            coord_rna[:, 1] = coord_rna[:, 1] - min_y + marge
            coord_rna[:, 2] = coord_rna[:, 2] - min_x + marge
            rna[coord_rna[:, 1], coord_rna[:, 2]] = True
        else:
            coord_rna[:, 0] = coord_rna[:, 0] - min_y + marge
            coord_rna[:, 1] = coord_rna[:, 1] - min_x + marge
            rna[coord_rna[:, 0], coord_rna[:, 1]] = True

    return cyt, nuc, rna


def from_boundaries_to_surface(binary_boundaries):
    """Fill in the binary matrix representing the boundaries of an object.

    Parameters
    ----------
    binary_boundaries : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x).

    Returns
    -------
    binary_surface : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x).

    """
    # TODO check dtype input & output
    # check parameters
    check_array(binary_boundaries,
                ndim=2,
                dtype=[np.uint8, np.uint16, np.int64, bool])

    # from binary boundaries to binary surface
    binary_surface = ndi.binary_fill_holes(binary_boundaries)

    return binary_surface


def from_coord_to_surface(coord_cyt, coord_nuc=None, coord_rna=None):
    """Convert 2-d coordinates to a binary matrix with the surface of the
    object.

    As we manipulate the coordinates of the external boundaries, the relative
    binary matrix has two extra pixels in each dimension. We compensate by
    keeping only the inside pixels of the object surface.
    If others coordinates are provided, the relative binary matrix is build
    with the same shape as the main coordinates.

    Parameters
    ----------
    coord_cyt : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    coord_nuc : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    coord_rna : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).

    Returns
    -------
    cyt_surface : np.ndarray, np.uint or np.int or bool
        Binary image of cytoplasm surface with shape (y, x).
    nuc_surface : np.ndarray, np.uint or np.int or bool
        Binary image of nucleus surface with shape (y, x).
    rna : np.ndarray, np.uint or np.int or bool
        Binary image of mRNAs localizations with shape (y, x).

    """
    # check parameters
    check_array(coord_cyt,
                ndim=2,
                dtype=[np.int64])
    if coord_nuc is not None:
        check_array(coord_nuc,
                    ndim=2,
                    dtype=[np.int64])
    if coord_rna is not None:
        check_array(coord_rna,
                    ndim=2,
                    dtype=[np.int64])

    # from coordinates to binary boundaries
    cyt, nuc, rna = _from_coord_to_boundaries(coord_cyt, coord_nuc, coord_rna)

    # from binary boundaries to binary surface
    cyt_surface = from_boundaries_to_surface(cyt)
    nuc_surface = from_boundaries_to_surface(nuc)

    return cyt_surface, nuc_surface, rna
