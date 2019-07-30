# -*- coding: utf-8 -*-

"""
Functions used to format and clean any input loaded in bigfish.
"""

import numpy as np

from .utils import check_array, check_parameter

from skimage.segmentation import find_boundaries
from skimage.measure import regionprops


# TODO use skimage.measure.find_contours

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
    foci_cleaned = foci.copy()
    spots_in_foci_cleaned = spots_in_foci.copy()
    for (_, y, x, _, i_foci) in foci:
        if mask_nuc[y, x]:
            foci_cleaned = foci_cleaned[foci_cleaned[:, 4] != i_foci]
            spots_in_foci_cleaned = spots_in_foci_cleaned[
                spots_in_foci_cleaned[:, 3] != i_foci]

    return spots_in_foci_cleaned, foci_cleaned


# ### Cell extraction ###

def extract_spots(spots, z_lim=None, y_lim=None, x_lim=None):
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
    2-d coordinates.

    Parameters
    ----------
    cyt_labelled : np.ndarray, np.uint or np.int
        Labelled cytoplasms image with shape (y, x).
    nuc_labelled : np.ndarray, np.uint or np.int
        Labelled nuclei image with shape (y, x).
    spots_out : np.ndarray, np.int64
        Coordinate of the spots detected outside foci, with shape
        (nb_spots, 3). One coordinate per dimension (zyx coordinates).
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
            Coordinates of the RNA spots with shape (nb_spots, 3). One
            coordinate per dimension (yx dimension), plus the index of a
            potential foci.
        - cell_foci : np.ndarray, np.int64
            Array with shape (nb_foci, 5). One coordinate per dimension for the
            foci centroid (zyx coordinates), the number of RNAs detected in the
            foci and its index.
        - cell : Tuple[int]
            Box coordinate of the cell in the original image (min_y, min_x,
            max_y and max_x).

    """
    # TODO implement several smaller functions
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

        # check cell is not cropped by the borders
        crop = cyt & borders
        if np.any(crop):
            continue

        # check nucleus is in the cytoplasm
        diff = cyt | nuc
        if np.any(diff != cyt):
            continue

        # get boundaries coordinates
        # TODO replace by find_contour
        cyt_coord = find_boundaries(cyt, mode='inner')
        cyt_coord = np.nonzero(cyt_coord)
        cyt_coord = np.column_stack(cyt_coord)
        nuc_coord = find_boundaries(nuc, mode='inner')
        nuc_coord = np.nonzero(nuc_coord)
        nuc_coord = np.column_stack(nuc_coord)

        # filter foci
        cell_foci = foci.copy()
        cell_spots_in = spots_in.copy()
        for (_, y, x, _, i_foci) in foci:
            if cyt_labelled[y, x] != label:
                cell_foci = cell_foci[cell_foci[:, 4] != i_foci]
                cell_spots_in = cell_spots_in[cell_spots_in[:, 3] != i_foci]

        # get rna coordinates
        image_shape = cyt_labelled.shape
        rna_out = np.zeros(image_shape, dtype=bool)
        rna_out[spots_out[:, 1], spots_out[:, 2]] = True
        rna_out = (rna_out & cyt)
        rna_out = np.nonzero(rna_out)
        rna_out = np.column_stack(rna_out)
        rna_in = np.zeros(image_shape, dtype=bool)
        rna_in[cell_spots_in[:, 1], cell_spots_in[:, 2]] = True
        rna_in = (rna_in & cyt)
        rna_in = np.nonzero(rna_in)
        rna_in = np.column_stack(rna_in)
        rna_coord = np.concatenate([rna_out, rna_in], axis=0)

        # filter cell without enough spots
        if len(rna_coord) < 30:
            continue

        # initialize cell coordinates
        cyt_coord[:, 0] -= min_y
        cyt_coord[:, 1] -= min_x
        nuc_coord[:, 0] -= min_y
        nuc_coord[:, 1] -= min_x
        rna_coord[:, 0] -= min_y
        rna_coord[:, 1] -= min_x
        cell_foci[:, 1] -= min_y
        cell_foci[:, 2] -= min_x

        results.append((cyt_coord, nuc_coord, rna_coord, cell_foci, cell.bbox))

    return results
