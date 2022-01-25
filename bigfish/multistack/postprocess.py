# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions used to format, merge and clean intermediate results from different
channels processed with bigfish.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import bigfish.stack as stack

from skimage.measure import regionprops
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter


# ### Object identification to sub-cellular regions ###

def identify_objects_in_region(mask, coord, ndim):
    """Identify cellular objects in specific region.

    Parameters
    ----------
    mask : np.ndarray, bool
        Binary mask of the targeted region with shape (y, x).
    coord : np.ndarray
        Array with two dimensions. One object per row, zyx or yx coordinates
        in the first 3 or 2 columns.
    ndim : int
        Number of spatial dimensions to consider (2 or 3).

    Returns
    -------
    coord_in : np.ndarray
        Coordinates of the objects detected inside the region.
    coord_out : np.ndarray
        Coordinates of the objects detected outside the region.

    """
    # check parameters
    stack.check_parameter(ndim=int)
    stack.check_array(
        mask,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_array(coord, ndim=2, dtype=[np.int64, np.float64])

    # check number of dimensions
    if ndim not in [2, 3]:
        raise ValueError("The number of spatial dimension requested should be "
                         "2 or 3, not {0}.".format(ndim))
    if coord.shape[1] < ndim:
        raise ValueError("Coord array should have at least {0} features to "
                         "match the number of spatial dimensions requested. "
                         "Currently {1} is not enough."
                         .format(ndim, coord.shape[1]))

    # binarize nuclei mask if needed
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # cast coordinates dtype if necessary
    if coord.dtype == np.int64:
        coord_int = coord
    else:
        coord_int = np.round(coord).astype(np.int64)

    # remove objects inside the region
    mask_in = mask[coord_int[:, ndim - 2], coord_int[:, ndim - 1]]
    coord_in = coord[mask_in]
    coord_out = coord[~mask_in]

    return coord_in, coord_out


def remove_transcription_site(rna, clusters, nuc_mask, ndim):
    """Distinguish RNA molecules detected in a transcription site from the
    rest.

    A transcription site is defined as as a foci detected within the nucleus.

    Parameters
    ----------
    rna : np.ndarray
        Coordinates of the detected RNAs with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the cluster assigned to the RNA. If no cluster was
        assigned, value is -1.
    clusters : np.ndarray
        Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate
        per dimension for the clusters centroid (zyx or yx coordinates),
        the number of RNAs detected in the clusters and their index.
    nuc_mask : np.ndarray, bool
        Binary mask of the nuclei region with shape (y, x).
    ndim : int
        Number of spatial dimensions to consider (2 or 3).

    Returns
    -------
    rna_out_ts : np.ndarray
        Coordinates of the detected RNAs with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates)
        plus the index of the foci assigned to the RNA. If no foci was
        assigned, value is -1. RNAs from transcription sites are removed.
    foci : np.ndarray
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension for the foci centroid (zyx or yx coordinates),
        the number of RNAs detected in the foci and its index.
    ts : np.ndarray
        Array with shape (nb_ts, 5) or (nb_ts, 4). One coordinate per
        dimension for the transcription site centroid (zyx or yx coordinates),
        the number of RNAs detected in the transcription site and its index.

    """
    # check parameters
    stack.check_array(rna, ndim=2, dtype=[np.int64, np.float64])

    # discriminate foci from transcription sites
    ts, foci = identify_objects_in_region(
        nuc_mask, clusters, ndim)

    # filter out rna from transcription sites
    rna_in_ts = ts[:, ndim + 1]
    mask_rna_in_ts = np.isin(rna[:, ndim], rna_in_ts)
    rna_out_ts = rna[~mask_rna_in_ts]

    return rna_out_ts, foci, ts


# ### Nuclei-cells matching

def match_nuc_cell(nuc_label, cell_label, single_nuc, cell_alone):
    """Match each nucleus instance with the most overlapping cell instance.

    Parameters
    ----------
    nuc_label : np.ndarray, np.int or np.uint
        Labelled image of nuclei with shape (z, y, x) or (y, x).
    cell_label : np.ndarray, np.int or np.uint
        Labelled image of cells with shape (z, y, x) or (y, x).
    single_nuc : bool
        Authorized only one nucleus in a cell.
    cell_alone : bool
        Authorized cell without nucleus.

    Returns
    -------
    new_nuc_label : np.ndarray, np.int or np.uint
        Labelled image of nuclei with shape (z, y, x) or (y, x).
    new_cell_label : np.ndarray, np.int or np.uint
        Labelled image of cells with shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(
        nuc_label,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.int64])
    stack.check_array(
        cell_label,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.int64])

    # initialize new labelled images
    new_nuc_label = np.zeros_like(nuc_label)
    new_cell_label = np.zeros_like(cell_label)
    remaining_cell_label = cell_label.copy()

    # loop over nuclei
    i_instance = 1
    max_nuc_label = nuc_label.max()
    for i_nuc in range(1, max_nuc_label + 1):

        # get nuc mask
        nuc_mask = nuc_label == i_nuc

        # check if a nucleus is labelled with this value
        if nuc_mask.sum() == 0:
            continue

        # check if a cell is labelled with this value
        i_cell = _get_most_frequent_value(cell_label[nuc_mask])
        if i_cell == 0:
            continue

        # get cell mask
        cell_mask = cell_label == i_cell

        # ensure nucleus is totally included in cell
        cell_mask |= nuc_mask
        cell_label[cell_mask] = i_cell
        remaining_cell_label[cell_mask] = i_cell

        # assign cell and nucleus
        new_nuc_label[nuc_mask] = i_instance
        new_cell_label[cell_mask] = i_instance
        i_instance += 1

        # remove pixel already assigned
        remaining_cell_label[cell_mask] = 0

        # if one nucleus per cell only, we remove the cell as candidate
        if single_nuc:
            cell_label[cell_mask] = 0

    # if only cell with nucleus are authorized we stop here
    if not cell_alone:
        return new_nuc_label, new_cell_label

    # loop over remaining cells
    max_remaining_cell_label = remaining_cell_label.max()
    for i_cell in range(1, max_remaining_cell_label + 1):

        # get cell mask
        cell_mask = remaining_cell_label == i_cell

        # check if a cell is labelled with this value
        if cell_mask.sum() == 0:
            continue

        # add cell in the result
        new_cell_label[cell_mask] = i_instance
        i_instance += 1

    return new_nuc_label, new_cell_label


def _get_most_frequent_value(array):
    """Count the most frequent value in a array.

    Parameters
    ----------
    array : np.ndarray, np.uint or np.int
        Array-like object.

    Returns
    -------
    value : int
        Most frequent integer in the array.

    """
    value = np.argmax(np.bincount(array))

    return value


# ### Cell extraction ###

def extract_cell(cell_label, ndim, nuc_label=None, rna_coord=None,
                 others_coord=None, image=None, others_image=None,
                 remove_cropped_cell=True, check_nuc_in_cell=True):
    """Extract cell-level results for an image.

    The function gathers different segmentation and detection results obtained
    at the image level and assigns each of them to the individual cells.

    Parameters
    ----------
    cell_label : np.ndarray, np.uint or np.int
        Image with labelled cells and shape (y, x).
    ndim : int
        Number of spatial dimensions to consider (2 or 3).
    nuc_label : np.ndarray, np.uint or np.int
        Image with labelled nuclei and shape (y, x). If None, individual
        nuclei are not assigned to each cell.
    rna_coord : np.ndarray
        Coordinates of the detected RNAs with zyx or yx coordinates in the
        first 3 or 2 columns. If None, RNAs are not assigned to individual
        cells.
    others_coord : Dict[np.ndarray]
        Dictionary of coordinates arrays. For each array of the dictionary,
        the different elements are assigned to individual cells. Arrays should
        be organized the same way than spots: zyx or yx coordinates in the
        first 3 or 2 columns, np.int64 dtype, one element per row. Can be used
        to assign different detected elements to the segmented cells along with
        the spots. If None, no others elements are assigned to the individual
        cells.
    image : np.ndarray, np.uint
        Image in 2-d. If None, image of the individual cells are not extracted.
    others_image : Dict[np.ndarray]
        Dictionary of images to crop. If None, no others image of the
        individual cells are extracted.
    remove_cropped_cell : bool
        Remove cells cropped by the FoV frame.
    check_nuc_in_cell : bool
        Check that each nucleus is entirely localized within a cell.

    Returns
    -------
    fov_results : List[Dict]
        List of dictionaries, one per cell segmented in the image. Each
        dictionary includes information about the cell (image, masks,
        coordinates arrays). Minimal information are:

        * `cell_id`: Unique id of the cell.
        * `bbox`: bounding box coordinates with the order (`min_y`, `min_x`,
          `max_y`, `max_x`).
        * `cell_coord`: boundary coordinates of the cell.
        * `cell_mask`: mask of the cell.

    """
    # check parameters
    stack.check_parameter(
        ndim=int,
        others_coord=(dict, type(None)),
        others_image=(dict, type(None)),
        remove_cropped_cell=bool,
        check_nuc_in_cell=bool)
    stack.check_array(
        cell_label,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64])
    if nuc_label is not None:
        stack.check_array(
            nuc_label,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int64, np.float64])
    if image is not None:
        stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16])
    actual_keys = ["cell_id", "bbox", "cell_coord", "cell_mask", "nuc_coord",
                   "nuc_mask", "rna_coord", "image"]
    if others_coord is not None:
        for key in others_coord:
            if key in actual_keys:
                raise KeyError("Key {0} in 'others_coord' is already taken. "
                               "Please choose another one.".format(key))
            else:
                actual_keys.append(key)
            array = others_coord[key]
            stack.check_array(array, ndim=2, dtype=[np.int64, np.float64])
            if array.shape[1] < ndim:
                warnings.warn("Array in 'others_coord' have less coordinates "
                              "({0}) than the minimum number of spatial "
                              "dimension we consider ({1})."
                              .format(array.shape[1], ndim),
                              UserWarning)
    # TODO allow boolean for 'others_image'
    # TODO bug if 'image' is None but not 'others_image'
    if others_image is not None:
        for key in others_image:
            if key in actual_keys:
                raise KeyError("Key {0} in 'others_image' is already taken. "
                               "Please choose another one.".format(key))
            else:
                actual_keys.append(key)
            image_ = others_image[key]
            stack.check_array(image_, ndim=2, dtype=[np.uint8, np.uint16])
            if image_.shape != image.shape:
                warnings.warn("Image in 'others_image' does not have the same "
                              "shape ({0}) than original image ({1})."
                              .format(image_.shape, image.shape),
                              UserWarning)
    if rna_coord is not None and rna_coord.shape[1] < ndim:
        warnings.warn("'rna_coord' have less coordinates ({0}) than the "
                      "minimum number of spatial dimension we "
                      "consider ({1}).".format(rna_coord.shape[1], ndim),
                      UserWarning)

    # initialize FoV results
    fov_results = []

    # initialize a mask to detect cells at the FoV borders
    fov_borders = np.zeros(cell_label.shape, dtype=bool)
    if remove_cropped_cell:
        fov_borders[:, 0] = True
        fov_borders[0, :] = True
        fov_borders[:, cell_label.shape[1] - 1] = True
        fov_borders[cell_label.shape[0] - 1, :] = True

    # iterate over each segmented cell
    cells = regionprops(cell_label)
    for cell in cells:

        # initialize cell results
        cell_results = {}

        # get the bounding box of the cell
        label = cell.label
        cell_results["cell_id"] = label
        (min_y, min_x, max_y, max_x) = cell.bbox
        cell_results["bbox"] = cell.bbox

        # get binary masks of the cell
        cell_mask = cell_label.copy()
        cell_mask = (cell_mask == label)

        # check if cell is not cropped by the borders
        if remove_cropped_cell and _check_cropped_cell(cell_mask, fov_borders):
            continue

        # get boundaries coordinates for cell
        cell_coord = from_binary_to_coord(cell_mask)
        cell_coord = complete_coord_boundaries(cell_coord)
        cell_coord[:, 0] -= min_y
        cell_coord[:, 1] -= min_x
        cell_results["cell_coord"] = cell_coord

        # crop binary mask of the cell
        cell_mask_cropped = cell_mask[min_y: max_y, min_x: max_x]
        cell_results["cell_mask"] = cell_mask_cropped

        # get binary mask of the nucleus
        if nuc_label is not None:
            nuc_mask = nuc_label.copy()
            nuc_mask = (nuc_mask == label)

            # check if nucleus is in the cell
            if (check_nuc_in_cell
                    and not _check_nucleus_in_cell(cell_mask, nuc_mask)):
                continue

            # get boundaries coordinates for nucleus
            nuc_coord = from_binary_to_coord(nuc_mask)
            nuc_coord = complete_coord_boundaries(nuc_coord)
            nuc_coord[:, 0] -= min_y
            nuc_coord[:, 1] -= min_x
            cell_results["nuc_coord"] = nuc_coord

            # crop binary mask of the nucleus
            nuc_mask_cropped = nuc_mask[min_y: max_y, min_x: max_x]
            cell_results["nuc_mask"] = nuc_mask_cropped

        # get coordinates of the spots detected in the cell
        if rna_coord is not None:
            rna_in_cell, _ = identify_objects_in_region(
                cell_mask,
                rna_coord,
                ndim)
            rna_in_cell[:, ndim - 2] -= min_y
            rna_in_cell[:, ndim - 1] -= min_x
            cell_results["rna_coord"] = rna_in_cell

        # get coordinates of the other detected elements
        if others_coord is not None:
            for key in others_coord:
                array = others_coord[key]
                element_in_cell, _ = identify_objects_in_region(
                    cell_mask,
                    array,
                    ndim)
                element_in_cell[:, ndim - 2] -= min_y
                element_in_cell[:, ndim - 1] -= min_x
                cell_results[key] = element_in_cell

        # crop cell image
        if image is not None:
            image_cropped = image[min_y: max_y, min_x: max_x]
            cell_results["image"] = image_cropped

        # get crops of the other images
        if others_image is not None:
            for key in others_image:
                image_ = others_image[key]
                image_cropped_ = image_[min_y: max_y, min_x: max_x]
                cell_results[key] = image_cropped_

        fov_results.append(cell_results)

    return fov_results


def _check_cropped_cell(cell_mask, border_frame):
    """
    Check if a cell is cropped by the border frame.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell cytoplasm.

    border_frame : np.ndarray, bool
        Binary mask of the border frame.

    Returns
    -------
    _ : bool
        True if cell is cropped.

    """
    # check cell is not cropped by the borders
    crop = cell_mask & border_frame
    if np.any(crop):
        return True
    else:
        return False


def _check_nucleus_in_cell(cell_mask, nuc_mask):
    """
    Check if the nucleus is properly contained in the cell cytoplasm.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell cytoplasm.

    nuc_mask : np.ndarray, bool
        Binary mask of the nucleus cytoplasm.

    Returns
    -------
    _ : bool
        True if the nucleus is in the cell.

    """
    diff = cell_mask | nuc_mask
    if np.any(diff != cell_mask):
        return False
    else:
        return True


def extract_spots_from_frame(spots, z_lim=None, y_lim=None, x_lim=None):
    """Get spots coordinates within a given frame.

    Parameters
    ----------
    spots : np.ndarray
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
    extracted_spots : np.ndarray
        Coordinate of the spots detected inside foci, with shape (nb_spots, 3)
        or (nb_spots, 4). One coordinate per dimension (zyx coordinates) plus
        the index of the foci if necessary.

    """
    # check parameters
    stack.check_array(spots, ndim=2, dtype=[np.int64, np.float64])
    stack.check_parameter(
        z_lim=(tuple, type(None)),
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


def summarize_extraction_results(fov_results, ndim, path_output=None):
    """Summarize results extracted from an image and store them in a dataframe.

    Parameters
    ----------
    fov_results : List[Dict]
        List of dictionaries, one per cell segmented in the image. Each
        dictionary includes information about the cell (image, masks,
        coordinates arrays). Minimal information are:

        * `cell_id`: Unique id of the cell.
        * `bbox`: bounding box coordinates with the order (`min_y`, `min_x`,
          `max_y`, `max_x`).
        * `cell_coord`: boundary coordinates of the cell.
        * `cell_mask`: mask of the cell.
    ndim : int
        Number of spatial dimensions to consider (2 or 3).
    path_output : str or None
        Path to save the dataframe in a csv file.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with summarized results from the field of view, at the cell
        level. At least `cell_id` (Unique id of the cell) is returned. Other
        indicators are summarized if available:

        * `nb_rna`: Number of detected rna in the cell.
        * `nb_rna_in_nuc`: Number of detected rna inside the nucleus.
        * `nb_rna_out_nuc`: Number of detected rna outside the nucleus.

        Extra coordinates elements detected are counted in the cell and
        summarized as well.

    """
    # check parameters
    stack.check_parameter(
        fov_results=list,
        ndim=int,
        path_output=(str, type(None)))

    # case if no cell were detected
    # TODO make it consistent with the case where there are cells
    if len(fov_results) == 0:
        df = pd.DataFrame({"cell_id": []})
        if path_output is not None:
            stack.save_data_to_csv(df, path_output)
        return df

    # check extra coordinates to summarize
    cell_results = fov_results[0]
    _extra_coord = {}
    for key in cell_results:
        if key in ["cell_id", "bbox", "cell_coord", "cell_mask",
                   "nuc_coord", "nuc_mask", "rna_coord", "image"]:
            continue
        others_coord = cell_results[key]
        if (not isinstance(others_coord, np.ndarray)
                or others_coord.dtype not in [np.int64, np.float64]):
            continue
        _extra_coord[key] = []

    # summarize results at the cell level
    _cell_id = []
    _nb_rna = []
    _nb_rna_in_nuc = []
    _nb_rna_out_nuc = []
    for cell_results in fov_results:
        # get cell id
        _cell_id.append(cell_results["cell_id"])

        # get rna coordinates and relative results
        if "rna_coord" in cell_results:
            rna_coord = cell_results["rna_coord"]
            _nb_rna.append(len(rna_coord))

            # get rna in nucleus
            if "nuc_mask" in cell_results:
                nuc_mask = cell_results["nuc_mask"]
                rna_in_nuc, rna_out_nuc = identify_objects_in_region(
                    nuc_mask,
                    rna_coord,
                    ndim)
                _nb_rna_in_nuc.append(len(rna_in_nuc))
                _nb_rna_out_nuc.append(len(rna_out_nuc))

        # get others coordinates
        for key in _extra_coord:
            others_coord = cell_results[key]
            _extra_coord[key].append(len(others_coord))

    # complete missing mandatory results
    n = len(_cell_id)
    if len(_nb_rna) == 0:
        _nb_rna = [np.nan] * n
    if len(_nb_rna_in_nuc) == 0:
        _nb_rna_in_nuc = [np.nan] * n
    if len(_nb_rna_out_nuc) == 0:
        _nb_rna_out_nuc = [np.nan] * n

    # store minimum results in a dataframe
    result_summary = {"cell_id": _cell_id,
                      "nb_rna": _nb_rna,
                      "nb_rna_in_nuc": _nb_rna_in_nuc,
                      "nb_rna_out_nuc": _nb_rna_out_nuc}

    # store available results on nucleus and rna
    if len(_nb_rna) > 0:
        result_summary["nb_rna"] = _nb_rna
    if len(_nb_rna_in_nuc) > 0:
        result_summary["nb_rna_in_nuc"] = _nb_rna_in_nuc
    if len(_nb_rna_out_nuc) > 0:
        result_summary["nb_rna_out_nuc"] = _nb_rna_out_nuc

    # store results from others elements detected in the cell
    for key in _extra_coord:
        result_summary["nb_{0}".format(key)] = _extra_coord[key]

    # instantiate dataframe
    df = pd.DataFrame(result_summary)

    # save dataframe
    if path_output is not None:
        stack.save_data_to_csv(df, path_output)

    return df


# Postprocessing

def center_mask_coord(main, others=None):
    """Center a 2-d binary mask (surface or boundaries) or a 2-d localization
    coordinates array and pad it.

    One mask or coordinates array should be at least provided (`main`). If
    others masks or arrays are provided (`others`), they will be transformed
    like `main`. All the provided masks should have the same shape.

    Parameters
    ----------
    main : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x) or array of coordinates with shape
        (nb_points, 2).
    others : List(np.ndarray)
        List of binary image with shape (y, x), array of coordinates with
        shape (nb_points, 2) or array of coordinates with shape (nb_points, 3).

    Returns
    -------
    main_centered : np.ndarray, np.uint or np.int or bool
        Centered binary image with shape (y, x).
    others_centered : List(np.ndarray)
        List of centered binary image with shape (y, x), centered array of
        coordinates with shape (nb_points, 2) or centered array of coordinates
        with shape (nb_points, 3).

    """
    # TODO allow the case when coordinates do not represent external boundaries
    # check parameters
    stack.check_array(
        main,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(others=(list, type(None)))
    if others is not None and len(others) != 0:
        for x in others:
            if x is None:
                continue
            stack.check_array(
                x,
                ndim=2,
                dtype=[np.uint8, np.uint16, np.int64, bool])

    # initialize parameter
    marge = stack.get_margin_value()

    # compute by how much we need to move the main object to center it
    if main.shape[1] == 2:
        # 'main' is already a 2-d coordinates array (probably locating the
        # external boundaries of the main object)
        main_coord = main.copy()
    else:
        # get external boundaries coordinates
        main_coord = from_binary_to_coord(main)

    # we get the minimum and maximum from external boundaries coordinates so
    # we add 1 to the minimum and substract 1 to the maximum
    min_y, max_y = main_coord[:, 0].min() + 1, main_coord[:, 0].max() - 1
    min_x, max_x = main_coord[:, 1].min() + 1, main_coord[:, 1].max() - 1

    # we compute the shape of the main object with a predefined marge
    shape_y = max_y - min_y + 1
    shape_x = max_x - min_x + 1
    main_centered_shape = (shape_y + 2 * marge, shape_x + 2 * marge)

    # center the main object
    if main.shape[1] == 2:
        # 'main' is a 2-d coordinates array
        main_centered = main.copy()
        main_centered[:, 0] = main_centered[:, 0] - min_y + marge
        main_centered[:, 1] = main_centered[:, 1] - min_x + marge
    else:
        # 'main' is a 2-d binary matrix
        main_centered = np.zeros(main_centered_shape, dtype=main.dtype)
        crop = main[min_y:max_y + 1, min_x:max_x + 1]
        main_centered[marge:shape_y + marge, marge:shape_x + marge] = crop

    if others is None or len(others) == 0:
        return main_centered, None

    # center the others objects with the same transformation
    others_centered = []
    for other in others:
        if other is None:
            other_centered = None
        elif other.shape[1] == 2:
            # 'other' is a 2-d coordinates array
            other_centered = other.copy()
            other_centered[:, 0] = other_centered[:, 0] - min_y + marge
            other_centered[:, 1] = other_centered[:, 1] - min_x + marge
        elif other.shape[1] == 3 or other.shape[1] == 4:
            # 'other' is a 3-d or 4-d coordinates
            other_centered = other.copy()
            other_centered[:, 1] = other_centered[:, 1] - min_y + marge
            other_centered[:, 2] = other_centered[:, 2] - min_x + marge
        else:
            # 'other' is a 2-d binary matrix
            other_centered = np.zeros(main_centered_shape, dtype=other.dtype)
            crop = other[min_y:max_y + 1, min_x:max_x + 1]
            other_centered[marge:shape_y + marge, marge:shape_x + marge] = crop
        others_centered.append(other_centered)

    return main_centered, others_centered


def from_boundaries_to_surface(binary_boundaries):
    """Fill in the binary matrix representing the boundaries of an object.

    Parameters
    ----------
    binary_boundaries : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x).

    Returns
    -------
    binary_surface : np.ndarray, bool
        Binary image with shape (y, x).

    """
    # check parameters
    stack.check_array(
        binary_boundaries,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, bool])

    # from binary boundaries to binary surface
    binary_surface = ndi.binary_fill_holes(binary_boundaries)

    return binary_surface


def from_surface_to_boundaries(binary_surface):
    """Convert the binary surface to binary boundaries.

    Parameters
    ----------
    binary_surface : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x).

    Returns
    -------
    binary_boundaries : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x).

    """
    # check parameters
    stack.check_array(
        binary_surface,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, bool])
    original_dtype = binary_surface.dtype

    # pad the binary surface in case object if on the edge
    binary_surface_ = np.pad(binary_surface, [(1, 1)], mode="constant")

    # compute distance map of the surface binary mask
    distance_map = ndi.distance_transform_edt(binary_surface_)

    # get binary boundaries
    binary_boundaries_ = (distance_map < 2) & (distance_map > 0)
    binary_boundaries_ = binary_boundaries_.astype(original_dtype)

    # remove pad
    binary_boundaries = binary_boundaries_[1:-1, 1:-1]

    return binary_boundaries


def from_binary_to_coord(binary):
    """Extract coordinates from a 2-d binary matrix.

    As the resulting coordinates represent the external boundaries of the
    object, the coordinates values can be negative.

    Parameters
    ----------
    binary : np.ndarray, np.uint or np.int or bool
        Binary image with shape (y, x).

    Returns
    -------
    coord : np.ndarray, np.int64
        Array of boundaries coordinates with shape (nb_points, 2).

    """
    # check parameters
    stack.check_array(
        binary,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, bool])

    # we enlarge the binary mask with one pixel to be sure the external
    # boundaries of the object still fit within the frame
    binary_ = np.pad(binary, [(1, 1)], mode="constant")

    # get external boundaries coordinates
    coord = find_contours(binary_, level=0)[0].astype(np.int64)

    # remove the pad
    coord -= 1

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
    stack.check_array(coord, ndim=2, dtype=[np.int64])

    # for each array in the list, complete its coordinates using the scikit
    # image method 'polygon_perimeter'
    coord_y, coord_x = polygon_perimeter(coord[:, 0], coord[:, 1])
    coord_y = coord_y[:, np.newaxis]
    coord_x = coord_x[:, np.newaxis]
    coord_completed = np.concatenate((coord_y, coord_x), axis=-1)

    return coord_completed


def from_coord_to_frame(coord, external_coord=True):
    """Initialize a frame shape to represent coordinates values in 2-d matrix.

    If coordinates represent the external boundaries of an object, we add 1 to
    the minimum coordinate and substract 1 to the maximum coordinate in order
    to build the frame. The frame centers the coordinates by default.

    Parameters
    ----------
    coord : np.ndarray, np.int64
        Array of cell boundaries coordinates with shape (nb_points, 2) or
        (nb_points, 3).
    external_coord : bool
        Coordinates represent external boundaries of object.

    Returns
    -------
    frame_shape : tuple
        Shape of the 2-d matrix.
    min_y : int
        Value tu substract from the y coordinate axis.
    min_x : int
        Value tu substract from the x coordinate axis.
    marge : int
        Value to add to the coordinates.

    """
    # check parameter
    stack.check_parameter(external_coord=bool)

    # initialize marge
    marge = stack.get_margin_value()

    # from 2D coordinates boundaries to binary boundaries
    if external_coord:
        min_y, max_y = coord[:, 0].min() + 1, coord[:, 0].max() - 1
        min_x, max_x = coord[:, 1].min() + 1, coord[:, 1].max() - 1
    else:
        min_y, max_y = coord[:, 0].min(), coord[:, 0].max()
        min_x, max_x = coord[:, 1].min(), coord[:, 1].max()
    shape_y = max_y - min_y + 1
    shape_x = max_x - min_x + 1
    frame_shape = (shape_y + 2 * marge, shape_x + 2 * marge)

    return frame_shape, min_y, min_x, marge


def from_coord_to_surface(cell_coord, nuc_coord=None, rna_coord=None,
                          external_coord=True):
    """Convert 2-d coordinates to a binary matrix with the surface of the
    object.

    If we manipulate the coordinates of the external boundaries, the relative
    binary matrix has two extra pixels in each dimension. We compensate by
    keeping only the inside pixels of the object surface.

    If others coordinates are provided (nucleus and mRNAs), the relative
    binary matrix is built with the same shape as the main coordinates (cell).

    Parameters
    ----------
    cell_coord : np.ndarray, np.int64
        Array of cell boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).
    external_coord : bool
        Coordinates represent external boundaries of object.

    Returns
    -------
    cell_surface : np.ndarray, bool
        Binary image of cell surface with shape (y, x).
    nuc_surface : np.ndarray, bool
        Binary image of nucleus surface with shape (y, x).
    rna_binary : np.ndarray, bool
        Binary image of mRNAs localizations with shape (y, x).
    new_rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or (nb_points, 3).

    """
    # check parameters
    stack.check_array(cell_coord, ndim=2, dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord, ndim=2, dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int64])
    stack.check_parameter(external_coord=bool)

    # center coordinates
    cell_coord_, [nuc_coord_, rna_coord_] = center_mask_coord(
        main=cell_coord,
        others=[nuc_coord, rna_coord])

    # get the binary frame
    frame_shape, min_y, min_x, marge = from_coord_to_frame(
        coord=cell_coord_,
        external_coord=external_coord)

    # from coordinates to binary external boundaries
    cell_boundaries_ext = np.zeros(frame_shape, dtype=bool)
    cell_boundaries_ext[cell_coord_[:, 0], cell_coord_[:, 1]] = True
    if nuc_coord_ is not None:
        nuc_boundaries_ext = np.zeros(frame_shape, dtype=bool)
        nuc_boundaries_ext[nuc_coord_[:, 0], nuc_coord_[:, 1]] = True
    else:
        nuc_boundaries_ext = None

    # from binary external boundaries to binary external surface
    cell_surface_ext = from_boundaries_to_surface(cell_boundaries_ext)
    if nuc_boundaries_ext is not None:
        nuc_surface_ext = from_boundaries_to_surface(nuc_boundaries_ext)
    else:
        nuc_surface_ext = None

    # from binary external surface to binary surface
    cell_surface = cell_surface_ext & (~cell_boundaries_ext)
    if nuc_surface_ext is not None:
        nuc_surface = nuc_surface_ext & (~nuc_boundaries_ext)
    else:
        nuc_surface = None

    # center mRNAs coordinates
    if rna_coord_ is not None:
        rna_binary = np.zeros(frame_shape, dtype=bool)
        if rna_coord_.shape[1] == 2:
            rna_binary[rna_coord_[:, 0], rna_coord_[:, 1]] = True
        else:
            rna_binary[rna_coord_[:, 1], rna_coord_[:, 2]] = True
        new_rna_coord = rna_coord_.copy()

    else:
        rna_binary = None
        new_rna_coord = None

    return cell_surface, nuc_surface, rna_binary, new_rna_coord
