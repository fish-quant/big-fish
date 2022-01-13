# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Postprocessing functions functions for bigfish.segmentation subpackage.
"""

import bigfish.stack as stack

import numpy as np
from scipy import ndimage as ndi

from skimage.measure import label, find_contours
from skimage.morphology import remove_small_objects
from skimage.draw import polygon_perimeter


# TODO make functions compatible with different type of integers

# ### Labelled images ###

def label_instances(image_binary):
    """Count and label the different instances previously segmented in an
    image.

    Parameters
    ----------
    image_binary : np.ndarray, bool
        Binary segmented image with shape (z, y, x) or (y, x).

    Returns
    -------
    image_label : np.ndarray, np.int64
        Labelled image. Each instance is characterized by the same pixel value.

    """
    # check parameters
    stack.check_array(image_binary, ndim=[2, 3], dtype=bool)

    # label instances
    image_label = label(image_binary)

    return image_label


def merge_labels(image_label_1, image_label_2):
    """Combine two partial labels of the same image.

    To prevent merging conflict, labels should not be rescale.

    Parameters
    ----------
    image_label_1 : np.ndarray, np.int64
        Labelled image with shape (z, y, x) or (y, x).
    image_label_2 : np.ndarray, np.int64
        Labelled image with shape (z, y, x) or (y, x).

    Returns
    -------
    image_label : np.ndarray, np.int64
        Labelled image with shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(image_label_1, ndim=[2, 3], dtype=np.int64)
    stack.check_array(image_label_2, ndim=[2, 3], dtype=np.int64)

    # count number of instances
    nb_instances_1 = image_label_1.max()
    nb_instances_2 = image_label_2.max()
    nb_instances = nb_instances_1 + nb_instances_2

    # check if labels can be merged
    if nb_instances > np.iinfo(np.int64).max:
        raise ValueError("Labels can not be merged. There are too many "
                         "instances for a 64 bit image, labels could overlap.")

    # merge labels
    image_label_2[image_label_2 > 0] += image_label_1
    image_label = np.maximum(image_label_1, image_label_2)

    return image_label


# ### Clean segmentation ###
# TODO make it available for 3D images
def clean_segmentation(image, small_object_size=None, fill_holes=False,
                       smoothness=None, delimit_instance=False):
    """Clean segmentation results (binary masks or integer labels).

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).
    small_object_size : int or None
        Areas with a smaller surface (in pixels) are removed.
    fill_holes : bool
        Fill holes within a labelled or masked area.
    smoothness : int or None
        Radius of a median kernel filter. The higher the smoother instance
        boundaries are.
    delimit_instance : bool
        Delimit clearly instances boundaries by preventing contact between each
        others.

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # check parameters
    stack.check_array(image, ndim=2, dtype=[np.int64, bool])
    stack.check_parameter(small_object_size=(int, type(None)),
                          fill_holes=bool,
                          smoothness=(int, type(None)),
                          delimit_instance=bool)

    # initialize cleaned image
    image_cleaned = image.copy()

    # remove small object
    if small_object_size is not None:
        image_cleaned = _remove_small_area(image_cleaned, small_object_size)

    # fill holes
    if fill_holes:
        image_cleaned = _fill_hole(image_cleaned)

    if smoothness:
        image_cleaned = _smooth_instance(image_cleaned, smoothness)

    if delimit_instance:
        image_cleaned = _delimit_instance(image_cleaned)

    return image_cleaned


def _remove_small_area(image, min_size):
    """Remove segmented areas with a small surface.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).
    min_size : int
        Areas with a smaller surface (in pixels) are removed.

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # remove small object
    image_cleaned = remove_small_objects(image, min_size=min_size)

    return image_cleaned


def _fill_hole(image):
    """Fill holes within the segmented areas.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # fill holes in a binary mask
    if image.dtype == bool:
        image_cleaned = ndi.binary_fill_holes(image)

    # fill holes in a labelled image
    else:
        image_cleaned = np.zeros_like(image)
        for i in range(1, image.max() + 1):
            image_binary = image == i
            image_binary = ndi.binary_fill_holes(image_binary)
            image_cleaned[image_binary] = i

    return image_cleaned


def _smooth_instance(image, radius):
    """Apply a median filter to smooth instance boundaries.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).
    radius : int
        Radius of the kernel for the median filter. The higher the smoother.

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # smooth instance boundaries for a binary mask
    if image.dtype == bool:
        image_cleaned = image.astype(np.uint8)
        image_cleaned = stack.median_filter(image_cleaned, "disk", radius)
        image_cleaned = image_cleaned.astype(bool)

    # smooth instance boundaries for a labelled image
    else:
        if image.max() <= 65535 and image.min() >= 0:
            image_cleaned = image.astype(np.uint16)
            image_cleaned = stack.median_filter(image_cleaned, "disk", radius)
            image_cleaned = image_cleaned.astype(np.int64)
        else:
            raise ValueError("Segmentation boundaries can't be smoothed "
                             "because more than 65535 has been detected in "
                             "the image. Smoothing is performed with 16-bit "
                             "unsigned integer images.")

    return image_cleaned


def _delimit_instance(image):
    """Subtract an eroded image to a dilated one in order to prevent
    boundaries contact.

    Parameters
    ----------
    image : np.ndarray, np.int64 or bool
        Labelled or masked image with shape (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int64 or bool
        Cleaned image with shape (y, x).

    """
    # handle 64 bit integer
    original_dtype = image.dtype
    if image.dtype == np.int64:
        image = image.astype(np.float64)

    # erode-dilate mask
    image_dilated = stack.dilation_filter(image, "disk", 1)
    image_eroded = stack.erosion_filter(image, "disk", 1)
    if original_dtype == bool:
        borders = image_dilated & ~image_eroded
        image_cleaned = image.copy()
        image_cleaned[borders] = False
    else:
        borders = image_dilated - image_eroded
        image_cleaned = image.copy()
        image_cleaned[borders > 0] = 0
        image_cleaned = image_cleaned.astype(original_dtype)

    return image_cleaned


def remove_disjoint(image):
    """For each instances with disconnected parts, keep the larger one.

    Parameters
    ----------
    image : np.ndarray, np.int, np.uint or bool
        Labelled image with shape (z, y, x) or (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int or np.uint
        Cleaned image with shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.int64, bool])

    # handle boolean array
    cast_to_bool = False
    if image.dtype == bool:
        cast_to_bool = bool
        image = image.astype(np.uint8)

    # initialize cleaned labels
    image_cleaned = np.zeros_like(image)

    # loop over instances
    max_label = image.max()
    for i in range(1, max_label + 1):

        # get instance mask
        mask = image == i

        # check if an instance is labelled with this value
        if mask.sum() == 0:
            continue

        # get an index for each disconnected part of the instance
        labelled_mask = label(mask)
        indices = sorted(list(set(labelled_mask.ravel())))
        if 0 in indices:
            indices = indices[1:]

        # keep the largest part of the instance
        max_area = 0
        mask_instance = None
        for j in indices:
            mask_part_j = labelled_mask == j
            area_j = mask_part_j.sum()
            if area_j > max_area:
                max_area = area_j
                mask_instance = mask_part_j

        # add instance in the final label
        image_cleaned[mask_instance] = i

    if cast_to_bool:
        image_cleaned = image_cleaned.astype(bool)

    return image_cleaned


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
    stack.check_array(main,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(others=(list, type(None)))
    if others is not None and len(others) != 0:
        for x in others:
            if x is None:
                continue
            stack.check_array(x,
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
    stack.check_array(binary_boundaries,
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
    stack.check_array(binary_surface,
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
    stack.check_array(binary,
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
    stack.check_array(coord,
                      ndim=2,
                      dtype=[np.int64])

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


# TODO replace 'cyt_coord' by 'cell_coord'
def from_coord_to_surface(cyt_coord, nuc_coord=None, rna_coord=None,
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
    cyt_coord : np.ndarray, np.int64
        Array of cytoplasm boundaries coordinates with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Array of nucleus boundaries coordinates with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or
        (nb_points, 3).
    external_coord : bool
        Coordinates represent external boundaries of object.

    Returns
    -------
    cyt_surface : np.ndarray, bool
        Binary image of cytoplasm surface with shape (y, x).
    nuc_surface : np.ndarray, bool
        Binary image of nucleus surface with shape (y, x).
    rna_binary : np.ndarray, bool
        Binary image of mRNAs localizations with shape (y, x).
    new_rna_coord : np.ndarray, np.int64
        Array of mRNAs coordinates with shape (nb_points, 2) or (nb_points, 3).

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
    stack.check_parameter(external_coord=bool)

    # center coordinates
    cyt_coord_, [nuc_coord_, rna_coord_] = center_mask_coord(
        main=cyt_coord,
        others=[nuc_coord, rna_coord])

    # get the binary frame
    frame_shape, min_y, min_x, marge = from_coord_to_frame(
        coord=cyt_coord_,
        external_coord=external_coord)

    # from coordinates to binary external boundaries
    cyt_boundaries_ext = np.zeros(frame_shape, dtype=bool)
    cyt_boundaries_ext[cyt_coord_[:, 0], cyt_coord_[:, 1]] = True
    if nuc_coord_ is not None:
        nuc_boundaries_ext = np.zeros(frame_shape, dtype=bool)
        nuc_boundaries_ext[nuc_coord_[:, 0], nuc_coord_[:, 1]] = True
    else:
        nuc_boundaries_ext = None

    # from binary external boundaries to binary external surface
    cyt_surface_ext = from_boundaries_to_surface(cyt_boundaries_ext)
    if nuc_boundaries_ext is not None:
        nuc_surface_ext = from_boundaries_to_surface(nuc_boundaries_ext)
    else:
        nuc_surface_ext = None

    # from binary external surface to binary surface
    cyt_surface = cyt_surface_ext & (~cyt_boundaries_ext)
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

    return cyt_surface, nuc_surface, rna_binary, new_rna_coord
