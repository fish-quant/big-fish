# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.detection subpackage.
"""

import warnings

import numpy as np

import bigfish.stack as stack


# ### Pixel - nanometer conversion

def convert_spot_coordinates(spots, voxel_size):
    """Convert spots coordinates from pixel to nanometer.

    Parameters
    ----------
    spots : np.ndarray
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 2).
    voxel_size : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.

    Returns
    -------
    spots_nanometer : np.ndarray
        Coordinates of the detected spots with shape (nb_spots, 3) or
        (nb_spots, 3), in nanometer.

    """
    # check parameters
    stack.check_parameter(voxel_size=(int, float, tuple, list))
    stack.check_array(spots, ndim=2, dtype=[np.float64, np.int64])
    dtype = spots.dtype

    # check consistency between parameters
    ndim = spots.shape[1]
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError("'voxel_size' must be a scalar or a sequence "
                             "with {0} elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim

    # convert spots coordinates in nanometer
    spots_nanometer = spots.copy()
    if ndim == 3:
        spots_nanometer[:, 0] *= voxel_size[0]
        spots_nanometer[:, 1:] *= voxel_size[-1]

    else:
        spots_nanometer *= voxel_size[-1]
    spots_nanometer = spots_nanometer.astype(dtype)

    return spots_nanometer


def get_object_radius_pixel(voxel_size_nm, object_radius_nm, ndim):
    """Convert the object radius in pixel.

    When the object considered is a spot this value can be interpreted as the
    standard deviation of the spot PSF, in pixel. For any object modelled with
    a gaussian signal, this value can be interpreted as the standard deviation
    of the gaussian.

    Parameters
    ----------
    voxel_size_nm : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    object_radius_nm : int, float, Tuple(int, float) or List(int, float)
        Radius of the object, in nanometer. One value per spatial dimension
        (zyx or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions.
    ndim : int
        Number of spatial dimension to consider.

    Returns
    -------
    object_radius_px : Tuple[float]
        Radius of the object in pixel, one element per dimension (zyx or yx
        dimensions).

    """
    # check parameters
    stack.check_parameter(
        voxel_size_nm=(int, float, tuple, list),
        object_radius_nm=(int, float, tuple, list),
        ndim=int)

    # check consistency between parameters
    if isinstance(voxel_size_nm, (tuple, list)):
        if len(voxel_size_nm) != ndim:
            raise ValueError("'voxel_size_nm' must be a scalar or a sequence "
                             "with {0} elements.".format(ndim))
    else:
        voxel_size_nm = (voxel_size_nm,) * ndim
    if isinstance(object_radius_nm, (tuple, list)):
        if len(object_radius_nm) != ndim:
            raise ValueError("'object_radius_nm' must be a scalar or a "
                             "sequence with {0} elements.".format(ndim))
    else:
        object_radius_nm = (object_radius_nm,) * ndim

    # get radius in pixel
    object_radius_px = [b / a for a, b in zip(voxel_size_nm, object_radius_nm)]
    object_radius_px = tuple(object_radius_px)

    return object_radius_px


def get_object_radius_nm(voxel_size_nm, object_radius_px, ndim):
    """Convert the object radius in nanometer.

    When the object considered is a spot this value can be interpreted as the
    standard deviation of the spot PSF, in nanometer. For any object modelled
    with a gaussian signal, this value can be interpreted as the standard
    deviation of the gaussian.

    Parameters
    ----------
    voxel_size_nm : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    object_radius_px : int, float, Tuple(int, float) or List(int, float)
        Radius of the object, in pixel. One value per spatial dimension
        (zyx or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions.
    ndim : int
        Number of spatial dimension to consider.

    Returns
    -------
    object_radius_nm : Tuple[float]
        Radius of the object in nanometer, one element per dimension (zyx or yx
        dimensions).

    """
    # check parameters
    stack.check_parameter(
        voxel_size_nm=(int, float, tuple, list),
        object_radius_px=(int, float, tuple, list),
        ndim=int)

    # check consistency between parameters
    if isinstance(voxel_size_nm, (tuple, list)):
        if len(voxel_size_nm) != ndim:
            raise ValueError("'voxel_size_nm' must be a scalar or a sequence "
                             "with {0} elements.".format(ndim))
    else:
        voxel_size_nm = (voxel_size_nm,) * ndim
    if isinstance(object_radius_px, (tuple, list)):
        if len(object_radius_px) != ndim:
            raise ValueError("'object_radius_px' must be a scalar or a "
                             "sequence with {0} elements.".format(ndim))
    else:
        object_radius_px = (object_radius_px,) * ndim

    # get radius in pixel
    object_radius_nm = [a * b for a, b in zip(voxel_size_nm, object_radius_px)]
    object_radius_nm = tuple(object_radius_nm)

    return object_radius_nm


# ### Reference spot ###

def build_reference_spot(image, spots, voxel_size, spot_radius, alpha=0.5):
    """Build a median or mean spot in 3 or 2 dimensions as reference.

    Reference spot is computed from a sample of uncropped detected spots. If
    such sample is not possible, an empty frame is returned.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray
        Coordinate of the spots with shape (nb_spots, 3) for 3-d images or
        (nb_spots, 2) for 2-d images.
    voxel_size : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    spot_radius : int, float, Tuple(int, float) or List(int, float)
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions.
    alpha : int or float
        Intensity score of the reference spot, between 0 and 1. If 0, reference
        spot approximates the spot with the lowest intensity. If 1, reference
        spot approximates the brightest spot. Default is 0.5.

    Returns
    -------
    reference_spot : np.ndarray
        Reference spot in 3-d or 2-d.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(spots, ndim=2, dtype=[np.float64, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list),
        alpha=(int, float))
    if alpha < 0 or alpha > 1:
        raise ValueError("'alpha' should be a value between 0 and 1, not {0}"
                         .format(alpha))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError(
                "'spot_radius' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim

    # compute radius used to crop spot image
    radius_pixel = get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=ndim)
    radius = [np.sqrt(ndim) * r for r in radius_pixel]
    radius = tuple(radius)

    # build reference spot
    if image.ndim == 3:
        reference_spot = _build_reference_spot_3d(image, spots, radius, alpha)
    else:
        reference_spot = _build_reference_spot_2d(image, spots, radius, alpha)

    return reference_spot


def _build_reference_spot_3d(image, spots, radius, alpha):
    """Build a median or mean spot in 3 dimensions as reference.

    Reference spot is computed from a sample of uncropped detected spots. If
    such sample is not possible, an empty frame is returned.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) for 3-d images.
    radius : Tuple[float]
        Radius in pixel of the detected spots, one element per dimension.
    alpha : int or float
        Intensity score of the reference spot, between 0 and 1. If 0, reference
        spot approximates the spot with the lowest intensity. If 1, reference
        spot approximates the brightest spot.

    Returns
    -------
    reference_spot : np.ndarray
        Reference spot in 3-d.

    """
    # get a rounded radius for each dimension
    radius_z = np.ceil(radius[0]).astype(np.int64)
    z_shape = radius_z * 2 + 1
    radius_yx = np.ceil(radius[-1]).astype(np.int64)
    yx_shape = radius_yx * 2 + 1

    # randomly choose some spots to aggregate
    indices = [i for i in range(spots.shape[0])]
    np.random.shuffle(indices)
    indices = indices[:min(2000, spots.shape[0])]
    candidate_spots = spots[indices, :]

    # collect area around each spot
    l_reference_spot = []
    for i_spot in range(candidate_spots.shape[0]):

        # get spot coordinates
        spot_z, spot_y, spot_x = candidate_spots[i_spot, :]

        # get the volume of the spot
        image_spot, _, = _get_spot_volume(image, spot_z, spot_y, spot_x,
                                          radius_z, radius_yx)

        # keep images that are not cropped by the borders
        if image_spot.shape == (z_shape, yx_shape, yx_shape):
            l_reference_spot.append(image_spot)

    # if not enough spots are detected
    if len(l_reference_spot) <= 30:
        warnings.warn("Problem occurs during the computation of a reference "
                      "spot. Not enough (uncropped) spots have been detected.",
                      UserWarning)
    if len(l_reference_spot) == 0:
        reference_spot = np.zeros(
            (z_shape, yx_shape, yx_shape), dtype=image.dtype)
        return reference_spot

    # project the different spot images
    l_reference_spot = np.stack(l_reference_spot, axis=0)
    alpha_ = alpha * 100
    reference_spot = np.percentile(l_reference_spot, alpha_, axis=0)
    reference_spot = reference_spot.astype(image.dtype)

    return reference_spot


def _get_spot_volume(image, spot_z, spot_y, spot_x, radius_z, radius_yx):
    """Get a subimage of a detected spot in 3 dimensions.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x).
    spot_z : np.int64
        Coordinate of the detected spot along the z axis.
    spot_y : np.int64
        Coordinate of the detected spot along the y axis.
    spot_x : np.int64
        Coordinate of the detected spot along the x axis.
    radius_z : int
        Radius in pixel of the detected spot, along the z axis.
    radius_yx : int
        Radius in pixel of the detected spot, on the yx plan.

    Returns
    -------
    image_spot : np.ndarray
        Reference spot in 3-d.
    _ : Tuple[int]
        Lower zyx coordinates of the crop.

    """
    # get boundaries of the volume surrounding the spot
    z_spot_min = max(0, int(spot_z - radius_z))
    z_spot_max = min(image.shape[0], int(spot_z + radius_z))
    y_spot_min = max(0, int(spot_y - radius_yx))
    y_spot_max = min(image.shape[1], int(spot_y + radius_yx))
    x_spot_min = max(0, int(spot_x - radius_yx))
    x_spot_max = min(image.shape[2], int(spot_x + radius_yx))

    # get the volume of the spot
    image_spot = image[z_spot_min:z_spot_max + 1,
                       y_spot_min:y_spot_max + 1,
                       x_spot_min:x_spot_max + 1]

    return image_spot, (z_spot_min, y_spot_min, x_spot_min)


def _build_reference_spot_2d(image, spots, radius, alpha):
    """Build a median or mean spot in 2 dimensions as reference.

    Reference spot is computed from a sample of uncropped detected spots. If
    such sample is not possible, an empty frame is returned.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 2) for 2-d images.
    radius : Tuple[float]
        Radius in pixel of the detected spots, one element per dimension.
    alpha : int or float
        Intensity score of the reference spot, between 0 and 1. If 0, reference
        spot approximates the spot with the lowest intensity. If 1, reference
        spot approximates the brightest spot.

    Returns
    -------
    reference_spot : np.ndarray
        Reference spot in 2-d.

    """
    # get a rounded radius for each dimension
    radius_yx = np.ceil(radius[-1]).astype(np.int64)
    yx_shape = radius_yx * 2 + 1

    # randomly choose some spots to aggregate
    indices = [i for i in range(spots.shape[0])]
    np.random.shuffle(indices)
    indices = indices[:min(2000, spots.shape[0])]
    candidate_spots = spots[indices, :]

    # collect area around each spot
    l_reference_spot = []
    for i_spot in range(candidate_spots.shape[0]):

        # get spot coordinates
        spot_y, spot_x = candidate_spots[i_spot, :]

        # get the volume of the spot
        image_spot, _ = _get_spot_surface(image, spot_y, spot_x, radius_yx)

        # keep images that are not cropped by the borders
        if image_spot.shape == (yx_shape, yx_shape):
            l_reference_spot.append(image_spot)

    # if not enough spots are detected
    if len(l_reference_spot) <= 30:
        warnings.warn("Problem occurs during the computation of a reference "
                      "spot. Not enough (uncropped) spots have been detected.",
                      UserWarning)
    if len(l_reference_spot) == 0:
        reference_spot = np.zeros((yx_shape, yx_shape), dtype=image.dtype)
        return reference_spot

    # project the different spot images
    l_reference_spot = np.stack(l_reference_spot, axis=0)
    alpha_ = alpha * 100
    reference_spot = np.percentile(l_reference_spot, alpha_, axis=0)
    reference_spot = reference_spot.astype(image.dtype)

    return reference_spot


def _get_spot_surface(image, spot_y, spot_x, radius_yx):
    """Get a subimage of a detected spot in 2 dimensions.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (y, x).
    spot_y : np.int64
        Coordinate of the detected spot along the y axis.
    spot_x : np.int64
        Coordinate of the detected spot along the x axis.
    radius_yx : int
        Radius in pixel of the detected spot, on the yx plan.

    Returns
    -------
    image_spot : np.ndarray
        Reference spot in 2-d.
    _ : Tuple[int]
        Lower yx coordinates of the crop.

    """
    # get boundaries of the surface surrounding the spot
    y_spot_min = max(0, int(spot_y - radius_yx))
    y_spot_max = min(image.shape[0], int(spot_y + radius_yx))
    x_spot_min = max(0, int(spot_x - radius_yx))
    x_spot_max = min(image.shape[1], int(spot_x + radius_yx))

    # get the surface of the spot
    image_spot = image[y_spot_min:y_spot_max + 1,
                       x_spot_min:x_spot_max + 1]

    return image_spot, (y_spot_min, x_spot_min)


# ### SNR ###

def compute_snr_spots(image, spots, voxel_size, spot_radius):
    """Compute signal-to-noise ratio (SNR) based on spot coordinates.

    .. math::

        \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
        \\mbox{mean(background)}}{\\mbox{std(background)}}

    Background is a region twice larger surrounding the spot region. Only the
    y and x dimensions are taking into account to compute the SNR.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray
        Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
        One coordinate per dimension (zyx or yx coordinates).
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.

    Returns
    -------
    snr : float
        Median signal-to-noise ratio computed for every spots.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_range_value(image, min_=0)
    stack.check_array(spots, ndim=2, dtype=[np.float64, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError(
                "'spot_radius' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim

    # cast spots coordinates if needed
    if spots.dtype == np.float64:
        spots = np.round(spots).astype(np.int64)

    # cast image if needed
    image_to_process = image.copy().astype(np.float64)

    # clip coordinate if needed
    if ndim == 3:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
        spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
    else:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

    # compute radius used to crop spot image
    radius_pixel = get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=ndim)
    radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
    radius_signal_ = tuple(radius_signal_)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * 2 for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(np.int)
    radius_background = np.ceil(radius_background_).astype(np.int)

    # loop over spots
    snr_spots = []
    for spot in spots:

        # extract spot images
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx
        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            spot_background_, _ = _get_spot_volume(
                image_to_process, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border (along y and x dimensions)
            expected_size = (2 * radius_background_yx + 1) ** 2
            actual_size = spot_background.shape[1] * spot_background.shape[2]
            if expected_size != actual_size:
                continue

            # remove signal from background crop
            spot_background[:,
                            edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = _get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border
            expected_size = (2 * radius_background_yx + 1) ** 2
            if expected_size != spot_background.size:
                continue

            # remove signal from background crop
            spot_background[edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        # compute mean background
        mean_background = np.mean(spot_background)

        # compute standard deviation background
        std_background = np.std(spot_background)

        # compute SNR
        snr = (max_signal - mean_background) / std_background
        snr_spots.append(snr)

    #  average SNR
    if len(snr_spots) == 0:
        snr = 0.
    else:
        snr = np.median(snr_spots)

    return snr


# ### Miscellaneous ###

def get_breaking_point(x, y):
    """Select the x-axis value where a L-curve has a kink.

    Assuming a L-curve from A to B, the 'breaking_point' is the more distant
    point to the segment [A, B].

    Parameters
    ----------
    x : np.array
        X-axis values.
    y : np.array
        Y-axis values.

    Returns
    -------
    breaking_point : float
        X-axis value at the kink location.
    x : np.array
        X-axis values.
    y : np.array
        Y-axis values.

    """
    # check parameters
    stack.check_array(x, ndim=1, dtype=[np.float64, np.int64])
    stack.check_array(y, ndim=1, dtype=[np.float64, np.int64])

    # select threshold where curve break
    slope = (y[-1] - y[0]) / len(y)
    y_grad = np.gradient(y)
    m = list(y_grad >= slope)
    j = m.index(False)
    m = m[j:]
    x = x[j:]
    y = y[j:]
    if True in m:
        i = m.index(True)
    else:
        i = -1
    breaking_point = float(x[i])

    return breaking_point, x, y
