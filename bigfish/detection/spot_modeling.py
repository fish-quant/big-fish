# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to model spots by fitting gaussian parameters.
"""

import warnings

import numpy as np

import bigfish.stack as stack

from scipy.special import erf
from scipy.optimize import curve_fit


# ### Reference spot ###

def build_reference_spot(image, spots, voxel_size_z=None, voxel_size_yx=100,
                         psf_z=None, psf_yx=200, alpha=0.5):
    """Build a median or mean spot in 3 or 2 dimensions as reference.

    Reference spot is computed from a sample of uncropped detected spots. If
    such sample is not possible, an empty frame is returned.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) for 3-d images or
        (nb_spots, 2) for 2-d images.
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, image is
        considered in 2-d.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan, in
        nanometer. If None, image is considered in 2-d.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan, in
        nanometer.
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
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(spots, ndim=2, dtype=np.int64)
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float),
                          alpha=(int, float))
    if alpha < 0 or alpha > 1:
        raise ValueError("'alpha' should be a value between 0 and 1, not {0}"
                         .format(alpha))

    # check number of dimensions
    ndim = image.ndim
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and psf_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'psf_z' parameter is missing.".format(ndim))
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if ndim == 2:
        voxel_size_z, psf_z = None, None

    # compute radius
    radius = stack.get_radius(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

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
        Radius in pixels of the detected spots, one element per dimension.
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
        reference_spot = np.zeros((z_shape, yx_shape, yx_shape),
                                  dtype=image.dtype)
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
        Radius in pixels of the detected spot, along the z axis.
    radius_yx : int
        Radius in pixels of the detected spot, on the yx plan.

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
        Radius in pixels of the detected spots, one element per dimension.
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
        Radius in pixels of the detected spot, on the yx plan.

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


# ### Spot modelization ###

def modelize_spot(reference_spot, voxel_size_z=None, voxel_size_yx=100,
                  psf_z=None, psf_yx=200, return_coord=False):
    """Fit a gaussian function on the reference spot.

    Parameters
    ----------
    reference_spot : np.ndarray
        A 3-d or 2-d image with detected spot and shape (z, y, x) or (y, x).
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, reference
        spot is considered in 2-d.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan, in
        nanometer. If None, reference spot is considered in 2-d.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan, in
        nanometer.
    return_coord : bool
        Return spot coordinates.

    Returns
    -------
    parameters_fitted : Tuple[float]
        * mu_z : float (optional)
            Coordinate of the spot center along the z axis, in pixel.
        * mu_y : float (optional)
            Coordinate of the spot center along the y axis, in pixel.
        * mu_x : float (optional)
            Coordinate of the spot center along the x axis, in pixel.
        * sigma_z : float
            Standard deviation of the spot along the z axis, in pixel.
            Available only for a 3-d modelization.
        * sigma_yx : float
            Standard deviation of the spot along the yx axis, in pixel.
        * amplitude : float
            Amplitude of the spot.
        * background : float
            Background minimum value of the image.

    """
    # check parameters
    stack.check_array(reference_spot,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float),
                          return_coord=bool)

    # check number of dimensions
    ndim = reference_spot.ndim
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and psf_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'psf_z' parameter is missing.".format(ndim))
    if ndim == 2:
        voxel_size_z, psf_z = None, None

    # initialize a grid representing the reference spot
    grid, centroid_coord = initialize_grid(
        image_spot=reference_spot,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx,
        return_centroid=True)

    # compute amplitude and background of the reference spot
    amplitude, background = _initialize_background_amplitude(reference_spot)

    # initialize parameters of the gaussian function
    f = _objective_function(
        nb_dimension=ndim,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx,
        psf_z=None,
        psf_yx=None,
        psf_amplitude=None)
    if ndim == 3:
        # parameters to fit: mu_z, mu_y, mu_x, sigma_z, sigma_yx, amplitude
        # and background
        centroid_z, centroid_y, centroid_x = centroid_coord
        p0 = [centroid_z, centroid_y, centroid_x, psf_z, psf_yx, amplitude,
              background]
        l_bound = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0]
        u_bound = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    else:
        # parameters to fit: mu_y, mu_x, sigma_yx, amplitude and background
        centroid_y, centroid_x = centroid_coord
        p0 = [centroid_y, centroid_x, psf_yx, amplitude, background]
        l_bound = [-np.inf, -np.inf, -np.inf, -np.inf, 0]
        u_bound = [np.inf, np.inf, np.inf, np.inf, np.inf]

    # fit a gaussian function on this reference spot
    popt, pcov = _fit_gaussian(f, grid, reference_spot, p0,
                               lower_bound=l_bound,
                               upper_bound=u_bound)

    # get optimized parameters to modelize the reference spot as a gaussian
    if ndim == 3:
        mu_z = popt[0]
        mu_y = popt[1]
        mu_x = popt[2]
        sigma_z = popt[3]
        sigma_yx = popt[4]
        amplitude = popt[5]
        background = popt[6]

        if return_coord:
            return mu_z, mu_y, mu_x, sigma_z, sigma_yx, amplitude, background
        else:
            return sigma_z, sigma_yx, amplitude, background

    else:
        mu_y = popt[0]
        mu_x = popt[1]
        sigma_yx = popt[2]
        amplitude = popt[3]
        background = popt[4]

        if return_coord:
            return mu_y, mu_x, sigma_yx, amplitude, background
        else:
            return sigma_yx, amplitude, background


# ### Spot modelization: initialization ###

def initialize_grid(image_spot, voxel_size_z, voxel_size_yx,
                    return_centroid=False):
    """Build a grid in nanometer to compute gaussian function values over a
    full volume or surface.

    Parameters
    ----------
    image_spot : np.ndarray
        An image with detected spot and shape (z, y, x) or (y, x).
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, image spot
        is considered in 2-d.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    return_centroid : bool
        Compute centroid estimation of the grid.

    Returns
    -------
    grid : np.ndarray, np.float32
        A grid with the shape (3, z * y * x) or (2, y * x), in nanometer.
    centroid_coord : Tuple[float]
        Estimated centroid of the spot, in nanometer. One element per
        dimension.

    """
    # initialize grid in 2-d...
    if image_spot.ndim == 2:
        if return_centroid:
            grid, centroid_y, centroid_x = _initialize_grid_2d(
                image_spot, voxel_size_yx, return_centroid)
            return grid, (centroid_y, centroid_x)
        else:
            grid = _initialize_grid_2d(
                image_spot, voxel_size_yx, return_centroid)
            return grid

    # ... or 3-d
    else:
        if return_centroid:
            grid, centroid_z, centroid_y, centroid_x = _initialize_grid_3d(
                image_spot, voxel_size_z, voxel_size_yx, return_centroid)
            return grid, (centroid_z, centroid_y, centroid_x)
        else:
            grid = _initialize_grid_3d(
                image_spot, voxel_size_z, voxel_size_yx, return_centroid)
            return grid


def _initialize_grid_3d(image_spot, voxel_size_z, voxel_size_yx,
                        return_centroid=False):
    """Build a grid in nanometer to compute gaussian function values over a
    full volume.

    Parameters
    ----------
    image_spot : np.ndarray
        A 3-d image with detected spot and shape (z, y, x).
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    return_centroid : bool
        Compute centroid estimation of the grid.

    Returns
    -------
    grid : np.ndarray, np.float32
        A grid with the shape (3, z * y * x), in nanometer.
    centroid_z : float
        Estimated centroid of the spot, in nanometer, along the z axis.
    centroid_y : float
        Estimated centroid of the spot, in nanometer, along the y axis.
    centroid_x : float
        Estimated centroid of the spot, in nanometer, along the x axis.

    """
    # get targeted size
    nb_z, nb_y, nb_x = image_spot.shape
    nb_pixels = image_spot.size

    # build meshgrid
    zz, yy, xx = np.meshgrid(np.arange(nb_z), np.arange(nb_y), np.arange(nb_x),
                             indexing="ij")
    zz = zz.astype(np.float32) * float(voxel_size_z)
    yy = yy.astype(np.float32) * float(voxel_size_yx)
    xx = xx.astype(np.float32) * float(voxel_size_yx)

    # format result
    grid = np.zeros((3, nb_pixels), dtype=np.float32)
    grid[0] = np.reshape(zz, (1, nb_pixels))
    grid[1] = np.reshape(yy, (1, nb_pixels))
    grid[2] = np.reshape(xx, (1, nb_pixels))

    # compute centroid of the grid
    if return_centroid:
        area = np.sum(image_spot)
        dz = image_spot * zz
        dy = image_spot * yy
        dx = image_spot * xx
        centroid_z = np.sum(dz) / area
        centroid_y = np.sum(dy) / area
        centroid_x = np.sum(dx) / area
        return grid, centroid_z, centroid_y, centroid_x

    else:
        return grid


def _initialize_grid_2d(image_spot, voxel_size_yx, return_centroid=False):
    """Build a grid in nanometer to compute gaussian function values over a
    full surface.

    Parameters
    ----------
    image_spot : np.ndarray
        A 2-d image with detected spot and shape (y, x).
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    return_centroid : bool
        Compute centroid estimation of the grid.

    Returns
    -------
    grid : np.ndarray, np.float32
        A grid with the shape (2, y * x), in nanometer.
    centroid_y : float
        Estimated centroid of the spot, in nanometer, along the y axis.
    centroid_x : float
        Estimated centroid of the spot, in nanometer, along the x axis.

    """
    # get targeted size
    nb_y, nb_x = image_spot.shape
    nb_pixels = image_spot.size

    # build meshgrid
    yy, xx = np.meshgrid(np.arange(nb_y), np.arange(nb_x), indexing="ij")
    yy = yy.astype(np.float32) * float(voxel_size_yx)
    xx = xx.astype(np.float32) * float(voxel_size_yx)

    # format result
    grid = np.zeros((2, nb_pixels), dtype=np.float32)
    grid[0] = np.reshape(yy, (1, nb_pixels))
    grid[1] = np.reshape(xx, (1, nb_pixels))

    # compute centroid of the grid
    if return_centroid:
        area = np.sum(image_spot)
        dy = image_spot * yy
        dx = image_spot * xx
        centroid_y = np.sum(dy) / area
        centroid_x = np.sum(dx) / area
        return grid, centroid_y, centroid_x

    else:
        return grid


def _initialize_background_amplitude(image_spot):
    """Compute amplitude of a spot and background minimum value.

    Parameters
    ----------
    image_spot : np.ndarray, np.uint
        Image with detected spot and shape (z, y, x) or (y, x).

    Returns
    -------
    psf_amplitude : float
        Amplitude of the spot.
    psf_background : float
        Background minimum value of the voxel.

    """
    # compute values
    image_min, image_max = image_spot.min(), image_spot.max()
    psf_amplitude = image_max - image_min
    psf_background = image_min

    return psf_amplitude, psf_background


# ### Spot modelization: fitting ###

def _objective_function(nb_dimension, voxel_size_z, voxel_size_yx, psf_z,
                        psf_yx, psf_amplitude=None):
    """Design the objective function used to fit the gaussian function.

    Parameters
    ----------
    nb_dimension : int
        Number of dimensions to consider (2 or 3).
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d gaussian function.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer. If None, we consider a 2-d gaussian function.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    psf_amplitude : int or float
        Amplitude of the spot.

    Returns
    -------
    f : func
        A 3-d or 2-d gaussian function with some parameters fixed.

    """
    # define objective gaussian function
    if nb_dimension == 3:
        f = _objective_function_3d(voxel_size_z=voxel_size_z,
                                   voxel_size_yx=voxel_size_yx,
                                   psf_z=psf_z,
                                   psf_yx=psf_yx,
                                   psf_amplitude=psf_amplitude)
    else:
        f = _objective_function_2d(voxel_size_yx=voxel_size_yx,
                                   psf_yx=psf_yx,
                                   psf_amplitude=psf_amplitude)

    return f


def _objective_function_3d(voxel_size_z, voxel_size_yx, psf_z, psf_yx,
                           psf_amplitude=None):
    """Design the objective function used to fit the gaussian function.

    Parameters
    ----------
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    psf_amplitude : int or float
        Amplitude of the spot.

    Returns
    -------
    f : func
        A 3-d gaussian function with some parameters fixed.

    """
    # sigma is known, we fit mu, amplitude and background
    if (psf_z is not None
            and psf_yx is not None
            and psf_amplitude is None):
        def f(grid, mu_z, mu_y, mu_x, psf_amplitude, psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=psf_z,
                                 sigma_yx=psf_yx,
                                 voxel_size_z=voxel_size_z,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # amplitude is known, we fit sigma, mu and background
    elif (psf_amplitude is not None
          and psf_z is None
          and psf_yx is None):
        def f(grid, mu_z, mu_y, mu_x, psf_z, psf_yx, psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=psf_z,
                                 sigma_yx=psf_yx,
                                 voxel_size_z=voxel_size_z,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # amplitude and sigma are known, we fit mu and background
    elif (psf_amplitude is not None
          and psf_z is not None
          and psf_yx is not None):
        def f(grid, mu_z, mu_y, mu_x, psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=psf_z,
                                 sigma_yx=psf_yx,
                                 voxel_size_z=voxel_size_z,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # we fit mu, sigma, amplitude and background
    elif (psf_amplitude is None
          and psf_z is None
          and psf_yx is None):
        def f(grid, mu_z, mu_y, mu_x, psf_z, psf_yx, psf_amplitude,
              psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=psf_z,
                                 sigma_yx=psf_yx,
                                 voxel_size_z=voxel_size_z,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    else:
        raise ValueError("Parameters 'psf_z' and 'psf_yx' should be "
                         "fixed or optimized together.")

    return f


# TODO add equations in the docstring
def gaussian_3d(grid, mu_z, mu_y, mu_x, sigma_z, sigma_yx, voxel_size_z,
                voxel_size_yx, psf_amplitude, psf_background,
                precomputed=None):
    """Compute the gaussian function over the grid representing a volume V
    with shape (V_z, V_y, V_x).

    Parameters
    ----------
    grid : np.ndarray, np.float
        Grid data to compute the gaussian function for different voxel within
        a volume V. In nanometer, with shape (3, V_z * V_y * V_x).
    mu_z : float
        Estimated mean of the gaussian signal along z axis, in nanometer.
    mu_y : float
        Estimated mean of the gaussian signal along y axis, in nanometer.
    mu_x : float
        Estimated mean of the gaussian signal along x axis, in nanometer.
    sigma_z : int or float
        Standard deviation of the gaussian along the z axis, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    voxel_size_z : int or float
        Height of a voxel, in nanometer.
    voxel_size_yx : int or float
        size of a voxel, in nanometer.
    psf_amplitude : float
        Estimated pixel intensity of a spot.
    psf_background : float
        Estimated pixel intensity of the background.
    precomputed : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    values : np.ndarray, np.float
        Value of each voxel within the volume V according to the 3-d gaussian
        parameters. Shape (V_z * V_y * V_x,).

    """
    # get grid data to design a volume V
    meshgrid_z = grid[0]
    meshgrid_y = grid[1]
    meshgrid_x = grid[2]

    # use precomputed tables
    if precomputed is not None:
        # get tables
        table_erf_z = precomputed[0]
        table_erf_y = precomputed[1]
        table_erf_x = precomputed[2]

        # get indices for the tables
        i_z = np.abs(meshgrid_z - mu_z).astype(np.int64)
        i_y = np.abs(meshgrid_y - mu_y).astype(np.int64)
        i_x = np.abs(meshgrid_x - mu_x).astype(np.int64)

        # get precomputed values
        voxel_integral_z = table_erf_z[i_z, 1]
        voxel_integral_y = table_erf_y[i_y, 1]
        voxel_integral_x = table_erf_x[i_x, 1]

    # compute erf value
    else:
        # get voxel coordinates
        meshgrid_z_minus = meshgrid_z - voxel_size_z / 2
        meshgrid_z_plus = meshgrid_z + voxel_size_z / 2
        meshgrid_y_minus = meshgrid_y - voxel_size_yx / 2
        meshgrid_y_plus = meshgrid_y + voxel_size_yx / 2
        meshgrid_x_minus = meshgrid_x - voxel_size_yx / 2
        meshgrid_x_plus = meshgrid_x + voxel_size_yx / 2

        # compute gaussian function for each voxel (i, j, k) of volume V
        voxel_integral_z = _rescaled_erf(low=meshgrid_z_minus,
                                         high=meshgrid_z_plus,
                                         mu=mu_z,
                                         sigma=sigma_z)
        voxel_integral_y = _rescaled_erf(low=meshgrid_y_minus,
                                         high=meshgrid_y_plus,
                                         mu=mu_y,
                                         sigma=sigma_yx)
        voxel_integral_x = _rescaled_erf(low=meshgrid_x_minus,
                                         high=meshgrid_x_plus,
                                         mu=mu_x,
                                         sigma=sigma_yx)

    # compute 3-d gaussian values
    factor = psf_amplitude / (voxel_size_yx ** 2 * voxel_size_z)
    voxel_integral = voxel_integral_z * voxel_integral_y * voxel_integral_x
    values = psf_background + factor * voxel_integral

    return values


def _objective_function_2d(voxel_size_yx, psf_yx, psf_amplitude=None):
    """Design the objective function used to fit a 2-d gaussian function.

    Parameters
    ----------
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    psf_amplitude : float
        Amplitude of the spot.

    Returns
    -------
    f : func
        A 2-d gaussian function with some parameters fixed.

    """
    # sigma is known, we fit mu, amplitude and background
    if psf_yx is not None and psf_amplitude is None:
        def f(grid, mu_y, mu_x, psf_amplitude, psf_background):
            values = gaussian_2d(grid=grid,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_yx=psf_yx,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # amplitude is known, we fit sigma, mu and background
    elif psf_amplitude is not None and psf_yx is None:
        def f(grid, mu_y, mu_x, psf_yx, psf_background):
            values = gaussian_2d(grid=grid,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_yx=psf_yx,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # amplitude and sigma are known, we fit mu and background
    elif psf_amplitude is not None and psf_yx is not None:
        def f(grid, mu_y, mu_x, psf_background):
            values = gaussian_2d(grid=grid,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_yx=psf_yx,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # we fit mu, sigma, amplitude and background
    else:
        def f(grid, mu_y, mu_x, psf_yx, psf_amplitude, psf_background):
            values = gaussian_2d(grid=grid,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_yx=psf_yx,
                                 voxel_size_yx=voxel_size_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    return f


# TODO add equations in the docstring
def gaussian_2d(grid, mu_y, mu_x, sigma_yx, voxel_size_yx, psf_amplitude,
                psf_background, precomputed=None):
    """Compute the gaussian function over the grid representing a surface S
    with shape (S_y, S_x).

    Parameters
    ----------
    grid : np.ndarray, np.float
        Grid data to compute the gaussian function for different voxel within
        a surface S. In nanometer, with shape (2, S_y * S_x).
    mu_y : float
        Estimated mean of the gaussian signal along y axis, in nanometer.
    mu_x : float
        Estimated mean of the gaussian signal along x axis, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    voxel_size_yx : int or float
        size of a voxel, in nanometer.
    psf_amplitude : float
        Estimated pixel intensity of a spot.
    psf_background : float
        Estimated pixel intensity of the background.
    precomputed : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    values : np.ndarray, np.float
        Value of each voxel within the surface S according to the 2-d gaussian
        parameters. Shape (S_y * S_x,).

    """
    # get grid data to design a surface S
    meshgrid_y = grid[0]
    meshgrid_x = grid[1]

    # use precomputed tables
    if precomputed is not None:
        # get tables
        table_erf_y = precomputed[0]
        table_erf_x = precomputed[1]

        # get indices for the tables
        i_y = np.abs(meshgrid_y - mu_y).astype(np.int64)
        i_x = np.abs(meshgrid_x - mu_x).astype(np.int64)

        # get precomputed values
        voxel_integral_y = table_erf_y[i_y, 1]
        voxel_integral_x = table_erf_x[i_x, 1]

    # compute erf value
    else:
        # get voxel coordinates
        meshgrid_y_minus = meshgrid_y - voxel_size_yx / 2
        meshgrid_y_plus = meshgrid_y + voxel_size_yx / 2
        meshgrid_x_minus = meshgrid_x - voxel_size_yx / 2
        meshgrid_x_plus = meshgrid_x + voxel_size_yx / 2

        # compute gaussian function for each voxel (i, j) of surface S
        voxel_integral_y = _rescaled_erf(low=meshgrid_y_minus,
                                         high=meshgrid_y_plus,
                                         mu=mu_y,
                                         sigma=sigma_yx)
        voxel_integral_x = _rescaled_erf(low=meshgrid_x_minus,
                                         high=meshgrid_x_plus,
                                         mu=mu_x,
                                         sigma=sigma_yx)

    # compute 2-d gaussian values
    factor = psf_amplitude / (voxel_size_yx ** 2)
    voxel_integral = voxel_integral_y * voxel_integral_x
    values = psf_background + factor * voxel_integral

    return values


def _rescaled_erf(low, high, mu, sigma):
    """Rescaled the Error function along a specific axis.

    # TODO add equations

    Parameters
    ----------
    low : np.ndarray, np.float
        Lower bound of the voxel along a specific axis.
    high : np.ndarray, np.float
        Upper bound of the voxel along a specific axis.
    mu : int or float
        Estimated mean of the gaussian signal along a specific axis.
    sigma : int or float
        Estimated standard deviation of the gaussian signal along a specific
        axis.

    Returns
    -------
    rescaled_erf : np.ndarray, np.float
        Rescaled erf along a specific axis.

    """
    # compute erf and normalize it
    low_ = (low - mu) / (np.sqrt(2) * sigma)
    high_ = (high - mu) / (np.sqrt(2) * sigma)
    rescaled_erf = sigma * np.sqrt(np.pi / 2) * (erf(high_) - erf(low_))

    return rescaled_erf


def _fit_gaussian(f, grid, image_spot, p0, lower_bound=None, upper_bound=None):
    """Fit a gaussian function to a 3-d or 2-d image.

    # TODO add equations and algorithm

    Parameters
    ----------
    f : func
        A 3-d or 2-d gaussian function with some parameters fixed.
    grid : np.ndarray, np.float
        Grid data to compute the gaussian function for different voxel within
        a volume V or surface S. In nanometer, with shape (3, V_z * V_y * V_x),
        or (2, S_y * S_x).
    image_spot : np.ndarray, np.uint
        A 3-d or 2-d image with detected spot and shape (z, y, x) or (y, x).
    p0 : List
        List of parameters to estimate.
    lower_bound : List
        List of lower bound values for the different parameters.
    upper_bound : List
        List of upper bound values for the different parameters.

    Returns
    -------
    popt : np.ndarray
        Fitted parameters.
    pcov : np.ndarray
        Estimated covariance of 'popt'.

    """
    # TODO check that we do not fit a 2-d gaussian function to a 3-d image or
    #  the opposite

    # compute lower bound and upper bound
    if lower_bound is None:
        lower_bound = [-np.inf for _ in p0]
    if upper_bound is None:
        upper_bound = [np.inf for _ in p0]
    bounds = (lower_bound, upper_bound)

    # Apply non-linear least squares to fit a gaussian function to a 3-d image
    y = np.reshape(image_spot, (image_spot.size,)).astype(np.float32)
    popt, pcov = curve_fit(f=f, xdata=grid, ydata=y, p0=p0, bounds=bounds)

    return popt, pcov


def precompute_erf(voxel_size_z=None, voxel_size_yx=100, sigma_z=None,
                   sigma_yx=200, max_grid=200):
    """Precompute different values for the erf with a nanometer resolution.

    Parameters
    ----------
    voxel_size_z : float or int or None
        Height of a voxel, in nanometer. If None, we consider a 2-d erf.
    voxel_size_yx : float or int
        size of a voxel, in nanometer.
    sigma_z : float or int or None
        Standard deviation of the gaussian along the z axis, in nanometer. If
        None, we consider a 2-d erf.
    sigma_yx : float or int
        Standard deviation of the gaussian along the yx axis, in nanometer.
    max_grid : int
        Maximum size of the grid on which we precompute the erf, in pixel.

    Returns
    -------
    table_erf : Tuple[np.ndarray]
        Tuple with tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension. First column is the coordinate
        along the table dimension. Second column is the precomputed erf value.

    """
    # check parameters
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          sigma_z=(int, float, type(None)),
                          sigma_yx=(int, float),
                          max_grid=int)

    # build a grid with a spatial resolution of 1 nm and a size of
    # max_grid * resolution nm
    max_size_yx = np.ceil(max_grid * voxel_size_yx).astype(np.int64)
    yy = np.array([i for i in range(0, max_size_yx)])
    xx = np.array([i for i in range(0, max_size_yx)])
    mu_y, mu_x = 0, 0

    # compute erf values for this grid
    erf_y = _rescaled_erf(low=yy - voxel_size_yx / 2,
                          high=yy + voxel_size_yx / 2,
                          mu=mu_y,
                          sigma=sigma_yx)
    erf_x = _rescaled_erf(low=xx - voxel_size_yx / 2,
                          high=xx + voxel_size_yx / 2,
                          mu=mu_x,
                          sigma=sigma_yx)

    table_erf_y = np.array([yy, erf_y]).T
    table_erf_x = np.array([xx, erf_x]).T

    # precompute erf along z axis if needed
    if voxel_size_z is None or sigma_z is None:
        return table_erf_y, table_erf_x

    else:
        max_size_z = np.ceil(max_grid * voxel_size_z).astype(np.int64)
        zz = np.array([i for i in range(0, max_size_z)])
        mu_z = 0
        erf_z = _rescaled_erf(low=zz - voxel_size_z / 2,
                              high=zz + voxel_size_z / 2,
                              mu=mu_z,
                              sigma=sigma_z)
        table_erf_z = np.array([zz, erf_z]).T
        return table_erf_z, table_erf_y, table_erf_x


# ### Subpixel fitting ###

def fit_subpixel(image, spots, voxel_size_z=None, voxel_size_yx=100,
                 psf_z=None, psf_yx=200):
    """Fit gaussian signal on every spot to find a subpixel coordinates.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots detected, with shape (nb_spots, 3) or
        (nb_spots, 2). One coordinate per dimension (zyx or yx coordinates).
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, image is
        considered in 2-d.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan, in
        nanometer. If None, image is considered in 2-d.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan, in
        nanometer.

    Returns
    -------
    spots_subpixel : np.ndarray, np.float64
        Coordinate of the spots detected, with shape (nb_spots, 3) or
        (nb_spots, 2). One coordinate per dimension (zyx or yx coordinates).

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(spots, ndim=2, dtype=np.int64)
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))

    # check number of dimensions
    ndim = image.ndim
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and psf_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'psf_z' parameter is missing.".format(ndim))
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if ndim == 2:
        voxel_size_z, psf_z = None, None

    # compute radius
    radius = stack.get_radius(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    # loop over every spot
    spots_subpixel = []
    for coord in spots[:, :ndim]:

        # fit subpixel coordinates
        if ndim == 3:
            subpixel_coord = _fit_subpixel_3d(
                image, coord, radius,
                voxel_size_z, voxel_size_yx, psf_z, psf_yx)

        else:
            subpixel_coord = _fit_subpixel_2d(
                image, coord, radius, voxel_size_yx, psf_yx)

        spots_subpixel.append(subpixel_coord)

    # format results
    spots_subpixel = np.stack(spots_subpixel)

    return spots_subpixel


def _fit_subpixel_3d(image, coord, radius, voxel_size_z, voxel_size_yx, psf_z,
                     psf_yx):
    """Fit a gaussian in a 3-d image.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x).
    coord : np.ndarray, np.int64
        Coordinate of the spot detected, with shape (3,). One coordinate per
        dimension (zyx coordinates).
    radius : Tuple[float]
        Radius in pixels of the detected spots, one element per dimension.
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    new_coord : List[float]
        Coordinates of the spot centroid with a subpixel accuracy (one element
        per dimension).

    """
    # extract spot image
    image_spot, bbox_low = _get_spot_volume(
        image, coord[0], coord[1], coord[2], radius[0], radius[1])

    # fit gaussian
    try:
        parameters = modelize_spot(image_spot,
                                   voxel_size_z=voxel_size_z,
                                   voxel_size_yx=voxel_size_yx,
                                   psf_z=psf_z, psf_yx=psf_yx,
                                   return_coord=True)

        # format coordinates and ensure it is fitted within the spot image
        z_max, y_max, x_max = image_spot.shape
        coord_z = parameters[0] / voxel_size_z
        if coord_z < 0 or coord_z > z_max:
            coord_z = coord[0]
        else:
            coord_z += bbox_low[0]
        coord_y = parameters[1] / voxel_size_yx
        if coord_y < 0 or coord_y > y_max:
            coord_y = coord[1]
        else:
            coord_y += bbox_low[1]
        coord_x = parameters[2] / voxel_size_yx
        if coord_x < 0 or coord_x > x_max:
            coord_x = coord[2]
        else:
            coord_x += bbox_low[2]
        new_coord = [coord_z, coord_y, coord_x]

    # if a spot is ill-conditioned, we simply keep its original coordinates
    except RuntimeError:
        new_coord = list(coord)

    return new_coord


def _fit_subpixel_2d(image, coord, radius, voxel_size_yx, psf_yx):
    """Fit a gaussian in a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (y, x).
    coord : np.ndarray, np.int64
        Coordinate of the spot detected, with shape (2,). One coordinate per
        dimension (yx coordinates).
    radius : Tuple[float]
        Radius in pixels of the detected spots, one element per dimension.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    new_coord : List[float]
        Coordinates of the spot centroid with a subpixel accuracy (one element
        per dimension).

    """
    # extract spot image
    image_spot, bbox_low = _get_spot_surface(
        image, coord[0], coord[1], radius[0])

    # fit gaussian
    try:
        parameters = modelize_spot(image_spot,
                                   voxel_size_z=None,
                                   voxel_size_yx=voxel_size_yx,
                                   psf_z=None, psf_yx=psf_yx,
                                   return_coord=True)

        # format coordinates and ensure it is fitted within the spot image
        y_max, x_max = image_spot.shape
        coord_y = parameters[0] / voxel_size_yx
        if coord_y < 0 or coord_y > y_max:
            coord_y = coord[0]
        else:
            coord_y += bbox_low[0]
        coord_x = parameters[1] / voxel_size_yx
        if coord_x < 0 or coord_x > x_max:
            coord_x = coord[1]
        else:
            coord_x += bbox_low[1]
        new_coord = [coord_y, coord_x]

    # if a spot is ill-conditioned, we simply keep its original coordinates
    except RuntimeError:
        new_coord = list(coord)

    return new_coord
