# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to fit gaussian functions to the detected RNA spots, especially in
clustered regions.
"""

import warnings

import numpy as np

import bigfish.stack as stack
from .utils import get_sigma, get_radius

from scipy.special import erf
from scipy.optimize import curve_fit
from skimage.measure import regionprops
from skimage.measure import label


# TODO add parameter to detect more or less clusters
# TODO add parameter to fit more or less spots in a cluster

# ### Main function ###

def decompose_cluster(image, spots, voxel_size_z=None, voxel_size_yx=100,
                      psf_z=None, psf_yx=200):
    """Detect potential regions with clustered spots and fit as many reference
    spots as possible in these regions.

    1) We estimate image background with a large gaussian filter. We then
    remove the background from the original image to denoise it.
    2) We build a reference spot by aggregating predetected spots.
    3) We fit a gaussian function on the reference spots.
    4) We detect potential clustered regions to decompose.
    5) We simulate as many gaussians as possible in the candidate regions.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2)
        for 3-d or 2-d images respectively.
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    psf_z : int or float
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.

    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the spots detected, with shape (nb_spots, 3) or
        (nb_spots, 2). One coordinate per dimension (zyx or yx coordinates).
    clusters : np.ndarray, np.int64
        Array with shape (nb_cluster, 7) or (nb_cluster, 6). One coordinate
        per dimension for the cluster centroid (zyx or yx coordinates), the
        number of RNAs detected in the cluster, the area of the cluster
        region, its average intensity value and its index.
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

    # compute expected standard deviation of the spots
    sigma = get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)
    large_sigma = tuple([sigma_ * 5 for sigma_ in sigma])

    # denoise the image
    image_denoised = stack.remove_background_gaussian(
        image,
        sigma=large_sigma)

    # build a reference median spot
    reference_spot = build_reference_spot(
        image_denoised,
        spots,
        voxel_size_z, voxel_size_yx, psf_z, psf_yx)
    threshold_cluster = int(reference_spot.max())

    # case where no spot were detected
    if spots.size == 0:
        spots_out_cluster = np.array([], dtype=np.int64).reshape((0, ndim))
        spots_in_cluster = np.array([], dtype=np.int64).reshape((0, ndim + 1))
        cluster = np.array([], dtype=np.int64).reshape((0, ndim + 4))
        return spots_out_cluster, spots_in_cluster, cluster, reference_spot

    # case with an empty frame as reference spot
    if reference_spot.sum() == 0:
        spots_in_cluster = np.array([], dtype=np.int64).reshape((0, ndim + 1))
        cluster = np.array([], dtype=np.int64).reshape((0, ndim + 4))
        return spots, spots_in_cluster, cluster, reference_spot

    # fit a gaussian function on the reference spot to be able to simulate it
    parameters_fitted = modelize_spot(
        reference_spot, voxel_size_z, voxel_size_yx, psf_z, psf_yx)
    if ndim == 3:
        sigma_z, sigma_yx, amplitude, background = parameters_fitted
    else:
        sigma_z = None
        sigma_yx, amplitude, background = parameters_fitted

    # use connected components to detect potential clusters
    cluster_regions, spots_out_cluster, cluster_size = get_clustered_region(
        image_denoised, spots, threshold_cluster)

    # case where no cluster where detected
    if cluster_regions.size == 0:
        spots_in_cluster = np.array([], dtype=np.int64).reshape((0, ndim + 1))
        cluster = np.array([], dtype=np.int64).reshape((0, ndim + 4))
        return spots, spots_in_cluster, cluster, reference_spot

    # precompute gaussian function values
    max_grid = max(200, cluster_size + 1)
    precomputed_gaussian = precompute_erf(
        voxel_size_z, voxel_size_yx, sigma_z, sigma_yx, max_grid=max_grid)

    # fit gaussian mixtures in the cluster regions
    spots_in_cluster, clusters = fit_gaussian_mixture(
        image=image_denoised,
        cluster_regions=cluster_regions,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx,
        sigma_z=sigma_z,
        sigma_yx=sigma_yx,
        amplitude=amplitude,
        background=background,
        precomputed_gaussian=precomputed_gaussian)

    # normally the number of detected spots should increase
    if len(spots_out_cluster) + len(spots_in_cluster) < len(spots):
        warnings.warn("Problem occurs during the decomposition of clusters. "
                      "Less spots are detected after the decomposition than "
                      "before.",
                      UserWarning)

    # merge outside and inside spots
    spots = np.concatenate((spots_out_cluster, spots_in_cluster[:, :ndim]),
                           axis=0)

    return spots, clusters, reference_spot


# ### Reference spot ###

def build_reference_spot(image, spots, voxel_size_z=None, voxel_size_yx=100,
                         psf_z=None, psf_yx=200, method="median"):
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
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    psf_z : int or float
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.
    method : str
        Method use to compute the reference spot (a 'mean' or 'median' spot).

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
                          method=str)
    if method not in ['mean', 'median']:
        raise ValueError("'{0}' is not a valid value for parameter 'method'. "
                         "Use 'mean' or 'median' instead.".format(method))

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
    radius = get_radius(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    # build reference spot
    if image.ndim == 3:
        reference_spot = _build_reference_spot_3d(image, spots, radius,
                                                  method="median")
    else:
        reference_spot = _build_reference_spot_2d(image, spots, radius,
                                                  method="median")

    return reference_spot


def _build_reference_spot_3d(image, spots, radius, method="median"):
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
    method : str
        Method use to compute the reference spot (a 'mean' or 'median' spot).

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
        image_spot = _get_spot_volume(image, spot_z, spot_y, spot_x,
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
    if method == "mean":
        reference_spot = np.mean(l_reference_spot, axis=0)
    else:
        reference_spot = np.median(l_reference_spot, axis=0)
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

    return image_spot


def _build_reference_spot_2d(image, spots, radius, method="median"):
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
    method : str
        Method use to compute the reference spot (a 'mean' or 'median' spot).

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
        image_spot = _get_spot_surface(image, spot_y, spot_x, radius_yx)

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
    if method == "mean":
        reference_spot = np.mean(l_reference_spot, axis=0)
    else:
        reference_spot = np.median(l_reference_spot, axis=0)
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

    """
    # get boundaries of the surface surrounding the spot
    y_spot_min = max(0, int(spot_y - radius_yx))
    y_spot_max = min(image.shape[0], int(spot_y + radius_yx))
    x_spot_min = max(0, int(spot_x - radius_yx))
    x_spot_max = min(image.shape[1], int(spot_x + radius_yx))

    # get the surface of the spot
    image_spot = image[y_spot_min:y_spot_max + 1,
                       x_spot_min:x_spot_max + 1]

    return image_spot


# ### Spot modelization ###

def modelize_spot(reference_spot, voxel_size_z=None, voxel_size_yx=100,
                  psf_z=None, psf_yx=200):
    """Fit a gaussian function on the reference spot.

    Parameters
    ----------
    reference_spot : np.ndarray
        A 3-d or 2-d image with detected spot and shape (z, y, x) or (y, x).
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    psf_z : int or float
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.

    Returns
    -------
    parameters_fitted : Tuple[float]
        - sigma_z : float
            Standard deviation of the spot along the z axis, in pixel.
            Available only for a 3-d modelization.
        - sigma_yx : float
            Standard deviation of the spot along the yx axis, in pixel.
        - amplitude : float
            Amplitude of the spot.
        - background : float
            Background minimum value of the image.

    """
    # check parameters
    stack.check_array(reference_spot,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))

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
    grid, centroid_coord = _initialize_grid(
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
    else:
        # parameters to fit: mu_y, mu_x, sigma_yx, amplitude and background
        centroid_y, centroid_x = centroid_coord
        p0 = [centroid_y, centroid_x, psf_yx, amplitude, background]

    # fit a gaussian function on this reference spot
    popt, pcov = _fit_gaussian(f, grid, reference_spot, p0)

    # get optimized parameters to modelize the reference spot as a gaussian
    if ndim == 3:
        sigma_z = popt[3]
        sigma_yx = popt[4]
        amplitude = popt[5]
        background = popt[6]

        return sigma_z, sigma_yx, amplitude, background

    else:
        sigma_yx = popt[2]
        amplitude = popt[3]
        background = popt[4]

        return sigma_yx, amplitude, background


# ### Spot modelization: initialization ###

def _initialize_grid(image_spot, voxel_size_z, voxel_size_yx,
                     return_centroid=False):
    """Build a grid in nanometer to compute gaussian function values over a
    full volume or surface.

    Parameters
    ----------
    image_spot : np.ndarray
        An image with detected spot and shape (z, y, x) or (y, x).
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
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
    zz *= voxel_size_z
    yy *= voxel_size_yx
    xx *= voxel_size_yx

    # format result
    grid = np.zeros((3, nb_pixels), dtype=np.float32)
    grid[0] = np.reshape(zz, (1, nb_pixels)).astype(np.float32)
    grid[1] = np.reshape(yy, (1, nb_pixels)).astype(np.float32)
    grid[2] = np.reshape(xx, (1, nb_pixels)).astype(np.float32)

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
    yy *= voxel_size_yx
    xx *= voxel_size_yx

    # format result
    grid = np.zeros((2, nb_pixels), dtype=np.float32)
    grid[0] = np.reshape(yy, (1, nb_pixels)).astype(np.float32)
    grid[1] = np.reshape(xx, (1, nb_pixels)).astype(np.float32)

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
            values = _gaussian_3d(grid=grid,
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
            values = _gaussian_3d(grid=grid,
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
            values = _gaussian_3d(grid=grid,
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
            values = _gaussian_3d(grid=grid,
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


def _gaussian_3d(grid, mu_z, mu_y, mu_x, sigma_z, sigma_yx, voxel_size_z,
                 voxel_size_yx, psf_amplitude, psf_background,
                 precomputed=None):
    """Compute the gaussian function over the grid 'xdata' representing a
    volume V with shape (V_z, V_y, V_x).

    # TODO add equations

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
        i_z = np.around(np.abs(meshgrid_z - mu_z) / 5).astype(np.int64)
        i_y = np.around(np.abs(meshgrid_y - mu_y) / 5).astype(np.int64)
        i_x = np.around(np.abs(meshgrid_x - mu_x) / 5).astype(np.int64)

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
            values = _gaussian_2d(grid=grid,
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
            values = _gaussian_2d(grid=grid,
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
            values = _gaussian_2d(grid=grid,
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
            values = _gaussian_2d(grid=grid,
                                  mu_y=mu_y,
                                  mu_x=mu_x,
                                  sigma_yx=psf_yx,
                                  voxel_size_yx=voxel_size_yx,
                                  psf_amplitude=psf_amplitude,
                                  psf_background=psf_background)
            return values

    return f


def _gaussian_2d(grid, mu_y, mu_x, sigma_yx, voxel_size_yx, psf_amplitude,
                 psf_background, precomputed=None):
    """Compute the gaussian function over the grid 'xdata' representing a
    surface S with shape (S_y, S_x).

    # TODO add equations

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
        i_y = np.around(np.abs(meshgrid_y - mu_y) / 5).astype(np.int64)
        i_x = np.around(np.abs(meshgrid_x - mu_x) / 5).astype(np.int64)

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
    """Precompute different values for the erf with a resolution of 5 nm.

    Parameters
    ----------
    voxel_size_z : float, int
        Height of a voxel, in nanometer.
    voxel_size_yx : float, int
        size of a voxel, in nanometer.
    sigma_z : float or int
        Standard deviation of the gaussian along the z axis, in nanometer.
    sigma_yx : float or int
        Standard deviation of the gaussian along the yx axis, in nanometer.
    max_grid : int
        Maximum size of the grid on which we precompute the erf, in pixel.

    Returns
    -------
    table_erf : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    """
    # check parameters
    stack.check_parameter(voxel_size_z=(int, type(None)),
                          voxel_size_yx=int,
                          sigma_z=(float, int, type(None)),
                          sigma_yx=(float, int),
                          max_grid=int)

    # build a grid with a spatial resolution of 5 nm and a size of
    # max_grid * resolution nm
    yy = np.array([i for i in range(0, max_grid * voxel_size_yx, 5)])
    xx = np.array([i for i in range(0, max_grid * voxel_size_yx, 5)])
    mu_y, mu_x = 0, 0

    # compute erf values for this grid
    erf_y = _rescaled_erf(low=yy - voxel_size_yx/2,
                          high=yy + voxel_size_yx/2,
                          mu=mu_y,
                          sigma=sigma_yx)
    erf_x = _rescaled_erf(low=xx - voxel_size_yx/2,
                          high=xx + voxel_size_yx/2,
                          mu=mu_x,
                          sigma=sigma_yx)

    table_erf_y = np.array([yy, erf_y]).T
    table_erf_x = np.array([xx, erf_x]).T

    # precompute erf along z axis if needed
    if voxel_size_z is None or sigma_z is None:
        return table_erf_y, table_erf_x

    else:
        zz = np.array([i for i in range(0, max_grid * voxel_size_z, 5)])
        mu_z = 0
        erf_z = _rescaled_erf(low=zz - voxel_size_z / 2,
                              high=zz + voxel_size_z / 2,
                              mu=mu_z,
                              sigma=sigma_z)
        table_erf_z = np.array([zz, erf_z]).T
        return table_erf_z, table_erf_y, table_erf_x


# ### Clustered regions ###

def get_clustered_region(image, spots, threshold):
    """Detect and filter potential clustered regions.

    A candidate region follows two criteria:
        - at least 2 connected pixels above a specific threshold.
        - among the 50% brightest regions.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    threshold : int or float
        A threshold to detect peaks.

    Returns
    -------
    cluster_regions : np.ndarray
        Array with filtered skimage.measure._regionprops._RegionProperties.
    spots_out_region : np.ndarray, np.int64
        Coordinate of the spots detected out of cluster, with shape
        (nb_spots, 3) or (nb_spots, 2). One coordinate per dimension (zyx or
        yx coordinates).
    max_size : int
        Maximum size of the regions.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(spots, ndim=2, dtype=np.int64)
    stack.check_parameter(threshold=(int, float))

    # check number of dimensions
    if image.ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(image.ndim, spots.shape[1]))

    # get connected regions
    connected_regions = _get_connected_region(image, threshold)

    # filter connected regions
    (cluster_regions, spots_out_region, max_size) = _filter_connected_region(
        image, connected_regions, spots)

    return cluster_regions, spots_out_region, max_size


def _get_connected_region(image, threshold):
    """Find connected regions above a fixed threshold.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    threshold : int or float
        A threshold to detect peaks.

    Returns
    -------
    cc : np.ndarray, np.int64
        Image labelled with shape (z, y, x) or (y, x).

    """
    # Compute binary mask of the filtered image
    mask = image > threshold

    # find connected components
    cc = label(mask)

    return cc


def _filter_connected_region(image, connected_component, spots):
    """Filter clustered regions (defined as connected component regions).

    A candidate region follows two criteria:
        - at least 2 connected pixels above a specific threshold.
        - among the 50% brightest regions.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    connected_component : np.ndarray, np.int64
        Image labelled with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2).

    Returns
    -------
    regions_filtered : np.ndarray
        Array with filtered skimage.measure._regionprops._RegionProperties.
    spots_out_region : np.ndarray, np.int64
        Coordinate of the spots outside the regions with shape (nb_spots, 3)
        or (nb_spots, 2).
    max_region_size : int
        Maximum size of the regions.

    """
    # get properties of the different connected regions
    regions = regionprops(connected_component, intensity_image=image)

    # get different features of the regions
    area = []
    intensity = []
    bbox = []
    for i, region in enumerate(regions):
        area.append(region.area)
        intensity.append(region.mean_intensity)
        bbox.append(region.bbox)
    regions = np.array(regions)
    area = np.array(area)
    intensity = np.array(intensity)
    bbox = np.array(bbox)

    # keep regions with a minimum size
    big_area = area >= 2
    regions = regions[big_area]
    intensity = intensity[big_area]
    bbox = bbox[big_area]

    # case where no region big enough were detected
    if regions.size == 0:
        regions_filtered = np.array([])
        return regions_filtered, spots, 0

    # keep the brightest regions
    high_intensity = intensity >= np.median(intensity)
    regions_filtered = regions[high_intensity]
    bbox = bbox[high_intensity]

    # case where no region bright enough were detected
    if regions_filtered.size == 0:
        return regions_filtered, spots, 0

    spots_out_region, max_region_size = _filter_spot_out_candidate_regions(
        bbox, spots, nb_dim=image.ndim)

    return regions_filtered, spots_out_region, max_region_size


def _filter_spot_out_candidate_regions(candidate_bbox, spots, nb_dim):
    """Filter spots out of the potential clustered regions.

    Parameters
    ----------
    candidate_bbox : List[Tuple]
        List of Tuples with the bounding box coordinates.
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    nb_dim : int
        Number of dimensions to consider (2 or 3).

    Returns
    -------
    spots_out_region : np.ndarray, np.int64
        Coordinate of the spots outside the regions with shape (nb_spots, 3)
        or (nb_spots, 2).
    max_region_size : int
        Maximum size of the regions.

    """
    # initialization
    mask_spots_out = np.ones(spots[:, 0].shape, dtype=bool)
    max_region_size = 0

    # get detected spots outside 3-d regions
    if nb_dim == 3:
        for box in candidate_bbox:
            (min_z, min_y, min_x, max_z, max_y, max_x) = box

            # get the size of the biggest region
            size_z = max_z - min_z
            size_y = max_y - min_y
            size_x = max_x - min_x
            max_region_size = max(max_region_size, size_z, size_y, size_x)

            # get coordinates of spots inside the region
            mask_spots_in = spots[:, 0] < max_z
            mask_spots_in = (mask_spots_in & (spots[:, 1] < max_y))
            mask_spots_in = (mask_spots_in & (spots[:, 2] < max_x))
            mask_spots_in = (mask_spots_in & (min_z <= spots[:, 0]))
            mask_spots_in = (mask_spots_in & (min_y <= spots[:, 1]))
            mask_spots_in = (mask_spots_in & (min_x <= spots[:, 2]))
            mask_spots_out = mask_spots_out & (~mask_spots_in)

    # get detected spots outside 2-d regions
    else:
        for box in candidate_bbox:
            (min_y, min_x, max_y, max_x) = box

            # get the size of the biggest region
            size_y = max_y - min_y
            size_x = max_x - min_x
            max_region_size = max(max_region_size, size_y, size_x)

            # get coordinates of spots inside the region
            mask_spots_in = spots[:, 0] < max_y
            mask_spots_in = (mask_spots_in & (spots[:, 1] < max_x))
            mask_spots_in = (mask_spots_in & (min_y <= spots[:, 0]))
            mask_spots_in = (mask_spots_in & (min_x <= spots[:, 1]))
            mask_spots_out = mask_spots_out & (~mask_spots_in)

    # keep apart spots inside a region
    spots_out_region = spots.copy()
    spots_out_region = spots_out_region[mask_spots_out]

    return spots_out_region, int(max_region_size)


# ### Gaussian simulation ###

def fit_gaussian_mixture(image, cluster_regions, voxel_size_z=None,
                         voxel_size_yx=100, sigma_z=None, sigma_yx=200,
                         amplitude=100, background=0,
                         precomputed_gaussian=None):
    """Fit as many gaussians as possible in the candidate clustered regions.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    cluster_regions : np.ndarray
        Array with filtered skimage.measure._regionprops._RegionProperties.
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    sigma_z : int or float
        Standard deviation of the gaussian along the z axis, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    amplitude : float
        Amplitude of the gaussian.
    background : float
        Background minimum value of the image.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    spots_in_cluster : np.ndarray, np.int64
        spots_in_cluster : np.ndarray, np.int64
        Coordinate of the spots detected inside cluster, with shape
        (nb_spots, 4) or (nb_spots, 3). One coordinate per dimension (zyx or
        yx coordinates) plus the index of the cluster.
    clusters : np.ndarray, np.int64
        Array with shape (nb_cluster, 7) or (nb_cluster, 6). One coordinate
        per dimension for the cluster centroid (zyx or yx coordinates), the
        number of RNAs detected in the cluster, the area of the cluster
        region, its average intensity value and its index.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(cluster_regions=np.ndarray,
                          voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          sigma_z=(int, float, type(None)),
                          sigma_yx=(int, float),
                          amplitude=float,
                          background=float)

    # check number of dimensions
    ndim = image.ndim
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and sigma_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'sigma_z' parameter is missing.".format(ndim))
    if ndim == 2:
        voxel_size_z, sigma_z = None, None

    # fit gaussian mixtures in the cluster regions...
    spots_in_cluster = []
    clusters = []

    # ... for 3-d regions...
    if image.ndim == 3:

        for i_cluster, region in enumerate(cluster_regions):
            image_region, best_simulation, pos_gaussian = _gaussian_mixture_3d(
                image,
                region,
                voxel_size_z,
                voxel_size_yx,
                sigma_z,
                sigma_yx,
                amplitude,
                background,
                precomputed_gaussian)

            # get coordinates of spots and clusters in the original image
            box = region.bbox
            (min_z, min_y, min_x, _, _, _) = box
            pos_gaussian = np.array(pos_gaussian, dtype=np.float64)
            pos_gaussian[:, 0] = (pos_gaussian[:, 0] / voxel_size_z) + min_z
            pos_gaussian[:, 1] = (pos_gaussian[:, 1] / voxel_size_yx) + min_y
            pos_gaussian[:, 2] = (pos_gaussian[:, 2] / voxel_size_yx) + min_x
            spots_in_cluster_ = np.zeros((pos_gaussian.shape[0], 4),
                                         dtype=np.int64)
            spots_in_cluster_[:, :3] = pos_gaussian
            spots_in_cluster_[:, 3] = i_cluster
            spots_in_cluster.append(spots_in_cluster_)
            cluster_z, cluster_y, cluster_x = tuple(pos_gaussian[0])
            nb_rna_cluster = pos_gaussian.shape[0]
            cluster_area = region.area
            cluster_intensity = region.mean_intensity
            clusters.append([cluster_z, cluster_y, cluster_x, nb_rna_cluster,
                             cluster_area, cluster_intensity, i_cluster])

    # ... or 2-d regions
    else:

        for i_cluster, region in enumerate(cluster_regions):
            image_region, best_simulation, pos_gaussian = _gaussian_mixture_2d(
                image,
                region,
                voxel_size_yx,
                sigma_yx,
                amplitude,
                background,
                precomputed_gaussian)

            # get coordinates of spots and clusters in the original image
            box = region.bbox
            (min_y, min_x, _, _) = box
            pos_gaussian = np.array(pos_gaussian, dtype=np.float64)
            pos_gaussian[:, 0] = (pos_gaussian[:, 0] / voxel_size_yx) + min_y
            pos_gaussian[:, 1] = (pos_gaussian[:, 1] / voxel_size_yx) + min_x
            spots_in_cluster_ = np.zeros((pos_gaussian.shape[0], 3),
                                         dtype=np.int64)
            spots_in_cluster_[:, :2] = pos_gaussian
            spots_in_cluster_[:, 2] = i_cluster
            spots_in_cluster.append(spots_in_cluster_)
            cluster_y, cluster_x = tuple(pos_gaussian[0])
            nb_rna_cluster = pos_gaussian.shape[0]
            cluster_area = region.area
            cluster_intensity = region.mean_intensity
            clusters.append([cluster_y, cluster_x, nb_rna_cluster,
                             cluster_area, cluster_intensity, i_cluster])

    spots_in_cluster = np.concatenate(spots_in_cluster, axis=0)
    clusters = np.array(clusters, dtype=np.int64)

    return spots_in_cluster, clusters


def _gaussian_mixture_3d(image, region, voxel_size_z, voxel_size_yx, sigma_z,
                         sigma_yx, amplitude, background, precomputed_gaussian,
                         limit_gaussian=1000):
    """Fit as many 3-d gaussians as possible in a potential clustered region.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    region : skimage.measure._regionprops._RegionProperties
        Properties of a clustered region.
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    sigma_z : int or float
        Standard deviation of the gaussian along the z axis, in pixel.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in pixel.
    amplitude : float
        Amplitude of the gaussian.
    background : float
        Background minimum value of the image.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.
    limit_gaussian : int
        Limit number of gaussian to fit into this region.

    Returns
    -------
    image_region : np.ndarray, np.uint
        A 3-d image with detected spots and shape (z, y, x).
    best_simulation : np.ndarray, np.uint
        A 3-d image with simulated spots and shape (z, y, x).
    positions_gaussian : List[List]
        List of positions (as a list [z, y, x]) for the different gaussian
        simulations used in the mixture.

    """
    # get an image of the region
    box = tuple(region.bbox)
    image_region = image[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
    image_region_raw = np.reshape(image_region, image_region.size)

    # build a grid to represent this image
    grid = _initialize_grid_3d(image_region, voxel_size_z, voxel_size_yx)

    # add a gaussian for each local maximum while the RSS decreases
    simulation = np.zeros(image_region_raw.shape, dtype=np.float64)
    residual = image_region_raw - simulation
    ssr = np.sum(residual ** 2)
    diff_ssr = -1
    nb_gaussian = 0
    best_simulation = simulation.copy()
    positions_gaussian = []
    while diff_ssr < 0 or nb_gaussian == limit_gaussian:
        position_gaussian = np.argmax(residual)
        positions_gaussian.append(list(grid[:, position_gaussian]))
        simulation += _gaussian_3d(grid=grid,
                                   mu_z=float(positions_gaussian[-1][0]),
                                   mu_y=float(positions_gaussian[-1][1]),
                                   mu_x=float(positions_gaussian[-1][2]),
                                   sigma_z=sigma_z,
                                   sigma_yx=sigma_yx,
                                   voxel_size_z=voxel_size_z,
                                   voxel_size_yx=voxel_size_yx,
                                   psf_amplitude=amplitude,
                                   psf_background=background,
                                   precomputed=precomputed_gaussian)
        residual = image_region_raw - simulation
        new_ssr = np.sum(residual ** 2)
        diff_ssr = new_ssr - ssr
        ssr = new_ssr
        nb_gaussian += 1
        background = 0

        if diff_ssr < 0:
            best_simulation = simulation.copy()

    if 1 < nb_gaussian < limit_gaussian:
        positions_gaussian.pop(-1)
    elif nb_gaussian == limit_gaussian:
        warnings.warn("Problem occurs during the decomposition of a cluster. "
                      "More than {0} spots seem to be necessary to reproduce "
                      "the clustered region and decomposition was stopped "
                      "early. Set a higher limit or check a potential "
                      "artifact in the image.".format(limit_gaussian),
                      UserWarning)

    best_simulation = np.reshape(best_simulation, image_region.shape)
    best_simulation = best_simulation.astype(image_region_raw.dtype)

    return image_region, best_simulation, positions_gaussian


def _gaussian_mixture_2d(image, region, voxel_size_yx, sigma_yx, amplitude,
                         background, precomputed_gaussian,
                         limit_gaussian=1000):
    """Fit as many 2-d gaussians as possible in a potential clustered region.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 2-d image with detected spot and shape (y, x).
    region : skimage.measure._regionprops._RegionProperties
        Properties of a clustered region.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in pixel.
    amplitude : float
        Amplitude of the gaussian.
    background : float
        Background minimum value of the image.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.
    limit_gaussian : int
        Limit number of gaussian to fit into this region.

    Returns
    -------
    image_region : np.ndarray, np.uint
        A 2-d image with detected spots and shape (y, x).
    best_simulation : np.ndarray, np.uint
        A 2-d image with simulated spots and shape (y, x).
    positions_gaussian : List[List]
        List of positions (as a list [y, x]) for the different gaussian
        simulations used in the mixture.

    """
    # get an image of the region
    box = tuple(region.bbox)
    image_region = image[box[0]:box[2], box[1]:box[3]]
    image_region_raw = np.reshape(image_region, image_region.size)

    # build a grid to represent this image
    grid = _initialize_grid_2d(image_region, voxel_size_yx)

    # add a gaussian for each local maximum while the RSS decreases
    simulation = np.zeros(image_region_raw.shape, dtype=np.float64)
    residual = image_region_raw - simulation
    ssr = np.sum(residual ** 2)
    diff_ssr = -1
    nb_gaussian = 0
    best_simulation = simulation.copy()
    positions_gaussian = []
    while diff_ssr < 0 or nb_gaussian == limit_gaussian:
        position_gaussian = np.argmax(residual)
        positions_gaussian.append(list(grid[:, position_gaussian]))
        simulation += _gaussian_2d(grid=grid,
                                   mu_y=float(positions_gaussian[-1][0]),
                                   mu_x=float(positions_gaussian[-1][1]),
                                   sigma_yx=sigma_yx,
                                   voxel_size_yx=voxel_size_yx,
                                   psf_amplitude=amplitude,
                                   psf_background=background,
                                   precomputed=precomputed_gaussian)
        residual = image_region_raw - simulation
        new_ssr = np.sum(residual ** 2)
        diff_ssr = new_ssr - ssr
        ssr = new_ssr
        nb_gaussian += 1
        background = 0

        if diff_ssr < 0:
            best_simulation = simulation.copy()

    if 1 < nb_gaussian < limit_gaussian:
        positions_gaussian.pop(-1)
    elif nb_gaussian == limit_gaussian:
        warnings.warn("Problem occurs during the decomposition of a cluster. "
                      "More than {0} spots seem to be necessary to reproduce "
                      "the clustered region and decomposition was stopped "
                      "early. Set a higher limit or check a potential "
                      "artifact in the image.".format(limit_gaussian),
                      UserWarning)

    best_simulation = np.reshape(best_simulation, image_region.shape)
    best_simulation = best_simulation.astype(image_region_raw.dtype)

    return image_region, best_simulation, positions_gaussian
