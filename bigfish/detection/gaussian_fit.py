# -*- coding: utf-8 -*-

"""
Functions to fit gaussian functions to the detected RNA spots.
"""

from .detection import get_sigma

import numpy as np

from scipy.special import erf
from scipy.optimize import curve_fit


# TODO complete documentation methods
# TODO add sanity check functions

# ### Gaussian function ###

def _rescaled_erf(low, high, mu, sigma):
    """Rescaled the Error function along a specific axis.

    # TODO add equations

    Parameters
    ----------
    low : np.ndarray, np.float
        Lower bound of the voxel along a specific axis.
    high : np.ndarray, np.float
        Upper bound of the voxel along a specific axis.
    mu : float
        Estimated mean of the gaussian signal along a specific axis.
    sigma : float
        Estimated standard deviation of the gaussian signal along a specific
        axis.

    Returns
    -------
    new_erf : np.ndarray, np.float
        Rescaled erf along a specific axis.

    """
    low_ = (low - mu) / (np.sqrt(2) * sigma)
    high_ = (high - mu) / (np.sqrt(2) * sigma)
    new_erf = sigma * np.sqrt(np.pi / 2) * (erf(high_) - erf(low_))
    return new_erf


def gaussian_3d(grid, mu_z, mu_y, mu_x, sigma_z, sigma_yx, resolution_z,
                resolution_yx, psf_amplitude, psf_background):
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
    sigma_z : float
        Estimated standard deviation of the gaussian signal along z axis, in
        nanometer.
    sigma_yx : float
        Estimated standard deviation of the gaussian signal along y and x axis,
        in nanometer.
    resolution_z : float
        Height of a voxel, in nanometer.
    resolution_yx : float
        size of a voxel, in nanometer.
    psf_amplitude : float
        Estimated pixel intensity of a spot.
    psf_background : float
        Estimated pixel intensity of the background.

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

    # get voxel coordinates
    meshgrid_z_minus = meshgrid_z - resolution_z / 2
    meshgrid_z_plus = meshgrid_z + resolution_z / 2
    meshgrid_y_minus = meshgrid_y - resolution_yx / 2
    meshgrid_y_plus = meshgrid_y + resolution_yx / 2
    meshgrid_x_minus = meshgrid_x - resolution_yx / 2
    meshgrid_x_plus = meshgrid_x + resolution_yx / 2

    # compute gaussian function for each voxel (i, j, k) volume V
    factor = psf_amplitude / (resolution_yx ** 2 * resolution_z)
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
    voxel_integral = voxel_integral_z * voxel_integral_y * voxel_integral_x
    values = psf_background + factor * voxel_integral

    return values


# ### Spot parameter ###

def get_spot_volume(image, spot_z, spot_y, spot_x, radius_z, radius_yx,
                    return_center=False):
    """Get a subimage of a detected spot in 3-d.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    spot_z : np.int64
        Coordinate of the detected spot along the z axis.
    spot_y : np.int64
        Coordinate of the detected spot along the y axis.
    spot_x : np.int64
        Coordinate of the detected spot along the x axis.
    radius_z : float
        Estimated radius of the spot along the z-dimension.
    radius_yx : float
        Estimated radius of the spot on the yx-plan.
    return_center : bool
        Return center of the detected spot in the new volume.

    Returns
    -------
    image_spot : np.ndarray, np.uint
        A 3-d image with detected spot and shape (radius_z * 2, radius_yx * 2,
        radius_yx * 2).
    center_z : float
        Estimated centroid of the spot, in nanometer, along the z axis.
    center_y : float
        Estimated centroid of the spot, in nanometer, along the y axis.
    center_x : float
        Estimated centroid of the spot, in nanometer, along the x axis.

    """
    # get boundaries of the volume surrounding the spot
    z_spot_min = max(0, int(spot_z - 2 * radius_z))
    z_spot_max = min(image.shape[0], int(spot_z + 2 * radius_z) + 1)
    y_spot_min = max(0, int(spot_y - 2 * radius_yx))
    y_spot_max = min(image.shape[1], int(spot_y + 2 * radius_yx) + 1)
    x_spot_min = max(0, int(spot_x - 2 * radius_yx))
    x_spot_max = min(image.shape[2], int(spot_x + 2 * radius_yx) + 1)

    # get the volume of the spot
    image_spot = image[z_spot_min:z_spot_max + 1,
                       y_spot_min:y_spot_max + 1,
                       x_spot_min:x_spot_max + 1]

    # get center of the detected spot in the new volume
    if return_center:
        center_z = spot_z - z_spot_min
        center_y = spot_y - y_spot_min
        center_x = spot_x - x_spot_min

        return image_spot, center_z, center_y, center_x

    else:

        return image_spot


def get_spot_surface(image, z_spot, spot_y, spot_x, radius_yx):
    """Get a subimage of a detected spot from its supposed yx plan.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    spot_z : np.int64
        Coordinate of the detected spot along the z axis.
    spot_y : np.int64
        Coordinate of the detected spot along the y axis.
    spot_x : np.int64
        Coordinate of the detected spot along the x axis.
    radius_yx : float
        Estimated radius of the spot on the yx-plan.

    Returns
    -------
    image_spot_2d : np.ndarray, np.uint
        A 2-d image with detected spot and shape (radius_yx * 2,
        radius_yx * 2).

    """
    # get boundaries of the volume surrounding the spot
    y_spot_min = max(0, int(spot_y - 2 * radius_yx))
    y_spot_max = min(image.shape[1], int(spot_y + 2 * radius_yx) + 1)
    x_spot_min = max(0, int(spot_x - 2 * radius_yx))
    x_spot_max = min(image.shape[2], int(spot_x + 2 * radius_yx) + 1)

    # get the detected yx plan for the spot
    image_spot_2d = image[z_spot,
                          y_spot_min:y_spot_max + 1,
                          x_spot_min:x_spot_max + 1]

    return image_spot_2d


def build_grid(image_spot, resolution_z, resolution_yx, return_centroid=False):
    """Build a grid in nanometer to compute gaussian function over a full
    volume.

    Parameters
    ----------
    image_spot : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    resolution_z : float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : float
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
    zz *= resolution_z
    yy *= resolution_yx
    xx *= resolution_yx

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


def compute_background_amplitude(image_spot):
    """Compute amplitude of a spot and background minimum value.

    Parameters
    ----------
    image_spot : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).

    Returns
    -------
    psf_amplitude : float
        Amplitude of the spot.
    psf_background : float
        Background minimum value of the voxel.

    """
    image_min, image_max = image_spot.min(), image_spot.max()
    psf_amplitude = image_max - image_min
    psf_background = image_min

    return psf_amplitude, psf_background


def get_spot_parameter(image, spot_z, spot_y, spot_x, psf_z=400, psf_yx=200,
                       resolution_z=300, resolution_yx=103,
                       compute_centroid=False):
    """Initialize parameters to fit gaussian function on a spot.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    spot_z : np.int64
        Coordinate of the detected spot along the z axis.
    spot_y : np.int64
        Coordinate of the detected spot along the y axis.
    spot_x : np.int64
        Coordinate of the detected spot along the x axis.
    psf_z : int or float
        Theoretical height of the spot PSF along the z axis, in nanometer.
    psf_yx : int or float
        Theoretical diameter of the spot PSF on the yx plan, in nanometer.
    resolution_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    compute_centroid : bool
        Compute an estimation of the centroid of the spot from pixel intensity
        or use the center of the subimage.

    Returns
    -------
    image_spot : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    grid : np.ndarray, np.float32
        A grid with the shape (3, z * y * x), in nanometer.
    center_z : float
        Estimated centroid of the spot, in nanometer, along the z axis.
    center_y : float
        Estimated centroid of the spot, in nanometer, along the y axis.
    center_x : float
        Estimated centroid of the spot, in nanometer, along the x axis.
    psf_amplitude : float
        Amplitude of the spot.
    psf_background : float
        Background minimum value of the voxel.

    """
    # compute estimated radius of the spot
    sigma_z, sigma_yx = get_sigma(resolution_z=resolution_z,
                                  resolution_yx=resolution_yx,
                                  psf_z=psf_z,
                                  psf_yx=psf_yx)
    radius_z = np.sqrt(3) * sigma_z
    radius_yx = np.sqrt(3) * sigma_yx

    if compute_centroid:
        # get subimage of the spot
        image_spot = get_spot_volume(
            image=image,
            spot_z=spot_z,
            spot_y=spot_y,
            spot_x=spot_x,
            radius_z=radius_z,
            radius_yx=radius_yx)

        # build a grid to fit the gaussian values
        grid, center_z, center_y, center_x = build_grid(
            image_spot=image_spot,
            resolution_z=resolution_z,
            resolution_yx=resolution_yx,
            return_centroid=True)

    else:
        # get subimage of the spot
        image_spot, center_z, center_y, center_x = get_spot_volume(
            image=image,
            spot_z=spot_z,
            spot_y=spot_y,
            spot_x=spot_x,
            radius_z=radius_z,
            radius_yx=radius_yx,
            return_center=True)
        center_z = float(center_z * resolution_z)
        center_y = float(center_y * resolution_yx)
        center_x = float(center_x * resolution_yx)

        # build a grid to fit the gaussian values
        grid = build_grid(
            image_spot=image_spot,
            resolution_z=resolution_z,
            resolution_yx=resolution_yx,
            return_centroid=False)

    # compute amplitude and background values
    psf_amplitude, psf_background = compute_background_amplitude(image_spot)

    return (image_spot, grid, center_z, center_y, center_x, psf_amplitude,
            psf_background)


# ### Gaussian fitting ###

def objective_function(resolution_z=300, resolution_yx=103, sigma_z=400,
                       sigma_yx=200,  psf_amplitude=None):
    """Design the objective function used to fit the gaussian function.

    Parameters
    ----------
    resolution_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    sigma_z : float
        Theoretical height of the spot PSF along the z axis, in nanometer.
    sigma_yx : float
        Theoretical diameter of the spot PSF on the yx plan, in nanometer.
    psf_amplitude : float
        Amplitude of the spot.

    Returns
    -------
    f : func
        A 3-d gaussian function with some parameters fixed.

    """
    # sigma is a fixed and known parameter
    if (sigma_z is not None
            and sigma_yx is not None
            and psf_amplitude is None):
        def f(grid, mu_z, mu_y, mu_x, psf_amplitude, psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=sigma_z,
                                 sigma_yx=sigma_yx,
                                 resolution_z=resolution_z,
                                 resolution_yx=resolution_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # amplitude is a fixed and known parameter
    elif (psf_amplitude is not None
          and sigma_z is None
          and sigma_yx is None):
        def f(grid, mu_z, mu_y, mu_x, sigma_z, sigma_yx, psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=sigma_z,
                                 sigma_yx=sigma_yx,
                                 resolution_z=resolution_z,
                                 resolution_yx=resolution_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    # amplitude and sigma are fixed and known parameters
    elif (psf_amplitude is not None
          and sigma_z is not None
          and sigma_yx is not None):
        def f(grid, mu_z, mu_y, mu_x, psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=sigma_z,
                                 sigma_yx=sigma_yx,
                                 resolution_z=resolution_z,
                                 resolution_yx=resolution_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    elif (psf_amplitude is None
          and sigma_z is None
          and sigma_yx is None):
        def f(grid, mu_z, mu_y, mu_x, sigma_z, sigma_yx, psf_amplitude,
              psf_background):
            values = gaussian_3d(grid=grid,
                                 mu_z=mu_z,
                                 mu_y=mu_y,
                                 mu_x=mu_x,
                                 sigma_z=sigma_z,
                                 sigma_yx=sigma_yx,
                                 resolution_z=resolution_z,
                                 resolution_yx=resolution_yx,
                                 psf_amplitude=psf_amplitude,
                                 psf_background=psf_background)
            return values

    else:
        raise ValueError("Parameters 'sigma_z' and 'sigma_yx' should be "
                         "fixed or optimized together.")

    return f


def fit_gaussian(f, grid, image_spot, p0, lower_bound=None, upper_bound=None):
    """Fit a gaussian function to a 3-d image.

    # TODO add equations and algorithm

    Parameters
    ----------
    f : func
        A 3-d gaussian function with some parameters fixed.
    grid : np.ndarray, np.float
        Grid data to compute the gaussian function for different voxel within
        a volume V. In nanometer, with shape (3, V_z * V_y * V_x).
    image_spot : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
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


def simulate_fitted_gaussian(f, grid, popt, original_shape=None):
    """Use the optimized parameter to simulate a gaussian signal.

    Parameters
    ----------
    f : func
        A 3-d gaussian function with some parameters fixed.
    grid : np.ndarray, np.float
        Grid data to compute the gaussian function for different voxel within
        a volume V. In nanometer, with shape (3, V_z * V_y * V_x).
    popt : np.ndarray
        Fitted parameters.
    original_shape : Tuple
        Shape of the spot image to reshape the simulation.

    Returns
    -------
    values : np.ndarray, np.float
        Value of each voxel within the volume V according to the 3-d gaussian
        parameters. Shape (V_z, V_y, V_x,) or (V_z * V_y * V_x,).

    """
    values = f(grid, *popt)
    if original_shape is not None:
        values = np.reshape(values, original_shape).astype(np.float32)

    return values
