# -*- coding: utf-8 -*-

"""
Functions to fit gaussian functions to the detected RNA spots.
"""

import bigfish.stack as stack
from .spot_detection import get_sigma, get_cc, filter_cc

import numpy as np

from scipy.special import erf
from scipy.optimize import curve_fit


# TODO complete documentation methods
# TODO add sanity check functions

# ### Gaussian function ###

def gaussian_3d(grid, mu_z, mu_y, mu_x, sigma_z, sigma_yx, resolution_z,
                resolution_yx, psf_amplitude, psf_background,
                precomputed=None):
    """Compute the gaussian function over the grid 'xdata' representing a
    volume V with shape (V_z, V_y, V_x).

    # TODO add equations

    Parameters
    ----------
    grid : np.ndarray, np.float32
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
    precomputed : List[np.ndarray] or Tuple[np.ndarray]
        Precomputed tables values of erf for the different axis.

    Returns
    -------
    values : np.ndarray, np.float
        Value of each voxel within the volume V according to the 3-d gaussian
        parameters. Shape (V_z * V_y * V_x,).

    """
    # check parameters
    stack.check_array(grid,
                      ndim=2,
                      dtype=np.float32,
                      allow_nan=True)
    stack.check_parameter(mu_z=(float, int),
                          mu_y=(float, int),
                          mu_x=(float, int),
                          sigma_z=(float, int),
                          sigma_yx=(float, int),
                          resolution_z=(float, int),
                          resolution_yx=(float, int),
                          psf_amplitude=(float, int),
                          psf_background=(float, int),
                          precomputed=(type(None), tuple, list))

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
        meshgrid_z_minus = meshgrid_z - resolution_z / 2
        meshgrid_z_plus = meshgrid_z + resolution_z / 2
        meshgrid_y_minus = meshgrid_y - resolution_yx / 2
        meshgrid_y_plus = meshgrid_y + resolution_yx / 2
        meshgrid_x_minus = meshgrid_x - resolution_yx / 2
        meshgrid_x_plus = meshgrid_x + resolution_yx / 2

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
    factor = psf_amplitude / (resolution_yx ** 2 * resolution_z)
    voxel_integral = voxel_integral_z * voxel_integral_y * voxel_integral_x
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
    mu : float
        Estimated mean of the gaussian signal along a specific axis.
    sigma : float
        Estimated standard deviation of the gaussian signal along a specific
        axis.

    Returns
    -------
    rescaled_erf : np.ndarray, np.float
        Rescaled erf along a specific axis.

    """
    # check parameters
    stack.check_parameter(low=np.ndarray,
                          high=np.ndarray,
                          mu=(float, int),
                          sigma=(float, int))

    # compute erf and normalize it
    low_ = (low - mu) / (np.sqrt(2) * sigma)
    high_ = (high - mu) / (np.sqrt(2) * sigma)
    rescaled_erf = sigma * np.sqrt(np.pi / 2) * (erf(high_) - erf(low_))

    return rescaled_erf


def precompute_erf(resolution_z, resolution_yx, sigma_z, sigma_yx,
                   max_grid=200):
    """Precompute different values for the erf with a resolution of 5 nm.

    Parameters
    ----------
    resolution_z : float, int
        Height of a voxel, in nanometer.
    resolution_yx : float, int
        size of a voxel, in nanometer.
    sigma_z : float, int
        Estimated standard deviation of the gaussian signal along z axis, in
        nanometer.
    sigma_yx : float, int
        Estimated standard deviation of the gaussian signal along y and x axis,
        in nanometer.
    max_grid : int
        Maximum size of the grid on which we precompute the erf, in pixel.

    Returns
    -------
    table_erf_z : np.ndarray, np.float64
        Table of precomputed values for the erf along the z axis with shape
        (nb_value, 2).
    table_erf_y : np.ndarray, np.float64
        Table of precomputed values for the erf along the y axis with shape
        (nb_value, 2).
    table_erf_x : np.ndarray, np.float64
        Table of precomputed values for the erf along the x axis with shape
        (nb_value, 2).

    """
    # check parameters
    stack.check_parameter(resolution_z=(float, int),
                          resolution_yx=(float, int),
                          sigma_z=(float, int),
                          sigma_yx=(float, int),
                          max_grid=int)

    # build a grid with a spatial resolution of 5 nm and a size of
    # max_grid * resolution nm
    zz = np.array([i for i in range(0, max_grid * resolution_z, 5)])
    yy = np.array([i for i in range(0, max_grid * resolution_yx, 5)])
    xx = np.array([i for i in range(0, max_grid * resolution_yx, 5)])
    mu_z, mu_y, mu_x = 0, 0, 0

    # compute erf values for this grid
    erf_z = _rescaled_erf(low=zz - resolution_z/2,
                          high=zz + resolution_z/2,
                          mu=mu_z,
                          sigma=sigma_z)
    erf_y = _rescaled_erf(low=yy - resolution_yx/2,
                          high=yy + resolution_yx/2,
                          mu=mu_y,
                          sigma=sigma_yx)
    erf_x = _rescaled_erf(low=xx - resolution_yx/2,
                          high=xx + resolution_yx/2,
                          mu=mu_x,
                          sigma=sigma_yx)
    table_erf_z = np.array([zz, erf_z]).T
    table_erf_y = np.array([yy, erf_y]).T
    table_erf_x = np.array([xx, erf_x]).T

    return table_erf_z, table_erf_y, table_erf_x


# ### Spot parameter ###

def build_reference_spot_3d(image, spots, radius, method="median"):
    """Build a median or mean spot volume/surface as reference.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3).
    radius : Tuple[float]
        Radius of the detected peaks, one for each dimension.
    method : str
        Method use to compute the reference spot (a 'mean' or 'median' spot).

    Returns
    -------
    reference_spot : np.ndarray
        Reference spot with shape (2*radius_z+1, 2*radius_y+1, 2*radius_x+1).

    """
    # check parameters
    stack.check_array(image,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_array(spots,
                      ndim=2,
                      dtype=[np.int64],
                      allow_nan=False)
    stack.check_parameter(radius=(float, int, tuple),
                          method=str)
    if method not in ['mean', 'median']:
        raise ValueError("'{0}' is not a valid value for parameter 'method'. "
                         "Use 'mean' or 'median' instead.".format(method))

    # get a rounded radius for each dimension
    radius_z = int(radius[0]) + 1
    radius_yx = int(radius[1]) + 1
    z_shape = radius_z * 2 + 1
    yx_shape = radius_yx * 2 + 1
    # randomly choose some spots to aggregate
    indices = [i for i in range(spots.shape[0])]
    np.random.shuffle(indices)
    indices = indices[:min(2000, spots.shape[0])]
    candidate_spots = spots[indices, :]
    # TODO add a warning if not enough spots are detected

    # collect area around each spot
    l_reference_spot = []
    nb_spots = 0
    for i_spot in range(candidate_spots.shape[0]):

        # get spot coordinates
        spot_z, spot_y, spot_x = candidate_spots[i_spot, :]

        # get the volume of the spot
        image_spot = get_spot_volume(image, spot_z, spot_y, spot_x,
                                     radius_z, radius_yx)

        # remove the cropped images
        if image_spot.shape != (z_shape, yx_shape, yx_shape):
            continue
        else:
            nb_spots += 1
        l_reference_spot.append(image_spot)

    # if no spot where detected
    if len(l_reference_spot) == 0:
        return None

    # project the different spot images
    l_reference_spot = np.stack(l_reference_spot, axis=0)
    if method == "mean":
        reference_spot = np.mean(l_reference_spot, axis=0)
    else:
        reference_spot = np.median(l_reference_spot, axis=0)
    reference_spot = reference_spot.astype(image.dtype)

    return reference_spot


def get_spot_volume(image, spot_z, spot_y, spot_x, radius_z, radius_yx):
    """Get a subimage of a detected spot in 3-d.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x)).
    spot_z : np.int64
        Coordinate of the detected spot along the z axis.
    spot_y : np.int64
        Coordinate of the detected spot along the y axis.
    spot_x : np.int64
        Coordinate of the detected spot along the x axis.
    radius_z : float or int
        Estimated radius of the spot along the z-dimension.
    radius_yx : float or int
        Estimated radius of the spot on the yx-plan.

    Returns
    -------
    image_spot : np.ndarray
        Reference spot with shape (2*radius_z+1, 2*radius_y+1, 2*radius_x+1).

    """
    # check parameters
    stack.check_array(image,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=True)
    stack.check_parameter(spot_z=np.int64,
                          spot_y=np.int64,
                          spot_x=np.int64,
                          radius_z=(float, int),
                          radius_yx=(float, int))

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


def get_spot_surface(image, spot_y, spot_x, radius_yx):
    """Get a subimage of a detected spot from its supposed yx plan.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with detected spot and shape (y, x).
    spot_y : np.int64
        Coordinate of the detected spot along the y axis.
    spot_x : np.int64
        Coordinate of the detected spot along the x axis.
    radius_yx : float
        Estimated radius of the spot on the yx-plan.

    Returns
    -------
    image_spot : np.ndarray
        Reference spot with shape (2*radius_y+1, 2*radius_x+1).

    """
    # check parameters
    stack.check_array(image,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=True)
    stack.check_parameter(spot_y=np.int64,
                          spot_x=np.int64,
                          radius_yx=np.int64)

    # get boundaries of the volume surrounding the spot
    y_spot_min = max(0, int(spot_y - radius_yx))
    y_spot_max = min(image.shape[1], int(spot_y + radius_yx))
    x_spot_min = max(0, int(spot_x - radius_yx))
    x_spot_max = min(image.shape[2], int(spot_x + radius_yx))

    # get the volume of the spot
    image_spot = image[y_spot_min:y_spot_max + 1,
                       x_spot_min:x_spot_max + 1]

    return image_spot


def initialize_spot_parameter_3d(image, spot_z, spot_y, spot_x, psf_z=400,
                                 psf_yx=200, resolution_z=300,
                                 resolution_yx=103):
    """Initialize parameters to fit a 3-d gaussian function on a spot.

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
    # check parameters
    stack.check_array(image,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_parameter(spot_z=np.int64,
                          spot_y=np.int64,
                          spot_x=np.int64,
                          psf_z=(float, int),
                          psf_yx=(float, int),
                          resolution_z=(float, int),
                          resolution_yx=(float, int))

    # compute estimated radius of the spot
    sigma_z, sigma_yx = get_sigma(resolution_z=resolution_z,
                                  resolution_yx=resolution_yx,
                                  psf_z=psf_z,
                                  psf_yx=psf_yx)
    radius_z = np.sqrt(3) * sigma_z
    radius_yx = np.sqrt(3) * sigma_yx

    # get subimage of the spot
    image_spot = get_spot_volume(
        image=image,
        spot_z=spot_z,
        spot_y=spot_y,
        spot_x=spot_x,
        radius_z=radius_z,
        radius_yx=radius_yx)

    # build a grid to fit the gaussian values
    grid, center_z, center_y, center_x = initialize_grid_3d(
        image_spot=image_spot,
        resolution_z=resolution_z,
        resolution_yx=resolution_yx,
        return_centroid=True)

    # compute amplitude and background values
    psf_amplitude, psf_background = compute_background_amplitude(image_spot)

    return (image_spot, grid, center_z, center_y, center_x, psf_amplitude,
            psf_background)


def initialize_grid_3d(image_spot, resolution_z, resolution_yx,
                       return_centroid=False):
    """Build a grid in nanometer to compute gaussian function over a full
    volume.

    Parameters
    ----------
    image_spot : np.ndarray
        A 3-d image with detected spot and shape (z, y, x).
    resolution_z : float or int
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : float or int
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
    # check parameters
    stack.check_array(image_spot,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=True)
    stack.check_parameter(resolution_z=(float, int),
                          resolution_yx=(float, int),
                          return_centroid=bool)

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
    psf_amplitude : float or int
        Amplitude of the spot.
    psf_background : float or int
        Background minimum value of the voxel.

    """
    # check parameters
    stack.check_array(image_spot,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)

    # compute values
    image_min, image_max = image_spot.min(), image_spot.max()
    psf_amplitude = image_max - image_min
    psf_background = image_min

    return psf_amplitude, psf_background


# ### Gaussian fitting ###

def objective_function(resolution_z=300, resolution_yx=103, sigma_z=400,
                       sigma_yx=200, psf_amplitude=None):
    """Design the objective function used to fit the gaussian function.

    Parameters
    ----------
    resolution_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    sigma_z : int or float
        Theoretical height of the spot PSF along the z axis, in nanometer.
    sigma_yx : int or float
        Theoretical diameter of the spot PSF on the yx plan, in nanometer.
    psf_amplitude : int or float
        Amplitude of the spot.

    Returns
    -------
    f : func
        A 3-d gaussian function with some parameters fixed.

    """
    # TODO add precomputation
    # check parameters
    stack.check_parameter(resolution_z=(float, int),
                          resolution_yx=(float, int),
                          sigma_z=(float, int, type(None)),
                          sigma_yx=(float, int, type(None)),
                          psf_amplitude=(float, int, type(None)))

    # sigma is known, we fit mu, amplitude and background
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

    # amplitude is known, we fit sigma, mu and background
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

    # amplitude and sigma are known, we fit mu and background
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

    # we fit mu, sigma, amplitude and background
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


def fit_gaussian_3d(f, grid, image_spot, p0, lower_bound=None,
                    upper_bound=None):
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
    # check parameters
    stack.check_array(grid,
                      ndim=2,
                      dtype=np.float32,
                      allow_nan=False)
    stack.check_array(image_spot,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_parameter(p0=list,
                          lower_bound=(list, type(None)),
                          upper_bound=(list, type(None)))

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


def simulate_fitted_gaussian_3d(f, grid, popt, original_shape=None):
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
    # check parameters
    stack.check_array(grid,
                      ndim=2,
                      dtype=np.float32,
                      allow_nan=False)
    stack.check_parameter(popt=np.ndarray,
                          original_shape=(tuple, type(None)))

    # compute gaussian values
    values = f(grid, *popt)

    # reshape values if necessary
    if original_shape is not None:
        values = np.reshape(values, original_shape).astype(np.float32)

    return values


def fit_gaussian_mixture(image, region, resolution_z, resolution_yx, sigma_z,
                         sigma_yx, amplitude, background,
                         precomputed_gaussian):
    """Fit a mixture of gaussian to a potential foci region.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    region : skimage.measure._regionprops._RegionProperties
        Properties of a foci region.
    resolution_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    sigma_z : int or float
        Theoretical height of the spot PSF along the z axis, in nanometer.
    sigma_yx : int or float
        Theoretical diameter of the spot PSF on the yx plan, in nanometer.
    amplitude : int or float
        Amplitude of the spot.
    background : int of float
        Background intensity level of the spot.
    precomputed_gaussian : List[np.ndarray] or Tuple[np.ndarray]
        Precomputed tables values of erf for the different axis.

    Returns
    -------
    image_region : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    best_simulation : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    positions_gaussian : List[List]
        List of positions (as a list [z, y, x]) for the different gaussian
        simulations used in the mixture.

    """
    # TODO improve documentation
    # TODO make this function consistent
    # check parameters
    stack.check_array(image,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=True)
    stack.check_parameter(resolution_z=(float, int),
                          resolution_yx=(float, int),
                          sigma_z=(float, int),
                          sigma_yx=(float, int),
                          amplitude=(float, int),
                          background=(float, int),
                          precomputed_gaussian=(list, tuple))

    # get an image of the region
    box = tuple(region.bbox)
    image_region = image[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
    image_region_raw = np.reshape(image_region, image_region.size)

    # build a grid to represent this image
    grid = initialize_grid_3d(image_region, resolution_z, resolution_yx)

    # add a gaussian for each local maximum while the RSS decreases
    simulation = np.zeros(image_region_raw.shape, dtype=np.float64)
    residual = image_region_raw - simulation
    ssr = np.sum(residual ** 2)
    diff_ssr = -1
    nb_gaussian = 0
    best_simulation = simulation.copy()
    positions_gaussian = []
    while diff_ssr < 0 or nb_gaussian == 1000:
        position_gaussian = np.argmax(residual)
        positions_gaussian.append(list(grid[:, position_gaussian]))
        simulation += gaussian_3d(grid=grid,
                                  mu_z=float(positions_gaussian[-1][0]),
                                  mu_y=float(positions_gaussian[-1][1]),
                                  mu_x=float(positions_gaussian[-1][2]),
                                  sigma_z=sigma_z,
                                  sigma_yx=sigma_yx,
                                  resolution_z=resolution_z,
                                  resolution_yx=resolution_yx,
                                  psf_amplitude=amplitude,
                                  psf_background=background,
                                  precomputed=precomputed_gaussian)
        residual = image_region_raw - simulation
        new_ssr = np.sum(residual ** 2)
        diff_ssr = new_ssr - ssr
        ssr = new_ssr
        nb_gaussian += 1
        background = 0
        # print("NB spots {0} | Difference SSR {1} | SSR {2}"
        #       .format(nb_gaussian, int(diff_ssr), int(ssr)))

        if diff_ssr < 0:
            best_simulation = simulation.copy()

    if 1 < nb_gaussian < 1000:
        positions_gaussian.pop(-1)

    best_simulation = np.reshape(best_simulation, image_region.shape)
    best_simulation = best_simulation.astype(image_region_raw.dtype)

    return image_region, best_simulation, positions_gaussian


# ### Foci decomposition ###


def foci_decomposition(image_filtered_log, image_filtered_background,
                       threshold_spot, spots, radius, min_area, min_nb_spots,
                       min_intensity_factor, resolution_z=300,
                       resolution_yx=103, psf_z=400, psf_yx=200):
    """Detect regions with clustered spots (foci) and fit a mixture of
    gaussian to them.

    Parameters
    ----------
    image_filtered_log : np.ndarray
        Image with shape (z, y, x) and filter with LoG operator.
    image_filtered_background : np.ndarray
        Image with shape (z, y, x) and filter with gaussian operator to
        estimate then remove background.
    threshold_spot : float or int
        A threshold to detect spots.
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3).
    radius : Tuple[float]
        Radius of the detected peaks, one for each dimension.
    min_area : int
        Minimum number of pixels in the connected region.
    min_nb_spots : int
        Minimum number of spot detected in this region.
    min_intensity_factor : int or float
        Minimum pixel intensity in the connected region is equal to
        median(intensity) * min_intensity_factor.
    resolution_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float
        Theoretical height of the spot PSF along the z axis, in nanometer.
    psf_yx : int or float
        Theoretical diameter of the spot PSF on the yx plan, in nanometer.

    Returns
    -------
    spots_out_foci : np.ndarray, np.int64
        Coordinate of the spots detected out of foci, with shape (nb_spots, 3).
        One coordinate per dimension (zyx coordinates).
    spots_in_foci : np.ndarray, np.int64
        Coordinate of the spots detected inside foci, with shape (nb_spots, 4).
        One coordinate per dimension (zyx coordinates) plus the index of the
        foci.
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index.
    reference_spot : np.ndarray
        Reference spot with shape (2*radius_z+1, 2*radius_y+1, 2*radius_x+1).

    """
    # check parameters
    stack.check_array(image_filtered_log,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_array(image_filtered_background,
                      ndim=3,
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_array(spots,
                      ndim=2,
                      dtype=[np.int64],
                      allow_nan=False)
    stack.check_parameter(threshold_spot=(float, int),
                          radius=(tuple, list),
                          min_area=(float, int),
                          min_nb_spots=(float, int),
                          min_intensity_factor=(float, int),
                          resolution_z=(float, int),
                          resolution_yx=(float, int),
                          psf_z=(float, int),
                          psf_yx=(float, int))

    # case where no spot were detected
    if spots.size == 0:
        spots_out_foci = np.array([], dtype=np.int64).reshape((0, 3))
        spots_in_foci = np.array([], dtype=np.int64).reshape((0, 4))
        foci = np.array([], dtype=np.float32).reshape((0, 5))
        radius_z = int(radius[0]) + 1
        radius_yx = int(radius[1]) + 1
        z_shape = radius_z * 2 + 1
        yx_shape = radius_yx * 2 + 1
        reference_spot = np.zeros((z_shape, yx_shape, yx_shape),
                                  dtype=image_filtered_background.dtype)

        return spots_out_foci, spots_in_foci, foci, reference_spot

    # build a reference median spot
    reference_spot = build_reference_spot_3d(
        image_filtered_background,
        spots,
        radius,
        method="median")

    # initialize a grid representing the reference spot
    grid, centroid_z, centroid_y, centroid_x = initialize_grid_3d(
        image_spot=reference_spot,
        resolution_z=resolution_z,
        resolution_yx=resolution_yx,
        return_centroid=True)

    # compute amplitude and background of the reference spot
    amplitude, background = compute_background_amplitude(reference_spot)

    # TODO initialize the function multiple times
    # fit a 3-d gaussian function on this reference spot
    f = objective_function(
        resolution_z=resolution_z,
        resolution_yx=resolution_yx,
        sigma_z=None,
        sigma_yx=None,
        psf_amplitude=None)
    p0 = [centroid_z, centroid_y, centroid_x, psf_z, psf_yx, amplitude,
          background]
    popt, pcov = fit_gaussian_3d(f, grid, reference_spot, p0)

    # get reference parameters
    sigma_z = popt[3]
    sigma_yx = popt[4]
    amplitude = popt[5]
    background = popt[6]

    # use connected components to detect potential foci
    cc = get_cc(image_filtered_log, threshold_spot)
    regions_filtered, spots_out_foci = filter_cc(
        image_filtered_background,
        cc,
        spots,
        min_area=min_area,
        min_nb_spots=min_nb_spots,
        min_intensity_factor=min_intensity_factor)

    # case where no foci where detected
    if regions_filtered.size == 0:
        spots_in_foci = np.array([], dtype=np.int64).reshape((0, 4))
        foci = np.array([], dtype=np.float32).reshape((0, 5))
        return spots, spots_in_foci, foci, reference_spot

    # precompute gaussian function values
    table_erf_z, table_erf_y, table_erf_x = precompute_erf(
        resolution_z,
        resolution_yx,
        sigma_z,
        sigma_yx,
        max_grid=200)
    precomputed_gaussian = (table_erf_z, table_erf_y, table_erf_x)

    # fit gaussian mixtures in the foci regions
    spots_in_foci = []
    foci = []
    for i_foci, region in enumerate(regions_filtered):
        (image_region,
         best_simulation,
         pos_gaussian) = fit_gaussian_mixture(
            image_filtered_background,
            region,
            resolution_z,
            resolution_yx,
            sigma_z,
            sigma_yx,
            amplitude,
            background,
            precomputed_gaussian)

        # get coordinates of spots and foci in the original image
        box = region.bbox
        (min_z, min_y, min_x, _, _, _) = box
        pos_gaussian = np.array(pos_gaussian, dtype=np.float64)
        pos_gaussian[:, 0] = (pos_gaussian[:, 0] / resolution_z) + min_z
        pos_gaussian[:, 1] = (pos_gaussian[:, 1] / resolution_yx) + min_y
        pos_gaussian[:, 2] = (pos_gaussian[:, 2] / resolution_yx) + min_x
        spots_in_foci_ = np.zeros((pos_gaussian.shape[0], 4), dtype=np.int64)
        spots_in_foci_[:, :3] = pos_gaussian
        spots_in_foci_[:, 3] = i_foci
        spots_in_foci.append(spots_in_foci_)
        foci_z, foci_y, foci_x = tuple(pos_gaussian[0])
        nb_rna_foci = pos_gaussian.shape[0]
        foci.append([foci_z, foci_y, foci_x, nb_rna_foci, i_foci])

    spots_in_foci = np.concatenate(spots_in_foci, axis=0)
    foci = np.array(foci, dtype=np.int64)

    return spots_out_foci, spots_in_foci, foci, reference_spot
