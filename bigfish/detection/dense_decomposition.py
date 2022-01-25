# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to detect dense and bright regions (with potential clustered spots),
then use gaussian functions to correct a misdetection in these regions.
"""

import warnings

import numpy as np

import bigfish.stack as stack

from .utils import build_reference_spot
from .utils import get_object_radius_pixel
from .spot_modeling import modelize_spot
from .spot_modeling import precompute_erf
from .spot_modeling import gaussian_2d
from .spot_modeling import _initialize_grid_2d
from .spot_modeling import gaussian_3d
from .spot_modeling import _initialize_grid_3d

from skimage.measure import regionprops
from skimage.measure import label


# ### Main function ###

def decompose_dense(image, spots, voxel_size, spot_radius, kernel_size=None,
                    alpha=0.5, beta=1, gamma=5):
    """Detect dense and bright regions with potential clustered spots and
    simulate a more realistic number of spots in these regions.

    #. We estimate image background with a large gaussian filter. We then
       remove the background from the original image to denoise it.
    #. We build a reference spot by aggregating predetected spots.
    #. We fit gaussian parameters on the reference spots.
    #. We detect dense regions to decompose.
    #. We simulate as many gaussians as possible in the candidate regions.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2)
        for 3-d or 2-d images respectively.
    voxel_size : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    spot_radius : int, float, Tuple(int, float) or List(int, float)
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions.
    kernel_size : int, float, Tuple(float, int), List(float, int) or None
        Standard deviation used for the gaussian kernel (one for each
        dimension), in pixel. If it's a scalar, the same standard deviation is
        applied to every dimensions. If None, we estimate the kernel size from
        'spot_radius', 'voxel_size' and 'gamma'
    alpha : int or float
        Intensity percentile used to compute the reference spot, between 0
        and 1. The higher, the brighter are the spots simulated in the dense
        regions. Consequently, a high intensity score reduces the number of
        spots added. Default is 0.5, meaning the reference spot considered is
        the median spot.
    beta : int or float
        Multiplicative factor for the intensity threshold of a dense region.
        Default is 1. Threshold is computed with the formula:

        .. math::
            \\mbox{threshold} = \\beta * \\mbox{max(median spot)}

        With :math:`\\mbox{median spot}` the median value of all detected spot
        signals.
    gamma : int or float
        Multiplicative factor use to compute the gaussian kernel size:

        .. math::
            \\mbox{kernel size} = \\frac{\\gamma * \\mbox{spot radius}}{\\mbox{
            voxel size}}

        We perform a large gaussian filter with such scale to estimate image
        background and remove it from original image. A large gamma increases
        the scale of the gaussian filter and smooth the estimated background.
        To decompose very large bright areas, a larger gamma should be set.

    Notes
    -----
    If ``gamma = 0`` and ``kernel_size = None``, image is not denoised.

    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the spots detected, with shape (nb_spots, 3) or
        (nb_spots, 2). One coordinate per dimension (zyx or yx coordinates).
    dense_regions : np.ndarray, np.int64
        Array with shape (nb_regions, 7) or (nb_regions, 6). One coordinate
        per dimension for the region centroid (zyx or yx coordinates), the
        number of RNAs detected in the region, the area of the region, its
        average intensity value and its index.
    reference_spot : np.ndarray
        Reference spot in 3-d or 2-d.

    """
    # TODO allow/return float64 spots
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(spots, ndim=2, dtype=np.int64)
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list),
        kernel_size=(int, float, tuple, list, type(None)),
        alpha=(int, float),
        beta=(int, float),
        gamma=(int, float))
    if alpha < 0 or alpha > 1:
        raise ValueError("'alpha' should be a value between 0 and 1, not {0}"
                         .format(alpha))
    if beta < 0:
        raise ValueError("'beta' should be a positive value, not {0}"
                         .format(beta))
    if gamma < 0:
        raise ValueError("'gamma' should be a positive value, not {0}"
                         .format(gamma))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence "
                "with {0} elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError("'spot_radius' must be a scalar or a "
                             "sequence with {0} elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim
    if kernel_size is not None:
        if isinstance(kernel_size, (tuple, list)):
            if len(kernel_size) != ndim:
                raise ValueError("'kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            kernel_size = (kernel_size,) * ndim

    # case where no spot were detected
    if spots.size == 0:
        dense_regions = np.array([], dtype=np.int64).reshape((0, ndim + 4))
        reference_spot = np.zeros((5,) * ndim, dtype=image.dtype)
        return spots, dense_regions, reference_spot

    # get gaussian kernel to denoise the image
    if kernel_size is None and gamma > 0:
        spot_radius_px = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
        kernel_size = tuple([spot_radius_px_ * gamma
                             for spot_radius_px_ in spot_radius_px])

    # denoise the image
    if kernel_size is not None:
        image_denoised = stack.remove_background_gaussian(
            image=image,
            sigma=kernel_size)
    else:
        image_denoised = image.copy()

    # build a reference median spot
    reference_spot = build_reference_spot(
        image=image_denoised,
        spots=spots,
        voxel_size=voxel_size,
        spot_radius=spot_radius,
        alpha=alpha)

    # case with an empty frame as reference spot
    if reference_spot.sum() == 0:
        dense_regions = np.array([], dtype=np.int64).reshape((0, ndim + 4))
        return spots, dense_regions, reference_spot

    # fit a gaussian function on the reference spot to be able to simulate it
    parameters_fitted = modelize_spot(
        reference_spot=reference_spot,
        voxel_size=voxel_size,
        spot_radius=spot_radius)
    if ndim == 3:
        sigma_z, sigma_yx, amplitude, background = parameters_fitted
        sigma = (sigma_z, sigma_yx, sigma_yx)
    else:
        sigma_yx, amplitude, background = parameters_fitted
        sigma = (sigma_yx, sigma_yx)

    # use connected components to detect dense and bright regions
    regions_to_decompose, spots_out_regions, region_size = get_dense_region(
        image=image_denoised,
        spots=spots,
        voxel_size=voxel_size,
        spot_radius=spot_radius,
        beta=beta)

    # case where no region where detected
    if regions_to_decompose.size == 0:
        dense_regions = np.array([], dtype=np.int64).reshape((0, ndim + 4))
        return spots, dense_regions, reference_spot

    # precompute gaussian function values
    max_grid = region_size + 1
    precomputed_gaussian = precompute_erf(
        ndim=ndim,
        voxel_size=voxel_size,
        sigma=sigma,
        max_grid=max_grid)

    # simulate gaussian mixtures in the dense regions
    spots_in_regions, dense_regions = simulate_gaussian_mixture(
        image=image_denoised,
        candidate_regions=regions_to_decompose,
        voxel_size=voxel_size,
        sigma=sigma,
        amplitude=amplitude,
        background=background,
        precomputed_gaussian=precomputed_gaussian)

    # normally the number of detected spots should increase
    if len(spots_out_regions) + len(spots_in_regions) < len(spots):
        warnings.warn("Problem occurs during the decomposition of dense "
                      "regions. Less spots are detected after the "
                      "decomposition than before.",
                      UserWarning)

    # merge outside and inside spots
    spots = np.concatenate((spots_out_regions, spots_in_regions[:, :ndim]),
                           axis=0)

    return spots, dense_regions, reference_spot


# ### Dense regions ###

def get_dense_region(image, spots, voxel_size, spot_radius, beta=1):
    """Detect and filter dense and bright regions.

    A candidate region has at least 2 connected pixels above a specific
    threshold.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    voxel_size : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    spot_radius : int, float, Tuple(int, float) or List(int, float)
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions.
    beta : int or float
        Multiplicative factor for the intensity threshold of a dense region.
        Default is 1. Threshold is computed with the formula:

        .. math::
            \\mbox{threshold} = \\beta * \\mbox{max(median spot)}

        With :math:`\\mbox{median spot}` the median value of all detected spot
        signals.

    Returns
    -------
    dense_regions : np.ndarray
        Array with selected ``skimage.measure._regionprops._RegionProperties``
        objects.
    spots_out_region : np.ndarray, np.int64
        Coordinate of the spots detected out of dense regions, with shape
        (nb_spots, 3) or (nb_spots, 2). One coordinate per dimension (zyx or
        yx coordinates).
    max_size : int
        Maximum size of the regions.

    """
    # TODO allow/return float64 spots
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(spots, ndim=2, dtype=np.int64)
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list),
        beta=(int, float))
    if beta < 0:
        raise ValueError("'beta' should be a positive value, not {0}"
                         .format(beta))

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

    # estimate median spot value and a threshold to detect dense regions
    median_spot = build_reference_spot(
        image=image,
        spots=spots,
        voxel_size=voxel_size,
        spot_radius=spot_radius,
        alpha=0.5)
    threshold = int(median_spot.max() * beta)

    # get connected regions
    connected_regions = _get_connected_region(image, threshold)

    # filter connected regions
    (dense_regions, spots_out_region, max_size) = _filter_connected_region(
        image, connected_regions, spots)

    return dense_regions, spots_out_region, max_size


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
    # compute binary mask of the filtered image
    mask = image > threshold

    # find connected components
    cc = label(mask)

    return cc


def _filter_connected_region(image, connected_component, spots):
    """Filter dense and bright regions (defined as connected component
    regions).

    A candidate region has at least 2 connected pixels above a specific
    threshold.

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
    bbox = []
    for i, region in enumerate(regions):
        area.append(region.area)
        bbox.append(region.bbox)
    regions = np.array(regions)
    area = np.array(area)
    bbox = np.array(bbox)

    # keep regions with a minimum size
    big_area = area >= 2
    regions_filtered = regions[big_area]
    bbox_filtered = bbox[big_area]

    # case where no region big enough were detected
    if regions.size == 0:
        regions_filtered = np.array([])
        return regions_filtered, spots, 0

    spots_out_region, max_region_size = _filter_spot_out_candidate_regions(
        bbox_filtered, spots, nb_dim=image.ndim)

    return regions_filtered, spots_out_region, max_region_size


def _filter_spot_out_candidate_regions(candidate_bbox, spots, nb_dim):
    """Filter spots out of the dense regions.

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

def simulate_gaussian_mixture(image, candidate_regions, voxel_size, sigma,
                              amplitude=100, background=0,
                              precomputed_gaussian=None):
    """Simulate as many gaussians as possible in the candidate dense regions in
    order to get a more realistic number of spots.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    candidate_regions : np.ndarray
        Array with filtered skimage.measure._regionprops._RegionProperties.
    voxel_size : int, float, Tuple(int, float) or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    sigma : int, float, Tuple(int, float) or List(int, float)
        Standard deviation of the gaussian, in nanometer. One value per
        spatial dimension (zyx or yx dimensions). If it's a scalar, the same
        value is applied to every dimensions.
    amplitude : float
        Amplitude of the gaussian.
    background : float
        Background minimum value.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    spots_in_regions : np.ndarray, np.int64
        Coordinate of the spots detected inside dense regions, with shape
        (nb_spots, 4) or (nb_spots, 3). One coordinate per dimension (zyx
        or yx coordinates) plus the index of the region.
    regions : np.ndarray, np.int64
        Array with shape (nb_regions, 7) or (nb_regions, 6). One coordinate
        per dimension for the region centroid (zyx or yx coordinates), the
        number of RNAs detected in the region, the area of the region, its
        average intensity value and its index.

    """
    # TODO allow/return float64 spots
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(
        candidate_regions=np.ndarray,
        voxel_size=(int, float, tuple, list),
        sigma=(int, float, tuple, list),
        amplitude=float,
        background=float)
    if background < 0:
        raise ValueError("Background value can't be negative: {0}"
                         .format(background))

    # check consistency between parameters
    ndim = image.ndim
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(sigma, (tuple, list)):
        if len(sigma) != ndim:
            raise ValueError(
                "'sigma' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        sigma = (sigma,) * ndim

    # simulate gaussian mixtures in the candidate regions...
    spots_in_regions = []
    regions = []

    # ... for 3-d regions...
    if ndim == 3:

        for i_region, region in enumerate(candidate_regions):
            image_region, _, coord_gaussian = _gaussian_mixture_3d(
                image=image,
                region=region,
                voxel_size_z=voxel_size[0],
                voxel_size_yx=voxel_size[-1],
                sigma_z=sigma[0],
                sigma_yx=sigma[-1],
                amplitude=amplitude,
                background=background,
                precomputed_gaussian=precomputed_gaussian)

            # get coordinates of spots and regions in the original image
            box = region.bbox
            (min_z, min_y, min_x, _, _, _) = box
            coord = np.array(coord_gaussian, dtype=np.float64)
            coord[:, 0] = (coord[:, 0] / voxel_size[0]) + min_z
            coord[:, 1] = (coord[:, 1] / voxel_size[-1]) + min_y
            coord[:, 2] = (coord[:, 2] / voxel_size[-1]) + min_x
            spots_in_region = np.zeros((coord.shape[0], 4), dtype=np.int64)
            spots_in_region[:, :3] = coord
            spots_in_region[:, 3] = i_region
            spots_in_regions.append(spots_in_region)
            region_z, region_y, region_x = tuple(coord[0])
            nb_rna_region = coord.shape[0]
            region_area = region.area
            region_intensity = region.mean_intensity
            regions.append([region_z, region_y, region_x, nb_rna_region,
                            region_area, region_intensity, i_region])

    # ... or 2-d regions
    else:

        for i_region, region in enumerate(candidate_regions):
            image_region, _, coord_gaussian = _gaussian_mixture_2d(
                image=image,
                region=region,
                voxel_size_yx=voxel_size[-1],
                sigma_yx=sigma[-1],
                amplitude=amplitude,
                background=background,
                precomputed_gaussian=precomputed_gaussian)

            # get coordinates of spots and regions in the original image
            box = region.bbox
            (min_y, min_x, _, _) = box
            coord = np.array(coord_gaussian, dtype=np.float64)
            coord[:, 0] = (coord[:, 0] / voxel_size[-1]) + min_y
            coord[:, 1] = (coord[:, 1] / voxel_size[-1]) + min_x
            spots_in_region = np.zeros((coord.shape[0], 3), dtype=np.int64)
            spots_in_region[:, :2] = coord
            spots_in_region[:, 2] = i_region
            spots_in_regions.append(spots_in_region)
            region_y, region_x = tuple(coord[0])
            nb_rna_region = coord.shape[0]
            region_area = region.area
            region_intensity = region.mean_intensity
            regions.append([region_y, region_x, nb_rna_region,
                            region_area, region_intensity, i_region])

    spots_in_regions = np.concatenate(spots_in_regions, axis=0)
    regions = np.array(regions, dtype=np.int64)

    return spots_in_regions, regions


def _gaussian_mixture_3d(image, region, voxel_size_z, voxel_size_yx, sigma_z,
                         sigma_yx, amplitude, background, precomputed_gaussian,
                         limit_gaussian=1000):
    """Fit as many 3-d gaussians as possible in a candidate region.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with detected spot and shape (z, y, x).
    region : skimage.measure._regionprops._RegionProperties
        Properties of a candidate region.
    voxel_size_z : int or float
        Size of a voxel along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel in the yx plan, in nanometer.
    sigma_z : int or float
        Standard deviation of the gaussian along the z axis, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian in the yx plan, in nanometer.
    amplitude : float
        Amplitude of the gaussian.
    background : float
        Background minimum value.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with tables of precomputed values for the erf, with shape
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
    image_region_raw = image_region_raw.astype(np.float64)

    # build a grid to represent this image
    grid = _initialize_grid_3d(image_region, voxel_size_z, voxel_size_yx)

    # add a gaussian for each local maximum while the RSS decreases
    simulation = np.zeros_like(image_region_raw)
    residual = image_region_raw - simulation
    ssr = np.sum(residual ** 2)
    diff_ssr = -1
    nb_gaussian = 0
    best_simulation = simulation.copy()
    positions_gaussian = []
    while diff_ssr < 0 or nb_gaussian == limit_gaussian:
        position_gaussian = np.argmax(residual)
        positions_gaussian.append(list(grid[:, position_gaussian]))
        simulation += gaussian_3d(
            grid=grid,
            mu_z=float(positions_gaussian[-1][0]),
            mu_y=float(positions_gaussian[-1][1]),
            mu_x=float(positions_gaussian[-1][2]),
            sigma_z=sigma_z,
            sigma_yx=sigma_yx,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            amplitude=amplitude,
            background=background,
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
        warnings.warn("Problem occurs during the decomposition of a dense "
                      "region. More than {0} spots seem to be necessary to "
                      "reproduce the candidate region and decomposition was "
                      "stopped early. Set a higher limit or check a potential "
                      "artifact in the image if you do not expect such a "
                      "large region to be decomposed.".format(limit_gaussian),
                      UserWarning)

    best_simulation = np.reshape(best_simulation, image_region.shape)
    max_value_dtype = np.iinfo(image_region.dtype).max
    best_simulation = np.clip(best_simulation, 0, max_value_dtype)
    best_simulation = best_simulation.astype(image_region.dtype)

    return image_region, best_simulation, positions_gaussian


def _gaussian_mixture_2d(image, region, voxel_size_yx, sigma_yx, amplitude,
                         background, precomputed_gaussian,
                         limit_gaussian=1000):
    """Fit as many 2-d gaussians as possible in a candidate region.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 2-d image with detected spot and shape (y, x).
    region : skimage.measure._regionprops._RegionProperties
        Properties of a candidate region.
    voxel_size_yx : int or float
        Size of a voxel in the yx plan, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian in the yx plan, in nanometer.
    amplitude : float
        Amplitude of the gaussian.
    background : float
        Background minimum value.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with tables of precomputed values for the erf, with shape
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
    image_region_raw = image_region_raw.astype(np.float64)

    # build a grid to represent this image
    grid = _initialize_grid_2d(image_region, voxel_size_yx)

    # add a gaussian for each local maximum while the RSS decreases
    simulation = np.zeros_like(image_region_raw)
    residual = image_region_raw - simulation
    ssr = np.sum(residual ** 2)
    diff_ssr = -1
    nb_gaussian = 0
    best_simulation = simulation.copy()
    positions_gaussian = []
    while diff_ssr < 0 or nb_gaussian == limit_gaussian:
        position_gaussian = np.argmax(residual)
        positions_gaussian.append(list(grid[:, position_gaussian]))
        simulation += gaussian_2d(
            grid=grid,
            mu_y=float(positions_gaussian[-1][0]),
            mu_x=float(positions_gaussian[-1][1]),
            sigma_yx=sigma_yx,
            voxel_size_yx=voxel_size_yx,
            amplitude=amplitude,
            background=background,
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
        warnings.warn("Problem occurs during the decomposition of a dense "
                      "region. More than {0} spots seem to be necessary to "
                      "reproduce the candidate region and decomposition was "
                      "stopped early. Set a higher limit or check a potential "
                      "artifact in the image if you do not expect such a "
                      "large region to be decomposed.".format(limit_gaussian),
                      UserWarning)

    best_simulation = np.reshape(best_simulation, image_region.shape)
    max_value_dtype = np.iinfo(image_region.dtype).max
    best_simulation = np.clip(best_simulation, 0, max_value_dtype)
    best_simulation = best_simulation.astype(image_region.dtype)

    return image_region, best_simulation, positions_gaussian
