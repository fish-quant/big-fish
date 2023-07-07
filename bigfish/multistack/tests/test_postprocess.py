# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.multistack.postprocess module.
"""

import pytest

import numpy as np

import bigfish.multistack as multistack

from numpy.testing import assert_array_equal


# TODO add test bigfish.multistack.match_nuc_cell
# TODO add test bigfish.multistack.extract_spots_from_frame

# TODO add test bigfish.multistack.center_mask_coord
# TODO add test bigfish.multistack.from_boundaries_to_surface
# TODO add test bigfish.multistack.from_surface_to_boundaries
# TODO add test bigfish.multistack.from_coord_to_frame
# TODO add test bigfish.multistack.from_coord_to_surface


@pytest.mark.parametrize(
    "mask_dtype", [np.uint8, np.uint16, np.int32, np.int64, bool]
)
@pytest.mark.parametrize(
    "spot_dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_identify_objects_in_region_exceptions(mask_dtype, spot_dtype):
    # simulate mask and coordinates
    mask = np.zeros((10, 10, 10), dtype=mask_dtype)
    mask[0:3, 1:4, 1:5] = np.ones((3, 3, 4), dtype=mask_dtype)
    spots_in = [[2, 1, 1], [5, 2, 2], [2, 3, 1], [9, 3, 4]]
    spots_in = np.array(spots_in, dtype=spot_dtype)
    spots_out = [[1, 0, 0], [3, 7, 2], [2, 1, 8], [7, 8, 8]]
    spots_out = np.array(spots_out, dtype=spot_dtype)
    spots_3d = np.concatenate((spots_in, spots_out))
    spots_2d = spots_3d[:, 1:]

    # test
    with pytest.raises(ValueError):  # ndim 4
        multistack.identify_objects_in_region(mask, spots_3d, ndim=4)
    with pytest.raises(ValueError):  # coord 2d, ndim 3
        multistack.identify_objects_in_region(mask, spots_2d, ndim=3)
    with pytest.raises(ValueError):  # mask 3d, coord 2d, ndim 2
        multistack.identify_objects_in_region(mask, spots_2d, ndim=2)
    with pytest.raises(ValueError):  # mask 3d, coord 3d, ndim 2
        multistack.identify_objects_in_region(mask, spots_3d, ndim=2)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "mask_dtype", [np.uint8, np.uint16, np.int32, np.int64, bool]
)
@pytest.mark.parametrize(
    "spot_dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_identify_objects_in_region_2d_mask(ndim, mask_dtype, spot_dtype):
    # simulate mask and coordinates
    mask = np.zeros((10, 10), dtype=mask_dtype)
    mask[1:4, 1:5] = np.ones((3, 4), dtype=mask_dtype)
    spots_in = [[2, 1, 1], [5, 2, 2], [2, 3, 1], [9, 3, 4]]
    spots_in = np.array(spots_in, dtype=spot_dtype)
    spots_out = [[1, 0, 0], [3, 7, 2], [2, 1, 8], [7, 8, 8]]
    spots_out = np.array(spots_out, dtype=spot_dtype)
    spots_3d = np.concatenate((spots_in, spots_out))
    spots_2d = spots_3d[:, 1:]
    if ndim == 2:
        spots_in = spots_in[:, 1:]
        spots_out = spots_out[:, 1:]

    # test
    if ndim == 2:
        # mask 2d, coord 2d, ndim 2
        spots_in_, spots_out_ = multistack.identify_objects_in_region(
            mask, spots_2d, ndim
        )
        assert_array_equal(spots_in_, spots_in)
        assert_array_equal(spots_out_, spots_out)
        assert spots_in_.dtype == spots_in.dtype
        assert spots_out_.dtype == spots_out.dtype
    elif ndim == 3:
        # mask 2d, coord 3d, ndim 3
        spots_in_, spots_out_ = multistack.identify_objects_in_region(
            mask, spots_3d, ndim
        )
        assert_array_equal(spots_in_, spots_in)
        assert_array_equal(spots_out_, spots_out)
        assert spots_in_.dtype == spots_in.dtype
        assert spots_out_.dtype == spots_out.dtype


@pytest.mark.parametrize("ndim", [3])
@pytest.mark.parametrize(
    "mask_dtype", [np.uint8, np.uint16, np.int32, np.int64, bool]
)
@pytest.mark.parametrize(
    "spot_dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_identify_objects_in_region_3d_mask(ndim, mask_dtype, spot_dtype):
    # simulate mask and coordinates
    mask = np.zeros((10, 10, 10), dtype=mask_dtype)
    mask[0:3, 1:4, 1:5] = np.ones((3, 3, 4), dtype=mask_dtype)
    spots_in = [[2, 1, 1], [2, 3, 1]]
    spots_in = np.array(spots_in, dtype=spot_dtype)
    spots_out = [
        [1, 0, 0],
        [3, 7, 2],
        [2, 1, 8],
        [7, 8, 8],
        [5, 2, 2],
        [9, 3, 4],
    ]
    spots_out = np.array(spots_out, dtype=spot_dtype)
    spots_3d = np.concatenate((spots_in, spots_out))

    # test mask 3d, coord 3d, ndim 3
    spots_in_, spots_out_ = multistack.identify_objects_in_region(
        mask, spots_3d, ndim
    )
    assert_array_equal(spots_in_, spots_in)
    assert_array_equal(spots_out_, spots_out)
    assert spots_in_.dtype == spots_in.dtype
    assert spots_out_.dtype == spots_out.dtype


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "mask_dtype", [np.uint8, np.uint16, np.int32, np.int64, bool]
)
@pytest.mark.parametrize(
    "spot_dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_remove_transcription_site_2d_mask(ndim, mask_dtype, spot_dtype):
    # simulate mask and coordinates
    nuc_mask = np.zeros((10, 10), dtype=mask_dtype)
    nuc_mask[1:4, 1:5] = np.ones((3, 4), dtype=mask_dtype)

    spots_in_nuc_out_ts = [[2, 1, 1, -1], [5, 2, 2, -1], [2, 3, 1, -1]]
    spots_in_nuc_out_ts = np.array(spots_in_nuc_out_ts, dtype=spot_dtype)

    spots_in_nuc_in_ts = [[5, 2, 4, 2], [2, 3, 3, 3], [2, 1, 3, 3]]
    spots_in_nuc_in_ts = np.array(spots_in_nuc_in_ts, dtype=spot_dtype)

    spots_out_nuc_out_foci = [[1, 0, 0, -1], [3, 7, 2, -1], [2, 1, 8, -1]]
    spots_out_nuc_out_foci = np.array(spots_out_nuc_out_foci, dtype=spot_dtype)

    spots_out_nuc_in_foci = [[1, 0, 5, 0], [3, 0, 7, 0], [2, 1, 5, 1]]
    spots_out_nuc_in_foci = np.array(spots_out_nuc_in_foci, dtype=spot_dtype)

    if ndim == 2:
        spots_in_nuc_out_ts = spots_in_nuc_out_ts[:, 1:]
        spots_in_nuc_in_ts = spots_in_nuc_in_ts[:, 1:]
        spots_out_nuc_out_foci = spots_out_nuc_out_foci[:, 1:]
        spots_out_nuc_in_foci = spots_out_nuc_in_foci[:, 1:]
    spots = np.concatenate(
        (
            spots_in_nuc_out_ts,
            spots_in_nuc_in_ts,
            spots_out_nuc_out_foci,
            spots_out_nuc_in_foci,
        )
    )
    spots_out_ts = np.concatenate(
        (spots_in_nuc_out_ts, spots_out_nuc_out_foci, spots_out_nuc_in_foci)
    )

    ts = [[5, 2, 4, 1, 2], [2, 2, 3, 2, 3]]
    ts = np.array(ts, dtype=spot_dtype)
    foci = [[2, 0, 6, 2, 0], [2, 1, 5, 1, 1]]
    foci = np.array(foci, dtype=spot_dtype)
    if ndim == 2:
        ts = ts[:, 1:]
        foci = foci[:, 1:]
    all_clusters = np.concatenate((ts, foci))

    # test
    spots_out_ts_, foci_, ts_ = multistack.remove_transcription_site(
        spots, all_clusters, nuc_mask, ndim
    )
    assert_array_equal(spots_out_ts_, spots_out_ts)
    assert spots_out_ts_.dtype == spots_out_ts.dtype
    assert_array_equal(foci_, foci)
    assert foci_.dtype == foci.dtype
    assert_array_equal(ts_, ts)
    assert ts_.dtype == ts.dtype


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "mask_dtype", [np.uint8, np.uint16, np.int32, np.int64, bool]
)
@pytest.mark.parametrize(
    "spot_dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_remove_transcription_site_3d_mask(ndim, mask_dtype, spot_dtype):
    # simulate mask and coordinates
    nuc_mask = np.zeros((10, 10, 10), dtype=mask_dtype)
    nuc_mask[0:3, 1:4, 1:5] = np.ones((3, 3, 4), dtype=mask_dtype)

    spots_in_nuc_out_ts = [[2, 1, 1, -1], [2, 3, 1, -1]]
    spots_in_nuc_out_ts = np.array(spots_in_nuc_out_ts, dtype=spot_dtype)

    spots_in_nuc_in_ts = [[2, 3, 3, 3], [2, 1, 3, 3]]
    spots_in_nuc_in_ts = np.array(spots_in_nuc_in_ts, dtype=spot_dtype)

    spots_out_nuc_out_foci = [
        [1, 0, 0, -1],
        [3, 7, 2, -1],
        [2, 1, 8, -1],
        [5, 2, 2, -1],
    ]
    spots_out_nuc_out_foci = np.array(spots_out_nuc_out_foci, dtype=spot_dtype)

    spots_out_nuc_in_foci = [
        [1, 0, 5, 0],
        [3, 0, 7, 0],
        [2, 1, 5, 1],
        [5, 2, 4, 2],
    ]
    spots_out_nuc_in_foci = np.array(spots_out_nuc_in_foci, dtype=spot_dtype)

    spots = np.concatenate(
        (
            spots_in_nuc_out_ts,
            spots_in_nuc_in_ts,
            spots_out_nuc_out_foci,
            spots_out_nuc_in_foci,
        )
    )
    spots_out_ts = np.concatenate(
        (spots_in_nuc_out_ts, spots_out_nuc_out_foci, spots_out_nuc_in_foci)
    )

    ts = [[2, 2, 3, 2, 3]]
    ts = np.array(ts, dtype=spot_dtype)
    foci = [[2, 0, 6, 2, 0], [2, 1, 5, 1, 1], [5, 2, 4, 1, 2]]
    foci = np.array(foci, dtype=spot_dtype)
    all_clusters = np.concatenate((ts, foci))

    # test
    if ndim == 2:
        spots = spots[:, 1:]
        all_clusters = all_clusters[:, 1:]
        with pytest.raises(ValueError):
            multistack.remove_transcription_site(
                spots, all_clusters, nuc_mask, ndim
            )
    else:
        spots_out_ts_, foci_, ts_ = multistack.remove_transcription_site(
            spots, all_clusters, nuc_mask, ndim
        )
        assert_array_equal(spots_out_ts_, spots_out_ts)
        assert spots_out_ts_.dtype == spots_out_ts.dtype
        assert_array_equal(foci_, foci)
        assert foci_.dtype == foci.dtype
        assert_array_equal(ts_, ts)
        assert ts_.dtype == ts.dtype


@pytest.mark.parametrize(
    "binary_dtype", [np.uint8, np.uint16, np.int32, np.int64, bool]
)
def test_from_binary_to_coord(binary_dtype):
    # simulate binary
    binary_2d = np.zeros((10, 10), dtype=binary_dtype)
    binary_2d[1:4, 1:5] = np.ones((3, 4), dtype=binary_dtype)
    binary_3d = np.zeros((10, 10, 10), dtype=binary_dtype)
    binary_3d[1:6, 1:4, 1:5] = np.ones((5, 3, 4), dtype=binary_dtype)

    # test
    coord_2d = multistack.from_binary_to_coord(binary_2d)
    assert coord_2d.shape[1] == 2
    if binary_dtype == np.int32:
        assert coord_2d.dtype == np.int32
    else:
        assert coord_2d.dtype == np.int64

    coord_3d = multistack.from_binary_to_coord(binary_3d)
    assert coord_3d.shape[1] == 3
    if binary_dtype == np.int32:
        assert coord_3d.dtype == np.int32
    else:
        assert coord_3d.dtype == np.int64


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "mask_dtype", [np.uint8, np.uint16, np.int32, np.int64]
)
@pytest.mark.parametrize(
    "spot_dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_extract_cell_2d_mask(mask_dtype, spot_dtype, ndim):
    # simulate mask and coordinates
    cell_label = np.zeros((10, 10), dtype=mask_dtype)
    cell_label[1:4, 1:5] = np.ones((3, 4), dtype=mask_dtype)
    nuc_label = np.zeros((10, 10), dtype=mask_dtype)
    nuc_label[2:3, 2:4] = np.ones((1, 2), dtype=mask_dtype)

    spots_in = [[2, 1, 1, -1], [2, 3, 1, -1], [5, 2, 2, -1]]
    spots_in = np.array(spots_in, dtype=spot_dtype)
    spots_out = [[2, 1, 8, -1], [3, 7, 2, -1]]
    spots_out = np.array(spots_out, dtype=spot_dtype)

    if ndim == 2:
        spots_in = spots_in[:, 1:]
        spots_out = spots_out[:, 1:]
    spots = np.concatenate((spots_in, spots_out))

    # test
    spots_in[:, ndim - 2] -= 1
    spots_in[:, ndim - 1] -= 1
    fov_results = multistack.extract_cell(
        cell_label,
        ndim,
        nuc_label=nuc_label,
        rna_coord=spots,
        others_coord={"rna_coord_bis": spots},
        image=cell_label.astype(np.uint8),
        others_image={"image_bis": cell_label.astype(np.uint8)},
        remove_cropped_cell=True,
        check_nuc_in_cell=True,
    )
    assert len(fov_results) == 1
    for key in [
        "cell_id",
        "bbox",
        "cell_coord",
        "cell_mask",
        "nuc_coord",
        "nuc_mask",
        "rna_coord",
        "rna_coord_bis",
        "image",
        "image_bis",
    ]:
        assert key in fov_results[0]
    assert fov_results[0]["cell_id"] == 1
    assert fov_results[0]["bbox"] == (1, 1, 4, 5)
    assert fov_results[0]["cell_coord"].shape[1] == 2
    assert_array_equal(
        fov_results[0]["cell_mask"], cell_label[1:4, 1:5].astype(bool)
    )
    assert fov_results[0]["nuc_coord"].shape[1] == 2
    assert_array_equal(
        fov_results[0]["nuc_mask"], nuc_label[1:4, 1:5].astype(bool)
    )
    assert_array_equal(fov_results[0]["rna_coord"], spots_in)
    assert_array_equal(fov_results[0]["rna_coord_bis"], spots_in)
    assert_array_equal(
        fov_results[0]["image"], cell_label[1:4, 1:5].astype(np.uint8)
    )
    assert_array_equal(
        fov_results[0]["image_bis"], cell_label[1:4, 1:5].astype(np.uint8)
    )

    # test exceptions
    with pytest.raises(KeyError):  # duplicated key 'rna_coord'
        multistack.extract_cell(
            cell_label,
            ndim,
            nuc_label=nuc_label,
            rna_coord=spots,
            others_coord={"rna_coord": spots},
            image=cell_label.astype(np.uint8),
            remove_cropped_cell=False,
            check_nuc_in_cell=False,
        )
    with pytest.raises(KeyError):  # duplicated key 'image'
        multistack.extract_cell(
            cell_label,
            ndim,
            nuc_label=nuc_label,
            rna_coord=spots,
            image=cell_label.astype(np.uint8),
            others_image={"image": cell_label.astype(np.uint8)},
            remove_cropped_cell=False,
            check_nuc_in_cell=False,
        )
    with pytest.warns(UserWarning):  # 'image_bis' with different shape
        multistack.extract_cell(
            cell_label,
            ndim,
            nuc_label=nuc_label,
            rna_coord=spots,
            image=cell_label.astype(np.uint8),
            others_image={"image_bis": cell_label[1:, :].astype(np.uint8)},
            remove_cropped_cell=False,
            check_nuc_in_cell=False,
        )

    # test removed cropped cells
    cell_label = np.zeros((10, 10), dtype=mask_dtype)
    cell_label[0:4, 1:5] = np.ones((4, 4), dtype=mask_dtype)
    fov_results = multistack.extract_cell(
        cell_label,
        ndim,
        nuc_label=nuc_label,
        rna_coord=spots,
        image=cell_label.astype(np.uint8),
        remove_cropped_cell=True,
        check_nuc_in_cell=False,
    )
    assert len(fov_results) == 0

    # test nuc in cell
    cell_label = np.zeros((10, 10), dtype=mask_dtype)
    cell_label[1:4, 1:5] = np.ones((3, 4), dtype=mask_dtype)
    nuc_label = np.zeros((10, 10), dtype=mask_dtype)
    nuc_label[6:8, 5:8] = np.ones((2, 3), dtype=mask_dtype)
    fov_results = multistack.extract_cell(
        cell_label,
        ndim,
        nuc_label=nuc_label,
        rna_coord=spots,
        image=cell_label.astype(np.uint8),
        remove_cropped_cell=False,
        check_nuc_in_cell=True,
    )
    assert len(fov_results) == 0


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "mask_dtype", [np.uint8, np.uint16, np.int32, np.int64]
)
@pytest.mark.parametrize(
    "spot_dtype", [np.float32, np.float64, np.int32, np.int64]
)
def test_extract_cell_3d_mask(mask_dtype, spot_dtype, ndim):
    # simulate mask and coordinates
    cell_label = np.zeros((10, 10, 10), dtype=mask_dtype)
    cell_label[1:6, 1:4, 1:5] = np.ones((5, 3, 4), dtype=mask_dtype)
    nuc_label = np.zeros((10, 10, 10), dtype=mask_dtype)
    nuc_label[2:4, 2:3, 2:4] = np.ones((2, 1, 2), dtype=mask_dtype)

    spots_in = [[2, 1, 1, -1], [2, 3, 1, -1], [5, 2, 2, -1]]
    spots_in = np.array(spots_in, dtype=spot_dtype)
    spots_out = [[2, 1, 8, -1], [3, 7, 2, -1]]
    spots_out = np.array(spots_out, dtype=spot_dtype)

    if ndim == 2:
        spots_in = spots_in[:, 1:]
        spots_out = spots_out[:, 1:]
    spots = np.concatenate((spots_in, spots_out))

    # test
    if ndim == 3:
        spots_in[:, 0] -= 1
        spots_in[:, 1] -= 1
        spots_in[:, 2] -= 1
        fov_results = multistack.extract_cell(
            cell_label,
            ndim,
            nuc_label=nuc_label,
            rna_coord=spots,
            others_coord={"rna_coord_bis": spots},
            image=cell_label.astype(np.uint8),
            others_image={"image_bis": cell_label.astype(np.uint8)},
            remove_cropped_cell=True,
            check_nuc_in_cell=True,
        )
        assert len(fov_results) == 1
        for key in [
            "cell_id",
            "bbox",
            "cell_coord",
            "cell_mask",
            "nuc_coord",
            "nuc_mask",
            "rna_coord",
            "rna_coord_bis",
            "image",
            "image_bis",
        ]:
            assert key in fov_results[0]
        assert fov_results[0]["cell_id"] == 1
        assert fov_results[0]["bbox"] == (1, 1, 1, 6, 4, 5)
        assert fov_results[0]["cell_coord"].shape[1] == 3
        assert_array_equal(
            fov_results[0]["cell_mask"], cell_label[1:6, 1:4, 1:5].astype(bool)
        )
        assert fov_results[0]["nuc_coord"].shape[1] == 3
        assert_array_equal(
            fov_results[0]["nuc_mask"], nuc_label[1:6, 1:4, 1:5].astype(bool)
        )
        assert_array_equal(fov_results[0]["rna_coord"], spots_in)
        assert_array_equal(fov_results[0]["rna_coord_bis"], spots_in)
        assert_array_equal(
            fov_results[0]["image"], cell_label[1:6, 1:4, 1:5].astype(np.uint8)
        )
        assert_array_equal(
            fov_results[0]["image_bis"],
            cell_label[1:6, 1:4, 1:5].astype(np.uint8),
        )
    else:
        with pytest.raises(ValueError):
            multistack.extract_cell(
                cell_label,
                ndim,
                nuc_label=nuc_label,
                rna_coord=spots,
                image=cell_label.astype(np.uint8),
                remove_cropped_cell=False,
                check_nuc_in_cell=False,
            )

    # test exceptions
    with pytest.raises(ValueError):  # cell_label.ndim != nuc_label.ndim
        multistack.extract_cell(
            cell_label,
            ndim,
            nuc_label=nuc_label[3, :, :],
            rna_coord=spots,
            image=cell_label.astype(np.uint8),
            remove_cropped_cell=False,
            check_nuc_in_cell=False,
        )

    # test removed cropped cells
    cell_label = np.zeros((10, 10, 10), dtype=mask_dtype)
    cell_label[0:6, 1:4, 1:5] = np.ones((6, 3, 4), dtype=mask_dtype)
    fov_results = multistack.extract_cell(
        cell_label,
        ndim,
        nuc_label=nuc_label,
        rna_coord=spots,
        image=cell_label.astype(np.uint8),
        remove_cropped_cell=True,
        check_nuc_in_cell=False,
    )
    assert len(fov_results) == 0

    # test nuc in cell
    cell_label = np.zeros((10, 10, 10), dtype=mask_dtype)
    cell_label[1:6, 1:4, 1:5] = np.ones((5, 3, 4), dtype=mask_dtype)
    nuc_label = np.zeros((10, 10, 10), dtype=mask_dtype)
    nuc_label[2:4, 6:8, 5:8] = np.ones((2, 2, 3), dtype=mask_dtype)
    fov_results = multistack.extract_cell(
        cell_label,
        ndim,
        nuc_label=nuc_label,
        rna_coord=spots,
        image=cell_label.astype(np.uint8),
        remove_cropped_cell=False,
        check_nuc_in_cell=True,
    )
    assert len(fov_results) == 0


def test_summarize_extraction_results():
    # simulate mask and coordinates
    cell_label_3d = np.zeros((10, 10, 10), dtype=np.int64)
    cell_label_3d[1:6, 1:4, 1:5] = np.ones((5, 3, 4), dtype=np.int64)
    nuc_label_3d = np.zeros((10, 10, 10), dtype=np.int64)
    nuc_label_3d[2:4, 2:3, 2:4] = np.ones((2, 1, 2), dtype=np.int64)

    cell_label_2d = cell_label_3d[2, :, :]
    nuc_label_2d = nuc_label_3d[2, :, :]

    spots_in = [[2, 2, 2, -1], [2, 3, 1, -1], [5, 2, 2, -1]]
    spots_in = np.array(spots_in, dtype=np.int64)
    spots_out = [[2, 1, 8, -1], [3, 7, 2, -1]]
    spots_out = np.array(spots_out, dtype=np.int64)
    spots = np.concatenate((spots_in, spots_out))

    # test mask 2d, spots 2d
    fov_results = multistack.extract_cell(
        cell_label_2d,
        ndim=2,
        nuc_label=nuc_label_2d,
        rna_coord=spots[:, 1:],
        remove_cropped_cell=True,
        check_nuc_in_cell=True,
    )
    df = multistack.summarize_extraction_results(fov_results, ndim=2)
    for key in [
        "cell_id",
        "cell_area",
        "cell_volume",
        "nuc_area",
        "nuc_volume",
        "nb_rna",
        "nb_rna_in_nuc",
        "nb_rna_out_nuc",
    ]:
        assert key in df.columns
    assert df.shape == (1, 8)
    assert df.loc[0, "cell_id"] == 1
    assert df.loc[0, "cell_area"] == 12
    assert np.isnan(df.loc[0, "cell_volume"])
    assert df.loc[0, "nuc_area"] == 2
    assert np.isnan(df.loc[0, "nuc_volume"])
    assert df.loc[0, "nb_rna"] == 3
    assert df.loc[0, "nb_rna_in_nuc"] == 2
    assert df.loc[0, "nb_rna_out_nuc"] == 1

    # test mask 2d, spots 3d
    fov_results = multistack.extract_cell(
        cell_label_2d,
        ndim=3,
        nuc_label=nuc_label_2d,
        rna_coord=spots,
        remove_cropped_cell=True,
        check_nuc_in_cell=True,
    )
    df = multistack.summarize_extraction_results(fov_results, ndim=3)
    assert df.shape == (1, 8)
    assert df.loc[0, "cell_id"] == 1
    assert df.loc[0, "cell_area"] == 12
    assert np.isnan(df.loc[0, "cell_volume"])
    assert df.loc[0, "nuc_area"] == 2
    assert np.isnan(df.loc[0, "nuc_volume"])
    assert df.loc[0, "nb_rna"] == 3
    assert df.loc[0, "nb_rna_in_nuc"] == 2
    assert df.loc[0, "nb_rna_out_nuc"] == 1

    # test mask 3d, spots 3d
    fov_results = multistack.extract_cell(
        cell_label_3d,
        ndim=3,
        nuc_label=nuc_label_3d,
        rna_coord=spots,
        remove_cropped_cell=True,
        check_nuc_in_cell=True,
    )
    df = multistack.summarize_extraction_results(fov_results, ndim=3)
    for key in [
        "cell_id",
        "cell_area",
        "cell_volume",
        "nuc_area",
        "nuc_volume",
        "nb_rna",
        "nb_rna_in_nuc",
        "nb_rna_out_nuc",
    ]:
        assert key in df.columns
    assert df.shape == (1, 8)
    assert df.loc[0, "cell_id"] == 1
    assert df.loc[0, "cell_volume"] == 60
    assert df.loc[0, "cell_area"] == 12
    assert df.loc[0, "nuc_volume"] == 4
    assert df.loc[0, "nuc_area"] == 2
    assert df.loc[0, "nb_rna"] == 3
    assert df.loc[0, "nb_rna_in_nuc"] == 1
    assert df.loc[0, "nb_rna_out_nuc"] == 2

    # test empty
    cell_label_3d = np.zeros((10, 10, 10), dtype=np.int64)
    cell_label_3d[0:6, 1:4, 1:5] = np.ones((6, 3, 4), dtype=np.int64)
    fov_results = multistack.extract_cell(
        cell_label_3d,
        ndim=3,
        nuc_label=nuc_label_3d,
        rna_coord=spots,
        remove_cropped_cell=True,
        check_nuc_in_cell=True,
    )
    df = multistack.summarize_extraction_results(fov_results, ndim=3)
    assert "cell_id" in df.columns
    assert df.shape == (0, 8)
