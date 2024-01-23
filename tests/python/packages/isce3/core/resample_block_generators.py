import numpy as np
from pytest import mark
from isce3.core.resample_block_generators import get_blocks_by_offsets


def test_get_blocks_by_offsets_all_nans():
    dim = 200

    block_slice = np.index_exp[0:dim, 0:dim]

    off_block = np.full(shape=(dim, dim), fill_value=np.nan, dtype=np.float64)

    ret_val = get_blocks_by_offsets(
        out_block_slices=block_slice,
        y_offsets_block=off_block,
        x_offsets_block=off_block,
        in_grid_shape=(dim, dim),
        buffer=0,
    )

    assert ret_val is None


def test_get_blocks_by_offsets_all_off_grid():
    dim = 200

    block_slice = np.index_exp[0:dim, 0:dim]

    off_block = np.full(shape=(dim, dim), fill_value=dim * 2, dtype=np.float64)

    ret_val = get_blocks_by_offsets(
        out_block_slices=block_slice,
        y_offsets_block=off_block,
        x_offsets_block=off_block,
        in_grid_shape=(dim, dim),
        buffer=0,
    )

    assert ret_val is None


@mark.parametrize("buffer", [0, 5])
def test_get_blocks_by_offsets(buffer: int):
    dim = 300

    block_slice = np.index_exp[0 : dim // 3, 0 : dim // 3]
    off_shape = (dim // 3, dim // 3)

    off_block = np.full(shape=off_shape, fill_value=dim // 3, dtype=np.float64)

    ret_val = get_blocks_by_offsets(
        out_block_slices=block_slice,
        y_offsets_block=off_block,
        x_offsets_block=off_block,
        in_grid_shape=(dim, dim),
        buffer=buffer,
    )

    assert ret_val is not None

    slice_az, slice_rg = ret_val

    assert isinstance(slice_az, slice)
    assert slice_az.start == dim // 3 - buffer
    assert slice_az.stop == 2 * dim // 3 + buffer

    assert isinstance(slice_rg, slice)
    assert slice_rg.start == dim // 3 - buffer
    assert slice_rg.stop == 2 * dim // 3 + buffer
