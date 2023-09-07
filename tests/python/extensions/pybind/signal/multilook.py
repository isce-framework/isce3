import types

import numpy as np
import pytest

from isce3.signal import (
    multilook_summed,
    multilook_averaged,
    multilook_nodata,
)


@pytest.fixture(scope="module", params=[
    {"shape": (21, 20), "nlooks": (3, 5)},
    {"shape": (20, 21), "nlooks": (3, 5)}
])
def unit_test_params(request):
    params = types.SimpleNamespace()

    # Instantiate some parameters
    # Length and width of original and multilooked dataset
    # Number of looks along rows and columns
    # No data for floating 2D arrays
    length, width = request.param["shape"]
    params.shape = (length, width)

    row_looks, col_looks = request.param["nlooks"]
    params.looks = (row_looks, col_looks)

    length_looked = length // row_looks
    width_looked = width // col_looks
    params.looked_shape = (length_looked, width_looked)

    params.float_nodata_value = 0.0
    params.cpx_nodata_value = 0.0 + 0.0 * 1j

    # Generate a float and a complex 2D dataset to be used for testing
    length_grid, width_grid = np.meshgrid(
        np.arange(width), np.arange(length)
    )
    params.float_data = (length_grid * width_grid).astype(np.float32)
    params.cpx_data = (
        np.cos(params.float_data) + np.sin(params.float_data) * 1j
    ).astype(np.complex64)

    # Get a 2D array of weight which weights the center pixel more than the
    # pixels in its surroundings
    center_row = length // 2
    center_col = width // 2

    # Calculate maximum distance from the center pixel
    max_distance = max(center_row, center_col)

    # Compute distance from center per axis
    length_diff_grid = np.abs(length_grid - center_row)
    width_diff_grid = np.abs(width_grid - center_col)

    # Compute max distance and normalize with max distance from center pixel
    dist = np.maximum(length_diff_grid, width_diff_grid)
    params.weights = (1 - dist / max_distance).astype(np.float32)

    return params


def multilook(
    in_array,
    looks,
    looked_shape,
    weights=None,
    data_type=np.float32,
    multilook_type="summed",
):
    """
    Compute the expected multilooked array based on the type of multilook to
    perform
    """
    row_looks, col_looks = looks
    length_looked, width_looked = looked_shape

    out_array = np.zeros((length_looked, width_looked), dtype=data_type)
    for row in range(length_looked):
        for col in range(width_looked):
            if multilook_type == "summed":
                out_array[row, col] = np.sum(
                    in_array[
                        row * row_looks : (row + 1) * row_looks,
                        col * col_looks : (col + 1) * col_looks,
                    ]
                )
            elif multilook_type == "averaged":
                out_array[row, col] = np.mean(
                    in_array[
                        row * row_looks : (row + 1) * row_looks,
                        col * col_looks : (col + 1) * col_looks,
                    ]
                )
            elif multilook_type == "weighted_average":
                row_min = row * row_looks
                row_max = (row + 1) * row_looks
                col_min = col * col_looks
                col_max = (col + 1) * col_looks

                tile = in_array[row_min:row_max, col_min:col_max]
                wgts = weights[row_min:row_max, col_min:col_max]
                out_array[row, col] = np.sum(tile * wgts) / np.sum(wgts)
            else:
                err_str = f"{multilook_type} is not a valid type of multilook"
                raise ValueError(err_str)
    return out_array


def test_multilook_summed(unit_test_params):
    row_looks, col_looks = unit_test_params.looks

    for data, data_type in zip(
        [unit_test_params.float_data, unit_test_params.cpx_data],
        [np.float32, np.complex64],
    ):
        mlook_sum = multilook_summed(data, row_looks, col_looks)

        mlook_sum_exp = multilook(
            data,
            unit_test_params.looks,
            unit_test_params.looked_shape,
            data_type=data_type,
            multilook_type="summed",
        )

        max_err = np.max(np.abs(mlook_sum_exp - mlook_sum))
        assert max_err < 1.0e-6, f"{data_type} multilook sum max_err > 1.0e-6"


def test_multilook_averaged(unit_test_params):
    row_looks, col_looks = unit_test_params.looks

    for data, data_type in zip(
        [unit_test_params.float_data, unit_test_params.cpx_data],
        [np.float32, np.complex64],
    ):
        mlook_avg = multilook_averaged(data, row_looks, col_looks)

        mlook_avg_exp = multilook(
            data,
            unit_test_params.looks,
            unit_test_params.looked_shape,
            data_type=data_type,
            multilook_type="averaged",
        )

        max_err = np.max(np.abs(mlook_avg_exp - mlook_avg))
        assert max_err < 1.0e-6, f"{data_type} multilook avg max_err > 1.0e-6"


def test_multilook_nodata(unit_test_params):
    row_looks, col_looks = unit_test_params.looks

    for data, data_type, no_data_val in zip(
        [unit_test_params.float_data, unit_test_params.cpx_data],
        [np.float32, np.complex64],
        [
            unit_test_params.float_nodata_value,
            unit_test_params.cpx_nodata_value,
        ],
    ):
        no_data_nodata = multilook_nodata(
            unit_test_params.float_data, row_looks, col_looks, no_data_val
        )

        weights_float = unit_test_params.float_data != no_data_val

        no_data_nodata_exp = multilook(
            unit_test_params.float_data,
            unit_test_params.looks,
            unit_test_params.looked_shape,
            weights=weights_float,
            data_type=data_type,
            multilook_type="weighted_average",
        )

        no_data_max_err = np.max(np.abs(no_data_nodata_exp - no_data_nodata))
        assert (
            no_data_max_err < 1.0e-6
        ), f"{data_type} multilook no data max_err > 1.0e-6"
