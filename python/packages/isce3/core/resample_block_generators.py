from __future__ import annotations

import journal
import numpy as np
from collections.abc import Iterator


def get_blocks(
    block_max_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    quiet: bool = False,
) -> Iterator[tuple[slice, slice]]:
    """
    Returns an iterator over regularly-sized blocks on a given grid.

    Parameters
    ----------
    block_max_shape : tuple[int, int]
        The largest extent that any block should have in dimensions of length and width.
    grid_shape : tuple[int, int]
        The length and width of the entire grid to be blocked over
    quiet : bool, optional
        If True, don't log informational block numbers. Defaults to False.

    Yields
    ------
    out_block_slice : tuple[slice, slice]
        The index expression object that describes the extent of a slice on the grid.
    """
    info_channel = journal.info("resample_block_generators.get_blocks")

    block_max_length, block_max_width = block_max_shape

    # Get the dimensions of the grid.
    grid_length, grid_width = grid_shape

    # Get total number of blocks in both directions and in total.
    num_blocks_len = int(np.ceil(grid_length / block_max_length))
    num_blocks_wid = int(np.ceil(grid_width / block_max_width))
    blocks = num_blocks_len * num_blocks_wid

    # Iterate over the number of blocks in y/length dimension.
    for block_index_y in range(num_blocks_len):
        # Compute y dimensions of this block.
        y_start_index = block_max_length * block_index_y
        y_end_index = min(y_start_index + block_max_length, grid_length)

        # Iterate over the number of blocks in x/width dimension.
        for block_index_x in range(num_blocks_wid):
            if not quiet:
                # Get the block number.
                block_number = block_index_x + block_index_y * num_blocks_wid + 1
                info_channel.log(f"Block #{block_number} of {blocks} total.")

            # Compute x dimensions of this block.
            x_start_index = block_index_x * block_max_width
            x_end_index = min(x_start_index + block_max_width, grid_width)

            # Create the slice and shape for this block.
            grid_block_slice = np.index_exp[
                y_start_index:y_end_index, x_start_index:x_end_index
            ]

            # Return the output block shape and index expression.
            yield grid_block_slice


def get_blocks_by_offsets(
    out_block_slices: tuple[slice, slice],
    y_offsets_block: np.ndarray,
    x_offsets_block: np.ndarray,
    in_grid_shape: tuple[int, int],
    buffer: int = 0,
) -> tuple[slice, slice] | None:
    """
    Compute the input block of data that corresponds to the specified output
    block, given the pixel offsets between the input and output.

    Returns an empty slice in the case that there is no overlap between the input grid
    and output block in either direction.

    Parameters
    ----------
    out_block_slices : tuple[slice, slice]
        The slices of output grid data for which this input data is being collected.
    y_offsets_block : np.ndarray
        The array for this slice of the y offsets grid.
    x_offsets_block : np.ndarray
        The array for this slice of the x offsets grid.
    in_grid_shape : tuple[int, int]
        The shape of the input data raster.
    buffer : int, optional
        The size of the additional buffer to add to the slice in each direction.
        Must be >= 0. Defaults to 0.

    Returns
    -------
    data_slice : tuple[slice, slice] | None
        A tuple of slices that describes the extent of the slice on the data reading
        grid. Returns None if one or both offsets blocks contains all NaNs or all data
        points somewhere that is outside of the data grid.
    """
    warn_channel = journal.warning("resample_block_generators.get_blocks_offset")
    error_channel = journal.error("resample_block_generators.get_blocks_offset")

    if buffer < 0:
        err_str = f"Buffer given as {buffer}; must be at least 0."
        error_channel.log(err_str)
        raise ValueError(err_str)
    
    # Check for all-NaN slices
    if np.all(np.isnan(x_offsets_block)) or np.all(np.isnan(y_offsets_block)):
        warn_channel.log(f"All-NaN offsets encountered in block: {out_block_slices}")
        return None

    # Get the extremities of the slice.
    y_slice, x_slice = out_block_slices

    y_start_index = y_slice.start
    y_end_index = y_slice.stop
    x_start_index = x_slice.start
    x_end_index = x_slice.stop

    # Get the dimensions of the data grid.
    data_length, data_width = in_grid_shape

    # Compute the minimum and maximum possible extents from the offsets blocks by
    # getting their extreme values, taking the ceiling or floor, adding or subtracting
    # a buffer, and then adding these values to their respective start or end values.
    # In this way, the largest possible slice of the output grid can be calculated.
    y_off_min = y_start_index + np.floor(np.nanmin(y_offsets_block)) - buffer
    y_off_max = y_end_index + np.ceil(np.nanmax(y_offsets_block)) + buffer
    x_off_min = x_start_index + np.floor(np.nanmin(x_offsets_block)) - buffer
    x_off_max = x_end_index + np.ceil(np.nanmax(x_offsets_block)) + buffer

    # Clip the output shape to the dimensions of the data grid.
    data_block_y_start = int(max(0, y_off_min))
    data_block_y_end = int(min(data_length, y_off_max))
    data_block_x_start = int(max(0, x_off_min))
    data_block_x_end = int(min(data_width, x_off_max))

    # If there is no overlap between the input grid and the output block in either
    # direction, return None.
    if data_block_y_start >= data_block_y_end or data_block_x_start >= data_block_x_end:
        warn_channel.log(
            "No overlap between input grid and output block for block "
            f"{out_block_slices}"
        )
        return None

    # Return the data grid slice.
    data_slice = np.index_exp[
        data_block_y_start:data_block_y_end, data_block_x_start:data_block_x_end
    ]
    return data_slice
