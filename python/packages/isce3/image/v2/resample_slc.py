from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import journal
import numpy as np
from isce3.core import SINC_HALF, LUT2d
from isce3.core.resample_block_generators import get_blocks, get_blocks_by_offsets
from isce3.ext.isce3.image.v2 import resample_to_coords
from isce3.io.dataset import DatasetReader, DatasetWriter
from isce3.product import RadarGridParameters


def resample_slc_blocks(
    output_resampled_slcs: Sequence[DatasetWriter],
    input_slcs: Sequence[DatasetReader],
    az_offsets_dataset: DatasetReader,
    rg_offsets_dataset: DatasetReader,
    input_radar_grid: RadarGridParameters,
    doppler: LUT2d = LUT2d(),
    block_size_az: int = 1024,
    block_size_rg: int = 1024,
    quiet: bool = False,
    fill_value: np.complex64 = np.nan + 1.0j * np.nan,
) -> None:
    """
    Resamples one or more SLCs onto a geometry described by given offsets datasets.

    The dimensions of the input datasets should all be the same, and likewise with the
    output datasets. The `az_offsets_dataset` and `rg_offsets_dataset` should be the
    same shape as the datasets in `output_resampled_slcs`.

    Parameters
    ----------
    output_resampled_slcs : Sequence[DatasetWriter]
        A set of one or more datasets to output to. Will be overwritten.
    input_slcs : Sequence[DatasetReader]
        A set of the same number of SLC datasets on the input swath to resample from.
    az_offsets_dataset : DatasetReader
        A dataset containing azimuth offsets, in pixels. Each offset defines the
        azimuth component of the displacement from a pixel in the output grid to the
        corresponding pixel in the input grid.
    rg_offsets_dataset : DatasetReader
        A dataset containing range offsets, in pixels. Each offset defines the range
        component of the displacement from a pixel in the output grid to the
        corresponding pixel in the input grid.
    input_radar_grid : RadarGridParameters
        The RadarGridParameters for the input data swath.
    doppler : LUT2d, optional
        The doppler lookup table, in Hertz, as a function of azimuth and range. Defaults
        to an empty LUT2d.
    block_size_az : int, optional
        The length of the blocks in azimuth rows. Defaults to 1024.
    block_size_rg : int, optional
        The width of the blocks in range columns. Defaults to 1024.
    quiet : bool, optional
        If True, don't log informational statements about progress and benchmarking.
        Defaults to False.
    fill_value: complex, optional
        The value to fill out-of-bounds pixels with. Defaults to NaN + j*NaN.
    """
    info_channel = journal.info("resample_slc.resample_slc_blocks")
    warning_channel = journal.warning("resample_slc.resample_slc_blocks")
    error_channel = journal.error("resample_slc.resample_slc_blocks")

    if len(input_slcs) != len(output_resampled_slcs):
        err_log = "Number of input and output datasets do not match."
        error_channel.log(err_log)
        raise ValueError(err_log)

    for in_dataset in input_slcs:
        in_dataset_dtype = in_dataset.dtype
        if in_dataset_dtype not in [np.complex64, np.complex128]:
            err_log = (
                f"Input dataset given with unsupported data type: {in_dataset_dtype}."
                " Must be complex64 or complex128."
            )
            error_channel.log(err_log)
            raise ValueError(err_log)
        if in_dataset_dtype == np.complex128:
            warning_channel.log(
                "Warning: An input dataset was given to resample_slc which is a "
                "double-precision complex128 raster. This data will be truncated to "
                "complex64 during processing."
            )
    for out_dataset in output_resampled_slcs:
        out_dataset_dtype = out_dataset.dtype
        if out_dataset_dtype not in [np.complex64, np.complex128]:
            err_log = (
                f"Output dataset given with unsupported data type: {out_dataset_dtype}."
                " Must be complex64 or complex128."
            )
            error_channel.log(err_log)
            raise ValueError(err_log)

    # Check to make sure that all input datasets are the same size as each other,
    # and likewise with all output datasets.
    dataset_shapes_consistent(datasets=input_slcs, name="input_slcs")
    dataset_shapes_consistent(
        datasets=output_resampled_slcs, name="output_resampled_slcs"
    )

    output_shape = output_resampled_slcs[0].shape

    # Check to make sure that the overall azimuth and range datasets are the same shape
    # as the output SLC datasets, which are already confirmed above to be the same
    # sizes.
    for name, dataset in zip(
        ["Azimuth", "Range"], [az_offsets_dataset, rg_offsets_dataset]
    ):
        if dataset.shape != output_shape:
            err_log = (
                f"{name} offsets dataset given with shape {dataset.shape} - "
                f"does not match output dataset shape {output_shape}"
            )
            error_channel.log(err_log)
            raise ValueError(err_log)

    # Initialize the overall runtime timers, denoted in seconds, for benchmarking.
    offsets_read_timer = 0
    slc_read_timer = 0
    write_timer = 0
    processing_timer = 0

    # For each block in the processing set:
    for out_block_slice in get_blocks(
        block_max_shape=(block_size_az, block_size_rg),
        grid_shape=output_shape,
        quiet=quiet,
    ):
        # Initialize the per-block runtime timers.
        # These take the time of major I/O and processing tasks by subtracting
        # their beginning time and then adding their end time, which leaves the
        # time committed per task.
        block_offsets_read_timer = 0
        block_slc_read_timer = 0
        block_write_timer = 0
        block_processing_timer = 0

        # Get the offsets blocks using the slices.
        block_offsets_read_timer -= perf_counter()
        az_offsets_block = np.array(az_offsets_dataset[out_block_slice], np.float64)
        rg_offsets_block = np.array(rg_offsets_dataset[out_block_slice], np.float64)
        block_offsets_read_timer += perf_counter()

        # Interpret extremely low values as NaN
        block_processing_timer -= perf_counter()

        # This happens if geo2rdr does not converge - it defaults to very large
        # negative values (namely -1000000.0)
        # TODO: Fix this behavior in geo2rdr?
        az_offsets_block[az_offsets_block == -1e6] = np.nan
        rg_offsets_block[rg_offsets_block == -1e6] = np.nan

        # Get the shape of the slice to be read on the input raster.
        in_slices = get_blocks_by_offsets(
            out_block_slices=out_block_slice,
            y_offsets_block=az_offsets_block,
            x_offsets_block=rg_offsets_block,
            in_grid_shape=input_slcs[0].shape,
            buffer=SINC_HALF,
        )

        # Skip this block if the offsets are all-NaN or if they point somewhere outside
        # of the input grid. This has already been logged in `get_blocks_offset`.
        if in_slices is None:
            continue

        block_processing_timer += perf_counter()

        # Get the shape of the output block.
        az_slice, rg_slice = out_block_slice
        out_block_length = az_slice.stop - az_slice.start
        out_block_width = rg_slice.stop - rg_slice.start
        out_block_shape = (out_block_length, out_block_width)

        # The set of resampled processing blocks
        output_blocks: list[np.ndarray] = []
        # The set of input processing blocks
        input_blocks: list[np.ndarray] = []

        # For each input SLC given, acquire a block of data to the in_blocks list
        # and create one in output_blocks that is filled with zeros to populate.
        block_slc_read_timer -= perf_counter()
        for input_slc in input_slcs:
            input_blocks.append(np.array(input_slc[in_slices], dtype=np.complex64))
            output_block = np.full(
                out_block_shape, fill_value=fill_value, dtype=np.complex64
            )
            output_blocks.append(output_block)
        block_slc_read_timer += perf_counter()

        # Get the first positions in azimuth and range on both the resampled block
        # and input block.
        block_processing_timer -= perf_counter()

        # Convert the offset blocks to index blocks that map pixels on the output
        # data block to positions on the input data block.
        azimuth_index_grid, range_index_grid = offsets_to_indices(
            out_block_slice=out_block_slice,
            in_block_slice=in_slices,
            az_offsets=az_offsets_block,
            rg_offsets=rg_offsets_block,
        )

        # First, check that the indices arrays and input array have equal shapes.
        # We know at this point that output_blocks[0].shape is the size of all the
        # other output_blocks because it has already passed dataset_shapes_consistent.
        if (
            output_blocks[0].shape != azimuth_index_grid.shape
            or output_blocks[0].shape != range_index_grid.shape
        ):
            err_log = (
                "Output block, azimuth indices block, and range indices block must "
                "be the same shape. Shapes: "
                f"Output block: {output_blocks[0].shape} "
                f"Azimuth indices: {azimuth_index_grid.shape} "
                f"Range indices: {range_index_grid.shape}"
            )
            error_channel.log(err_log)
            raise ValueError(err_log)

        # Run the resampling algorithm on the given blocks.
        for output_block, input_block in zip(output_blocks, input_blocks):
            if not quiet:
                info_channel.log(
                    f"interpolating to output SLC for block {out_block_slice}..."
                )
                # Reporting input block shape for debugging
                info_channel.log(f"Input block: {in_slices}")
            resample_to_coords(
                output_block,
                input_block,
                range_index_grid,
                azimuth_index_grid,
                input_radar_grid[in_slices],
                doppler,
                fill_value,
            )

        block_processing_timer += perf_counter()

        # The resampling blocks have now been filled. For each output dataset, write
        # the associated block to it.
        block_write_timer -= perf_counter()
        for output_dataset, output_block in zip(output_resampled_slcs, output_blocks):
            output_dataset[out_block_slice] = output_block
        block_write_timer += perf_counter()

        # Add the accumulated times per block to the overall totals and report.
        offsets_read_timer += block_offsets_read_timer
        slc_read_timer += block_slc_read_timer
        write_timer += block_write_timer
        processing_timer += block_processing_timer
        if not quiet:
            info_channel.log(f"Block SLC I/O read time (sec): {block_slc_read_timer}")
            info_channel.log(
                f"Block Offsets I/O read time (sec): {block_offsets_read_timer}"
            )
            info_channel.log(f"Block I/O write time (sec): {block_write_timer}")
            info_channel.log(f"Block Processing time (sec): {block_processing_timer}")

    # Report the overall totals and return.
    if not quiet:
        info_channel.log(f"Total SLC I/O read time (sec): {slc_read_timer}")
        info_channel.log(f"Total Offsets I/O read time (sec): {offsets_read_timer}")
        info_channel.log(f"Total I/O write time (sec): {write_timer}")
        info_channel.log(f"Total Processing time (sec): {processing_timer}")


def dataset_shapes_consistent(
    datasets: Sequence[DatasetReader | DatasetWriter],
    name: str | None = None,
) -> None:
    """
    Asserts that all arrays in the given sequence have the same shape.

    Parameters
    ----------
    datasets : Sequence[DatasetReader | DatasetWriter]
        A sequence of datasets to check.
    name : str or None, optional
        The name of the array, used to log a more descriptive error on failure.
        If None, a less descriptive error message will be used. Defaults to None.

    Raises
    ------
    ValueError
        If any array within the sequence is not the same shape as the first.
    """
    error_channel = journal.error("resample_slc.dataset_shapes_consistent")

    shape = datasets[0].shape

    if name is None:
        name = "the given"

    if any(dataset.shape != shape for dataset in datasets):
        err_log = f"Dataset shapes in {name} sequence are unequal."
        error_channel.log(err_log)
        raise ValueError(err_log)


def offsets_to_indices(
    out_block_slice: tuple[slice, slice],
    in_block_slice: tuple[slice, slice],
    az_offsets: np.ndarray,
    rg_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert azimuth and range offsets to indices pointing from an output block to an
    input block.

    Assumes that the output and input block slices are integer-valued rather than None.

    Parameters
    ----------
    out_block_slice : tuple[slice, slice]
        A 2-dimensional slice on an output block.
    in_block_slice : tuple[slice, slice]
        A 2-dimensional slice on an input block.
    az_offsets : np.ndarray
        An ndarray of offsets in the azimuth direction. Each offset defines the
        displacement in azimuth pixels from a pixel in the broader output grid to the
        corresponding pixel in the broader input grid.
    rg_offsets : np.ndarray
        An ndarray of offsets in the range direction. Each offset defines the
        displacement in range pixels from a pixel in the broader output grid to the
        corresponding pixel in the broader input grid.

    Returns
    -------
    az_indices, rg_indices : np.ndarray
        Arrays of indices in azimuth and range for a position in the input block
        corresponding to each position on the output block.
    """

    # Get the start indices and lengths of the azimuth and range dimensions on the
    # output block.
    out_az_slice, out_rg_slice = out_block_slice
    az_out_start = out_az_slice.start
    rg_out_start = out_rg_slice.start
    out_block_length = out_az_slice.stop - az_out_start
    out_block_width = out_rg_slice.stop - rg_out_start

    # Get the start indices of the azimuth and range dimensions on the input block.
    az_in_slice, rg_in_slice = in_block_slice
    az_in_start = az_in_slice.start
    rg_in_start = rg_in_slice.start

    # Since the input and output slices don't start at the same pixel on their
    # respective grids, azimuth and range grid offsets are computed here to correct
    # the offset between the (0,0) point in the input block and that of the output block
    # relative to their broader grids.
    az_grid_offset = az_out_start - az_in_start
    rg_grid_offset = rg_out_start - rg_in_start

    # Indices can be gotten from the offsets by adding the azimuth or range pixel index
    # on the output grid to its' respective azimuth or range offset and grid offset.
    # For each pixel, this converts the displacement from one output pixel to the
    # absolute position that it points to on the input grid.
    azimuth_arange = np.arange(out_block_length, dtype=np.float64)
    range_arange = np.arange(out_block_width, dtype=np.float64)
    az_indices, rg_indices = np.meshgrid(azimuth_arange, range_arange, indexing="ij")
    az_indices += az_offsets + az_grid_offset
    rg_indices += rg_offsets + rg_grid_offset

    return az_indices, rg_indices
