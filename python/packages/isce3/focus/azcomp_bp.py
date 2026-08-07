#!/usr/bin/env python3
"""
Azimuth compression algorithms for SAR focusing.

Provides standard and factorized backprojection implementations.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import logging
import numpy as np
import h5py
import isce3
from isce3.core import LUT2d
from isce3.focus.serialization import BackprojectionStageParameters
from isce3.geometry import DEMInterpolator
from isce3.product import RadarGridParameters
from typing import Optional

log = logging.getLogger("isce3.focus.azcomp_bp")

# Type aliases for processing block plan
Selection2d = tuple[slice, slice]
TimeBounds = tuple[float, float]
BlockPlan = list[tuple[Selection2d, TimeBounds]]


def is_overlapping(a, b, c, d):
    """
    Check if two intervals overlap.

    Parameters
    ----------
    a : float
        Start of first interval.
    b : float
        End of first interval.
    c : float
        Start of second interval.
    d : float
        End of second interval.

    Returns
    -------
    bool
        True if intervals [a, b] and [c, d] overlap, False otherwise.
    """
    assert (b >= a) and (d >= c)
    return (d >= a) and (c <= b)


class Task:
    """
    Deferred task wrapper for lazy evaluation.

    Stores a function and its arguments for later execution, allowing
    tasks to be defined without immediate evaluation. Used in factorized
    backprojection to reduce memory pressure by only computing intermediate
    results when needed.

    Parameters
    ----------
    function : callable
        Function to execute when result() is called.
    *args
        Positional arguments to pass to function.
    **kwargs
        Keyword arguments to pass to function.
    """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def result(self):
        """
        Execute the stored function with its arguments.

        Returns
        -------
        Any
            Result of calling function(*args, **kwargs).
        """
        return self.function(*self.args, **self.kwargs)


def find_min_cache_size(key_lists):
    """
    Determine the minimum LRU cache size needed to avoid cache misses.

    Parameters
    ----------
    key_lists: Iterable[Iterable[Hashable]]
        List of jobs, where each job is a list of task keys, and
        keys may be shared between jobs.

    Returns
    -------
    n : int
        Minimum cache size required to hold shared keys in memory,
        assuming jobs are executed in the given order.
    """
    # Flatten the key sequence into a single ordered list of key accesses
    access_sequence = [key for keylist in key_lists for key in keylist]

    # For each key, record the indices where it's accessed
    access_indices = defaultdict(list)
    for i, key in enumerate(access_sequence):
        access_indices[key].append(i)

    min_size = 1

    for key, indices in access_indices.items():
        # Only care about keys accessed more than once (re-use case)
        for j in range(1, len(indices)):
            prev_idx = indices[j - 1]
            curr_idx = indices[j]

            # Count distinct keys in the window [prev_idx, curr_idx] inclusive.
            # If this many distinct keys were accessed, the LRU cache must hold
            # at least this many entries to avoid evicting `key` before reuse.
            window = access_sequence[prev_idx:curr_idx + 1]
            distinct_in_window = len(set(window))
            min_size = max(min_size, distinct_in_window)

    return min_size


def nfft_params_dict(p: isce3.focus.serialization.NonUniformFFT2DParams):
    return dict(
        rows = dict(
            m = p.azimuth.kernel_halfwidth,
            s = p.azimuth.zero_padding_factor),
        cols = dict(
            m = p.range.kernel_halfwidth,
            s = p.range.zero_padding_factor))


def azcomp_bp(azres, kernel, blocks_bounds, igeom, rcdata, ogrid, writer,
              height=None, dem=isce3.geometry.DEMInterpolator(),
              rdr2geo_params=dict(), geo2rdr_params=dict(), atmos="nodelay",
              use_gpu=False):
    """
    Perform azimuth compression using standard backprojection algorithm.

    Parameters
    ----------
    azres : float
        Desired azimuth resolution, in meters.
    kernel : isce3.core.Kernel
        Interpolation kernel for backprojection.
    blocks_bounds : BlockPlan
        List of tuples containing ((row_slice, col_slice), (t0, t1)) for each
        processing block, where slices define the output grid region and (t0, t1)
        are the required raw data time bounds in seconds.
    igeom : isce3.container.RadarGeometry
        Input radar geometry for range-compressed data.
    rcdata : array-like
        Range-compressed data, shape (azimuth, range).
    ogrid : RadarGridParameters
        Output zero-Doppler radar grid parameters.
    writer : BackgroundWriter
        Writer object.  Must have a method `queue_write(z, block)` for writing
        out image subset `z` into selection `block` of output image where `z`
        is a 2D numpy array and `block` is a tuple[slice, slice].
    height : array-like, optional
        Optional storage for height above ellipsoid (in meters) for each output
        pixel, shape matching ogrid.
    dem : isce3.geometry.DEMInterpolator, optional
        Digital elevation model. Default is ellipsoid (height=0).
    rdr2geo_params : dict, optional
        Parameters for rdr2geo_bracket solver.
    geo2rdr_params : dict, optional
        Parameters for geo2rdr_bracket solver.
    atmos : str, optional
        Atmospheric delay model. Default is "nodelay".
    use_gpu : bool, optional
        Use GPU acceleration if available. Default is False.
    """
    if use_gpu:
        backproject = isce3.cuda.focus.backproject
    else:
        backproject = isce3.focus.backproject
    fc = isce3.core.speed_of_light / ogrid.wavelength
    zerodop = isce3.core.LUT2d()
    for block, (t0, t1) in blocks_bounds:
        description = f"(i, j) = ({block[0].start}, {block[1].start})"
        if not is_overlapping(t0, t1, igeom.radar_grid.sensing_start,
                igeom.radar_grid.sensing_stop):
            log.info(f"Skipping inactive azcomp block at {description}")
            continue
        log.info(f"Azcomp block at {description}")
        bgrid = ogrid[block]
        ogeom = isce3.container.RadarGeometry(bgrid, igeom.orbit, zerodop)
        z = np.zeros(bgrid.shape, 'c8')
        hgt = height[block] if height is not None else None
        err = backproject(z, ogeom, rcdata, igeom, dem, fc, azres, kernel,
            atmos, rdr2geo_params, geo2rdr_params, height=hgt)
        if err:
            log.warning("azcomp block contains some invalid pixels")
        writer.queue_write(z, block)


def azcomp_fbp(factors: BackprojectionStageParameters,
        azres, kernel, blocks_bounds, igeom,
        rcdata, ogrid, writer, height=None, dem=isce3.geometry.DEMInterpolator(),
        rdr2geo_params=dict(), geo2rdr_params=dict(), atmos="nodelay",
        use_gpu=False, bandwidth=0.0, debugfile=None):
    """
    Perform azimuth compression using factorized backprojection algorithm.

    The factorized backprojection algorithm processes data in multiple stages,
    first focusing blocks of pulses to intermediate polar grids, then merging
    those grids hierarchically, and finally accumulating the results to the
    output zero-Doppler radar grid.

    Parameters
    ----------
    factors : BackprojectionStageParameters
        List of factorization stage parameters defining the processing hierarchy.
        Each stage specifies the number of pulses or polar images to combine,
        oversample factors, and NFFT interpolation parameters.
    azres : float
        Desired azimuth resolution, in meters.
    kernel : isce3.core.Kernel
        Interpolation kernel for backprojection.
    blocks_bounds : BlockPlan
        List of tuples containing ((row_slice, col_slice), (t0, t1)) for each
        processing block, where slices define the output grid region and (t0, t1)
        are the required raw data time bounds in seconds.
    igeom : isce3.container.RadarGeometry
        Input radar geometry for range-compressed data.
    rcdata : array-like
        Range-compressed data, shape (azimuth, range).
    ogrid : RadarGridParameters
        Output zero-Doppler radar grid parameters.
    writer : BackgroundWriter
        Writer object.  Must have a method `queue_write(z, block)` for writing
        out image subset `z` into selection `block` of output image where `z`
        is a 2D numpy array and `block` is a tuple[slice, slice].
    height : array-like, optional
        Height above ellipsoid (in meters) for each output pixel, shape matching
        ogrid. If None, uses DEM.
    dem : isce3.geometry.DEMInterpolator, optional
        Digital elevation model. Default is ellipsoid (height=0).
    rdr2geo_params : dict, optional
        Parameters for rdr2geo_bracket solver.
    geo2rdr_params : dict, optional
        Parameters for geo2rdr_bracket solver.
    atmos : str, optional
        Atmospheric delay model. Applied at final stage only to avoid phase
        modulation artifacts from DEM sampling across subimages. Default is "nodelay".
    use_gpu : bool, optional
        Use GPU acceleration if available. Default is False.
    bandwidth : float, optional
        Signal bandwidth in Hz, used to determine polar grid angular extent.
        Default is 0.0.
    debugfile : file-like, optional
        HDF5 file handle for writing intermediate polar grids for debugging.
        If None, no debug output is written.
    """
    fc = isce3.core.speed_of_light / ogrid.wavelength
    zerodop = isce3.core.LUT2d()

    if use_gpu:
        bp_to_polar_grid = isce3.cuda.focus.backproject_to_polar_grid
        merge_polar_images = isce3.cuda.focus.merge_polar_images
        add_to_radar_grid = isce3.cuda.focus.accumulate_polar_images_to_radar_grid
        # TODO could make this a separate option to conserve GPU memory.
        # That'd require a little finess in the bindings and CUDA side, though.
        make_image_nfft2d = isce3.cuda.signal.make_image_nfft2d
    else:
        bp_to_polar_grid = isce3.focus.backproject_to_polar_grid
        merge_polar_images = isce3.focus.merge_polar_images
        add_to_radar_grid = isce3.focus.accumulate_polar_images_to_radar_grid
        make_image_nfft2d = isce3.signal.make_image_nfft2d

    _, v = igeom.orbit.interpolate(igeom.orbit.mid_time)
    vs = np.linalg.norm(v)
    tq_max = isce3.focus.get_polar_angle_time_constant(fc, vs, bandwidth)

    if debugfile is not None:
        log.debug(f"Writing FBP metadata to file {debugfile.name}")
        with h5py.File(debugfile, "w") as h5:
            epoch = igeom.reference_epoch
            igeom.orbit.save_to_h5(h5.require_group("orbit"))
            igeom.doppler.save_to_h5(h5.require_group("doppler"), "doppler",
                epoch, "Hz")
            h5.create_dataset("epoch", data=np.bytes_(epoch))
            h5.create_dataset("wavelength", data=igeom.radar_grid.wavelength)

    # Focus to intermediate grids.
    # NOTE We'll actually just define the tasks and only evaluate them as needed
    # in order to reduce memory pressure.  We could process them in
    # parallel using concurrent.futures or dask, for for now just store them in
    # a dict keyed by the PolarGrid associated with the imagelets.
    tasks = dict()
    aztimes = np.array(igeom.radar_grid.sensing_times)
    pris = np.hstack((np.diff(aztimes), aztimes[-1] - aztimes[-2]))
    stage = factors[0]
    nfft2d_params = nfft_params_dict(stage.interpolation)
    pulse_starts = range(0, igeom.radar_grid.length, stage.size)
    log.info(f"Beginning initial factorizations of {stage.size} pulses")
    nblocks = len(pulse_starts)

    def process_pulses(iblock, nblocks, debugfile, fdata, sr, x, v,
                polar_grid, dem, fc, kernel, atmos, rdr2geo_params):
        log.info(f"Focusing {len(x)} pulses to polar image {iblock + 1} of {nblocks}")
        # NOTE Atmosphere will get applied (if requested) at final stage to
        # avoid phase modulation from DEM sampling issues across subimages.
        # Always "nodelay" in this stage.
        _, img, _ = bp_to_polar_grid(fdata, sr, x, v,
            polar_grid, dem, fc, kernel, "nodelay", rdr2geo_params)
        if debugfile is not None:
            import h5py
            log.debug(f"Dumping FBP factor with shape = {img.shape} to file.")
            with h5py.File(debugfile, "w") as h5:  # okay to reopen stream
                g = h5.require_group(f"stage_00/block_{iblock:06d}")
                isce3.focus.save_polar_image_to_h5(img, polar_grid, g)
        log.info("NFFT upsampling and filtering")
        return make_image_nfft2d(img, nfft2d_params, pad_input=True)

    for i in pulse_starts:
        pulses = slice(i, i + stage.size)
        ti = aztimes[pulses]
        if len(ti) < 2:
            log.info("Skipping FBP block containing only a single pulse.")
            continue
        fgrid = igeom.radar_grid[pulses, :]
        fgeom = isce3.container.RadarGeometry(fgrid, igeom.orbit, igeom.doppler)
        fdata = rcdata[pulses, :]
        iblock = i // stage.size
        polar_grid, x, v = isce3.focus.setup_polar_grid_for_pulses(fgeom, ti,
            bandwidth, azres, stage.oversample_range, stage.oversample_azimuth,
            pri=pris[pulses][-1])
        tasks[polar_grid] = Task(process_pulses, iblock, nblocks,
            debugfile, fdata, fgrid.slant_ranges, x, v,
            polar_grid, dem, fc, kernel, atmos, rdr2geo_params)

    grids = sorted(tasks.keys(), key = lambda grid: grid.aztime_start)

    # Merge polar grids to make bigger polar grids.
    # With Python 3.12 we could use itertools.batched
    def process_merge(i_stage, i_block, nblocks, in_grids, out_grid, nfft2d_params):
        # Process the input data we need.  Middle stages don't overlap, so no
        # harm in popping the task off the stack.
        in_images = [tasks.pop(grid).result() for grid in in_grids]
        log.info(f"Merging {len(in_grids)} polar images stage {i_stage} block "
            f"{i_block + 1} / {nblocks}")
        out_image = np.zeros(out_grid.shape, np.complex64)
        merge_polar_images(in_grids, in_images, out_grid, out_image,
            fc, dem, rdr2geo_params)
        if debugfile is not None:
            import h5py
            name = f"stage_{i_stage + 1:02d}/block_{i_block:06d}"
            with h5py.File(debugfile, "w") as h5:  # okay to reopen stream
                g = h5.require_group(name)
                isce3.focus.save_polar_image_to_h5(out_image, out_grid, g)
        log.info("NFFT upsampling and filtering")
        return make_image_nfft2d(out_image, nfft2d_params, pad_input=True)

    num_middle_stages = len(factors[1:])
    for i_stage, stage in enumerate(factors[1:]):
        log.info("Planning intermediate factorization stage "
            + f"{i_stage + 1} / {num_middle_stages}")

        # Don't let azimuth resolution grow finer than user requested one.
        dq_min = azres / (ogrid.slant_ranges[-1] * stage.oversample_azimuth)
        tq = tq_max / stage.oversample_azimuth

        nfft2d_params = nfft_params_dict(stage.interpolation)
        input_block_starts = range(0, len(grids), stage.size)
        nblocks = len(input_block_starts)
        stage_grids = []

        for i in input_block_starts:
            i_block = i // stage.size
            mask = slice(i, i + stage.size)
            input_grids = grids[mask]
            if len(input_grids) == 1:
                # No need to merge. Grid is already tasked, though be sure to
                # carry it forward to next stage.
                stage_grids.append(input_grids[0])
                continue
            my_grid = isce3.focus.merge_polar_grids(input_grids, dem,
                rdr2geo_params, dq_min, tq)
            tasks[my_grid] = Task(process_merge, i_stage, i_block, nblocks,
                input_grids, my_grid, nfft2d_params)
            stage_grids.append(my_grid)

        # Use this stage's grids as input for next stage.
        grids = stage_grids

    # Plan final stage to get bound on LRU cache size, assuming FIFO access
    # pattern.
    blocks_grids = list()
    for block, (t0, t1) in blocks_bounds:
        description = f"(i, j) = ({block[0].start}, {block[1].start})"
        if not is_overlapping(t0, t1, igeom.radar_grid.sensing_start,
                igeom.radar_grid.sensing_stop):
            log.info(f"Will skip inactive azcomp block at {description}")
            continue
        active_grids = [grid for grid in grids
            if is_overlapping(t0, t1, grid.aztime_start, grid.aztime_end)]
        blocks_grids.append((block, active_grids))

    max_images = max(len(grids) for (_, grids) in blocks_grids)
    log.info(f"Proceeding to final stage with max {max_images} sub-images per block")
    # Required cache size may be smaller than max_images when not all
    # sub-images in one block are used in the next block.  However, it can
    # also be more when we subdivide in range, since a far-range block may
    # need all the sub-images of a near-range block plus a few more.
    cache_size = find_min_cache_size([grid for (_, grid) in blocks_grids])
    log.info(f"Calculated min cache size = {cache_size}")

    @lru_cache(maxsize=cache_size)
    def get_image_iterpolator(polar_grid):
        # Using pop() to remove from stack requires that cache size is adequate
        # to avoid redundant computations, which we prioritize over generality.
        try:
            return tasks.pop(polar_grid).result()
        except KeyError as err:
            msg = ("Failed to retrieve sub-image spanning time interval "
                f"[{polar_grid.aztime_start}, {polar_grid.aztime_end}).  This "
                "could mean that the stripmap assumption was violated.")
            log.error(msg)
            raise

    # sum factors into final image
    for block, active_grids in blocks_grids:
        description = f"(i, j) = ({block[0].start}, {block[1].start})"
        active_images = [get_image_iterpolator(grid) for grid in active_grids]
        bgrid = ogrid[block]
        ogeom = isce3.container.RadarGeometry(bgrid, igeom.orbit, zerodop)
        z = np.zeros(bgrid.shape, 'c8')
        hgt = height[block] if height is not None else None
        log.info(f"Azcomp final sums for block at {description} using "
            f"{len(active_images)} sub-apertures")
        err = add_to_radar_grid(
            z, ogeom, igeom.orbit, igeom.doppler, active_grids, active_images,
            dem, fc, azres, atmos, rdr2geo_params, geo2rdr_params, hgt)
        if err:
            log.warning("azcomp block contains some invalid pixels")
        writer.queue_write(z, block)

    if len(tasks) != 0:
        log.warning(f"Queued {len(tasks)} tasks that were never needed.")
