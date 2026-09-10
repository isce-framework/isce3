import re
from datetime import datetime
from typing import Optional

import h5py
import isce3
import journal
import numpy as np
from isce3.core import crop_external_orbit
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from osgeo import gdal


def number_to_ordinal(number):
    """
    Convert an unsigned integer to its ordinal representation.

    Parameters
    ----------
    number : int
        The non-negative integer to be converted to its ordinal form.

    Returns
    -------
    str
        The ordinal representation of the input number.

    Notes
    -----
    The function appends the appropriate suffix ('st', 'nd', 'rd', or 'th')
    to the input number based on common English ordinal representations.
    Exceptions are made for numbers ending in 11, 12, and 13, which use 'th'.

    Examples
    --------
    >>> number_to_ordinal(1)
    '1st'

    >>> number_to_ordinal(22)
    '22nd'

    >>> number_to_ordinal(33)
    '33rd'

    >>> number_to_ordinal(104)
    '104th'
    """
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return f"{number}{suffix}"


def extract_datetime_from_string(date_string,
                                 prefix: Optional[str] = ''):
    """
    Extracts a datetime object from a string.

    Parameters
    ----------
    date_string : str
        The input string containing the datetime information.

    prefix : str, optional
        The prefix of the datatime. Defaults to ''.

    Returns
    -------
    string or None
        A string with format YYYY-mm-ddTHH:MM:SS if successful,
        or None if there was an error.

    Notes
    -----
    This function uses a regular expression to extract a datetime string
    from the input string and then converts it to a string
    with format YYYY-mm-ddTHH:MM:SS.

    Examples
    --------
    >>> date_string = "Some text here 2023-12-10 14:30:00 and more text"
    >>> result = extract_datetime_from_string(date_string)
    >>> print(result)
    2023-12-10T14:30:00

    """
    # Define a regular expression pattern for the datetime format
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

    # Search for the pattern in the string
    match = re.search(pattern, date_string)

    if match:
        # Extract the matched datetime string
        datetime_string = match.group(1)

        # Convert the datetime string to a datetime object
        try:
            datetime_object = \
                datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
            return f'{prefix}{datetime_object.strftime("%Y-%m-%dT%H:%M:%S")}'
        except ValueError:
            return None
    else:
        return None

def compute_number_of_elements(shape : tuple):
    """
    Compute the number of data elements from a given the shape

    Parameters
    ----------
    shape : tuple
        The shape of the h5py dataset

    Returns
    -------
    int
        the number of cells in the shape
    """

    # compute the product of all the entries
    return np.prod(shape)

def get_radar_grid_cube_shape(cfg : dict):
    """
    Get the radar grid cube shape

    Parameters
    ---------
    cfg : dict
        InSAR runconfig dictionary

    Returns
    ----------
    tuple
        (height, grid_length, grid_width):
    """
    proc_cfg = cfg["processing"]
    radar_grid_cubes_geogrid = proc_cfg["radar_grid_cubes"]["geogrid"]
    radar_grid_cubes_heights = proc_cfg["radar_grid_cubes"]["heights"]

    return (len(radar_grid_cubes_heights),
            radar_grid_cubes_geogrid.length,
            radar_grid_cubes_geogrid.width)

def get_geolocation_grid_cube_obj(cfg : dict):
    """
    Get the geolocation grid object

    Parameters
    ---------
    cfg : dict
        InSAR runconfig dictionary

    Returns
    ----------
    isce3.product.GeoGridParameters
        geolocation_radargrid
    """

    ref_h5_slc_file = cfg["input_file_group"]["reference_rslc_file"]
    ref_rslc = SLC(hdf5file=ref_h5_slc_file)

    # Pull the radar frequency
    radargrid = ref_rslc.getRadarGrid()
    external_ref_orbit_path = \
        cfg["dynamic_ancillary_file_group"]["orbit_files"]['reference_orbit_file']

    ref_orbit = ref_rslc.getOrbit()
    if external_ref_orbit_path is not None:
        ref_external_orbit = load_orbit_from_xml(external_ref_orbit_path,
                                                 radargrid.ref_epoch)
        ref_orbit = crop_external_orbit(ref_external_orbit,
                                        ref_orbit)

    # The maximum spacing here is to keep consistent with the RSLC product
    # where both the azimuth and slant range spacing are around 500 meters
    max_spacing = 500.0
    t = radargrid.sensing_mid + \
        (radargrid.ref_epoch - ref_orbit.reference_epoch).total_seconds()

    _, v = ref_orbit.interpolate(t)
    dx = np.linalg.norm(v) / radargrid.prf

    # Create a new geolocation radar grid with 5 extra points
    # before and after the starting and ending
    # zeroDopplerTime and slantRange
    extra_points = 5

    # Total number of samples along the azimuth and slant range
    # using around 500m sampling interval
    ysize = int(np.ceil(radargrid.length / (max_spacing / dx)))
    xsize = int(np.ceil(radargrid.width / \
        (max_spacing / radargrid.range_pixel_spacing)))

    # New geolocation grid
    geolocation_radargrid = \
        radargrid.resize_and_keep_startstop(ysize, xsize)
    geolocation_radargrid = \
        geolocation_radargrid.add_margin(extra_points,
                                         extra_points)

    return geolocation_radargrid

def get_geolocation_grid_cube_shape(cfg : dict):
    """
    Get the geolocation grid cube shape

    Parameters
    ---------
    cfg : dict
        InSAR runconfig dictionary

    Returns
    ----------
    tuple
        (height, grid_length, grid_width):
    """

    # Pull the heights and espg from the radar_grid_cubes group
    # in the runconfig
    radar_grid_cfg = cfg["processing"]["radar_grid_cubes"]
    heights = np.array(radar_grid_cfg["heights"])

    geolocation_radargrid = get_geolocation_grid_cube_obj(cfg)

    return (len(heights),
            geolocation_radargrid.length,
            geolocation_radargrid.width)

def get_interferogram_dataset_shape(cfg : dict, freq : str):
    """
    Get the interfergraom dataset shape at a given frequency

    Parameters
    ---------
    cfg : dict
        InSAR runconfig dictionary
    freq: str
        frequency ('A' or 'B')

    Returns
    ----------
    igram_shape : tuple
        interfergraom shape
    """
    # get the RSLC lines and columns
    ref_h5_slc_file = cfg["input_file_group"]["reference_rslc_file"]
    ref_rslc = SLC(hdf5file=ref_h5_slc_file)
    ref_rslc.parsePolarizations()

    proc_cfg = cfg["processing"]
    igram_range_looks = proc_cfg["crossmul"]["range_looks"]
    igram_azimuth_looks = proc_cfg["crossmul"]["azimuth_looks"]
    pol = ref_rslc.polarizations[freq][0]

    with h5py.File(ref_h5_slc_file, "r", libver="latest", swmr=True)\
        as ref_h5py_file_obj:
        slc_dset = ref_h5py_file_obj[
            f"{ref_rslc.SwathPath}/frequency{freq}/{pol}"]
        slc_lines, slc_cols = slc_dset.shape

        # shape of the interferogram product
        igram_shape = (slc_lines // igram_azimuth_looks,
                        slc_cols // igram_range_looks)

    return igram_shape


def get_unwrapped_interferogram_dataset_shape(cfg : dict, freq : str):
    """
    Get the unwrapped interfergraom dataset shape at a given frequency

    Parameters
    ---------
    cfg : dict
        InSAR runconfig dictionary
    freq: str
        frequency ('A' or 'B')

    Returns
    ----------
    igram_shape : tuple
        unwrapped interfergraom shape
    """
    # get the RSLC lines and columns
    ref_h5_slc_file = cfg["input_file_group"]["reference_rslc_file"]
    ref_rslc = SLC(hdf5file=ref_h5_slc_file)
    ref_rslc.parsePolarizations()

    proc_cfg = cfg["processing"]
    igram_range_looks = proc_cfg["crossmul"]["range_looks"]
    igram_azimuth_looks = proc_cfg["crossmul"]["azimuth_looks"]
    unwrap_rg_looks = proc_cfg["phase_unwrap"]["range_looks"]
    unwrap_az_looks = proc_cfg["phase_unwrap"]["azimuth_looks"]

    if (unwrap_az_looks != 1) or (unwrap_rg_looks != 1):
        igram_range_looks = unwrap_rg_looks
        igram_azimuth_looks = unwrap_az_looks
    pol = ref_rslc.polarizations[freq][0]

    with h5py.File(ref_h5_slc_file, "r", libver="latest", swmr=True)\
        as ref_h5py_file_obj:
        slc_dset = ref_h5py_file_obj[
            f"{ref_rslc.SwathPath}/frequency{freq}/{pol}"]
        slc_lines, slc_cols = slc_dset.shape

        # shape of the interferogram product
        igram_shape = (slc_lines // igram_azimuth_looks,
                        slc_cols // igram_range_looks)

    return igram_shape

def _compute_subswath_mask_id(azi_idx,
                              range_idx,
                              azi_offset,
                              range_offset,
                              ref_subswaths,
                              sec_subswaths):
    """
    Compute the subswath mask id between the reference and secondary RSLC
    using the range and azimuth offsets by the geometric coregistration where
    the offsets are used to compute the original azimuth and range indices of
    the secondary RSLC.

    Parameters
    ---------
    azi_idx : int
        Index along the azimuth of reference RSLC starting from 0
    range_idx: int
        Index along the slant range of reference RSLC starting from 0
    azi_offset: float
        The azimuth offset between the reference and secondary RSLC
    range_offset: float
        The range offset between the reference and secondary RSLC
    ref_subswaths : isce3.product.SubSwaths
        The subswath object of the reference RSLC
    sec_subswaths : isce3.product.SubSwaths
        The subswath object of the secondary RSLC

    Returns
    ----------
    subswath_mask_id : int
        The subswath mask id
    """

    # subswath number of the reference RSLC
    ref_subswath_num = \
        ref_subswaths.get_sample_sub_swath(azi_idx,range_idx)

    # Nearest neighbor to get the subswath number of the
    # secondary RSLC where offsets are used to compute the original
    # range and azimuth indices of the secondary RSLC.
    sec_subswath_num = \
        sec_subswaths.get_sample_sub_swath(
            int(azi_idx+azi_offset+0.5),
            int(range_idx+range_offset+0.5))

    # Compute the subswath mask id based on the subswath number of
    # reference and secondary RSLC. The mask id has 3 digits where
    # the last digit is the subswath number of secondary RSLC,
    # the second digit is the subswath number of reference RSLC,
    # and the first digit is reserved for the land (0) or water (1).

    # For example, 12 means land, subwath number of reference and secodnary
    # RSLC are 1 and 2 respectively.
    subswath_mask_id = \
        int(10 * ref_subswath_num + sec_subswath_num)

    return subswath_mask_id

def save_to_hdf5_ds(input_file_path,
                    hdf5_ds_obj,
                    lines_per_block = 1000):
    """
    Save the data to the HDF5 dataset

    Parameters
    ---------
    input_file_path : str
        Path of the input file
    hdf5_ds_obj : h5py.Dataset
        The HDF5 dataset object
    lines_per_block : integer (default: 1000)
         Lines per block to write the data to the hard drive
    """

    input_src = gdal.Open(input_file_path)
    width = input_src.RasterXSize
    length = input_src.RasterYSize

    # Write data block by block
    for line in range(0, length, lines_per_block):
        line_blocks = lines_per_block
        if (line + lines_per_block) > length:
            line_blocks = length - line
        data = input_src.GetRasterBand(1).ReadAsArray(0,line, width, line_blocks)
        hdf5_ds_obj.write_direct(data,
                                 dest_sel=np.s_[line : line + line_blocks, : width])

    input_src = None

def generate_dem_rdr(radar_grid_obj,
                     orbit_obj,
                     dem_file,
                     out_dem_rdr_path,
                     use_gpu = True,
                     dem_interp_method = 'BIQUINTIC',
                     threshold = 1.0e-7,
                     numiter = 25,
                     extraiter = 10,
                     lines_per_block = 1000):
    """
    Generate the DEM in radar grid

    Parameters
    ---------
    radar_grid_obj : isce3.product.RadarGridParameters
        The radar grid object for the reference RSLC
    orbit_obj : isce3.core.Orbit
        The SLC object for the secondary RSLC
    dem_file  : str
        Input DEM file in geocoded coordinates
    out_dem_rdr_path : str
        output path of the DEM in radar grid
    use_gpu : boolean (default: True)
        Indicator to use the GPU for rdr2geo computations
    dem_interp_method : str (default: BIQUINTIC)
        DEM interpolation method, one of 'BILINEAR', 'BICUBIC', 'NEAREST', and 'BIQUINTIC'
    threshold : float (default: 1.0e-7)
        The rdr2geo absolute slant range convergence tolerance (m)
    numiter : integer (default: 25)
        Maximum number of primary Newton-Raphson iterations
    extraiter : integer (default: 10)
         Maximum number of secondary iterations
    lines_per_block : integer (default: 1000)
         Lines per block to run rdr2geo
    """

    error_journal = journal.error('utils.generate_insar_dem')
    grid_doppler = isce3.core.LUT2d()

    dem_raster = isce3.io.Raster(dem_file)
    if dem_raster is None:
        err_str = f'Can not open the DEM file {dem_raster}'
        error_journal.log(err_str)
        raise ValueError(err_str)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    try:
         interp_method = getattr(isce3.core.DataInterpMethod, dem_interp_method)
    except AttributeError:
         err_str = f"invalid interpolation method: {dem_interp_method}"
         error_journal.log(err_str)
         raise ValueError(err_str)

    # Use the GPU or CPU version
    if use_gpu:
        Rdr2Geo = isce3.cuda.geometry.Rdr2Geo
    else:
        Rdr2Geo = isce3.geometry.Rdr2Geo

    # Create the DEM in the range Doppler coordinates
    dem_src = isce3.io.Raster(out_dem_rdr_path,
                              radar_grid_obj.width,
                              radar_grid_obj.length, 1,
                              gdal.GDT_Float32, 'ENVI')

    # Build the Rdr2Geo object
    rdr2geo_obj = Rdr2Geo(radar_grid_obj, orbit_obj, ellipsoid, grid_doppler,
                          dem_interp_method=interp_method,
                          threshold=threshold, numiter=numiter,
                          extraiter=extraiter,
                          lines_per_block=lines_per_block)

    x_raster, y_raster, incidence_raster,\
        heading_raster, local_incidence_raster, local_psi_raster,\
            simulated_amplitude_raster, shadow_raster,\
                ground_to_sat_x_ratser, ground_to_sat_y_raster= [None] * 10
    rdr2geo_obj.topo(dem_raster, x_raster, y_raster, dem_src,
                     incidence_raster, heading_raster, local_incidence_raster,
                     local_psi_raster, simulated_amplitude_raster,
                     shadow_raster,
                     ground_to_sat_x_ratser, ground_to_sat_y_raster)

    # Clean the memory
    dem_raster = None
    rdr2geo_obj = None
    dem_src = None


def _subswath_numbers(subswaths,
                      intervals,
                      azi_idx_arr,
                      rg_idx_arr):
    """
    Vectorized equivalent of SubSwaths.get_sample_sub_swath over index
    arrays.

    Returns 0 for out-of-swath samples, otherwise the 1-based number of
    the first sub-swath whose per-line valid-sample interval
    [start, end) contains the sample. An empty interval array claims
    every in-bounds sample (matching the scalar API's short-circuit),
    and a dataset without sub-swath information assigns 1 everywhere in
    bounds.

    Parameters
    ----------
    subswaths : isce3.product.SubSwaths
        The subswath object of the RSLC
    intervals : list of numpy.ndarray
        Per-sub-swath [start, end) valid-sample interval arrays, i.e.
        [subswaths.get_valid_samples_array(s) for s = 1..num_sub_swaths]
    azi_idx_arr : numpy.ndarray
        Integer azimuth indices
    rg_idx_arr : numpy.ndarray
        Integer slant range indices

    Returns
    ----------
    numpy.ndarray
        int64 sub-swath numbers, same shape as the index arrays
    """
    in_bounds = ((azi_idx_arr >= 0) & (azi_idx_arr < subswaths.length) &
                 (rg_idx_arr >= 0) & (rg_idx_arr < subswaths.width))
    numbers = np.zeros(azi_idx_arr.shape, dtype=np.int64)
    if not intervals:
        return np.where(in_bounds, np.int64(1), numbers)

    # Clipped so the per-line gather stays legal; out-of-bounds samples
    # are excluded through in_bounds
    azi_gather = np.clip(azi_idx_arr, 0, subswaths.length - 1)
    for number, interval in enumerate(intervals, start=1):
        if interval.size == 0:
            claimed = in_bounds
        else:
            claimed = (in_bounds &
                       (rg_idx_arr >= interval[azi_gather, 0]) &
                       (rg_idx_arr < interval[azi_gather, 1]))
        unassigned = numbers == 0
        numbers[unassigned & claimed] = number
        if not unassigned.any():
            break

    return numbers


class _RSLCInputDataExceptionMask:
    """
    Sliding-window reader for an RSLC inputDataExceptionMask dataset.

    Keeps one contiguous block of lines resident, in the dataset's
    native dtype, and reads a new block only when a request falls
    outside it. New blocks extend from the request in the direction
    the requests are moving and are snapped to the dataset's chunk
    rows, so a monotonic sweep through the radar grid reads each chunk
    once.

    Parameters
    ----------
    dataset : h5py.Dataset or None
        The inputDataExceptionMask dataset, of shape (lines, samples).
        None if the RSLC has no such dataset, in which case every
        request returns uint8 zeros without any I/O.
    lines : int
        Number of lines of the swath
    samples : int
        Number of samples of the swath
    block_lines : int, optional
        Nominal number of lines per block read. Raised to the chunk
        height of the dataset if that is larger.

    Raises
    ------
    ValueError
        If the dataset shape differs from (lines, samples)

    Notes
    -----
    Line indices are bounds-checked on every request; sample indices
    are not, so negative sample indices would wrap around as in NumPy.
    """

    def __init__(self, dataset, lines, samples, block_lines=256):
        self._dset = dataset
        self._lines = lines
        self._start = 0
        if dataset is None:
            # Zero-stride view of a single zero: spans the whole grid
            # without allocating it, so no request ever triggers a read
            self._block = np.broadcast_to(np.uint8(0), (lines, samples))
            return
        if dataset.shape != (lines, samples):
            raise ValueError(
                f"inputDataExceptionMask shape {dataset.shape} differs "
                f"from the swath shape {(lines, samples)}")
        self._chunk_lines = dataset.chunks[0] if dataset.chunks else 1
        self._block_lines = max(block_lines, self._chunk_lines)
        self._block = np.empty((0, samples), dtype=dataset.dtype)

    def _ensure(self, lo, hi):
        """
        Make lines lo..hi (inclusive) resident.

        Parameters
        ----------
        lo : int
            First line requested
        hi : int
            Last line requested, hi >= lo

        Raises
        ------
        IndexError
            If the line range is not within [0, lines)
        """
        if lo < 0 or hi >= self._lines:
            raise IndexError(
                f"lines {lo}..{hi} outside the radar grid "
                f"[0, {self._lines})")
        end = self._start + len(self._block)
        if self._start <= lo and hi < end:
            return
        if lo >= end:
            # Moving forward: read ahead of the request
            start = lo
        elif hi < self._start:
            # Moving backward: read behind the request
            start = min(lo, hi + 1 - self._block_lines)
        else:
            # Request straddles the block: center on it
            start = lo - max(0, self._block_lines - (hi - lo + 1)) // 2
        chunk = self._chunk_lines
        start = max(0, start) // chunk * chunk
        stop = max(hi + 1, start + self._block_lines)
        stop = min(self._lines, -(-stop // chunk) * chunk)
        self._block = self._dset[start:stop]
        self._start = start

    def row(self, i, rg_idx):
        """
        Values of one line at the given samples.

        Parameters
        ----------
        i : int
            Line index
        rg_idx : numpy.ndarray
            1-D integer sample indices, all in bounds

        Returns
        -------
        numpy.ndarray
            Values in the dataset's native dtype, same shape as rg_idx

        Raises
        ------
        IndexError
            If i is not within [0, lines)
        """
        self._ensure(i, i)
        return self._block[i - self._start, rg_idx]

    def gather(self, az_idx, rg_idx, valid):
        """
        Values at scattered (line, sample) positions.

        Parameters
        ----------
        az_idx : numpy.ndarray
            1-D integer line indices
        rg_idx : numpy.ndarray
            1-D integer sample indices, same shape as az_idx
        valid : numpy.ndarray
            1-D boolean array, same shape as az_idx. Indices only need
            to be in bounds where it is True.

        Returns
        -------
        numpy.ndarray
            Values in the dataset's native dtype where valid is True
            and 0 elsewhere, same shape as az_idx

        Raises
        ------
        IndexError
            If any valid line index is not within [0, lines)
        """
        out = np.zeros(az_idx.shape, dtype=self._block.dtype)
        sel = np.flatnonzero(valid)
        if sel.size:
            az = az_idx[sel]
            self._ensure(int(az.min()), int(az.max()))
            out[sel] = self._block[az - self._start, rg_idx[sel]]
        return out


def _subswath_numbers(subswaths, intervals, azi_idx, rg_idx):
    """
    Vectorized equivalent of SubSwaths.get_sample_sub_swath.

    Returns 0 for out-of-swath samples, otherwise the 1-based number of
    the first sub-swath whose per-line valid-sample interval
    [start, end) contains the sample. An empty interval array claims
    every in-bounds sample (matching the scalar API's short-circuit),
    and a dataset without sub-swath information assigns 1 everywhere in
    bounds.

    Parameters
    ----------
    subswaths : isce3.product.SubSwaths
        The sub-swaths object of the RSLC
    intervals : list of numpy.ndarray
        Per-sub-swath [start, end) valid-sample interval arrays, i.e.
        [subswaths.get_valid_samples_array(s) for s = 1..num_sub_swaths]
    azi_idx : int or numpy.ndarray
        Integer azimuth indices, broadcastable against rg_idx
    rg_idx : int or numpy.ndarray
        Integer slant range indices, broadcastable against azi_idx

    Returns
    -------
    numpy.ndarray
        np.byte sub-swath numbers, of the broadcast shape of the indices
    """
    in_bounds = ((azi_idx >= 0) & (azi_idx < subswaths.length) &
                 (rg_idx >= 0) & (rg_idx < subswaths.width))
    numbers = np.zeros_like(in_bounds, dtype=np.uint8)
    if not intervals:
        return np.where(in_bounds, np.uint8(1), numbers)

    # Clipped so the per-line gather stays legal; out-of-bounds samples
    # are excluded through in_bounds
    azi_gather = np.clip(azi_idx, 0, subswaths.length - 1)
    remaining = in_bounds
    for number, interval in enumerate(intervals, start=1):
        if interval.size == 0:
            claimed = remaining
        else:
            claimed = (remaining &
                       (rg_idx >= interval[azi_gather, 0]) &
                       (rg_idx < interval[azi_gather, 1]))
        numbers[claimed] = number
        remaining = remaining & ~claimed
        if not remaining.any():
            break

    return numbers


def generate_insar_mask(ref_rslc_obj,
                        sec_rslc_obj,
                        ref_rslc_h5_obj,
                        sec_rslc_h5_obj,
                        range_offset_path,
                        azimuth_offset_path,
                        freq,
                        azi_idx_arr,
                        rg_idx_arr):
    """
    Generate the InSAR mask on a grid of reference radar-grid indices.

    Each mask value is a uint32 packing:

    - bits 0-7: 10 * reference sub-swath number + secondary sub-swath
      number, where 0 means the sample is outside that RSLC's swath
    - bits 8-15: low 8 bits of the secondary inputDataExceptionMask
    - bits 16-23: low 8 bits of the reference inputDataExceptionMask

    The geometric coregistration offsets are read at the truncated
    reference indices of each output pixel and the secondary position
    is rounded to the nearest secondary sample. Output pixels outside
    the reference radar grid are 0.

    Parameters
    ----------
    ref_rslc_obj : SLC
        The SLC object for the reference RSLC
    sec_rslc_obj : SLC
        The SLC object for the secondary RSLC
    ref_rslc_h5_obj : h5py.File
        The opened HDF5 file of the reference RSLC
    sec_rslc_h5_obj : h5py.File
        The opened HDF5 file of the secondary RSLC
    range_offset_path : str
        The path of the range offset raster from geo2rdr, on the
        reference radar grid
    azimuth_offset_path : str
        The path of the azimuth offset raster from geo2rdr, on the
        reference radar grid
    freq : str
        The swath frequency ('A' or 'B')
    azi_idx_arr : numpy.ndarray
        1-D azimuth indices of the output rows in the reference radar
        grid; may be fractional or outside the grid
    rg_idx_arr : numpy.ndarray
        1-D slant range indices of the output columns in the reference
        radar grid; may be fractional or outside the grid

    Returns
    -------
    numpy.ndarray
        uint32 mask of shape (len(azi_idx_arr), len(rg_idx_arr))
    """

    # Reference and secondary RSLC swaths
    ref_swath = ref_rslc_obj.getSwathMetadata(freq)
    sec_swath = sec_rslc_obj.getSwathMetadata(freq)
    ref_subswaths = ref_swath.sub_swaths()
    sec_subswaths = sec_swath.sub_swaths()

    # Fetch each sub-swath's per-line valid-sample interval array once
    # (1-based API); the per-sample sub-swath tests then run as numpy
    # array operations instead of scalar SubSwaths.get_sample_sub_swath
    # calls per output pixel
    ref_intervals = [ref_subswaths.get_valid_samples_array(s)
                     for s in range(1, ref_subswaths.num_sub_swaths + 1)]
    sec_intervals = [sec_subswaths.get_valid_samples_array(s)
                     for s in range(1, sec_subswaths.num_sub_swaths + 1)]

    # Range and azimuth offset rasters, read one line at a time in the
    # loop below (the datasets are kept alive while the bands are used)
    src_range_offset = gdal.Open(range_offset_path)
    src_azimuth_offset = gdal.Open(azimuth_offset_path)
    range_offset_band = src_range_offset.GetRasterBand(1)
    azimuth_offset_band = src_azimuth_offset.GetRasterBand(1)

    # Input data exception masks, opened but not loaded; lines are read
    # in blocks as the loop below sweeps through the radar grid
    def _open_exception_mask(h5_obj, rslc_obj, swath):
        path = f"{rslc_obj.SwathPath}/frequency{freq}/inputDataExceptionMask"
        return _RSLCInputDataExceptionMask(
            h5_obj.get(path), swath.lines, swath.samples)

    ref_exception_mask = _open_exception_mask(ref_rslc_h5_obj,
                                              ref_rslc_obj,
                                              ref_swath)
    sec_exception_mask = _open_exception_mask(sec_rslc_h5_obj,
                                              sec_rslc_obj,
                                              sec_swath)

    # Integer reference indices of the output columns: int() truncates
    # toward zero, as does astype
    rg_idx_int = rg_idx_arr.astype(np.int64)
    col_out_of_swath = (rg_idx_arr < 0) | (rg_idx_arr >= ref_swath.samples)
    # Clipped copy so the per-line gathers stay legal; out-of-swath
    # columns are zeroed at the end of each row
    rg_gather = np.clip(rg_idx_int, 0, ref_swath.samples - 1)

    mask = np.zeros((len(azi_idx_arr), len(rg_idx_arr)), dtype=np.uint32)
    # Polarization dependent valid mask
    pol_valid_mask = np.zeros((len(azi_idx_arr), len(rg_idx_arr)), dtype=np.uint16)

    for row, i in enumerate(azi_idx_arr):
        # Rows outside the reference radar grid stay 0
        if not (0 <= i < ref_swath.lines):
            continue

        # Geometric coregistration offsets at the truncated reference
        # indices of the output pixels
        i_int = int(i)
        rg_off = range_offset_band.ReadAsArray(
            0, i_int, ref_swath.samples, 1)[0][rg_gather]
        az_off = azimuth_offset_band.ReadAsArray(
            0, i_int, ref_swath.samples, 1)[0][rg_gather]

        # Sub-swath numbers of the reference RSLC and, at the nearest
        # secondary sample (int(x + 0.5) of the scalar code, i.e.
        # truncation toward zero), of the secondary RSLC
        ref_num = _subswath_numbers(ref_subswaths, ref_intervals,
                                    i_int, rg_idx_int)
        sec_num = _subswath_numbers(
            sec_subswaths, sec_intervals,
            np.trunc(i_int + az_off + 0.5).astype(np.int64),
            np.trunc(rg_idx_int + rg_off + 0.5).astype(np.int64))
        mask_row = (10 * ref_num + sec_num).astype(np.uint32)

        # Reference RSLC input exception mask bits: keep the low 8 bits,
        # then widen to uint32 before the shift so the packing is safe
        # under NEP 50 scalar promotion as well
        ref_exception_mask_row = ref_exception_mask.row(i_int, rg_gather)
        mask_row |= (ref_exception_mask_row
                     .astype(np.uint8).astype(np.uint32) << 16)

        # polarization dependent mask for the reference RSLC
        pol_mask_row = ref_exception_mask_row & np.uint16(0xFF00)

        # Secondary RSLC input exception mask bits at the nearest
        # secondary sample (round() of the scalar code, i.e. half to
        # even, as np.rint); out-of-swath samples are zeroed by gather()
        sec_i = np.rint(i + az_off).astype(np.int64)
        sec_j = np.rint(rg_idx_arr + rg_off).astype(np.int64)
        sec_in_swath = ((sec_i >= 0) & (sec_i < sec_swath.lines) &
                        (sec_j >= 0) & (sec_j < sec_swath.samples))

        sec_exception_mask_row = sec_exception_mask.gather(sec_i, sec_j, sec_in_swath)
        mask_row |= (sec_exception_mask_row
                     .astype(np.uint8).astype(np.uint32) << 8)

        # polarization dependent mask combing with the secondary RSLC
        pol_mask_row |= (sec_exception_mask_row & np.uint16(0xFF00)) >> 8

        mask_row[col_out_of_swath] = 0
        pol_mask_row[col_out_of_swath] = 0

        mask[row] = mask_row
        pol_valid_mask[row] = pol_mask_row

    return mask, pol_valid_mask


def extract_pol_valid_mask(pol_valid_mask, pol):
    """
    Extract polarization-dependent valid mask from the combined mask.

    Creates a binary mask where bit 1 indicates reference polarization validity
    and bit 0 indicates secondary polarization validity.

    Parameters
    ----------
    pol_valid_mask : numpy.ndarray
        uint16 mask array where bits 8-15 are for reference polarization
        and bits 0-7 are for secondary polarization. Each bit corresponds
        to a polarization: HH(0), HV(1), VH(2), VV(3), LH(4), LV(5), RH(6), RV(7)
    pol : str
        Polarization identifier (e.g., 'HH', 'HV', 'VH', 'VV', 'LH', 'LV', 'RH', 'RV')

    Returns
    -------
    numpy.ndarray
        uint8 array where bit 1 = reference valid (1=valid, 0=invalid)
        and bit 0 = secondary valid (1=valid, 0=invalid)
    """
    # Map polarization to bit position (0-7) based on the standard order
    pol_to_bit = {'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3,
                  'LH': 4, 'LV': 5, 'RH': 6, 'RV': 7}
    bit_pos = pol_to_bit.get(pol, 0)

    # Extract reference (high byte) and secondary (low byte) bits
    ref_valid = (pol_valid_mask >> (bit_pos + 8)) & 1
    sec_valid = (pol_valid_mask >> bit_pos) & 1

    # Create binary mask: bit 1 = reference, bit 0 = secondary
    valid_mask = ((ref_valid << 1) | sec_valid).astype(np.uint8)

    return valid_mask