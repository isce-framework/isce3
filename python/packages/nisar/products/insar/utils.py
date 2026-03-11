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


def generate_insar_subswath_mask(ref_rslc_obj,
                                 sec_rslc_obj,
                                 range_offset_path,
                                 azimuth_offset_path,
                                 freq,
                                 azi_idx_arr,
                                 rg_idx_arr):

    """
    Generate the InSAR subswath 2d array mask

    Parameters
    ---------
    ref_rslc_obj : SLC
        The SLC object for the reference RSLC
    sec_rslc_obj : SLC
        The SLC object for the secondary RSLC
    range_offset_path : str
        The path of the range offset product from geo2rdr
    azimuth_offset_path : str
        The path of the azimuth offset product from the geo2r
    freq : str
        The swath frequency ('A' or 'B')
    azi_idx_arr : np.ndarray
        The index array along the azimuth direction
    rg_idx_arr : np.ndarray
        The index array along the range direction

    Returns
    ----------
    numpy.ndarray
        subswath mask at a given frequency
    """

    # Reference and Secondary RSLC files
    ref_swath = ref_rslc_obj.getSwathMetadata(freq)
    ref_subswaths = ref_rslc_obj.getSwathMetadata(freq).sub_swaths()
    sec_subswaths = sec_rslc_obj.getSwathMetadata(freq).sub_swaths()

    # Read the range and azimuth offsets products
    src_range_offset = gdal.Open(range_offset_path)
    src_azimuth_offset = gdal.Open(azimuth_offset_path)

    range_offset_band = src_range_offset.GetRasterBand(1)
    azimuth_offset_band = src_azimuth_offset.GetRasterBand(1)

    subswath_mask = []
    for i in azi_idx_arr:
        # Check if the azimuth index is within the radar grid
        if i >= 0 and i < ref_swath.lines:
            range_off = \
                range_offset_band.ReadAsArray(0,
                                            int(i),
                                            ref_swath.samples,
                                            1)
            azimuth_off = \
                azimuth_offset_band.ReadAsArray(0,
                                                int(i),
                                                ref_swath.samples,
                                                1)
            for j in rg_idx_arr:
                # Initialize the subswath mask id to be 0
                subswath_mask_id = 0
                # Check if the range index is within the swath
                if j >= 0 and j < ref_swath.samples:
                    subswath_mask_id =  _compute_subswath_mask_id(int(i),int(j),
                                            azimuth_off[0,int(j)],
                                            range_off[0,int(j)],
                                            ref_subswaths,
                                            sec_subswaths)
                subswath_mask.append(subswath_mask_id)
        # The azimuth index is not in the radar grid meaning no subswath mask
        else:
            subswath_mask += [0] * len(rg_idx_arr)

    return np.array(subswath_mask).reshape(
        (len(azi_idx_arr),
         len(rg_idx_arr))).astype(np.uint8)