from __future__ import annotations
from types import SimpleNamespace
from typing import Union

import isce3
from isce3.ext.isce3.geocode import _geocode_slc
import journal
import numpy as np


def _io_value_check(io: Union[np.ndarray, list[np.ndarray]]):
    '''
    Check validity of input or output array(s). Results are stored in a
    SimpleNamespace and returned for examination and comparison in _is_valid()
    below
    '''
    # default to checks to invalid state with validity determined below
    io_checks = SimpleNamespace(is_list=False, is_array=False, n_items=0,
                                shapes_consistent=False,
                                dtypes_consistent=False, is_valid=False)

    # if item is list, check number of elements inside
    if isinstance(io, list):
        io_checks.is_list = True
        io_checks.is_array = False

        # if list populated, inspect and save inspection results
        if io:
            # save number of items in list
            io_checks.n_items = len(io)

            # check if all items in list are np.ndarray
            all_arrays = all([isinstance(x, np.ndarray) for x in io])

            # if all arrays, check if shapes and dtypes are consistent
            if all_arrays:
                # get all np.ndarray shapes then check if they're the same
                all_shapes = [x.shape for x in io]
                io_checks.shapes_consistent = all([x == all_shapes[0]
                                                   for x in all_shapes])

                # get all np.ndarray dtypes then check if they're the same
                all_dtypes = [x.dtype for x in io]
                io_checks.dtypes_consistent = all([x == all_dtypes[0]
                                                   for x in all_dtypes])

                # valid if shapes and dtypes are consistent thru all arrays
                if io_checks.shapes_consistent and io_checks.dtypes_consistent:
                    io_checks.is_valid = True
    elif isinstance(io, np.ndarray):
        io_checks.is_list = False
        io_checks.is_array = True
        io_checks.n_items = 1
        io_checks.shapes_consistent = True
        io_checks.dtypes_consistent = True
        io_checks.is_valid = True

    return io_checks


def _io_valid(geo_io_checks, rdr_io_checks):
    '''
    Compare geo outputs and radar inputs checks to see if they are compatible
    to be passed to pybind geocodeSlc
    '''
    error_channel = journal.error('isce3.geocode.geocode_slc')

    # convenience function to strip starting 'namespace(' and ending ')' from
    # str(namespace)
    def io_check_contents(x): return str(x)[10:-1]

    if not rdr_io_checks.is_valid:
        err_str = f'radar input is invalid: {io_check_contents(rdr_io_checks)}'
        error_channel.log(err_str)
        raise ValueError(err_str)

    if not geo_io_checks.is_valid:
        err_str = f'geo input is invalid: {io_check_contents(geo_io_checks)}'
        error_channel.log(err_str)
        raise ValueError(err_str)

    if (geo_io_checks.is_array and rdr_io_checks.is_list) or \
            (geo_io_checks.is_list and rdr_io_checks.is_array):
        err_str = 'geo and radar inputs are not the same type'
        error_channel.log(err_str)
        raise ValueError(err_str)

    if (geo_io_checks.is_list and rdr_io_checks.is_list) and \
            geo_io_checks.n_items != rdr_io_checks.n_items:
        err_str = f'length of geo input list ({geo_io_checks.n_items}) and radar input list ({rdr_io_checks.n_items}) do not match'
        error_channel.log(err_str)
        raise ValueError(err_str)


def _output_array_valid(output_array, geo_array, which_output):
    '''
    Compare geo output array shape and output array shape to see if they match
    '''

    # Skip the size check when the output array is not provided
    if output_array is None:
        return

    # default output_array has size==0
    if output_array.size:
        if output_array.shape != geo_array.shape:
            err_str = f'{which_output} output array shape does not match geocoded array shape'
            error_channel.log(err_str)
            raise ValueError(err_str)


def geocode_slc(geo_data_blocks: Union[np.ndarray, list[np.ndarray]],
                rdr_data_blocks: Union[np.ndarray, list[np.ndarray]],
                dem_raster, radargrid,
                geogrid, orbit, native_doppler, image_grid_doppler,
                ellipsoid, threshold_geo2rdr, num_iter_geo2rdr,
                mask_block=None,
                sliced_radargrid=None, subswaths=None,
                first_azimuth_line=0,  first_range_sample=0, flatten=True,
                reramp=True,
                az_carrier=isce3.core.LUT2d(),
                rg_carrier=isce3.core.LUT2d(),
                az_time_correction=isce3.core.LUT2d(),
                srange_correction=isce3.core.LUT2d(),
                flatten_with_corrected_srange=False,
                invalid_value=np.nan + np.nan * 1j,
                carrier_phase_block: Optional[np.ndarray] = np.array([],
                                                                     dtype=np.float64),
                flatten_phase_block: Optional[np.ndarray] = np.array([],
                                                                     dtype=np.float64)):
    '''
    Geocode a subset of pixels for multiple radar SLC arrays to a given geogrid.
    All radar SLC arrays share a common radar grid. All output geocoded arrays
    share a common geogrid. Subset of pixels defined by a sliced radar grid - a
    radar grid contained within the common radar grid.

    Parameters
    ----------
    geo_data_blocks: list of numpy.ndarray
        List of output arrays containing geocoded SLC
    rdr_data_blocks: list of numpy.ndarray
        List of input arrays of the SLC in radar coordinates
    dem_raster: isce3.io.Raster
        Raster of the DEM
    radargrid: isce3.product.RadarGridParameters
        Radar grid parameters of input SLC raster
    geogrid: GeoGridParameters
        Geo grid parameters of output raster
    orbit: isce3.core.Orbit
        Orbit object associated with radar grid
    native_doppler: LUT2d
        2D LUT doppler of the SLC image
    image_grid_doppler: LUT2d
        2d LUT doppler of the image grid
    ellipsoid: Ellipsoid
        Ellipsoid object
    threshold_geo2rdr: float
        Threshold for geo2rdr computations
    num_iter_geo2rdr: int
        Maximum number of iterations for geo2rdr convergence
    mask_block: numpy.ndarray
        Output array containing masking values of geocoded SLC
    sliced_radargrid: RadarGridParameters
        Radar grid representing subset of radargrid
    subswaths: SubSwaths or None, default=None
        Subswath mask representing valid portions of a swath
    azimuth_first_line: int
        FIrst line of radar data block with respect to larger radar data raster, else 0
    range_first_pixel: int
        FIrst pixel of radar data block with respect to larger radar data raster, else 0
    flatten: bool
        Flag to flatten the geocoded SLC
    reramp: bool
        Flag to reramp the geocoded SLC
    azimuth_carrier: [LUT2d, Poly2d]
        Azimuth carrier phase of the SLC data, in radians, as a function of azimuth and range
    range_carrier: [LUT2d, Poly2d]
        Range carrier phase of the SLC data, in radians, as a function of azimuth and range
    az_time_correction: LUT2d
         geo2rdr azimuth additive correction, in seconds, as a function of azimuth and range
    srange_correction: LUT2d
        geo2rdr slant range additive correction, in meters, as a function of azimuth and range
    flatten_with_corrected_srange: bool
        flag to indicate whether geo2rdr slant-range additive values should be used for phase flattening
    invalid_value: complex
        invalid pixel fill value
    carrier_phase_block: numpy.ndarray
        Optional output array containing geocoded carrier phase
    flatten_phase_block: numpy.ndarray
        Optional output array containing geocoded range offset phase
    '''
    geo_io_checks = _io_value_check(geo_data_blocks)
    rdr_io_checks = _io_value_check(rdr_data_blocks)

    # check validity and consistency of input/output - raise errors if invalid
    _io_valid(geo_io_checks, rdr_io_checks)

    # if no sliced radar grid given, use full radar grid as sliced radar grid
    if sliced_radargrid is None:
        sliced_radargrid = radargrid

    # Pass the parameters with kwargs for better manifestation
    # Exclude `subswaths` and `mask_block` which will be provided only when
    # both of them are not None
    kwargs = {
        'geo_data_blocks': geo_data_blocks,
        'carrier_phase_block': carrier_phase_block,
        'flatten_phase_block': flatten_phase_block,
        'rdr_data_blocks': rdr_data_blocks,
        'dem_raster': dem_raster,
        'radargrid': radargrid,
        'sliced_radar_grid': sliced_radargrid,
        'geogrid': geogrid,
        'orbit': orbit,
        'native_doppler': native_doppler,
        'image_grid_doppler':image_grid_doppler,
        'ellipsoid': ellipsoid,
        'threshold_geo2rdr': threshold_geo2rdr,
        'numiter_geo2rdr': num_iter_geo2rdr,
        #'mask_block': mask_block,
        'azimuth_first_line': first_azimuth_line,
        'range_first_pixel': first_range_sample,
        'flatten': flatten,
        'reramp': reramp,
        'az_carrier': az_carrier,
        'rg_carrier': rg_carrier,
        'az_time_correction': az_time_correction,
        'srange_correction': srange_correction,
        'flatten_with_corrected_srange': flatten_with_corrected_srange,
        'invalid_value': invalid_value,
        #'subswaths': subswaths
    }

    if subswaths is not None:
        kwargs['subswaths'] = subswaths

    if mask_block is not None:
        kwargs['mask_block'] = mask_block

    # if both geo and radar blocks are np.ndarray, put them into a list
    # to match expected input type of list[np.ndarray]
    if geo_io_checks.is_array and rdr_io_checks.is_array:
        geo_data_blocks = [geo_data_blocks]
        rdr_data_blocks = [rdr_data_blocks]

    # check if non-geocoded output array shapes matches geocoded array shape.
    for output_arr, which_output in zip([carrier_phase_block,
                                         flatten_phase_block,
                                         mask_block],
                                        ['carrier', 'flattening', 'mask']):
        _output_array_valid(output_arr, geo_data_blocks[0], which_output)

    _geocode_slc(**kwargs)

    # TODO delete the string below when it is no longer needed.
    '''_geocode_slc(geo_data_blocks=geo_data_blocks,
                 carrier_phase_block=carrier_phase_block,
                 flatten_phase_block=flatten_phase_block,
                 rdr_data_blocks=rdr_data_blocks,
                 dem_raster=dem_raster,
                 radargrid=radargrid,
                 sliced_radar_grid=sliced_radargrid,
                 geogrid=geogrid,
                 orbit=orbit,
                 native_doppler=native_doppler,
                 image_grid_doppler=image_grid_doppler,
                 ellipsoid=ellipsoid,
                 threshold_geo2rdr=threshold_geo2rdr,
                 numiter_geo2rdr=num_iter_geo2rdr,
                 mask_block=mask_block,
                 azimuth_first_line=first_azimuth_line,
                 range_first_pixel=first_range_sample,
                 flatten=flatten,
                 reramp=reramp,
                 az_carrier=az_carrier,
                 rg_carrier=rg_carrier,
                 az_time_correction=az_time_correction,
                 srange_correction=srange_correction,
                 flatten_with_corrected_srange=flatten_with_corrected_srange,
                 invalid_value=invalid_value,
                 subswaths=subswaths)'''
