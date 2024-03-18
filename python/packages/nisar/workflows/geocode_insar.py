#!/usr/bin/env python3

"""
collection of functions for NISAR geocode workflow
"""
import os
import pathlib
import time
from enum import Enum

import h5py
import isce3
import journal
import numpy as np
from nisar.products.insar.product_paths import (CommonPaths, GOFFGroupsPaths,
                                                GUNWGroupsPaths,
                                                RIFGGroupsPaths,
                                                ROFFGroupsPaths,
                                                RUNWGroupsPaths)
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows import prepare_insar_hdf5
from nisar.workflows.compute_stats import (compute_layover_shadow_water_stats,
                                           compute_stats_real_data)
from nisar.workflows.geocode_corrections import get_az_srg_corrections
from nisar.workflows.geocode_insar_runconfig import GeocodeInsarRunConfig
from nisar.workflows.helpers import get_cfg_freq_pols
from nisar.workflows.yaml_argparse import YamlArgparse
from osgeo import gdal


class InputProduct(Enum):
    '''
    The input product type to geocode
    '''
    # RUWN product
    RUNW = 1
    # ROFF product
    ROFF = 2
    # RIFG product
    RIFG = 3


def run(cfg, input_hdf5, output_hdf5, input_product_type=InputProduct.RUNW):
    """ Run geocode insar on user specified hardware

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    input_hdf5 : str
        Path input RUNW, ROFF, or RIFG HDF5
    output_hdf5 : str
        Path to output GUNW or GOFF HDF5
    input_product_type: enum
        Input product type of the input_hdf5 to geocode
    """
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)
        gpu_run(cfg, input_hdf5, output_hdf5, input_product_type)
    else:
        cpu_run(cfg, input_hdf5, output_hdf5, input_product_type)


def get_shadow_input_output(scratch_path, freq, dst_freq_path):
    """ Create input raster object and output dataset path for shadow layover

    Parameters
    ----------
    scratch_path : pathlib.Path
        Scratch path to shadow layover mask rasters
    freq : str
        Frequency, A or B, of shadow layover mask raster
    dst_freq_path : str
        HDF5 path to destination frequency group of geocoded shadow layover

    Returns
    -------
    input_raster : isce3.io.Raster
        Shadow layover input raster object
    dataset_path : str
        HDF5 path to geocoded shadow layover dataset
    """
    raster_ref =f'{str(scratch_path)}/rdr2geo/freq{freq}/layoverShadowMask.rdr'
    input_raster = isce3.io.Raster(str(raster_ref))

    # access the HDF5 dataset for layover shadow mask
    dataset_path = f"{dst_freq_path}/unwrappedInterferogram/mask"

    return input_raster, dataset_path


def get_ds_input_output(src_freq_path, dst_freq_path, pol, input_hdf5,
                        dataset_name, off_layer=None,
                        input_product_type=InputProduct.RUNW):
    """ Create input raster object and output dataset path for datasets outside

    Parameters
    ----------
    src_freq_path : str
        HDF5 path to input frequency group of input dataset
    dst_freq_path : str
        HDF5 path to input frequency group of output dataset
    pol : str
        Polarity of input dataset
    input_hdf5 : str
        Path to input RUNW or ROFF HDF5
    dataset_name : str
        Name of dataset to be geocoded
    off_layer: str
        Name of offset layer
    input_product_type: enum
        Input product type, which is one of RUNW, ROFF, RIFG

    Returns
    -------
    input_raster : isce3.io.Raster
        Shadow layover input raster object
    dataset_path : str
        HDF5 path to geocoded shadow layover dataset
    """

    if dataset_name in ['alongTrackOffset', 'slantRangeOffset',
                        'correlationSurfacePeak'] and \
            input_product_type is InputProduct.RUNW:
        src_group_path = f'{src_freq_path}/pixelOffsets/{pol}'
        dst_group_path = f'{dst_freq_path}/pixelOffsets/{pol}'
    else:
        src_group_path = f'{src_freq_path}/interferogram/{pol}'
        dst_group_path = f'{dst_freq_path}/interferogram/{pol}'

        # RUNW and RIFG product
        if input_product_type is InputProduct.RUNW:
            dst_group_path = f'{dst_freq_path}/unwrappedInterferogram/{pol}'
        elif input_product_type is InputProduct.RIFG:
            dst_group_path = f'{dst_freq_path}/wrappedInterferogram/{pol}'

    if input_product_type is InputProduct.ROFF:
        src_group_path = f'{src_freq_path}/pixelOffsets/{pol}/{off_layer}'
        dst_group_path = f'{dst_freq_path}/pixelOffsets/{pol}/{off_layer}'

    # prepare input raster
    input_raster_str = f"HDF5:{input_hdf5}:/{src_group_path}/{dataset_name}"
    input_raster = isce3.io.Raster(input_raster_str)

    # access the HDF5 dataset for a given frequency and pol
    dataset_path = f"{dst_group_path}/{dataset_name}"

    return input_raster, dataset_path


def get_offset_radar_grid(cfg, radar_grid_slc):
    ''' Create radar grid object for offset datasets

    Parameters
    ----------
    cfg : dict
        Dictionary containing processing parameters
    radar_grid_slc : SLC
        Object containing SLC properties
    '''
    # Define margin used during dense offsets execution
    if cfg['processing']['dense_offsets']['enabled']:
        offset_cfg = cfg['processing']['dense_offsets']
    else:
        offset_cfg = cfg['processing']['offsets_product']
    error_channel = journal.error('geocode_insar.get_offset_radar_grid')
    margin = max(offset_cfg['margin'],
                 offset_cfg['gross_offset_range'],
                 offset_cfg['gross_offset_azimuth'])
    rg_start = offset_cfg['start_pixel_range']
    az_start = offset_cfg['start_pixel_azimuth']
    off_length = offset_cfg['offset_length']
    off_width = offset_cfg['offset_width']

    if cfg['processing']['offsets_product']['enabled']:
        # In case both offset_product and dense_offsets are enabled,
        # it is necessary to re-assgin the 'offsets_product' to offset_cfg
        offset_cfg = cfg['processing']['offsets_product']
        az_search = np.inf
        rg_search = np.inf
        az_window = np.inf
        rg_window = np.inf
        layer_names = [key for key in offset_cfg if key.startswith('layer')]
        if not layer_names:
            err_str = 'No offset layer found'
            error_channel.log(err_str)
            raise KeyError(err_str)
        # Extract search/chip windows per layer; default to inf if not found
        for key in layer_names:
            az_search = min(offset_cfg[key].get('half_search_azimuth', np.inf),
                            az_search)
            rg_search = min(offset_cfg[key].get('half_search_range', np.inf),
                            rg_search)
            az_window = min(offset_cfg[key].get('window_azimuth', np.inf),
                            az_window)
            rg_window = min(offset_cfg[key].get('window_range', np.inf),
                            rg_window)
        # Check if any value is Inf and raise exception
        if np.inf in [az_search, rg_search, az_window, rg_window]:
            err_str = "Half search or chip window is Inf"
            error_channel.log(err_str)
            raise ValueError(err_str)
    else:
        # In case both offset_product and dense_offsets are enabled,
        # it is necessary to re-assgin the 'dense_offsets' to offset_cfg
        offset_cfg = cfg['processing']['dense_offsets']
        az_search = offset_cfg['half_search_azimuth']
        rg_search = offset_cfg['half_search_range']
        az_window = offset_cfg['window_azimuth']
        rg_window = offset_cfg['window_range']

    # If not allocated, determine shape of the offsets
    if off_length is None:
        length_margin = 2 * margin + 2 * az_search + az_window
        off_length = (radar_grid_slc.length - length_margin) \
                     // offset_cfg['skip_azimuth']
    if off_width is None:
        width_margin = 2 * margin + 2 * rg_search + rg_window
        off_width = (radar_grid_slc.width - width_margin) // \
                    offset_cfg['skip_azimuth']
    # Determine the starting range and sensing start for the offset radar grid
    if rg_start is None:
        rg_start = margin + rg_search
    if az_start is None:
        az_start = margin + az_search
    offset_starting_range = radar_grid_slc.starting_range + \
                            (rg_start + rg_window//2)\
                            * radar_grid_slc.range_pixel_spacing
    offset_sensing_start = radar_grid_slc.sensing_start + \
                           (az_start + az_window//2)\
                           / radar_grid_slc.prf
    # Range spacing for offsets
    offset_range_spacing = radar_grid_slc.range_pixel_spacing * offset_cfg['skip_range']
    offset_prf = radar_grid_slc.prf / offset_cfg['skip_azimuth']

    # Create offset radar grid
    radar_grid = isce3.product.RadarGridParameters(offset_sensing_start,
                                                   radar_grid_slc.wavelength,
                                                   offset_prf,
                                                   offset_starting_range,
                                                   offset_range_spacing,
                                                   radar_grid_slc.lookside,
                                                   off_length,
                                                   off_width,
                                                   radar_grid_slc.ref_epoch)
    return radar_grid


def _project_water_to_geogrid(input_water_path, geogrid):
    """
    Project water mask to geogrid of GUNW product.

    Parameters
    ----------
    input_water_path : str
        file path for input water mask
    geogrid : isce3.product.GeoGridParameters
        geogrid to map the water mask

    Returns
    -------
    water_mask_interpret : numpy.ndarray
        boolean array (1: water)
    """
    inputraster = gdal.Open(input_water_path)
    output_extent = (geogrid.start_x,
                     geogrid.start_y + geogrid.length * geogrid.spacing_y,
                     geogrid.start_x + geogrid.width * geogrid.spacing_x,
                     geogrid.start_y)

    gdalwarp_options = gdal.WarpOptions(format="MEM",
                                        dstSRS=f"EPSG:{geogrid.epsg}",
                                        xRes=geogrid.spacing_x,
                                        yRes=np.abs(geogrid.spacing_y),
                                        resampleAlg='mode',
                                        outputBounds=output_extent)
    dst_ds = gdal.Warp("", inputraster, options=gdalwarp_options)

    projected_data = dst_ds.ReadAsArray()
    water_mask_interpret = projected_data.astype('uint8') != 0

    return water_mask_interpret


def add_water_to_mask(cfg, freq, geogrid, dst_h5):
    """
    Add water mask to mask layer in GUNW product.

    Parameters
    ----------
    cfg : dict
        Dictionary containing processing parameters
    freq : str
        Frequency, A or B, of water mask raster
    geogrid : isce3.product.GeoGridParameters
        geogrid to map the water mask
    dst_h5 : h5py.File
        h5py.File object where geocoded data is to be written
    """
    water_mask_path = cfg['dynamic_ancillary_file_group']['water_mask_file']

    if water_mask_path is not None:
        freq_path = f'{GUNWGroupsPaths().GridsPath}/frequency{freq}'
        mask_h5_path = f'{freq_path}/unwrappedInterferogram/mask'

        water_mask = _project_water_to_geogrid(water_mask_path, geogrid)
        mask_layer = dst_h5[mask_h5_path][()]

        # The mask layer has the shadow (1), layover (2), and both(3).
        # Here, the water mask (4) is added to the existing info.
        # If the water is coexist with the above (1-3), they will be assigned to
        # new values.
        # shadow + water : 5
        # layover + water : 6
        # layover + shadow + water : 7
        combo_pxl_mask = (mask_layer >= 0) & (mask_layer < 4) & water_mask
        mask_layer[combo_pxl_mask] += 4
        dst_h5[mask_h5_path].write_direct(mask_layer)

def _snake_to_camel_case(snake_case_str):
    splitted_snake_case_str = snake_case_str.split('_')
    return (splitted_snake_case_str[0] +
            ''.join(w.title() for w in splitted_snake_case_str[1:]))

def get_raster_lists(all_geocoded_dataset_flags,
                     desired_geo_dataset_names,
                     freq,
                     pol_list,
                     input_hdf5,
                     dst_h5,
                     offset_params=None,
                     scratch_path='',
                     input_product_type=InputProduct.RUNW,
                     iono_sideband=False,
                     possible_interp_methods=None,
                     possible_invalid_values=None):
    '''
    Get list of isce3.io.rasters to geocode to, corresponding h5py.Datasets,
    input isce3.io.Rasters, interpolation methods, and invalid values based on
    flags retrieved from the runconfig

    Parameters
    ----------
    all_geocoded_dataset_flags : dict
        key: dataset name
        value: whether or not dataset is to be geocoded
    desired_geo_dataset_names : list[str]
        List of names of datasets with a common radar grid that could be
        geocoded. A dataset listed here will be geocoded only if the associated
        value for it in all_geocoded_dataset_flags is True.
    freq : str
        Frequency of datasets to be geocoded
    pol_list : list
        List of polarizations of frequency to be geocoded
    input_hdf5: str
        Path to input RUNW or ROFF HDF5
    dst_h5 : h5py.File
        h5py.File object where geocoded data is to be written
    offset_params: list[tupel[str, isce3.core.DataInterpMethod, float]]
        List of offset layer geocoding params as tuples. Tuples with each tuple
        consisting of offset layer name, interpolation method, and invalid
        value.
    scratch_path : str
        Path to scratch where layover shadow raster is saved
    input_product_type : enum
        Product type of the input_hdf5
    iono_sideband : bool
        Flag to geocode ionosphere phase screen estimated from side-band
    possible_interp_methods: list[isce3.core.DataInterpMethod]
        Used for GPU geocode only. List of possible interpolation methods to be
        applied to possible rasters.
    possible_invalid_values: list[float]
        Used for GPU geocode only. List of invalid values to initialize
        possible rasters with.

    Returns
    -------
    geocoded_rasters = [isce3.io.Raster]
        List of output of to-be geocoded rasters as isce3.io.Rasters objects
    geocoded_datasets = [h5py.Dataset]
        List of h5py.Datasets of to-be geocoded rasters
    input_rasters: list[isce3.io.Raster]
        List of input rasters as isce3.io.Raster objects to be geocoded
    interp_methods = [isce3.core.DataInterpMethod]
        List of interpolation methods for geocoding each raster
    invalid_values: list[float]
        List of invalid values to initialize each raster with
    '''
    if input_product_type is InputProduct.ROFF:
        src_paths_obj = ROFFGroupsPaths()
        dst_paths_obj = GOFFGroupsPaths()
    else:
        src_paths_obj = RIFGGroupsPaths() if input_product_type is InputProduct.RIFG else RUNWGroupsPaths()
        dst_paths_obj = GUNWGroupsPaths()

    src_freq_path = f"{src_paths_obj.SwathsPath}/frequency{freq}"
    dst_freq_path = f"{dst_paths_obj.GridsPath}/frequency{freq}"

    # Ensure possible interpolation methods and invalid values are iterable
    # Following temp variables ensure default parameter not overwritten
    n_possible = len(desired_geo_dataset_names) * [[]]
    _possible_interp_methods = n_possible if possible_interp_methods is None \
        else possible_interp_methods
    _possible_invalid_values = n_possible if possible_invalid_values is None \
        else possible_invalid_values

    # List of input rasters as isce3.io.Raster objects
    input_rasters = []
    # List of output geocoded rasters as isce3.io.Rasters objects
    geocoded_rasters = []
    # List of h5py.Datasets of geocoded rasters
    geocoded_datasets = []
    # List of interpolation methods for geocoding each raster
    interp_methods = []
    # List of invalid values to initialize each raster with
    invalid_values = []

    # Following flag set to True to prevent geocoding layover shadow for
    # different polarizations. Layover shadow does not change with
    # polarization
    skip_layover_shadow = False

    # Iterate through dataset names and their respective interpolation methods
    # and invalid value. Only geocode one where flags for geocoding set to True
    # in runconfig
    for ds_name, interp_method, invalid_value in \
            zip(desired_geo_dataset_names, _possible_interp_methods,
                _possible_invalid_values):
        # Skip geocoding if flag passed all the way from runconfig is False
        if not all_geocoded_dataset_flags[ds_name]:
            continue

        for pol in pol_list:
            # Only geocode layover shadow once. Skip if already geocoded.
            if skip_layover_shadow:
                continue

            # Container for destination/output HDF5 paths of geocoded rasters
            pol_out_ds_paths = []

            # Append input raster object and output HDF5 paths based on product
            if ds_name == "mask":
                raster, path = get_shadow_input_output(
                    scratch_path, freq, dst_freq_path)

                # Set bool to True to ensure layover shadow only geocoded once
                skip_layover_shadow = True

                # Update geocoding parameters
                input_rasters.append(raster)
                pol_out_ds_paths.append(path)
                interp_methods.append(interp_method)
                invalid_values.append(invalid_value)
            elif input_product_type is InputProduct.ROFF:
                ds_name_camel_case = _snake_to_camel_case(ds_name)
                for lay_name, lay_interp_method, lay_invalid in offset_params:
                    raster, path = get_ds_input_output(src_freq_path,
                                                       dst_freq_path,
                                                       pol, input_hdf5,
                                                       ds_name_camel_case,
                                                       lay_name,
                                                       input_product_type)
                    # Update geocoding parameters
                    input_rasters.append(raster)
                    pol_out_ds_paths.append(path)
                    interp_methods.append(lay_interp_method)
                    invalid_values.append(lay_invalid)
            elif iono_sideband and ds_name in ['ionosphere_phase_screen',
                           'ionosphere_phase_screen_uncertainty']:
                # ionosphere_phase_screen from main_side_band or
                # main_diff_ms_band are computed on radargrid of frequencyB.
                # The ionosphere_phase_screen is geocoded on geogrid of
                # frequencyA.
                iono_src_freq_path = f"{src_paths_obj.SwathsPath}/frequencyB"
                iono_dst_freq_path = f"{dst_paths_obj.GridsPath}/frequencyA"
                ds_name_camel_case = _snake_to_camel_case(ds_name)

                raster, path = get_ds_input_output(
                    iono_src_freq_path, iono_dst_freq_path, pol, input_hdf5,
                        ds_name_camel_case)

                # Update geocoding parameters
                input_rasters.append(raster)
                pol_out_ds_paths.append(path)
                interp_methods.append(interp_method)
                invalid_values.append(invalid_value)
            else:
                ds_name_camel_case = _snake_to_camel_case(ds_name)
                raster, path = get_ds_input_output(
                    src_freq_path, dst_freq_path, pol, input_hdf5,
                    ds_name_camel_case, None, input_product_type)

                # Update geocoding parameters
                input_rasters.append(raster)
                pol_out_ds_paths.append(path)
                interp_methods.append(interp_method)
                invalid_values.append(invalid_value)

            # Prepare output raster access the HDF5 dataset for datasets to be
            # geocoded
            for path in pol_out_ds_paths:
                geocoded_dataset = dst_h5[path]
                geocoded_datasets.append(geocoded_dataset)

                # Construct the output raster directly from HDF5 dataset
                geocoded_raster = isce3.io.Raster(
                    f"IH5:::ID={geocoded_dataset.id.id}".encode("utf-8"),
                    update=True)

                geocoded_rasters.append(geocoded_raster)

    # Check all output lists have the same length
    output_lens = [len(x) == len(geocoded_rasters)
                   for x in (geocoded_datasets, input_rasters, interp_methods,
                             invalid_values)]
    if (not all(output_lens)):
        error_channel = journal.error('geocode_insar.get_raster_lists')
        err_str = 'Not all output lists have the same length'
        error_channel.log(err_str)
        raise RunTimeError(err_str)

    return (geocoded_rasters, geocoded_datasets, input_rasters, interp_methods,
            invalid_values)

def cpu_geocode_rasters(cpu_geo_obj, geo_datasets, desired, freq, pol_list,
                        input_hdf5, dst_h5, radar_grid, dem_raster,
                        block_size, offset_params=None, scratch_path='',
                        compute_stats=True, input_product_type = InputProduct.RUNW,
                        iono_sideband=False, az_correction=isce3.core.LUT2d(),
                        srg_correction=isce3.core.LUT2d(), subswaths=None):

    geocoded_rasters, geocoded_datasets, input_rasters, *_ = \
        get_raster_lists(geo_datasets, desired, freq, pol_list, input_hdf5,
                         dst_h5, offset_params, scratch_path, input_product_type,
                         iono_sideband)

    if input_rasters:
        geocode_tuples = zip(input_rasters, geocoded_rasters)
        for input_raster, geocoded_raster in geocode_tuples:
            cpu_geo_obj.geocode(
                radar_grid=radar_grid,
                input_raster=input_raster,
                output_raster=geocoded_raster,
                dem_raster=dem_raster,
                output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
                min_block_size=block_size,
                max_block_size=block_size,
                az_time_correction=az_correction,
                slant_range_correction=srg_correction,
                sub_swaths=subswaths)

        if compute_stats:
            for raster, ds in zip(geocoded_rasters, geocoded_datasets):
                if os.path.basename(ds.name) not in ['wrappedInterferogram',
                                                     'connectedComponents']:
                    compute_stats_real_data(raster, ds)

def cpu_run(cfg, input_hdf5, output_hdf5, input_product_type=InputProduct.RUNW):
    """ Geocode RUNW products on CPU

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    input_hdf5 : str
        Path input RUNW or ROFF HDF5
    output_hdf5 : str
        Path to output GUNW HDF5
    input_product_type: enum
        Input product type
    """
    # pull parameters from cfg
    ref_hdf5 = cfg["input_file_group"]["reference_rslc_file"]
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    if input_product_type is InputProduct.RIFG:
        geogrids = cfg["processing"]["geocode"]["wrapped_igram_geogrids"]
    dem_file = cfg["dynamic_ancillary_file_group"]["dem_file"]
    ref_orbit = cfg["dynamic_ancillary_file_group"]['orbit_files']['reference_orbit_file']
    threshold_geo2rdr = cfg["processing"]["geo2rdr"]["threshold"]
    iteration_geo2rdr = cfg["processing"]["geo2rdr"]["maxiter"]
    lines_per_block = cfg["processing"]["geocode"]["lines_per_block"]
    interp_method = cfg["processing"]["geocode"]["interp_method"]
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    rg_looks = cfg['processing']['crossmul']['range_looks']
    az_looks = cfg['processing']['crossmul']['azimuth_looks']
    unwrap_rg_looks = cfg['processing']['phase_unwrap']['range_looks']
    unwrap_az_looks = cfg['processing']['phase_unwrap']['azimuth_looks']

    if input_product_type is InputProduct.RUNW:
        if unwrap_rg_looks != 1 or unwrap_az_looks != 1:
            rg_looks = unwrap_rg_looks
            az_looks = unwrap_az_looks

    if input_product_type is InputProduct.ROFF:
        geo_datasets = cfg["processing"]["geocode"]["goff_datasets"]
    elif input_product_type is InputProduct.RUNW:
        geo_datasets = cfg["processing"]["geocode"]["gunw_datasets"]
    else:
        # RIFG
        geo_datasets = cfg["processing"]["geocode"]["wrapped_datasets"]

    # if bool for all geocoded datasets is False return - no need to process
    if not any(geo_datasets.values()):
        return

    iono_args = cfg['processing']['ionosphere_phase_correction']
    iono_enabled = iono_args['enabled']
    iono_method = iono_args['spectral_diversity']
    is_iono_method_sideband = iono_method in ['main_side_band',
                                              'main_diff_ms_band']
    freq_pols_iono = iono_args["list_of_frequencies"]

    slc = SLC(hdf5file=ref_hdf5)
    info_channel = journal.info("geocode.run")
    info_channel.log("starting geocode")

    # NISAR products are always zero doppler
    grid_zero_doppler = isce3.core.LUT2d()

    # set defaults shared by both frequencies
    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # init geocode object
    geocode_obj = isce3.geocode.GeocodeFloat32()
    geocode_cplx_obj = isce3.geocode.GeocodeCFloat32()

    # init geocode members
    if ref_orbit is not None:
        # SLC will get first radar grid whose frequency is available.
        # Reference epoch and orbit have no frequency dependency.
        orbit = load_orbit_from_xml(ref_orbit, slc.getRadarGrid().ref_epoch)
    else:
        orbit = slc.getOrbit()

    geocode_obj.orbit = orbit
    geocode_obj.ellipsoid = ellipsoid
    geocode_obj.doppler = grid_zero_doppler
    geocode_obj.threshold_geo2rdr = threshold_geo2rdr
    geocode_obj.numiter_geo2rdr = iteration_geo2rdr
    geocode_obj.data_interpolator = interp_method

    geocode_cplx_obj.orbit = orbit
    geocode_cplx_obj.ellipsoid = ellipsoid
    geocode_cplx_obj.doppler = grid_zero_doppler
    geocode_cplx_obj.threshold_geo2rdr = threshold_geo2rdr
    geocode_cplx_obj.numiter_geo2rdr = iteration_geo2rdr
    geocode_cplx_obj.data_interpolator = interp_method

    t_all = time.time()
    with h5py.File(output_hdf5, "a") as dst_h5:
        for freq, pol_list, offset_pol_list in get_cfg_freq_pols(cfg):
            # Get azimuth and slant range corrections
            az_correction, srg_correction = \
                get_az_srg_corrections(cfg, slc, freq, orbit)
            radar_grid_slc = slc.getRadarGrid(freq)
            if az_looks > 1 or rg_looks > 1:
                radar_grid_mlook = radar_grid_slc.multilook(az_looks, rg_looks)

            geo_grid = geogrids[freq]
            geocode_obj.geogrid(geo_grid.start_x, geo_grid.start_y,
                        geo_grid.spacing_x, geo_grid.spacing_y,
                        geo_grid.width, geo_grid.length, geo_grid.epsg)

            geocode_cplx_obj.geogrid(geo_grid.start_x, geo_grid.start_y,
                                     geo_grid.spacing_x, geo_grid.spacing_y,
                                     geo_grid.width, geo_grid.length, geo_grid.epsg)

            subswaths = slc.getSwathMetadata(freq).sub_swaths()

            # Assign correct radar grid
            if az_looks > 1 or rg_looks > 1:
                radar_grid = radar_grid_mlook
            else:
                radar_grid = radar_grid_slc

            # set min/max block size from lines_per_block
            type_size = 4  # float32
            block_size = lines_per_block * geo_grid.width * type_size
            if input_product_type is InputProduct.RUNW:
                desired = ['coherence_magnitude', 'unwrapped_phase']

                geocode_obj.data_interpolator = interp_method
                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    pol_list, input_hdf5, dst_h5, radar_grid,
                                    dem_raster, block_size, az_correction=az_correction,
                                    srg_correction=srg_correction,
                                    subswaths=subswaths)
                if iono_enabled:
                    # polarizations for ionosphere can be independent to insar pol
                    pol_list_iono = freq_pols_iono[freq]
                    desired = ['ionosphere_phase_screen',
                               'ionosphere_phase_screen_uncertainty']
                    geocode_iono_bool = True
                    input_hdf5_iono = input_hdf5
                    if is_iono_method_sideband and freq == 'A':
                        # ionosphere_phase_screen from main_side_band or
                        # main_diff_ms_band are computed on radargrid of frequencyB.
                        # The ionosphere_phase_screen is geocoded on geogrid of
                        # frequencyA. Instead of geocoding ionosphere in the RUNW
                        # standard product (frequencyA), geocode the frequencyB in
                        # scratch/ionosphere/method/RUNW.h5 to avoid additional
                        # interpolation.
                        radar_grid_iono = slc.getRadarGrid('B')
                        iono_sideband_bool = True
                        if az_looks > 1 or rg_looks > 1:
                            radar_grid_iono = radar_grid_iono.multilook(
                                az_looks, rg_looks)
                        input_hdf5_iono = f'{scratch_path}/ionosphere/{iono_method}/RUNW.h5'
                    if is_iono_method_sideband and freq == 'B':
                        geocode_iono_bool = False

                    if not is_iono_method_sideband:
                        radar_grid_iono = radar_grid
                        iono_sideband_bool = False
                        if pol_list_iono is None:
                            geocode_iono_bool = False

                    if geocode_iono_bool:
                        cpu_geocode_rasters(geocode_obj, geo_datasets, desired,
                                            freq, pol_list_iono, input_hdf5_iono,
                                            dst_h5, radar_grid_iono, dem_raster,
                                            block_size,
                                            iono_sideband=iono_sideband_bool,
                                            az_correction=az_correction,
                                            srg_correction=srg_correction,
                                            subswaths=subswaths)

                # reset geocode_obj geogrid
                if is_iono_method_sideband and freq == 'B':
                    geo_grid = geogrids['B']
                    geocode_obj.geogrid(geo_grid.start_x, geo_grid.start_y,
                                geo_grid.spacing_x, geo_grid.spacing_y,
                                geo_grid.width, geo_grid.length,
                                geo_grid.epsg)

                desired = ["connected_components"]
                geocode_obj.data_interpolator = 'NEAREST'
                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    pol_list, input_hdf5, dst_h5, radar_grid,
                                    dem_raster, block_size, az_correction=az_correction,
                                    srg_correction=srg_correction,
                                    subswaths=subswaths)

                desired = ['along_track_offset', 'slant_range_offset',
                           'correlation_surface_peak']
                geocode_obj.data_interpolator = interp_method
                radar_grid_offset = get_offset_radar_grid(cfg,
                                                          radar_grid_slc)

                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    offset_pol_list, input_hdf5, dst_h5,
                                    radar_grid_offset, dem_raster,
                                    block_size, az_correction=az_correction,
                                    srg_correction=srg_correction)
                if cfg['processing']['rdr2geo']['write_layover_shadow']:
                    desired = ["mask"]
                    geocode_obj.data_interpolator = 'NEAREST'
                    cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                        pol_list, input_hdf5, dst_h5,
                                        radar_grid_slc, dem_raster, block_size,
                                        scratch_path=scratch_path,
                                        compute_stats=False,
                                        az_correction=az_correction,
                                        srg_correction=srg_correction,
                                        subswaths=subswaths)

                    # add water mask to GUNW product
                    add_water_to_mask(cfg, freq, geo_grid, dst_h5)
                    mask_path = f'{GUNWGroupsPaths().GridsPath}/frequency{freq}/unwrappedInterferogram/mask'
                    mask_ds = dst_h5[mask_path]
                    compute_layover_shadow_water_stats(mask_ds)

            elif input_product_type is InputProduct.ROFF:
                offset_cfg = cfg['processing']['offsets_product']
                desired = ['along_track_offset', 'slant_range_offset',
                           'along_track_offset_variance',
                           'correlation_surface_peak',
                           'cross_offset_variance', 'slant_range_offset',
                           'snr']

                # Create list to tuples containing offset layer name with
                # corresponding interpolation method and invalid value
                # Interpolation method and invalid value are not used for
                # cpu geocode. These params are added to as None for
                # consistency with gpu_geocode_rasters, who needs it for
                # get_raster_lists
                layer_geocode_params = [(layer_name, None, None)
                                        for layer_name in offset_cfg.keys() if
                                        layer_name.startswith('layer')]

                radar_grid = get_offset_radar_grid(cfg,
                                                   slc.getRadarGrid(freq))

                geocode_obj.data_interpolator = interp_method
                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    offset_pol_list, input_hdf5, dst_h5,
                                    radar_grid, dem_raster, block_size,
                                    offset_params=layer_geocode_params,
                                    input_product_type=InputProduct.ROFF,
                                    az_correction=az_correction,
                                    srg_correction=srg_correction,
                                    subswaths=subswaths)
            else:
                #RIFG
                # Geocode the coherence
                desired = ['coherence_magnitude']
                geocode_obj.data_interpolator = interp_method
                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    pol_list,input_hdf5, dst_h5, radar_grid,
                                    dem_raster, block_size,
                                    input_product_type=InputProduct.RIFG,
                                    az_correction=az_correction,
                                    srg_correction=srg_correction,
                                    subswaths=subswaths)

                # Geocode the wrapped interferogram
                desired = ['wrapped_interferogram']
                geocode_cplx_obj.data_interpolator = cfg["processing"]["geocode"]\
                        ['wrapped_interferogram']['interp_method']
                cpu_geocode_rasters(geocode_cplx_obj, geo_datasets, desired, freq,
                                    pol_list,input_hdf5, dst_h5, radar_grid,
                                    dem_raster, block_size * 2,
                                    input_product_type=InputProduct.RIFG,
                                    az_correction=az_correction,
                                    srg_correction=srg_correction,
                                    subswaths=subswaths)

            # spec for NISAR GUNW does not require freq B so skip radar cube
            if freq.upper() == 'B':
                continue

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geocode in {t_all_elapsed:.3f} seconds")

def gpu_geocode_rasters(geocoded_dataset_flags,
                        desired_geo_dataset_names,
                        interpolation_methods,
                        invalid_values,
                        freq,
                        pol_list,
                        geogrid,
                        rdr_geometry,
                        dem_raster,
                        lines_per_block,
                        input_hdf5,
                        dst_h5,
                        subswaths,
                        offset_layers=None,
                        scratch_path='',
                        compute_stats=True,
                        input_product_type=InputProduct.RUNW,
                        iono_sideband=False,
                        az_correction=isce3.core.LUT2d(),
                        srg_correction=isce3.core.LUT2d()):
    '''
    Geocode datasets with common geogrid and radar geometry.

    Parameters
    ----------
    geocoded_dataset_flags: dict
        Dict describing which datasets are to be geocoded.
        key: dataset name
        value: True if dataset is to be geocoded
    desired_geo_dataset_names: list[str]
        List of dataset names that could be geocoded
    interpolation_methods: list[isce3.core.interp_method.DataInterpMethod]
        List of data interpolation methods to be used per dataset
    invalid_values: list[float]
        List of invalid values to be used initialized each dataset raster
    freq: ['A', 'B']
        Used to determine path the dataset in HDF5
    pol_list: list(str)
        List of polarizations for current frequency to be geocoded. Used to
        determine path to dataset in HDF5.
    geogrid: isce3.product.GeoGridParameters
        Geogrid to geocode rasters to
    rdr_geometry: isce3.container.RadarGeometry
        Radar grid, orbit, image doppler describing scene geometry in then
        radar coordinate system
    dem_raster: isce3.io.Raster
        DEM containing radar grid to be geocoded
    lines_per_block: int
        Number of lines per block to be processed.
    input_hdf5: str
        Path to HDF5 with radar datasets to be geocoded.
    dst_h5: str
        Path to HDF5 where geocoded datasets are to be placed.
    subswaths: isce3.product.SubSwaths
        Possible subswath that could be used to mask geocoding.
    offset_layers: list[str]
        List of names of offset layers.
    scratch_path: str
        Path to scratch directory.
    compute_stats: bool
        True if stats are to be computed rasters.
    input_product_type: isce3.io.gdal.GDALDataType
        Enum describing type of product to geocoded.
    iono_sideband: bool
        True if iono rasters are to be geocoded.
    az_correction: isce3.core.LUT2d()
        Low-res LUT containing azimuth timing correction for geocoding
    srg_correction: isce3.core.LUT2d()
        Low-res LUT containing slant range timing correction for geocoding
    '''
    # Get:
    # 1. List of output geocoded rasters asisce3.io.Rasters objects
    # 2. List of h5py.Datasets of geocoded rasters (for stats computation)
    # 3. List of input rasters as isce3.io.Raster objects
    # 4. List of interpolation methods for geocoding each raster
    # 5. List of invalid values to initialize each raster with
    (geocoded_rasters, geocoded_datasets, input_rasters,
     interpolation_methods, invalid_values) = \
        get_raster_lists(geocoded_dataset_flags, desired_geo_dataset_names, freq,
                         pol_list, input_hdf5, dst_h5, offset_layers,
                         scratch_path, input_product_type, iono_sideband,
                         interpolation_methods, invalid_values)

    if input_rasters:
        # Get raster types and convert to isce3.io.gdal.GDALDataType
        convert_dtypes = {gdal.GDT_Unknown:  isce3.io.gdal.GDT_Unknown,
                          gdal.GDT_Byte:     isce3.io.gdal.GDT_Byte,
                          gdal.GDT_UInt16:   isce3.io.gdal.GDT_UInt16,
                          gdal.GDT_Int16:    isce3.io.gdal.GDT_Int16,
                          gdal.GDT_UInt32:   isce3.io.gdal.GDT_UInt32,
                          gdal.GDT_Int32:    isce3.io.gdal.GDT_Unknown,
                          gdal.GDT_Float32:  isce3.io.gdal.GDT_Float32,
                          gdal.GDT_Float64:  isce3.io.gdal.GDT_Float64,
                          gdal.GDT_CInt16:   isce3.io.gdal.GDT_CInt16,
                          gdal.GDT_CInt32:   isce3.io.gdal.GDT_CInt32,
                          gdal.GDT_CFloat32: isce3.io.gdal.GDT_CFloat32,
                          gdal.GDT_CFloat64: isce3.io.gdal.GDT_CFloat64}
        raster_types = [convert_dtypes[input_raster.datatype()]
                        for input_raster in input_rasters]

        # Create geocode object to perform geocoding
        gpu_geocode_obj = \
            isce3.cuda.geocode.Geocode(geogrid, rdr_geometry,
                                       lines_per_block)

        gpu_geocode_obj.geocode_rasters(geocoded_rasters, input_rasters,
                                        interpolation_methods,
                                        raster_types,
                                        invalid_values,
                                        dem_raster,
                                        subswaths=subswaths,
                                        az_time_correction=az_correction,
                                        srange_correction=srg_correction)

        if compute_stats:
            for raster, ds in zip(geocoded_rasters, geocoded_datasets):
                if os.path.basename(ds.name) not in ['connectedComponents',
                                                     'wrappedInterferogram']:
                    compute_stats_real_data(raster, ds)

def gpu_run(cfg, input_hdf5, output_hdf5, input_product_type=InputProduct.RUNW):
    """ Geocode RUNW products on GPU

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    prof_hdf5 : str
        Path input RUNW or ROFF HDF5
    output_hdf5 : str
        Path to output GUNW HDF5
    input_product_type: enum
        Input product type
    """
    t_all = time.time()

    # Extract parameters from cfg dictionary
    ref_hdf5 = cfg["input_file_group"]["reference_rslc_file"]
    dem_file = cfg["dynamic_ancillary_file_group"]["dem_file"]
    ref_orbit = cfg["dynamic_ancillary_file_group"]['orbit_files']['reference_orbit_file']
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    if input_product_type is InputProduct.RIFG:
        geogrids = cfg["processing"]["geocode"]["wrapped_igram_geogrids"]
    lines_per_block = cfg["processing"]["geocode"]["lines_per_block"]
    interp_method = cfg["processing"]["geocode"]["interp_method"]
    rg_looks = cfg['processing']['crossmul']['range_looks']
    az_looks = cfg['processing']['crossmul']['azimuth_looks']
    unwrap_rg_looks = cfg['processing']['phase_unwrap']['range_looks']
    unwrap_az_looks = cfg['processing']['phase_unwrap']['azimuth_looks']

    # Only when the input product is RUNW, then we ajust the range and azimuth looks
    if input_product_type is InputProduct.RUNW:
        if unwrap_rg_looks != 1 or unwrap_az_looks != 1:
            rg_looks = unwrap_rg_looks
            az_looks = unwrap_az_looks

    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])

    # Retrieve enabled/disabled flags for each geocoded InSAR product
    if input_product_type is InputProduct.ROFF:
        geocoded_dataset_flags = cfg["processing"]["geocode"]["goff_datasets"]
    elif input_product_type is InputProduct.RUNW:
        geocoded_dataset_flags = cfg["processing"]["geocode"]["gunw_datasets"]
    else:
        # RIFG
        geocoded_dataset_flags = cfg["processing"]["geocode"]["wrapped_datasets"]

    iono_args = cfg['processing']['ionosphere_phase_correction']
    iono_enabled = iono_args['enabled']
    iono_method = iono_args['spectral_diversity']
    freq_pols_iono = iono_args["list_of_frequencies"]
    is_iono_method_sideband = iono_method in ['main_side_band',
                                              'main_diff_ms_band']

    if interp_method == 'BILINEAR':
        interp_method = isce3.core.DataInterpMethod.BILINEAR
    if interp_method == 'BICUBIC':
        interp_method = isce3.core.DataInterpMethod.BICUBIC
    if interp_method == 'NEAREST':
        interp_method = isce3.core.DataInterpMethod.NEAREST
    if interp_method == 'BIQUINTIC':
        interp_method = isce3.core.DataInterpMethod.BIQUINTIC

    # Interpolation method for the wrapped interferogram
    wrapped_igram_interp_method = interp_method

    if input_product_type is InputProduct.RIFG:
        wrapped_igram_interp_method = cfg["processing"]["geocode"]\
                ['wrapped_interferogram']['interp_method']

        if wrapped_igram_interp_method == 'SINC':
            wrapped_igram_interp_method = isce3.core.DataInterpMethod.SINC
        if wrapped_igram_interp_method == 'BILINEAR':
            wrapped_igram_interp_method = isce3.core.DataInterpMethod.BILINEAR
        if wrapped_igram_interp_method == 'BICUBIC':
            wrapped_igram_interp_method = isce3.core.DataInterpMethod.BICUBIC
        if wrapped_igram_interp_method == 'NEAREST':
            wrapped_igram_interp_method = isce3.core.DataInterpMethod.NEAREST
        if wrapped_igram_interp_method == 'BIQUINTIC':
            wrapped_igram_interp_method = isce3.core.DataInterpMethod.BIQUINTIC

    info_channel = journal.info("geocode.run")
    info_channel.log("starting geocode")

    # Init frequency independent objects
    slc = SLC(hdf5file=ref_hdf5)
    grid_zero_doppler = isce3.core.LUT2d()
    dem_raster = isce3.io.Raster(dem_file)

    # init geocode members
    if ref_orbit is not None:
        # SLC will get first radar grid whose frequency is available.
        # Reference epoch and orbit have no frequency dependency.
        orbit = load_orbit_from_xml(ref_orbit, slc.getRadarGrid().ref_epoch)
    else:
        orbit = slc.getOrbit()

    with h5py.File(output_hdf5, "a", libver='latest', swmr=True) as dst_h5:

        # Based on runconfig iterate over frequencies and their polarizations
        for freq, pol_list, offset_pol_list in get_cfg_freq_pols(cfg):

            # Get azimuth and slant range LUT corrections
            az_correction, srg_correction = \
                get_az_srg_corrections(cfg, slc, freq, orbit
                                       )
            geogrid = geogrids[freq]

            subswaths = slc.getSwathMetadata(freq).sub_swaths()

            # Create frequency based radar grid
            radar_grid = slc.getRadarGrid(freq)
            if az_looks > 1 or rg_looks > 1:
                # Multilook radar grid if needed
                radar_grid = radar_grid.multilook(az_looks, rg_looks)

            if input_product_type is InputProduct.RUNW:
                desired_geo_dataset_names = \
                    ['coherence_magnitude', 'unwrapped_phase',
                     'connected_components']

                # Interpolation methods for respective datasets named above
                interpolation_methods = [interp_method, interp_method,
                                         isce3.core.DataInterpMethod.NEAREST]

                # Invalid values for respective datasets named above
                # connected_components raster has type unsigned char and an invalid
                # value of NaN becomes 0 which conflicts with 0 being used to indicate
                # an unmasked value/pixel. 65535 is chosen as the max mappable connected
                # component
                invalid_values = [np.nan, np.nan, 65535]

                # Create radar grid geometry used by most datasets
                rdr_geometry = isce3.container.RadarGeometry(radar_grid, orbit,
                                                             grid_zero_doppler)

                gpu_geocode_rasters(geocoded_dataset_flags,
                                    desired_geo_dataset_names,
                                    interpolation_methods, invalid_values,
                                    freq, pol_list,
                                    geogrid, rdr_geometry, dem_raster,
                                    lines_per_block, input_hdf5, dst_h5,
                                    subswaths=subswaths,
                                    az_correction=az_correction,
                                    srg_correction=srg_correction)

                if iono_enabled:
                    desired_geo_dataset_names = ['ionosphere_phase_screen',
                               'ionosphere_phase_screen_uncertainty']

                    # Interpolation methods for respective datasets named above
                    interpolation_methods = [interp_method, interp_method]

                    # Invalid values for respective datasets named above
                    invalid_values = [np.nan, np.nan]

                    geocode_iono_bool = True
                    pol_list_iono = freq_pols_iono[freq]
                    input_hdf5_iono = input_hdf5
                    if is_iono_method_sideband:
                        # ionosphere_phase_screen from main_side_band or
                        # main_diff_ms_band are computed on radargrid of frequencyB.
                        # The ionosphere_phase_screen is geocoded on geogrid of
                        # frequencyA. Instead of geocoding ionosphere in the RUNW standard
                        # product (frequencyA), geocode the frequencyB in ionosphere/RUNW.h5
                        # to avoid additional interpolation.
                        input_hdf5_iono = \
                            f'{scratch_path}/ionosphere/{iono_method}/RUNW.h5'
                        if freq == 'A':
                            radar_grid_iono = slc.getRadarGrid('B')
                            if az_looks > 1 or rg_looks > 1:
                                radar_grid_iono = radar_grid_iono.multilook(
                                    az_looks, rg_looks)
                            iono_sideband_bool = True
                            iono_freq = 'B'
                            rdr_geometry_iono = \
                                isce3.container.RadarGeometry(
                                    radar_grid_iono,
                                    slc.getOrbit(),
                                    grid_zero_doppler)
                        else:
                            # The methods using sideband (e.g., main_side_band,
                            # and main_ms_diff_band) produce only one
                            # ionosphere from frequency A and B interferogram.
                            # The ionosphere of radargrid (frequency B) is
                            # geocoded only to geogrid in frequency A.
                            geocode_iono_bool = False
                    else:
                        # The method using split_main_band produces
                        # can have two ionosphere layers in A and B.
                        iono_sideband_bool = False
                        iono_freq = freq
                        rdr_geometry_iono = rdr_geometry
                        if pol_list_iono is None:
                            geocode_iono_bool = False

                    if geocode_iono_bool:
                        gpu_geocode_rasters(geocoded_dataset_flags,
                                            desired_geo_dataset_names,
                                            interpolation_methods,
                                            invalid_values,
                                            iono_freq, pol_list_iono,
                                            geogrid, rdr_geometry, dem_raster,
                                            lines_per_block, input_hdf5_iono, dst_h5,
                                            subswaths,
                                            iono_sideband=iono_sideband_bool,
                                            az_correction=az_correction,
                                            srg_correction=srg_correction)

                desired_geo_dataset_names = [
                   'along_track_offset', 'slant_range_offset',
                    'correlation_surface_peak']
                n_desired_geo_dataset_names = len(desired_geo_dataset_names)

                # Interpolation methods for respective datasets named above
                interpolation_methods = [interp_method] * n_desired_geo_dataset_names

                # Invalid values for respective datasets named above
                invalid_values = [np.nan] * n_desired_geo_dataset_names

                # If needed create geocode object for offset datasets
                # Create offset unique radar grid
                radar_grid = get_offset_radar_grid(cfg,
                                                   slc.getRadarGrid(freq))

                # Create radar grid geometry required by offset datasets
                rdr_geometry = isce3.container.RadarGeometry(radar_grid, orbit,
                                                             grid_zero_doppler)

                gpu_geocode_rasters(geocoded_dataset_flags,
                                    desired_geo_dataset_names,
                                    interpolation_methods, invalid_values,
                                    freq, offset_pol_list,
                                    geogrid, rdr_geometry, dem_raster,
                                    lines_per_block, input_hdf5, dst_h5,
                                    subswaths=subswaths,
                                    az_correction=az_correction,
                                    srg_correction=srg_correction)

                # Geocode layover shadow mask
                if cfg['processing']['rdr2geo']['write_layover_shadow']:
                    desired_geo_dataset_names = ["mask"]

                    # Interpolation methods for dataset above
                    interpolation_methods = [isce3.core.DataInterpMethod.NEAREST]

                    # Invalid values for dataset above
                    # layover shadow raster has type char and an invalid
                    # value of NaN becomes 0 which conflicts with 0 being used
                    # to indicate an unmasked value/pixel. 127 is chosen as it is
                    # the most distant value from the allowed set of [0, 1, 2, 3].
                    invalid_values = [127]

                    # If needed create geocode object for shadow layover dataset
                    # Create radar grid geometry required by layover shadow
                    rdr_geometry = isce3.container.RadarGeometry(slc.getRadarGrid(freq),
                                                                orbit,
                                                                grid_zero_doppler)

                    gpu_geocode_rasters(geocoded_dataset_flags,
                                        desired_geo_dataset_names,
                                        interpolation_methods, invalid_values,
                                        freq, pol_list,
                                        geogrid, rdr_geometry, dem_raster,
                                        lines_per_block, input_hdf5, dst_h5,
                                        subswaths,
                                        scratch_path=scratch_path,
                                        compute_stats=False, az_correction=az_correction,
                                        srg_correction=srg_correction)

                    # add water mask to GUNW product
                    add_water_to_mask(cfg, freq, geogrid, dst_h5)
                    mask_path = f'{GUNWGroupsPaths().GridsPath}/frequency{freq}/unwrappedInterferogram/mask'
                    mask_ds = dst_h5[mask_path]
                    compute_layover_shadow_water_stats(mask_ds)

            elif input_product_type is InputProduct.ROFF:
                offset_cfg = cfg['processing']['offsets_product']

                desired_geo_dataset_names = [
                    'along_track_offset',
                    'slant_range_offset',
                    'along_track_offset_variance',
                    'correlation_surface_peak',
                    'cross_offset_variance',
                    'slant_range_offset_variance',
                    'snr']

                # Create list to tuples containing offset layer name with
                # corresponding interpolation method and invalid value
                layer_geocode_params = [(layer_name, interp_method, np.nan)
                                        for layer_name in offset_cfg.keys() if
                                        layer_name.startswith('layer')]

                # Interpolation methods for datasets above all the same
                interpolation_methods = \
                    [interp_method] * len(desired_geo_dataset_names)

                # Invalid values for datasets above all the same
                invalid_values = [np.nan] * len(desired_geo_dataset_names)

                # Create radar grid geometry required by offset datasets
                radar_grid = get_offset_radar_grid(cfg,
                                                   slc.getRadarGrid(freq))
                rdr_geometry = isce3.container.RadarGeometry(radar_grid,
                                                             orbit,
                                                             grid_zero_doppler)

                gpu_geocode_rasters(geocoded_dataset_flags,
                                    desired_geo_dataset_names,
                                    interpolation_methods, invalid_values,
                                    freq, offset_pol_list,
                                    geogrid, rdr_geometry, dem_raster,
                                    lines_per_block, input_hdf5, dst_h5,
                                    subswaths,
                                    offset_layers=layer_geocode_params,
                                    input_product_type=InputProduct.ROFF,
                                    az_correction=az_correction,
                                    srg_correction=srg_correction)
            else:
                # Datasets from RIFG to be geocoded
                desired_geo_dataset_names = ['coherence_magnitude',
                                             'wrapped_interferogram']

                # Interpolation method for respective datasets named above
                interpolation_methods = [interp_method,
                                         wrapped_igram_interp_method]

                # Invalid values for respective datasets named above
                invalid_values = [np.nan, np.nan]

                # Create radar grid geometry required by RIFG product
                rdr_geometry = isce3.container.RadarGeometry(radar_grid, orbit,
                                                             grid_zero_doppler)

                # Geocode the coherence and wrapped interferogram
                gpu_geocode_rasters(geocoded_dataset_flags,
                                    desired_geo_dataset_names,
                                    interpolation_methods, invalid_values,
                                    freq, pol_list,
                                    geogrid, rdr_geometry, dem_raster,
                                    lines_per_block, input_hdf5, dst_h5,
                                    subswaths,
                                    input_product_type=InputProduct.RIFG,
                                    az_correction=az_correction,
                                    srg_correction=srg_correction)

            # spec for NISAR GUNW does not require freq B so skip radar cube
            if freq.upper() == 'B':
                continue

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geocode in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    # run geocode from command line

    # load command line args
    geocode_insar_parser = YamlArgparse()
    args = geocode_insar_parser.parse()

    # Get a runconfig dictionary from command line args
    geocode_insar_runconfig = GeocodeInsarRunConfig(args)

    # prepare the HDF5
    geocode_insar_runconfig.cfg['primary_executable']['product_type'] = 'GUNW'
    out_paths = prepare_insar_hdf5.run(geocode_insar_runconfig.cfg)
    runw_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'runw_path']
    if runw_path is not None:
        out_paths['RUNW'] = runw_path

    # Run geocode RUNW
    run(geocode_insar_runconfig.cfg, out_paths["RUNW"], out_paths["GUNW"],
        input_product_type=InputProduct.RUNW)

    rifg_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'rifg_path']
    if rifg_path is not None:
        out_paths['RIFG'] = rifg_path
    # Run geocode RIFG
    run(geocode_insar_runconfig.cfg, out_paths["RIFG"], out_paths["GUNW"],
        input_product_type=InputProduct.RIFG)

    # Check if need to geocode offset product
    enabled = geocode_insar_runconfig.cfg['processing']['offsets_product']['enabled']
    # Prepare the GOFF product
    if enabled:
        geocode_insar_runconfig.cfg['primary_executable']['product_type'] = 'GOFF'
        out_paths = prepare_insar_hdf5.run(geocode_insar_runconfig.cfg)
    roff_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'roff_path']
    if roff_path is not None:
        out_paths['ROFF'] = roff_path
    if enabled:
        run(geocode_insar_runconfig.cfg, out_paths['ROFF'],
            out_paths['GOFF'], InputProduct.ROFF)
