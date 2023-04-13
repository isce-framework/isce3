#!/usr/bin/env python3

"""
collection of functions for NISAR geocode workflow
"""

from enum import Enum
import numpy as np

import pathlib
import time

import h5py
import journal
import isce3
import numpy as np

from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows import h5_prep
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from nisar.workflows.helpers import get_cfg_freq_pols
from nisar.workflows.geocode_insar_runconfig import \
    GeocodeInsarRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.compute_stats import compute_stats_real_data, \
    compute_water_mask_stats, compute_layover_shadow_stats


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
    dataset_path = f"{dst_freq_path}/interferogram/layoverShadowMask"

    return input_raster, dataset_path


def get_ds_input_output(src_freq_path, dst_freq_path, pol, input_hdf5,
                        dataset_name, off_layer=None, input_product_type=InputProduct.RUNW):
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
    input_product_type: enum
        Input product type, which is one of RUNW, ROFF, RIFG

    Returns
    -------
    input_raster : isce3.io.Raster
        Shadow layover input raster object
    dataset_path : str
        HDF5 path to geocoded shadow layover dataset
    """

    if dataset_name in ['alongTrackOffset', 'slantRangeOffset'] and \
            input_product_type is InputProduct.RUNW:
        src_group_path = f'{src_freq_path}/pixelOffsets/{pol}'
        dst_group_path = f'{dst_freq_path}/pixelOffsets/{pol}'
    else:
        src_group_path = f'{src_freq_path}/interferogram/{pol}'
        dst_group_path = f'{dst_freq_path}/interferogram/{pol}'

        if input_product_type is InputProduct.RIFG:
            dst_group_path = f'{dst_freq_path}/wrappedInterferogram/{pol}'

    if input_product_type is InputProduct.ROFF:
        src_group_path = f'{src_freq_path}/pixelOffsets/{pol}/{off_layer}'
        dst_group_path = f'{dst_freq_path}/pixelOffsets/{pol}/{off_layer}'

    # prepare input raster
    input_raster_str = (f"HDF5:{input_hdf5}:/{src_group_path}/{dataset_name}")
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
        layer_keys = [key for key in offset_cfg if key.startswith('layer')]
        if not layer_keys:
            err_str = 'No offset layer found'
            error_channel.log(err_str)
            raise KeyError(err_str)
        # Extract search/chip windows per layer; default to inf if not found
        for key in layer_keys:
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

    return projected_data


def add_water_mask(cfg, freq, geogrid, dst_h5):
    """
    Create water mask to HDF5 from given water mask

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
        freq_path = f'/science/LSAR/GUNW/grids/frequency{freq}'
        water_mask_h5_path = f'{freq_path}/interferogram/waterMask'
        water_mask = _project_water_to_geogrid(water_mask_path, geogrid)
        water_mask_interpret = water_mask.astype('uint8') != 0
        dst_h5[water_mask_h5_path].write_direct(water_mask_interpret)


def add_radar_grid_cube(cfg, freq, radar_grid, orbit, dst_h5, input_product_type):
    ''' Write radar grid cube to HDF5

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    freq : str
        Frequency in HDF5 to add cube to
    orbit : isce3.core.orbit
        Orbit object of current SLC
    dst_h5: str
        Path to output GUNW HDF5
    input_product_type: enum
        Input product type
    '''
    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']
    threshold_geo2rdr = cfg["processing"]["geo2rdr"]["threshold"]
    iteration_geo2rdr = cfg["processing"]["geo2rdr"]["maxiter"]

    ref_hdf5 = cfg["input_file_group"]["reference_rslc_file"]
    slc = SLC(hdf5file=ref_hdf5)

    # get doppler centroid
    cube_geogrid_param = isce3.product.GeoGridParameters(
        start_x=radar_grid_cubes_geogrid.start_x,
        start_y=radar_grid_cubes_geogrid.start_y,
        spacing_x=radar_grid_cubes_geogrid.spacing_x,
        spacing_y=radar_grid_cubes_geogrid.spacing_y,
        width=int(radar_grid_cubes_geogrid.width),
        length=int(radar_grid_cubes_geogrid.length),
        epsg=radar_grid_cubes_geogrid.epsg)

    product = 'GOFF' if input_product_type is InputProduct.ROFF else 'GUNW'
    cube_group_path = f'/science/LSAR/{product}/metadata/radarGrid'

    radar_grid = slc.getRadarGrid(freq)
    native_doppler = slc.getDopplerCentroid(frequency=freq)
    grid_zero_doppler = isce3.core.LUT2d()
    '''
    The native-Doppler LUT bounds error is turned off to
    computer cubes values outside radar-grid boundaries
    '''
    native_doppler.bounds_error = False
    add_radar_grid_cubes_to_hdf5(dst_h5, cube_group_path,
                                 cube_geogrid_param, radar_grid_cubes_heights,
                                 radar_grid, orbit, native_doppler,
                                 grid_zero_doppler, threshold_geo2rdr,
                                 iteration_geo2rdr)

def _snake_to_camel_case(snake_case_str):
    splitted_snake_case_str = snake_case_str.split('_')
    return (splitted_snake_case_str[0] +
            ''.join(w.title() for w in splitted_snake_case_str[1:]))

def get_raster_lists(geo_datasets, desired, freq, pol_list, input_hdf5, dst_h5,
                     off_layer_dict=None, scratch_path='',
                     input_product_type=InputProduct.RUNW,
                     iono_sideband=False):
    '''
    Geocode rasters with a shared geogrid.

    Parameters
    ----------
    geo_datasets : dict
        Dict of all dataset names and whether or not to geocode as key/value
    desired : list
        List of dataset names to be geocoded for ?
    freq : str
        Frequency of datasets to be geocoded
    pol_list : list
        List of polarizations of frequency to be geocoded
    input_hdf5: str
        Path to input RUNW or ROFF HDF5
    dst_h5 : h5py.File
        h5py.File object where geocoded data is to be written
    scratch_path : str
        Path to scratch where layover shadow raster is saved
    input_product_type : enum
        Product type of the input_hdf5
    iono_sideband : bool
        Flag to geocode ionosphere phase screen estimated from
        side-band

    Returns
    -------
    '''
    get_ds_names = lambda ds_dict, desired: [
        x for x, y in ds_dict.items() if y and x in desired]

    if input_product_type is InputProduct.ROFF:
        src_product = 'OFF'
        dst_product = 'OFF'
    else:
        src_product = 'IFG' if input_product_type is InputProduct.RIFG else 'UNW'
        dst_product = 'UNW'

    src_freq_path = f"/science/LSAR/R{src_product}/swaths/frequency{freq}"
    dst_freq_path = f"/science/LSAR/G{dst_product}/grids/frequency{freq}"

    input_rasters = []
    geocoded_rasters = []
    geocoded_datasets = []

    skip_layover_shadow = False
    ds_names = [x for x, y in geo_datasets.items() if y and x in desired]
    for ds_name in ds_names:
        for pol in pol_list:
            if skip_layover_shadow:
                continue
            input_raster = []
            out_ds_path = []
            if ds_name == "layover_shadow_mask":
                raster, path = get_shadow_input_output(
                    scratch_path, freq, dst_freq_path)
                skip_layover_shadow = True
                input_raster.append(raster)
                out_ds_path.append(path)
            elif input_product_type is InputProduct.ROFF:
                ds_name_camel_case = _snake_to_camel_case(ds_name)
                for layer in off_layer_dict:
                    raster, path = get_ds_input_output(src_freq_path,
                                                       dst_freq_path,
                                                       pol, input_hdf5, ds_name_camel_case,
                                                       layer, input_product_type)
                    input_raster.append(raster)
                    out_ds_path.append(path)
            elif iono_sideband and ds_name in ['ionosphere_phase_screen',
                           'ionosphere_phase_screen_uncertainty']:
                '''
                ionosphere_phase_screen from main_side_band or
                main_diff_ms_band are computed on radargrid of frequencyB.
                The ionosphere_phase_screen is geocoded on geogrid of
                frequencyA.
                '''
                iono_src_freq_path = f"/science/LSAR/R{src_product}/swaths/frequencyB"
                iono_dst_freq_path = f"/science/LSAR/G{src_product}/grids/frequencyA"
                ds_name_camel_case = _snake_to_camel_case(ds_name)
                raster, path = get_ds_input_output(
                    iono_src_freq_path, iono_dst_freq_path, pol, input_hdf5,
                        ds_name_camel_case)
                input_raster.append(raster)
                out_ds_path.append(path)
            else:
                ds_name_camel_case = _snake_to_camel_case(ds_name)
                raster, path = get_ds_input_output(
                    src_freq_path, dst_freq_path, pol, input_hdf5, ds_name_camel_case,
                    None, input_product_type)
                input_raster.append(raster)
                out_ds_path.append(path)
            for input, path in zip(input_raster, out_ds_path):
                input_rasters.append(input)

                # Prepare output raster access the HDF5 dataset for a given frequency and pol
                geocoded_dataset = dst_h5[path]
                geocoded_datasets.append(geocoded_dataset)

                # Construct the output raster directly from HDF5 dataset
                geocoded_raster = isce3.io.Raster(
                    f"IH5:::ID={geocoded_dataset.id.id}".encode("utf-8"),
                    update=True)

                geocoded_rasters.append(geocoded_raster)

    return geocoded_rasters, geocoded_datasets, input_rasters

def cpu_geocode_rasters(cpu_geo_obj, geo_datasets, desired, freq, pol_list,
                        input_hdf5, dst_h5, radar_grid, dem_raster,
                        block_size, off_layer_dict=None, scratch_path='',
                        compute_stats=True, input_product_type = InputProduct.RUNW,
                        iono_sideband=False):

    geocoded_rasters, geocoded_datasets, input_rasters = \
        get_raster_lists(geo_datasets, desired, freq, pol_list, input_hdf5,
                         dst_h5, off_layer_dict, scratch_path, input_product_type,
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
                max_block_size=block_size)

        if compute_stats:
            for raster, ds in zip(geocoded_rasters, geocoded_datasets):
                compute_stats_real_data(raster, ds)
            if input_product_type != InputProduct.ROFF:
                water_mask_ds = dst_h5['/science/LSAR/GUNW/grids/frequencyA/interferogram/waterMask']
                compute_water_mask_stats(water_mask_ds)
                lay_shadow_ds = dst_h5['/science/LSAR/GUNW/grids/frequencyA/interferogram/layoverShadowMask']
                compute_layover_shadow_stats(lay_shadow_ds)


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
    freq_pols = cfg["processing"]["input_subset"]["list_of_frequencies"]
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    if input_product_type is InputProduct.RIFG:
        geogrids = cfg["processing"]["geocode"]["wrapped_igram_geogrids"]
    dem_file = cfg["dynamic_ancillary_file_group"]["dem_file"]
    ref_orbit = cfg["dynamic_ancillary_file_group"]['orbit']['reference_orbit_file']
    threshold_geo2rdr = cfg["processing"]["geo2rdr"]["threshold"]
    iteration_geo2rdr = cfg["processing"]["geo2rdr"]["maxiter"]
    lines_per_block = cfg["processing"]["geocode"]["lines_per_block"]
    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    interp_method = cfg["processing"]["geocode"]["interp_method"]
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
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
        orbit = load_orbit_from_xml(ref_orbit)
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
                                    pol_list,input_hdf5, dst_h5, radar_grid,
                                    dem_raster, block_size)
                if iono_enabled:
                    # polarizations for ionosphere can be independent to insar pol
                    pol_list_iono = freq_pols_iono[freq]
                    desired = ['ionosphere_phase_screen',
                               'ionosphere_phase_screen_uncertainty']
                    geocode_iono_bool = True
                    input_hdf5_iono = input_hdf5
                    if is_iono_method_sideband and freq == 'A':
                        '''
                        ionosphere_phase_screen from main_side_band or
                        main_diff_ms_band are computed on radargrid of frequencyB.
                        The ionosphere_phase_screen is geocoded on geogrid of
                        frequencyA. Instead of geocoding ionosphere in the RUNW
                        standard product (frequencyA), geocode the frequencyB in
                        scratch/ionosphere/method/RUNW.h5 to avoid additional
                        interpolation.
                        '''
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
                                            iono_sideband=iono_sideband_bool)

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
                                    dem_raster, block_size)

                if cfg['processing']['dense_offsets']['enabled']:
                   desired = ['along_track_offset', 'slant_range_offset']
                   geocode_obj.data_interpolator = interp_method
                   radar_grid_offset = get_offset_radar_grid(cfg,
                                                             radar_grid_slc)

                   cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                       offset_pol_list, input_hdf5, dst_h5,
                                       radar_grid_offset, dem_raster,
                                       block_size)

                desired = ["layover_shadow_mask"]
                geocode_obj.data_interpolator = 'NEAREST'
                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    pol_list, input_hdf5, dst_h5,
                                    radar_grid_slc, dem_raster, block_size,
                                    scratch_path=scratch_path,
                                    compute_stats=False)
            elif input_product_type is InputProduct.ROFF:
                offset_cfg = cfg['processing']['offsets_product']
                desired = ['along_track_offset', 'slant_range_offset',
                           'along_track_offset_variance',
                           'correlation_surface_peak',
                           'cross_offset_variance', 'slant_range_offset',
                           'snr']
                layer_keys = [key for key in offset_cfg.keys() if
                              key.startswith('layer')]

                radar_grid = get_offset_radar_grid(cfg,
                                                   slc.getRadarGrid(freq))

                geocode_obj.data_interpolator = interp_method
                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    offset_pol_list, input_hdf5, dst_h5,
                                    radar_grid, dem_raster, block_size,
                                    off_layer_dict=layer_keys,
                                    input_product_type=InputProduct.ROFF)
            else:
                #RIFG
                # Geocode the coherence
                desired = ['coherence_magnitude']
                geocode_obj.data_interpolator = interp_method
                cpu_geocode_rasters(geocode_obj, geo_datasets, desired, freq,
                                    pol_list,input_hdf5, dst_h5, radar_grid,
                                    dem_raster, block_size,
                                    input_product_type=InputProduct.RIFG)

                # Geocode the wrapped interferogram
                desired = ['wrapped_interferogram']
                geocode_cplx_obj.data_interpolator = cfg["processing"]["geocode"]\
                        ['wrapped_interferogram']['interp_method']
                cpu_geocode_rasters(geocode_cplx_obj, geo_datasets, desired, freq,
                                    pol_list,input_hdf5, dst_h5, radar_grid,
                                    dem_raster, block_size * 2,
                                    input_product_type=InputProduct.RIFG)

            # add water mask to GUNW product
            add_water_mask(cfg, freq, geo_grid, dst_h5)

            # spec for NISAR GUNW does not require freq B so skip radar cube
            if freq.upper() == 'B':
                continue

            add_radar_grid_cube(cfg, freq, radar_grid, orbit, dst_h5, input_product_type)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geocode in {t_all_elapsed:.3f} seconds")

def gpu_geocode_rasters(geo_datasets, desired, freq, pol_list,
                        input_hdf5, dst_h5, gpu_geocode_obj,
                        off_layer_dict=None, scratch_path='',
                        compute_stats=True,
                        input_product_type=InputProduct.RUNW,
                        iono_sideband=False):
    geocoded_rasters, geocoded_datasets, input_rasters = \
        get_raster_lists(geo_datasets, desired, freq, pol_list, input_hdf5,
                         dst_h5, off_layer_dict, scratch_path, input_product_type,
                         iono_sideband)

    if input_rasters:
        gpu_geocode_obj.geocode_rasters(geocoded_rasters, input_rasters)

        if compute_stats:
            for raster, ds in zip(geocoded_rasters, geocoded_datasets):
                compute_stats_real_data(raster, ds)
            if input_product_type != InputProduct.ROFF:
                water_mask_ds = dst_h5['/science/LSAR/GUNW/grids/frequencyA/interferogram/waterMask']
                compute_water_mask_stats(water_mask_ds)
                lay_shadow_ds = dst_h5['/science/LSAR/GUNW/grids/frequencyA/interferogram/layoverShadowMask']
                compute_layover_shadow_stats(lay_shadow_ds)


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
    ref_orbit = cfg["dynamic_ancillary_file_group"]['orbit']['reference_orbit_file']
    freq_pols = cfg["processing"]["input_subset"]["list_of_frequencies"]
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    if input_product_type is InputProduct.RIFG:
        geogrids = cfg["processing"]["geocode"]["wrapped_igram_geogrids"]
    lines_per_block = cfg["processing"]["geocode"]["lines_per_block"]
    interp_method = cfg["processing"]["geocode"]["interp_method"]
    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])

    if input_product_type is InputProduct.ROFF:
        geo_datasets = cfg["processing"]["geocode"]["goff_datasets"]
    elif input_product_type is InputProduct.RUNW:
        geo_datasets = cfg["processing"]["geocode"]["gunw_datasets"]
    else:
        # RIFG
        geo_datasets = cfg["processing"]["geocode"]["wrapped_datasets"]

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

        if wrapped_igram_interp_method  == 'SINC':
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
        orbit = load_orbit_from_xml(ref_orbit)
    else:
        orbit = slc.getOrbit()

    with h5py.File(output_hdf5, "a", libver='latest', swmr=True) as dst_h5:

        get_ds_names = lambda ds_dict, desired: [
            x for x, y in ds_dict.items() if y and x in desired]

        for freq, pol_list, offset_pol_list in get_cfg_freq_pols(cfg):

            geogrid = geogrids[freq]

            # Create frequency based radar grid
            radar_grid = slc.getRadarGrid(freq)
            if az_looks > 1 or rg_looks > 1:
                # Multilook radar grid if needed
                radar_grid = radar_grid.multilook(az_looks, rg_looks)

            if input_product_type is InputProduct.RUNW:
                desired = ['coherence_magnitude', 'unwrapped_phase']
                # Create radar grid geometry used by most datasets
                rdr_geometry = isce3.container.RadarGeometry(radar_grid, orbit,
                                                             grid_zero_doppler)

                # Create geocode object other than offset and shadow layover datasets
                geocode_obj = isce3.cuda.geocode.Geocode(geogrid, rdr_geometry,
                                                         dem_raster,
                                                         lines_per_block,
                                                         interp_method,
                                                         invalid_value=np.nan)

                gpu_geocode_rasters(geo_datasets, desired, freq, pol_list,
                                    input_hdf5, dst_h5, geocode_obj)

                if iono_enabled:
                    desired = ['ionosphere_phase_screen',
                               'ionosphere_phase_screen_uncertainty']
                    geocode_iono_bool = True
                    pol_list_iono = freq_pols_iono[freq]
                    input_hdf5_iono = input_hdf5
                    if is_iono_method_sideband:
                        '''
                        ionosphere_phase_screen from main_side_band or
                        main_diff_ms_band are computed on radargrid of frequencyB.
                        The ionosphere_phase_screen is geocoded on geogrid of
                        frequencyA. Instead of geocoding ionosphere in the RUNW standard
                        product (frequencyA), geocode the frequencyB in ionosphere/RUNW.h5
                        to avoid additional interpolation.
                        '''
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
                            '''
                            The methods using sideband (e.g., main_side_band,
                            and main_ms_diff_band) produce only one
                            ionosphere from frequency A and B interferogram.
                            The ionosphere of radargrid (frequency B) is
                            geocoded only to geogrid in frequency A.
                            '''
                            geocode_iono_bool = False
                    else:
                        '''
                        The method using split_main_band produces
                        can have two ionosphere layers in A and B.
                        '''
                        iono_sideband_bool = False
                        iono_freq = freq
                        rdr_geometry_iono = rdr_geometry
                        if pol_list_iono == None:
                            geocode_iono_bool = False

                    if geocode_iono_bool:
                        geocode_iono_obj = \
                            isce3.cuda.geocode.Geocode(geogrid,
                                                    rdr_geometry_iono,
                                                    dem_raster,
                                                    lines_per_block,
                                                    interp_method,
                                                    invalid_value=np.nan)

                        gpu_geocode_rasters(geo_datasets, desired,
                                            iono_freq, pol_list_iono,
                                            input_hdf5_iono, dst_h5,
                                            geocode_iono_obj,
                                            iono_sideband=iono_sideband_bool)

                desired = ["connected_components"]
                '''
                connected_components raster has type unsigned char and an invalid
                value of NaN becomes 0 which conflicts with 0 being used to indicate
                an unmasked value/pixel. 255 is chosen as it is the most distant
                value from components assigned in ascending order [0, 1, ...)
                '''
                geocode_conn_comp_obj = \
                    isce3.cuda.geocode.Geocode(geogrid, rdr_geometry,
                                               dem_raster,
                                               lines_per_block,
                                               isce3.core.DataInterpMethod.NEAREST,
                                               invalid_value=255)

                gpu_geocode_rasters(geo_datasets, desired, freq, pol_list,
                                    input_hdf5, dst_h5, geocode_conn_comp_obj)
                if cfg['processing']['dense_offsets']['enabled']:
                   desired = ['along_track_offset', 'slant_range_offset']

                   # If needed create geocode object for offset datasets
                   # Create offset unique radar grid
                   radar_grid = get_offset_radar_grid(cfg,
                                                      slc.getRadarGrid(freq))

                   # Create radar grid geometry required by offset datasets
                   rdr_geometry = isce3.container.RadarGeometry(radar_grid, orbit,
                                                                grid_zero_doppler)

                   geocode_offset_obj = isce3.cuda.geocode.Geocode(geogrid,
                                                                   rdr_geometry,
                                                                   dem_raster,
                                                                   lines_per_block,
                                                                   interp_method,
                                                                   invalid_value=np.nan)
                   gpu_geocode_rasters(geo_datasets, desired, freq,
                                       offset_pol_list, input_hdf5, dst_h5,
                                       geocode_offset_obj),

                desired = ["layover_shadow_mask"]
                # If needed create geocode object for shadow layover dataset
                # Create radar grid geometry required by layover shadow
                rdr_geometry = isce3.container.RadarGeometry(slc.getRadarGrid(freq),
                                                             orbit,
                                                             grid_zero_doppler)

                '''
                layover shadow raster has type char and an invalid
                value of NaN becomes 0 which conflicts with 0 being used
                to indicate an unmasked value/pixel. 127 is chosen as it is
                the most distant value from the allowed set of [0, 1, 2, 3].
                '''
                geocode_shadow_obj = isce3.cuda.geocode.Geocode(geogrid,
                                                                rdr_geometry,
                                                                dem_raster,
                                                                lines_per_block,
                                                                isce3.core.DataInterpMethod.NEAREST,
                                                                invalid_value=127)
                gpu_geocode_rasters(geo_datasets, desired, freq, pol_list,
                                    input_hdf5, dst_h5, geocode_shadow_obj,
                                    scratch_path=scratch_path, compute_stats=False)
            elif input_product_type is InputProduct.ROFF:
                offset_cfg = cfg['processing']['offsets_product']
                desired=['along_track_offset', 'slant_range_offset',
                         'along_track_offset_variance',
                         'correlation_surface_peak',
                         'cross_offset_variance',
                         'slant_range_offset_variance', 'snr']
                layer_keys = [key for key in offset_cfg.keys() if
                              key.startswith('layer')]

                radar_grid = get_offset_radar_grid(cfg,
                                                   slc.getRadarGrid(freq))
                #  Create radar grid geometry required by offset datasets
                rdr_geometry = isce3.container.RadarGeometry(radar_grid,
                                                             orbit,
                                                             grid_zero_doppler)

                geocode_obj = isce3.cuda.geocode.Geocode(geogrid,
                                                         rdr_geometry,
                                                         dem_raster,
                                                         lines_per_block,
                                                         interp_method,
                                                         invalid_value=np.nan)

                gpu_geocode_rasters(geo_datasets, desired, freq, pol_list,
                                    input_hdf5,
                                    dst_h5, geocode_obj,
                                    off_layer_dict=layer_keys,
                                    input_product_type=InputProduct.ROFF)
            else:
                #RIFG
                desired = ['coherence_magnitude', 'wrapped_interferogram']
                interp_methods = [interp_method, wrapped_igram_interp_method]

                # Create radar grid geometry required by RIFG product
                rdr_geometry = isce3.container.RadarGeometry(radar_grid, orbit,
                                                             grid_zero_doppler)

                # Iterate over desired unwrapped datasets to account for
                # possible use of different interpolation methods
                for desired_ds, ds_interp_method in zip(desired,
                                                        interp_methods):
                    # Create geocode object
                    geocode_obj = isce3.cuda.geocode.Geocode(geogrid, rdr_geometry,
                                                             dem_raster,
                                                             lines_per_block,
                                                             data_interp_method=ds_interp_method,
                                                             invalid_value=np.nan)

                    # Geocode the coherence and wrapped interferogram
                    gpu_geocode_rasters(geo_datasets, [desired_ds], freq, pol_list,
                                        input_hdf5, dst_h5, geocode_obj,
                                        input_product_type = InputProduct.RIFG)

            # add water mask to GUNW product
            add_water_mask(cfg, freq, geogrid, dst_h5)

            # spec for NISAR GUNW does not require freq B so skip radar cube
            if freq.upper() == 'B':
                continue

            add_radar_grid_cube(cfg, freq, radar_grid, slc.getOrbit(), dst_h5, input_product_type)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geocode in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    """
    run geocode from command line
    """

    # load command line args
    geocode_insar_parser = YamlArgparse()
    args = geocode_insar_parser.parse()

    # Get a runconfig dictionary from command line args
    geocode_insar_runconfig = GeocodeInsarRunConfig(args)

    # prepare the HDF5
    geocode_insar_runconfig.cfg['primary_executable']['product_type'] = 'GUNW_STANDALONE'
    out_paths = h5_prep.run(geocode_insar_runconfig.cfg)
    runw_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'runw_path']
    if runw_path is not None:
        out_paths['RUNW'] = runw_path

    # Run geocode RUNW
    run(geocode_insar_runconfig.cfg, out_paths["RUNW"], out_paths["GUNW"], input_product_type=InputProduct.RUNW)

    rifg_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'rifg_path']
    if rifg_path is not None:
        out_paths['RIFG'] = rifg_path
    # Run geocode RIFG
    run(geocode_insar_runconfig.cfg, out_paths["RIFG"], out_paths["GUNW"], input_product_type=InputProduct.RIFG)

    # Check if need to geocode offset product
    enabled = geocode_insar_runconfig.cfg['processing']['offsets_product']['enabled']
    # Prepare the GOFF product
    if enabled:
        geocode_insar_runconfig.cfg['primary_executable']['product_type'] = 'GOFF'
        out_paths = h5_prep.run(geocode_insar_runconfig.cfg)
    roff_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'roff_path']
    if roff_path is not None:
        out_paths['ROFF'] = roff_path
    if enabled:
        run(geocode_insar_runconfig.cfg, out_paths['ROFF'],
            out_paths['GOFF'], InputProduct.ROFF)
