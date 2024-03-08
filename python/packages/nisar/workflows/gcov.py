#!/usr/bin/env python3

'''
collection of functions for NISAR GCOV workflow
'''

import datetime
import os
import tempfile
import time
from osgeo import gdal
import h5py
import journal
import numpy as np

import isce3
from isce3.core.types import complex32, read_complex_dataset
from nisar.products.readers import SLC
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from isce3.atmosphere.tec_product import (tec_lut2d_from_json_srg,
                                          tec_lut2d_from_json_az)
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.gcov_runconfig import GCOVRunConfig
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.products.writers.BaseL2WriterSingleInput import (save_dataset,
                                                            get_file_extension)
from nisar.products.writers import GcovWriter


def read_rslc_backscatter(ds: h5py.Dataset, key=np.s_[...]):
    """
    Read a complex HDF5 dataset and return the square of its magnitude

    Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32 conversions
    which are about 10x faster than HDF5 ones.

    Parameters
    ----------
    ds: h5py.Dataset
        Path to RSLC HDF5 dataset
    key: numpy.s_
        Numpy slice to subset input dataset

    Returns
    -------
    numpy.ndarray(numpy.float32)
        RSLC backscatter as a numpy.ndarray(numpy.float32)
    """
    try:
        backscatter = np.absolute(ds[key][:]) ** 2
        return backscatter
    except TypeError:
        pass

    # This context manager handles h5py exception:
    # TypeError: data type '<c4' not understood
    data_block_c4 = ds.astype(complex32)[key]

    # backscatter = sqrt(r^2 + i^2) ^ 2
    #             = r^2 + i^2
    backscatter = data_block_c4['r'].astype(np.float32) ** 2
    backscatter += data_block_c4['i'].astype(np.float32) ** 2
    return backscatter


def prepare_rslc(in_file, freq, pol, out_file, lines_per_block,
                 flag_rslc_to_backscatter, pol_2=None, format="ENVI"):
    '''
    Copy RSLC dataset to GDAL format converting RSLC real and
    imaginary parts from float16 to float32. If the flag
    flag_rslc_to_backscatter is enabled, the 
    RSLC complex values are converted to radar backscatter (square of
    the RSLC magnitude): out = abs(RSLC)**2

    Optionally, the output raster can be created from the
    symmetrization of two polarimetric channels
    (`pol` and `pol_2`).

    Parameters
    ----------
    in_file: str
        Path to RSLC HDF5
    freq: str
        RSLC frequency band to process ('A' or 'B')
    pol: str
        RSLC polarization to process
    out_file: str
        Output filename
    lines_per_block: int
        Number of lines per block
    flag_rslc_to_backscatter: bool
        Indicates if the RSLC complex values should be
        convered to radar backscatter (square of
        the RSLC magnitude)
    pol_2: str, optional
        Polarization associated with the second RSLC
    format: str, optional
        GDAL-friendly format

    Returns
    -------
    isce3.io.Raster
        output raster object
    '''

    # open RSLC HDF5 file dataset
    rslc = SLC(hdf5file=in_file)
    hdf5_ds = rslc.getSlcDataset(freq, pol)

    # if provided, open RSLC HDF5 file dataset 2
    if pol_2:
        hdf5_ds_2 = rslc.getSlcDataset(freq, pol_2)

    # get RSLC dimension through GDAL
    gdal_ds = gdal.Open(f'HDF5:{in_file}:/{rslc.slcPath(freq, pol)}')
    rslc_length, rslc_width = gdal_ds.RasterYSize, gdal_ds.RasterXSize

    if flag_rslc_to_backscatter:
        gdal_dtype = gdal.GDT_Float32
    else:
        gdal_dtype = gdal.GDT_CFloat32

    # create output file
    driver = gdal.GetDriverByName(format)
    out_ds = driver.Create(out_file, rslc_width, rslc_length, 1, gdal_dtype)

    # start block processing
    lines_per_block = min(rslc_length, lines_per_block)
    num_blocks = int(np.ceil(rslc_length / lines_per_block))

    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = rslc_length - line_start
        else:
            block_length = lines_per_block

        # read a block of data from the RSLC
        if flag_rslc_to_backscatter:
            data_block = read_rslc_backscatter(
                hdf5_ds, np.s_[line_start:line_start + block_length, :])
        else:
            data_block = read_complex_dataset(
                hdf5_ds, np.s_[line_start:line_start + block_length, :])

        if pol_2:
            # compute average with a block of data from the second RSLC
            if flag_rslc_to_backscatter:
                data_block += read_rslc_backscatter(
                    hdf5_ds_2, np.s_[line_start:line_start + block_length, :])
            else:
                data_block += read_complex_dataset(
                    hdf5_ds_2, np.s_[line_start:line_start + block_length, :])
            data_block /= 2.0

        # write to GDAL raster
        out_ds.GetRasterBand(1).WriteArray(data_block, yoff=line_start, xoff=0)

    out_ds.FlushCache()
    return isce3.io.Raster(out_file)


def read_and_validate_rtc_anf_flags(geocode_dict, flag_apply_rtc):
    '''
    Read and validate radiometric terrain correction (RTC) area
    normalization factor (ANF) flags

    Parameters
    ----------
    geocode_dict: dict
        Runconfig geocode namespace
    flag_apply_rtc: bool
        Flag apply RTC (radiometric terrain correction)
    output_terrain_radiometry: isce3.geometry.RtcOutputTerrainRadiometry
        Output terrain radiometry (backscatter coefficient convention)

    Returns
    -------
    save_rtc_anf: bool
        Flag indicating whether the radiometric terrain correction (RTC)
        area normalization factor (ANF) layer should be created.
        This RTC ANF layer provides the conversion factor from
        from gamma0 backscatter normalization convention 
        to input backscatter normalization convention
        (e.g., beta0 or sigma0-ellipsoid)
    save_rtc_anf_gamma0_to_sigma0: bool
        Flag indicating whether the radiometric terrain correction (RTC)
        area normalization factor (ANF) gamma0 to sigma0 layer should be
        created
    '''

    error_channel = journal.error(
        "gcov.read_and_validate_rtc_anf_flags")

    save_rtc_anf = geocode_dict['save_rtc_anf']
    save_rtc_anf_gamma0_to_sigma0 = \
        geocode_dict['save_rtc_anf_gamma0_to_sigma0']

    if not flag_apply_rtc and save_rtc_anf:
        error_msg = (
            "the option `save_rtc_anf` is not available"
            " with radiometric terrain correction"
            " disabled (`apply_rtc = False`).")
        error_channel.log(error_msg)
        raise ValueError(error_msg)

    if not flag_apply_rtc and save_rtc_anf_gamma0_to_sigma0:
        error_msg = (
            "the option `save_rtc_anf_gamma0_to_sigma0`"
            " is not available with radiometric terrain"
            " correction disabled (`apply_rtc = False`).")
        error_channel.log(error_msg)
        raise ValueError(error_msg)

    return save_rtc_anf, save_rtc_anf_gamma0_to_sigma0


def run(cfg):
    '''
    run GCOV

    Parameters
    ----------
    cfg : dict
        Dictionary containing processing parameters

    Returns
    -------
    output_files_list: list(str)
        List of output files

    '''
    scratch_path = cfg['product_path_group']['scratch_path']
    with tempfile.TemporaryDirectory(dir=scratch_path) \
            as raster_scratch_dir:
        output_files_list = _run(cfg, raster_scratch_dir)
        return output_files_list


def _run(cfg, raster_scratch_dir):
    '''
    run GCOV with a scratch directory

    Parameters
    ----------
    cfg : dict
        Dictionary containing processing parameters
    raster_scratch_dir: str
        Scratch directory

    Returns
    -------
    output_files_list: list(str)
        List of output files

    '''
    info_channel = journal.info("gcov.run")
    error_channel = journal.error("gcov.run")
    info_channel.log("Starting GCOV workflow")

    # pull parameters from cfg
    input_hdf5 = cfg['input_file_group']['input_file_path']
    output_hdf5 = cfg['product_path_group']['sas_output_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    flag_fullcovariance = cfg['processing']['input_subset']['fullcovariance']
    flag_symmetrize_cross_pol_channels = \
        cfg['processing']['input_subset']['symmetrize_cross_pol_channels']

    output_gcov_terms_kwargs = cfg['output_gcov_terms']
    output_secondary_layers_kwargs = cfg['output_secondary_layers']

    # Raster files format (output of GeocodeCov).
    # Cannot use HDF5 because we cannot save multiband HDF5 datasets
    # simultaneously, therefore we use "ENVI" instead
    output_gcov_terms_raster_files_format = \
        output_gcov_terms_kwargs['format']
    if output_gcov_terms_raster_files_format == 'HDF5':
        output_gcov_terms_raster_files_format = 'ENVI'
    secondary_layer_files_raster_files_format = \
        output_secondary_layers_kwargs['format']
    if secondary_layer_files_raster_files_format == 'HDF5':
        secondary_layer_files_raster_files_format = 'ENVI'

    # Set raster file extensions for GCOV terms and secondary layers
    gcov_terms_file_extension = get_file_extension(
        output_gcov_terms_raster_files_format)
    secondary_layers_file_extension = get_file_extension(
        secondary_layer_files_raster_files_format)

    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']

    # DEM parameters
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    dem_interp_method_enum = cfg['processing']['dem_interpolation_method_enum']

    orbit_file = cfg["dynamic_ancillary_file_group"]['orbit_file']
    tec_file = cfg["dynamic_ancillary_file_group"]['tec_file']

    # unpack RTC run parameters
    rtc_dict = cfg['processing']['rtc']
    output_terrain_radiometry = rtc_dict['output_type_enum']
    rtc_algorithm = rtc_dict['algorithm_type_enum']
    input_terrain_radiometry = rtc_dict['input_terrain_radiometry_enum']
    rtc_min_value_db = rtc_dict['rtc_min_value_db']
    rtc_upsampling = rtc_dict['dem_upsampling']

    rtc_area_beta_mode = rtc_dict['area_beta_mode']
    if rtc_area_beta_mode == 'pixel_area':
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.PIXEL_AREA
    elif rtc_area_beta_mode == 'projection_angle':
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.PROJECTION_ANGLE
    elif (rtc_area_beta_mode == 'auto' or
            rtc_area_beta_mode is None):
        rtc_area_beta_mode_enum = \
            isce3.geometry.RtcAreaBetaMode.AUTO
    else:
        err_msg = ('ERROR invalid area beta mode:'
                   f' {rtc_area_beta_mode}')
        raise ValueError(err_msg)

    # unpack geocode run parameters
    geocode_dict = cfg['processing']['geocode']
    geocode_algorithm = geocode_dict['algorithm_type']
    output_mode = geocode_dict['output_mode']
    flag_apply_rtc = geocode_dict['apply_rtc']

    apply_range_ionospheric_delay_correction = \
        geocode_dict['apply_range_ionospheric_delay_correction']

    apply_azimuth_ionospheric_delay_correction = \
        geocode_dict['apply_azimuth_ionospheric_delay_correction']

    apply_valid_samples_sub_swath_masking = \
        geocode_dict['apply_valid_samples_sub_swath_masking']
    memory_mode = geocode_dict['memory_mode_enum']
    geogrid_upsampling = geocode_dict['geogrid_upsampling']
    abs_cal_factor = geocode_dict['abs_rad_cal']
    clip_max = geocode_dict['clip_max']
    clip_min = geocode_dict['clip_min']
    geogrids = geocode_dict['geogrids']
    flag_upsample_radar_grid = geocode_dict['upsample_radargrid']
    save_nlooks = geocode_dict['save_nlooks']
    save_rtc_anf, save_rtc_anf_gamma0_to_sigma0 = \
        read_and_validate_rtc_anf_flags(geocode_dict, flag_apply_rtc)
    save_dem = geocode_dict['save_dem']
    min_block_size_mb = cfg["processing"]["geocode"]['min_block_size']
    max_block_size_mb = cfg["processing"]["geocode"]['max_block_size']

    # optional keyword arguments , i.e. arguments that may or may not be
    # included in the call to geocode()
    optional_geo_kwargs = {}

    # declare list of output and temporary files
    output_files_list = [output_hdf5]

    output_dir = os.path.dirname(output_hdf5)
    output_gcov_terms_kwargs['output_dir'] = output_dir
    output_secondary_layers_kwargs['output_dir'] = output_dir
    output_gcov_terms_kwargs['output_files_list'] = output_files_list
    output_secondary_layers_kwargs['output_files_list'] = output_files_list

    # read min/max block size converting MB to B
    if min_block_size_mb is not None:
        optional_geo_kwargs['min_block_size'] = min_block_size_mb * (2**20)
    if max_block_size_mb is not None:
        optional_geo_kwargs['max_block_size'] = max_block_size_mb * (2**20)

    # unpack geo2rdr parameters
    geo2rdr_dict = cfg['processing']['geo2rdr']
    threshold = geo2rdr_dict['threshold']
    maxiter = geo2rdr_dict['maxiter']

    if (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT):
        output_radiometry_str = "radar backscatter sigma0"
    elif (flag_apply_rtc and output_terrain_radiometry ==
            isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT):
        output_radiometry_str = 'radar backscatter gamma0'
    elif input_terrain_radiometry == \
            isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT:
        output_radiometry_str = 'radar backscatter beta0'
    else:
        output_radiometry_str = 'radar backscatter sigma0'

    # unpack pre-processing
    preprocess = cfg['processing']['pre_process']
    rg_look = preprocess['range_looks']
    az_look = preprocess['azimuth_looks']
    radar_grid_nlooks = rg_look * az_look

    # init parameters shared between frequencyA and frequencyB sub-bands
    slc = SLC(hdf5file=input_hdf5)
    dem_raster = isce3.io.Raster(dem_file)
    zero_doppler = isce3.core.LUT2d()
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    if flag_fullcovariance:
        flag_rslc_to_backscatter = False
    else:
        flag_rslc_to_backscatter = True

    for frequency, input_pol_list in freq_pols.items():

        # do no processing if no polarizations specified for current frequency
        if not input_pol_list:
            continue

        t_freq = time.time()

        # get sub_swaths metadata
        if apply_valid_samples_sub_swath_masking:
            sub_swaths = slc.getSwathMetadata(frequency).sub_swaths()
        else:
            sub_swaths = None

        # unpack frequency dependent parameters
        radar_grid = slc.getRadarGrid(frequency)
        if radar_grid_nlooks > 1:
            radar_grid = radar_grid.multilook(az_look, rg_look)
        geogrid = geogrids[frequency]

        # set dict of input rasters
        input_raster_dict = {}

        # `input_pol_list` is the input list of polarizations that may include
        # HV and VH. `pol_list` is the actual list of polarizations to be
        # geocoded. It may include HV but it will not include VH if the
        # polarimetric symmetrization is performed
        pol_list = input_pol_list.copy()
        for pol in pol_list:
            temp_ref = f'HDF5:"{input_hdf5}":/{slc.slcPath(frequency, pol)}'
            input_raster_dict[pol] = temp_ref

        # symmetrize cross-polarimetric channels (if applicable)
        if (flag_symmetrize_cross_pol_channels and
                'HV' in input_pol_list and
                'VH' in input_pol_list):

            # temporary file for the symmetrized HV polarization
            symmetrized_hv_temp = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir, suffix=gcov_terms_file_extension)

            # call symmetrization function
            info_channel.log('Symmetrizing polarization channels HV and VH')
            input_raster = prepare_rslc(
                input_hdf5, frequency, 'HV',
                symmetrized_hv_temp.name, 2**11,  # 2**11 = 2048 lines
                flag_rslc_to_backscatter=flag_rslc_to_backscatter,
                pol_2='VH', format=output_gcov_terms_raster_files_format)

            # Since HV and VH were symmetrized into HV, remove VH from
            # `pol_list` and `from input_raster_dict`.
            pol_list.remove('VH')
            input_raster_dict.pop('VH')

            # Update `input_raster_dict` with the symmetrized polarization
            input_raster_dict['HV'] = input_raster

        # construct input rasters
        input_raster_list = []
        for pol, input_raster in input_raster_dict.items():

            # GDAL reference starting with HDF5 (RSLCs) are
            # converted to backscatter. This step is only
            # applied because h5py reads the NISAR C4 (4 bytes)
            # format faster than GDAL reads it. We take
            # advantage that we are reading the RLSC using
            # h5py to convert the data to backscatter (square)
            # and save it as float32.

            if isinstance(input_raster, str):
                info_channel.log(
                    f'Computing radar samples backscatter ({pol})')
                temp_pol_file = tempfile.NamedTemporaryFile(
                    dir=raster_scratch_dir,
                    suffix=gcov_terms_file_extension)
                input_raster = prepare_rslc(
                    input_hdf5, frequency, pol,
                    temp_pol_file.name, 2**12,  # 2**12 = 4096 lines
                    flag_rslc_to_backscatter=flag_rslc_to_backscatter,
                    format=output_gcov_terms_raster_files_format)

            input_raster_list.append(input_raster)

        info_channel.log('Preparing multi-band raster for geocoding')

        # set paths temporary files
        input_temp = tempfile.NamedTemporaryFile(
            dir=raster_scratch_dir, suffix='.vrt')
        input_raster_obj = isce3.io.Raster(
            input_temp.name, raster_list=input_raster_list)

        # init Geocode object depending on raster type
        if input_raster_obj.datatype() == gdal.GDT_Float32:
            geo = isce3.geocode.GeocodeFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_Float64:
            geo = isce3.geocode.GeocodeFloat64()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat32:
            geo = isce3.geocode.GeocodeCFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat64:
            geo = isce3.geocode.GeocodeCFloat64()
        else:
            err_str = 'Unsupported raster type for geocoding'
            error_channel.log(err_str)
            raise NotImplementedError(err_str)

        # if provided, load an external orbit from the runconfig file;
        # othewise, load the orbit from the RSLC metadata
        if orbit_file is not None:
            orbit = load_orbit_from_xml(orbit_file)
        else:
            orbit = slc.getOrbit()

        # get azimuth ionospheric delay LUTs (if applicable)
        center_freq = \
            slc.getSwathMetadata(frequency).processed_center_frequency

        if apply_azimuth_ionospheric_delay_correction:
            az_correction = tec_lut2d_from_json_az(tec_file, center_freq,
                                                   orbit, radar_grid)
            optional_geo_kwargs['az_time_correction'] = az_correction

        # get slant-range ionospheric delay LUTs (if applicable)
        if apply_range_ionospheric_delay_correction:
            rg_correction = tec_lut2d_from_json_srg(tec_file, center_freq,
                                                    orbit, radar_grid,
                                                    zero_doppler, dem_file)
            optional_geo_kwargs['slant_range_correction'] = rg_correction

        # init geocode members
        geo.orbit = orbit
        geo.ellipsoid = ellipsoid
        geo.doppler = zero_doppler
        geo.threshold_geo2rdr = threshold
        geo.numiter_geo2rdr = maxiter

        # set data interpolator based on the geocode algorithm
        if output_mode == isce3.geocode.GeocodeOutputMode.INTERP:
            geo.data_interpolator = geocode_algorithm

        geo.geogrid(geogrid.start_x, geogrid.start_y,
                    geogrid.spacing_x, geogrid.spacing_y,
                    geogrid.width, geogrid.length, geogrid.epsg)

        # create output raster
        temp_output = tempfile.NamedTemporaryFile(
            dir=raster_scratch_dir, suffix=gcov_terms_file_extension)

        output_raster_obj = isce3.io.Raster(
            temp_output.name,
            geogrid.width, geogrid.length,
            input_raster_obj.num_bands,
            gdal.GDT_Float32, output_gcov_terms_raster_files_format)

        nbands_off_diag_terms = 0
        out_off_diag_terms_obj = None
        if flag_fullcovariance:
            nbands = input_raster_obj.num_bands
            nbands_off_diag_terms = (nbands**2 - nbands) // 2
            if nbands_off_diag_terms > 0:
                temp_off_diag = tempfile.NamedTemporaryFile(
                    dir=raster_scratch_dir,
                    suffix=gcov_terms_file_extension)
                out_off_diag_terms_obj = isce3.io.Raster(
                    temp_off_diag.name,
                    geogrid.width, geogrid.length,
                    nbands_off_diag_terms,
                    gdal.GDT_CFloat32, output_gcov_terms_raster_files_format)

        if save_nlooks:
            temp_nlooks = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir,
                suffix=secondary_layers_file_extension)
            out_geo_nlooks_obj = isce3.io.Raster(
                temp_nlooks.name,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, secondary_layer_files_raster_files_format)
        else:
            temp_nlooks = None
            out_geo_nlooks_obj = None

        if save_rtc_anf:
            temp_rtc_anf = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir,
                suffix=secondary_layers_file_extension)
            out_geo_rtc_obj = isce3.io.Raster(
                temp_rtc_anf.name,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, secondary_layer_files_raster_files_format)
        else:
            temp_rtc_anf = None
            out_geo_rtc_obj = None

        if save_rtc_anf_gamma0_to_sigma0:
            temp_rtc_anf_gamma0_to_sigma0 = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir,
                suffix=secondary_layers_file_extension)
            out_geo_rtc_gamma0_to_sigma0_obj = isce3.io.Raster(
                temp_rtc_anf_gamma0_to_sigma0.name,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, secondary_layer_files_raster_files_format)
        else:
            temp_rtc_anf_gamma0_to_sigma0 = None
            out_geo_rtc_gamma0_to_sigma0_obj = None

        if save_dem:
            temp_interpolated_dem = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir,
                suffix=secondary_layers_file_extension)
            if (output_mode ==
                    isce3.geocode.GeocodeOutputMode.AREA_PROJECTION):
                interpolated_dem_width = geogrid.width + 1
                interpolated_dem_length = geogrid.length + 1
            else:
                interpolated_dem_width = geogrid.width
                interpolated_dem_length = geogrid.length
            out_geo_dem_obj = isce3.io.Raster(
                temp_interpolated_dem.name,
                interpolated_dem_width,
                interpolated_dem_length, 1,
                gdal.GDT_Float32, secondary_layer_files_raster_files_format)
        else:
            temp_interpolated_dem = None
            out_geo_dem_obj = None

        # geocode rasters
        geo.geocode(radar_grid=radar_grid,
                    input_raster=input_raster_obj,
                    output_raster=output_raster_obj,
                    dem_raster=dem_raster,
                    output_mode=output_mode,
                    geogrid_upsampling=geogrid_upsampling,
                    flag_apply_rtc=flag_apply_rtc,
                    input_terrain_radiometry=input_terrain_radiometry,
                    output_terrain_radiometry=output_terrain_radiometry,
                    rtc_min_value_db=rtc_min_value_db,
                    rtc_upsampling=rtc_upsampling,
                    rtc_algorithm=rtc_algorithm,
                    abs_cal_factor=abs_cal_factor,
                    flag_upsample_radar_grid=flag_upsample_radar_grid,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    radargrid_nlooks=radar_grid_nlooks,
                    out_off_diag_terms=out_off_diag_terms_obj,
                    out_geo_nlooks=out_geo_nlooks_obj,
                    out_geo_rtc=out_geo_rtc_obj,
                    rtc_area_beta_mode=rtc_area_beta_mode_enum,
                    out_geo_rtc_gamma0_to_sigma0=out_geo_rtc_gamma0_to_sigma0_obj,
                    out_geo_dem=out_geo_dem_obj,
                    input_rtc=None,
                    output_rtc=None,
                    sub_swaths=sub_swaths,
                    dem_interp_method=dem_interp_method_enum,
                    memory_mode=memory_mode,
                    **optional_geo_kwargs)

        del input_raster_obj
        del output_raster_obj

        if save_nlooks:
            del out_geo_nlooks_obj

        if save_rtc_anf:
            del out_geo_rtc_obj

        if save_rtc_anf_gamma0_to_sigma0:
            del out_geo_rtc_gamma0_to_sigma0_obj

        if save_dem:
            del out_geo_dem_obj

        if flag_fullcovariance:
            # out_off_diag_terms_obj.close_dataset()
            del out_off_diag_terms_obj

        with h5py.File(output_hdf5, 'a') as hdf5_obj:
            root_ds = f'/science/LSAR/GCOV/grids/frequency{frequency}'

            h5_ds = os.path.join(root_ds, 'listOfPolarizations')
            if h5_ds in hdf5_obj:
                del hdf5_obj[h5_ds]
            pol_list_s2 = np.array(pol_list, dtype='S2')
            dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
            dset.attrs['description'] = np.string_(
                'List of processed polarization layers with frequency ' +
                frequency)

            # save GCOV diagonal elements
            yds, xds = set_get_geo_info(hdf5_obj, root_ds, geogrid)
            cov_elements_list = [p.upper()+p.upper() for p in pol_list]

            output_gcov_terms_kwargs['output_file_prefix'] = \
                f'frequency{frequency}_'
            output_secondary_layers_kwargs['output_file_prefix'] = \
                f'frequency{frequency}_'

            # save GCOV imagery
            save_dataset(temp_output.name, hdf5_obj, root_ds,
                         yds, xds, cov_elements_list,
                         long_name=output_radiometry_str,
                         units='1',
                         valid_min=clip_min,
                         valid_max=clip_max,
                         **output_gcov_terms_kwargs)

            # save listOfCovarianceTerms
            freq_group = hdf5_obj[root_ds]
            if not flag_fullcovariance:
                _save_list_cov_terms(cov_elements_list, freq_group)

            # save nlooks
            if save_nlooks:
                save_dataset(temp_nlooks.name, hdf5_obj, root_ds,
                             yds, xds, 'numberOfLooks',
                             long_name='number of looks',
                             units='1',
                             valid_min=0,
                             **output_secondary_layers_kwargs)

            # save rtc
            if save_rtc_anf:
                save_dataset(temp_rtc_anf.name, hdf5_obj, root_ds,
                             yds, xds,
                             'rtcAreaNormalizationFactor',
                             long_name='RTC area factor',
                             units='1',
                             valid_min=0,
                             **output_secondary_layers_kwargs)

            # save rtc
            if save_rtc_anf_gamma0_to_sigma0:
                save_dataset(temp_rtc_anf_gamma0_to_sigma0.name,
                             hdf5_obj, root_ds,
                             yds, xds,
                             'rtcGammaToSigmaFactor',
                             long_name=('RTC Area Normalization Factor'
                                        'Gamma0 to Sigma0'),
                             units='1',
                             valid_min=0,
                             **output_secondary_layers_kwargs)

            # save interpolated DEM
            if save_dem:

                '''
                The DEM is interpolated over the geogrid pixels vertices
                rather than the pixels centers.
                '''
                if (output_mode ==
                        isce3.geocode.GeocodeOutputMode.AREA_PROJECTION):
                    dem_geogrid = isce3.product.GeoGridParameters(
                        start_x=geogrid.start_x - geogrid.spacing_x / 2,
                        start_y=geogrid.start_y - geogrid.spacing_y / 2,
                        spacing_x=geogrid.spacing_x,
                        spacing_y=geogrid.spacing_y,
                        width=int(geogrid.width) + 1,
                        length=int(geogrid.length) + 1,
                        epsg=geogrid.epsg)
                    yds_dem, xds_dem = \
                        set_get_geo_info(hdf5_obj, root_ds, dem_geogrid)
                else:
                    yds_dem = yds
                    xds_dem = xds

                save_dataset(temp_interpolated_dem.name, hdf5_obj,
                             root_ds, yds_dem, xds_dem,
                             'interpolatedDem',
                             long_name='Interpolated DEM',
                             units='1',
                             **output_secondary_layers_kwargs)

            # save GCOV off-diagonal elements
            if flag_fullcovariance:
                off_diag_terms_list = []
                for b1, p1 in enumerate(pol_list):
                    for b2, p2 in enumerate(pol_list):
                        if (b2 <= b1):
                            continue
                        off_diag_terms_list.append(p1.upper()+p2.upper())
                _save_list_cov_terms(cov_elements_list + off_diag_terms_list,
                                     freq_group)

                save_dataset(temp_off_diag.name, hdf5_obj, root_ds,
                             yds, xds, off_diag_terms_list,
                             long_name=output_radiometry_str,
                             units='1',
                             valid_min=clip_min,
                             valid_max=clip_max,
                             **output_gcov_terms_kwargs)

            t_freq_elapsed = time.time() - t_freq
            info_channel.log(f'frequency {frequency} ran in {t_freq_elapsed:.3f} seconds')

            if frequency.upper() == 'B' and 'A' in freq_pols:
                continue

            cube_geogrid = isce3.product.GeoGridParameters(
                start_x=radar_grid_cubes_geogrid.start_x,
                start_y=radar_grid_cubes_geogrid.start_y,
                spacing_x=radar_grid_cubes_geogrid.spacing_x,
                spacing_y=radar_grid_cubes_geogrid.spacing_y,
                width=int(radar_grid_cubes_geogrid.width),
                length=int(radar_grid_cubes_geogrid.length),
                epsg=radar_grid_cubes_geogrid.epsg)

            cube_group_name = '/science/LSAR/GCOV/metadata/radarGrid'
            native_doppler = slc.getDopplerCentroid()
            '''
            The native-Doppler LUT bounds error is turned off to
            computer cubes values outside radar-grid boundaries
            '''
            native_doppler.bounds_error = False
            add_radar_grid_cubes_to_hdf5(hdf5_obj, cube_group_name,
                                         cube_geogrid,
                                         radar_grid_cubes_heights,
                                         radar_grid, orbit, native_doppler,
                                         zero_doppler, threshold, maxiter)

    return output_files_list


def _save_list_cov_terms(cov_elements_list, dataset_group):

    name = "listOfCovarianceTerms"
    cov_elements_list.sort()
    cov_elements_array = np.array(cov_elements_list, dtype="S4")
    dset = dataset_group.create_dataset(name, data=cov_elements_array)
    desc = "List of processed covariance terms"
    dset.attrs["description"] = np.string_(desc)


if __name__ == "__main__":

    t_all = time.time()
    info_channel = journal.info("gcov.run")

    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    gcov_runconfig = GCOVRunConfig(args)

    sas_output_file = gcov_runconfig.cfg[
        'product_path_group']['sas_output_file']

    if os.path.isfile(sas_output_file):
        os.remove(sas_output_file)

    output_files_list = run(gcov_runconfig.cfg)

    with GcovWriter(runconfig=gcov_runconfig) as gcov_obj:
        gcov_obj.populate_metadata()

    info_channel.log('output file(s):')
    for filename in output_files_list:
        info_channel.log(f'    {filename}')

    t_all_elapsed = time.time() - t_all
    hms_str = str(datetime.timedelta(seconds=int(t_all_elapsed)))
    t_all_elapsed_str = f'elapsed time: {hms_str}s ({t_all_elapsed:.3f}s)'

    info_channel.log(t_all_elapsed_str)