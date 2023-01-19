#!/usr/bin/env python3

'''
collection of functions for NISAR GCOV workflow
'''

import os
import tempfile
import time

from osgeo import gdal
import h5py
import journal
import numpy as np

import isce3
from nisar.types import complex32, read_complex_dataset
from nisar.products.readers import SLC
from nisar.workflows import h5_prep
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.gcov_runconfig import GCOVRunConfig
from nisar.workflows.h5_prep import set_get_geo_info
from nisar.products.readers.orbit import load_orbit_from_xml




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


def run(cfg):
    '''
    run GCOV
    '''

    # pull parameters from cfg
    input_hdf5 = cfg['input_file_group']['input_file_path']
    output_hdf5 = cfg['product_path_group']['sas_output_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    flag_fullcovariance = cfg['processing']['input_subset']['fullcovariance']
    flag_symmetrize_cross_pol_channels = \
        cfg['processing']['input_subset']['symmetrize_cross_pol_channels']
    scratch_path = cfg['product_path_group']['scratch_path']

    output_data_compression = cfg['processing']['output_data_compression']
    if not output_data_compression or output_data_compression.lower == 'none':
        output_data_compression = None
    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']

    # DEM parameters
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    dem_interp_method_enum = cfg['processing']['dem_interpolation_method_enum']

    orbit_file = cfg["dynamic_ancillary_file_group"]['orbit_file']

    # unpack geocode run parameters
    geocode_dict = cfg['processing']['geocode']
    geocode_algorithm = geocode_dict['algorithm_type']
    output_mode = geocode_dict['output_mode']
    flag_apply_rtc = geocode_dict['apply_rtc']
    flag_apply_valid_samples_sub_swath_masking = \
        geocode_dict['apply_valid_samples_sub_swath_masking']
    memory_mode = geocode_dict['memory_mode']
    geogrid_upsampling = geocode_dict['geogrid_upsampling']
    abs_cal_factor = geocode_dict['abs_rad_cal']
    clip_max = geocode_dict['clip_max']
    clip_min = geocode_dict['clip_min']
    geogrids = geocode_dict['geogrids']
    flag_upsample_radar_grid = geocode_dict['upsample_radargrid']
    flag_save_nlooks = geocode_dict['save_nlooks']
    flag_save_rtc = geocode_dict['save_rtc']
    flag_save_dem = geocode_dict['save_dem']
    min_block_size_mb = cfg["processing"]["geocode"]['min_block_size']
    max_block_size_mb = cfg["processing"]["geocode"]['max_block_size']

    # optional keyword arguments , i.e. arguments that may or may not be
    # included in the call to geocode()
    optional_geo_kwargs = {}

    # read min/max block size converting MB to B
    if min_block_size_mb is not None:
        optional_geo_kwargs['min_block_size'] = min_block_size_mb * (2**20)
    if max_block_size_mb is not None:
        optional_geo_kwargs['max_block_size'] = max_block_size_mb * (2**20)

    # unpack RTC run parameters
    rtc_dict = cfg['processing']['rtc']
    output_terrain_radiometry = rtc_dict['output_type']
    rtc_algorithm = rtc_dict['algorithm_type']
    input_terrain_radiometry = rtc_dict['input_terrain_radiometry']
    rtc_min_value_db = rtc_dict['rtc_min_value_db']
    rtc_upsampling = rtc_dict['dem_upsampling']

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
    elif input_terrain_radiometry == isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT:
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

    info_channel = journal.info("gcov.run")
    error_channel = journal.error("gcov.run")
    info_channel.log("starting geocode COV")

    t_all = time.time()
    for frequency in freq_pols.keys():

        t_freq = time.time()

        # get sub_swaths metadata
        if flag_apply_valid_samples_sub_swath_masking:
            sub_swaths = slc.getSwathMetadata(frequency).sub_swaths()
        else:
            sub_swaths = None

        # unpack frequency dependent parameters
        radar_grid = slc.getRadarGrid(frequency)
        if radar_grid_nlooks > 1:
            radar_grid = radar_grid.multilook(az_look, rg_look)
        geogrid = geogrids[frequency]
        input_pol_list = freq_pols[frequency]

        # do no processing if no polarizations specified for current frequency
        if not input_pol_list:
            continue

        # set dict of input rasters
        input_raster_dict = {}

        # `input_pol_list` is the input list of polarizations that may include
        # HV and VH. `pol_list` is the actual list of polarizations to be
        # geocoded. It may include HV but it will not include VH if the
        # polarimetric symmetrization is performed
        pol_list = input_pol_list
        for pol in pol_list:
            temp_ref = f'HDF5:"{input_hdf5}":/{slc.slcPath(frequency, pol)}'
            input_raster_dict[pol] = temp_ref

        # symmetrize cross-polarimetric channels (if applicable)
        if (flag_symmetrize_cross_pol_channels and
                'HV' in input_pol_list and
                'VH' in input_pol_list):

            # temporary file for the symmetrized HV polarization
            symmetrized_hv_temp = tempfile.NamedTemporaryFile(
                dir=scratch_path, suffix='.tif')

            # call symmetrization function
            info_channel.log('Symmetrizing polarization channels HV and VH')
            input_raster = prepare_rslc(
                input_hdf5, frequency, 'HV',
                symmetrized_hv_temp.name, 2**11,  # 2**11 = 2048 lines
                flag_rslc_to_backscatter=flag_rslc_to_backscatter,
                pol_2='VH', format="ENVI")

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
                    dir=scratch_path, suffix='.tif')
                input_raster = prepare_rslc(
                    input_hdf5, frequency, pol,
                    temp_pol_file.name, 2**12,  # 2**12 = 4096 lines
                    flag_rslc_to_backscatter=flag_rslc_to_backscatter,
                    format="ENVI")

            input_raster_list.append(input_raster)

        info_channel.log(f'Preparing multi-band raster for geocoding')

        # set paths temporary files
        input_temp = tempfile.NamedTemporaryFile(
            dir=scratch_path, suffix='.vrt')
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
            dir=scratch_path, suffix='.tif')

        output_raster_obj = isce3.io.Raster(temp_output.name,
                geogrid.width, geogrid.length,
                input_raster_obj.num_bands,
                gdal.GDT_Float32, 'GTiff')

        nbands_off_diag_terms = 0
        out_off_diag_terms_obj = None
        if flag_fullcovariance:
            nbands = input_raster_obj.num_bands
            nbands_off_diag_terms = (nbands**2 - nbands) // 2
            if nbands_off_diag_terms > 0:
                temp_off_diag = tempfile.NamedTemporaryFile(
                    dir=scratch_path, suffix='.tif')
                out_off_diag_terms_obj = isce3.io.Raster(
                    temp_off_diag.name,
                    geogrid.width, geogrid.length,
                    nbands_off_diag_terms,
                    gdal.GDT_CFloat32, 'GTiff')

        if flag_save_nlooks:
            temp_nlooks = tempfile.NamedTemporaryFile(
                dir=scratch_path, suffix='.tif')
            out_geo_nlooks_obj = isce3.io.Raster(
                temp_nlooks.name,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, "GTiff")
        else:
            temp_nlooks = None
            out_geo_nlooks_obj = None

        if flag_save_rtc:
            temp_rtc = tempfile.NamedTemporaryFile(
                dir=scratch_path, suffix='.tif')
            out_geo_rtc_obj = isce3.io.Raster(
                temp_rtc.name,
                geogrid.width, geogrid.length, 1,
                gdal.GDT_Float32, "GTiff")
        else:
            temp_rtc = None
            out_geo_rtc_obj = None

        if flag_save_dem:
            temp_interpolated_dem = tempfile.NamedTemporaryFile(
                dir=scratch_path, suffix='.tif')
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
                gdal.GDT_Float32, "GTiff")
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
                    out_geo_dem=out_geo_dem_obj,
                    input_rtc=None,
                    output_rtc=None,
                    sub_swaths=sub_swaths,
                    dem_interp_method=dem_interp_method_enum,
                    memory_mode=memory_mode,
                    **optional_geo_kwargs)

        del output_raster_obj

        if flag_save_nlooks:
            del out_geo_nlooks_obj

        if flag_save_rtc:
            del out_geo_rtc_obj

        if flag_save_dem:
            del out_geo_dem_obj

        if flag_fullcovariance:
            # out_off_diag_terms_obj.close_dataset()
            del out_off_diag_terms_obj

        with h5py.File(output_hdf5, 'a') as hdf5_obj:
            hdf5_obj.attrs['Conventions'] = np.string_("CF-1.8")
            root_ds = f'/science/LSAR/GCOV/grids/frequency{frequency}'

            h5_ds = os.path.join(root_ds, 'listOfPolarizations')
            if h5_ds in hdf5_obj:
                del hdf5_obj[h5_ds]
            pol_list_s2 = np.array(pol_list, dtype='S2')
            dset = hdf5_obj.create_dataset(h5_ds, data=pol_list_s2)
            dset.attrs['description'] = np.string_(
                'List of processed polarization layers with frequency ' +
                frequency)

            h5_ds = os.path.join(root_ds, 'radiometricTerrainCorrectionFlag')
            if h5_ds in hdf5_obj:
                del hdf5_obj[h5_ds]
            dset = hdf5_obj.create_dataset(h5_ds, data=bool(flag_apply_rtc))

            # save GCOV diagonal elements
            xds = hdf5_obj[os.path.join(root_ds, 'xCoordinates')]
            yds = hdf5_obj[os.path.join(root_ds, 'yCoordinates')]
            cov_elements_list = [p.upper()+p.upper() for p in pol_list]

            # save GCOV imagery
            _save_hdf5_dataset(temp_output.name, hdf5_obj, root_ds,
                               yds, xds, cov_elements_list,
                               output_data_compression=output_data_compression,
                               long_name=output_radiometry_str,
                               units='',
                               valid_min=clip_min,
                               valid_max=clip_max)

            # save listOfCovarianceTerms
            freq_group = hdf5_obj[root_ds]
            if not flag_fullcovariance:
                _save_list_cov_terms(cov_elements_list, freq_group)

            # save nlooks
            if flag_save_nlooks:
                _save_hdf5_dataset(temp_nlooks.name, hdf5_obj, root_ds,
                                   yds, xds, 'numberOfLooks',
                                   output_data_compression=output_data_compression,
                                   long_name = 'number of looks',
                                   units = '',
                                   valid_min = 0)

            # save rtc
            if flag_save_rtc:
                _save_hdf5_dataset(temp_rtc.name, hdf5_obj, root_ds,
                                   yds, xds, 'areaNormalizationFactor',
                                   output_data_compression=output_data_compression,
                                   long_name = 'RTC area factor',
                                   units = '',
                                   valid_min = 0)

            # save interpolated DEM
            if flag_save_dem:

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

                _save_hdf5_dataset(temp_interpolated_dem.name, hdf5_obj,
                                   root_ds, yds_dem, xds_dem,
                                   'interpolatedDem',
                                   output_data_compression=output_data_compression,
                                   long_name='Interpolated DEM',
                                   units='')

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
                _save_hdf5_dataset(temp_off_diag.name, hdf5_obj, root_ds,
                                   yds, xds, off_diag_terms_list,
                                   output_data_compression=output_data_compression,
                                   long_name = output_radiometry_str,
                                   units = '',
                                   valid_min = clip_min,
                                   valid_max = clip_max)

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
                                         cube_geogrid, radar_grid_cubes_heights,
                                         radar_grid, orbit, native_doppler,
                                         zero_doppler, threshold, maxiter)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran geocode COV in {t_all_elapsed:.3f} seconds")


def _save_list_cov_terms(cov_elements_list, dataset_group):

    name = "listOfCovarianceTerms"
    cov_elements_list.sort()
    cov_elements_array = np.array(cov_elements_list, dtype="S4")
    dset = dataset_group.create_dataset(name, data=cov_elements_array)
    desc = f"List of processed covariance terms"
    dset.attrs["description"] = np.string_(desc)


def _save_hdf5_dataset(ds_filename, h5py_obj, root_path,
                       yds, xds, ds_name,
                       output_data_compression=None,
                       standard_name=None, long_name=None, units=None,
                       fill_value=None, valid_min=None, valid_max=None,
                       compute_stats=True):
    '''
    write temporary raster file contents to HDF5

    Parameters
    ----------
    ds_filename : string
        source raster file
    h5py_obj : h5py object
        h5py object of destination HDF5
    root_path : string
        path of output raster data
    yds : h5py dataset object
        y-axis dataset
    xds : h5py dataset object
        x-axis dataset
    ds_name : string
        name of dataset to be added to root_path
    output_data_compression : str or None, optional
        Output imagery and secondary layers' compression
    standard_name : string, optional
    long_name : string, optional
    units : string, optional
    fill_value : float, optional
    valid_min : float, optional
    valid_max : float, optional
    '''
    if not os.path.isfile(ds_filename):
        return

    stats_real_imag_vector = None
    stats_vector = None
    if compute_stats:
        raster = isce3.io.Raster(ds_filename)

        if (raster.datatype() == gdal.GDT_CFloat32 or
                raster.datatype() == gdal.GDT_CFloat64):
            stats_real_imag_vector = \
                isce3.math.compute_raster_stats_real_imag(raster)
        elif raster.datatype() == gdal.GDT_Float64:
            stats_vector = isce3.math.compute_raster_stats_float64(raster)
        else:
            stats_vector = isce3.math.compute_raster_stats_float32(raster)

    compression_kwargs = {}
    if (output_data_compression == 'gzip9'):
        # maximum compression
        compression_kwargs['compression'] = 'gzip'
        # maximum compression
        compression_kwargs['compression_opts'] = 9
    elif output_data_compression is not None:
        compression_kwargs['compression'] = output_data_compression

    gdal_ds = gdal.Open(ds_filename)
    nbands = gdal_ds.RasterCount
    for band in range(nbands):
        data = gdal_ds.GetRasterBand(band+1).ReadAsArray()

        if isinstance(ds_name, str):
            h5_ds = os.path.join(root_path, ds_name)
        else:
            h5_ds = os.path.join(root_path, ds_name[band])

        if h5_ds in h5py_obj:
            del h5py_obj[h5_ds]

        dset = h5py_obj.create_dataset(h5_ds, data=data, **compression_kwargs)

        dset.dims[0].attach_scale(yds)
        dset.dims[1].attach_scale(xds)
        dset.attrs['grid_mapping'] = np.string_("projection")

        if standard_name is not None:
            dset.attrs['standard_name'] = np.string_(standard_name)

        if long_name is not None:
            dset.attrs['long_name'] = np.string_(long_name)

        if units is not None:
            dset.attrs['units'] = np.string_(units)

        if fill_value is not None:
            dset.attrs.create('_FillValue', data=fill_value)
        elif 'cfloat' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan + 1j * np.nan)
        elif 'float' in gdal.GetDataTypeName(raster.datatype()).lower():
            dset.attrs.create('_FillValue', data=np.nan)

        if stats_vector is not None:
            stats_obj = stats_vector[band]
            dset.attrs.create('min_value', data=stats_obj.min)
            dset.attrs.create('mean_value', data=stats_obj.mean)
            dset.attrs.create('max_value', data=stats_obj.max)
            dset.attrs.create('sample_standard_deviation', data=stats_obj.sample_stddev)

        elif stats_real_imag_vector is not None:

            stats_obj = stats_real_imag_vector[band]
            dset.attrs.create('min_real_value', data=stats_obj.real.min)
            dset.attrs.create('mean_real_value', data=stats_obj.real.mean)
            dset.attrs.create('max_real_value', data=stats_obj.real.max)
            dset.attrs.create('sample_standard_deviation_real',
                              data=stats_obj.real.sample_stddev)

            dset.attrs.create('min_imag_value', data=stats_obj.imag.min)
            dset.attrs.create('mean_imag_value', data=stats_obj.imag.mean)
            dset.attrs.create('max_imag_value', data=stats_obj.imag.max)
            dset.attrs.create('sample_standard_deviation_imag',
                              data=stats_obj.imag.sample_stddev)

        if valid_min is not None:
            dset.attrs.create('valid_min', data=valid_min)

        if valid_max is not None:
            dset.attrs.create('valid_max', data=valid_max)

    del gdal_ds


if __name__ == "__main__":
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    gcov_runcfg = GCOVRunConfig(args)
    h5_prep.run(gcov_runcfg.cfg)
    run(gcov_runcfg.cfg)
